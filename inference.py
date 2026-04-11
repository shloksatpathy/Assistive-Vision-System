import argparse
import torch
from pathlib import Path
from PIL import Image

from models.image_caption_model import ImageCaptioningModel
from preprocessing.vocabulary import Vocabulary
from preprocessing.preprocess_images import get_image_transforms
from models.yolo_detector import YoloDetector
from models.caption_optimizer import CaptionOptimizer

def load_model(checkpoint_path, vocab_size, device):
    """
    Loads the model weights from a checkpoint.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    # Fix for newer PyTorch version requiring weights_only=True for safe loading
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    embed_size = checkpoint.get('embed_size', 256)
    hidden_size = checkpoint.get('hidden_size', 512)
    num_layers = checkpoint.get('num_layers', 1)
    
    model = ImageCaptioningModel(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        train_cnn=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def generate_caption(image_path, model, vocab, device, max_length=20, beam_size=3):
    """
    Generates a single caption for a given image (standard flow).
    """
    transform = get_image_transforms()
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image '{image_path}': {e}")
        return None
    
    # Preprocess image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate indices
    predicted_indices = model.generate_caption(image_tensor, vocab, max_length, beam_size=beam_size)
    
    # Convert indices to words
    words = []
    for idx in predicted_indices:
        word = vocab.itos[idx]
        if word == "<end>":
            break
        if word not in ["<start>", "<pad>"]:
            words.append(word)
            
    caption = " ".join(words)
    return caption

def generate_optimized_caption(image_path, model, vocab, device, detector, 
                                max_length=20, beam_size=5, top_k=5):
    """
    Generates an optimized caption by:
    1. Running YOLO object detection
    2. Generating top-K caption candidates via beam search
    3. Re-ranking candidates based on alignment with detected objects
    4. Returning the best result with a validation report
    """
    transform = get_image_transforms()
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image '{image_path}': {e}")
        return None
    
    # Step 1: YOLO Object Detection
    print(f"\n{'=' * 50}")
    print(f"  YOLO Object Detection")
    print(f"{'=' * 50}")
    
    detected_objs = detector.detect_objects(image_path)
    
    if not detected_objs:
        print("  No objects detected by YOLO.")
        print("  Falling back to standard caption generation.\n")
        # Fall back to standard captioning
        image_tensor = transform(image).unsqueeze(0).to(device)
        predicted_indices = model.generate_caption(image_tensor, vocab, max_length, beam_size=beam_size)
        words = []
        for idx in predicted_indices:
            word = vocab.itos[idx]
            if word == "<end>":
                break
            if word not in ["<start>", "<pad>"]:
                words.append(word)
        return " ".join(words)
    
    # Print detected objects summary
    class_counts = {}
    for obj in detected_objs:
        cls = obj['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    detected_summary = ", ".join(f"{count}x {cls}" for cls, count in class_counts.items())
    print(f"  Detected: {detected_summary}\n")
    
    # Step 2: Generate top-K caption candidates
    image_tensor = transform(image).unsqueeze(0).to(device)
    candidates = model.generate_captions_top_k(
        image_tensor, vocab, max_length=max_length, 
        beam_size=beam_size, top_k=top_k
    )
    
    if not candidates:
        print("  No caption candidates generated.")
        return None
    
    # Step 3: Re-rank with CaptionOptimizer
    optimizer = CaptionOptimizer(alpha=0.6, beta=0.4)
    ranked_results = optimizer.optimize(candidates, detected_objs, vocab)
    
    # Step 4: Display results
    print(f"{'=' * 50}")
    print(f"  Caption Candidates (re-ranked by object alignment)")
    print(f"{'=' * 50}")
    
    for i, result in enumerate(ranked_results):
        star = "\u2605" if i == 0 else " "
        print(f"  [{i+1}] {star} \"{result['caption']}\"")
        print(f"       (combined: {result['combined_score']:.2f} | "
              f"beam: {result['norm_beam_score']:.2f} | "
              f"alignment: {result['alignment_score']:.2f})")
    
    # Step 5: Validation report for the best caption
    best = ranked_results[0]
    print(f"\n{'=' * 50}")
    print(f"  Validation Report")
    print(f"{'=' * 50}")
    
    report, confidence = optimizer.validate(best["words"], detected_objs)
    print(report)
    
    # Step 6: Refine the caption — remove hallucinations, apply object grounding
    refined_caption = optimizer.refine_caption(best["caption"], detected_objs)
    
    if refined_caption != best["caption"]:
        print(f"\n  Refinement applied:")
        print(f"    Before: \"{best['caption']}\"")
        print(f"    After:  \"{refined_caption}\"")
    
    return refined_caption


def main():
    parser = argparse.ArgumentParser(description="Generate a caption for an image using a trained model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.pth).")
    parser.add_argument("--vocab", type=str, required=True, help="Path to the vocabulary file (.pth).")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the generated caption.")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of caption candidates to consider for re-ranking.")
    parser.add_argument("--detect_objects", action="store_true", 
                        help="Run YOLOv8 detection to validate and optimize the generated caption.")
    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load vocabulary
    vocab_path = Path(args.vocab)
    if not vocab_path.exists():
        print(f"Error: Vocabulary file not found at {vocab_path}")
        return
    vocab = Vocabulary.load_vocabulary(vocab_path)
    print(f"Vocabulary loaded. Size: {len(vocab)}")
    
    # Load captioning model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
        
    model = load_model(checkpoint_path, len(vocab), device)
    print("Captioning model loaded successfully.")
    
    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
         print(f"Error: Image file not found at {image_path}")
         return
    
    if args.detect_objects:
        # ── Optimized flow: YOLO + Caption re-ranking ──
        print(f"\nGenerating optimized caption for {image_path.name}...")
        try:
            detector = YoloDetector(device=device)
            caption = generate_optimized_caption(
                image_path, model, vocab, device, detector,
                max_length=args.max_length, beam_size=args.beam_size, top_k=args.top_k
            )
        except ImportError as e:
            print(f"Error: {e}")
            print("Falling back to standard caption generation.")
            caption = generate_caption(image_path, model, vocab, device,
                                       max_length=args.max_length, beam_size=args.beam_size)
        except Exception as e:
            print(f"An error occurred during optimized generation: {e}")
            print("Falling back to standard caption generation.")
            caption = generate_caption(image_path, model, vocab, device,
                                       max_length=args.max_length, beam_size=args.beam_size)
    else:
        # ── Standard flow: Caption only ──
        print(f"\nGenerating caption for {image_path.name} (Beam Size: {args.beam_size})...")
        caption = generate_caption(image_path, model, vocab, device, 
                                   max_length=args.max_length, beam_size=args.beam_size)
    
    if caption is not None:
        print(f"\n{'=' * 50}")
        print(f"  Final Caption: {caption}")
        print(f"{'=' * 50}\n")

if __name__ == "__main__":
    main()
