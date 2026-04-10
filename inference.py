import argparse
import torch
from pathlib import Path
from PIL import Image

from models.image_caption_model import ImageCaptioningModel
from preprocessing.vocabulary import Vocabulary
from preprocessing.preprocess_images import get_image_transforms

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
    Generates a caption for a given image.
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

def main():
    parser = argparse.ArgumentParser(description="Generate a caption for an image using a trained model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.pth).")
    parser.add_argument("--vocab", type=str, required=True, help="Path to the vocabulary file (.pth).")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the generated caption.")
    parser.add_argument("--beam_size", type=int, default=3, help="Beam size for beam search. Set to 1 for greedy decoding.")
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
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
        
    model = load_model(checkpoint_path, len(vocab), device)
    print("Model loaded successfully.")
    
    # Generate caption
    image_path = Path(args.image)
    if not image_path.exists():
         print(f"Error: Image file not found at {image_path}")
         return
         
    print(f"\nGenerating caption for {image_path.name} (Beam Size: {args.beam_size})...")
    caption = generate_caption(image_path, model, vocab, device, max_length=args.max_length, beam_size=args.beam_size)
    
    if caption is not None:
        print(f"\nPredicted Caption: {caption}\n")

if __name__ == "__main__":
    main()
