import argparse
import torch
from pathlib import Path

from models.vlm import VLMCaptionGenerator
from models.yolo_detector import YoloDetector
from models.caption_optimizer import CaptionOptimizer


def generate_vlm_caption(image_path, captioner, max_length=50, num_beams=4):
    """
    Generates a caption using the VLM (BLIP) model.
    """
    caption = captioner.generate_caption(image_path, max_length=max_length, num_beams=num_beams)
    return caption


def generate_optimized_caption(image_path, captioner, detector,
                                max_length=50, num_beams=5, top_k=5):
    """
    Generates an optimized caption by:
    1. Running YOLO object detection
    2. Generating multiple VLM caption candidates
    3. Scoring candidates against detected objects
    4. Refining the best caption to remove hallucinations
    """
    # Step 1: YOLO Object Detection
    print(f"\n{'=' * 55}")
    print(f"  YOLO Object Detection")
    print(f"{'=' * 55}")

    detected_objs = detector.detect_objects(image_path)

    if not detected_objs:
        print("  No objects detected by YOLO.")
        print("  Returning raw VLM caption.\n")
        return captioner.generate_caption(image_path, max_length=max_length, num_beams=num_beams)

    # Print detected objects summary
    class_counts = {}
    for obj in detected_objs:
        cls = obj['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1

    detected_summary = ", ".join(f"{count}x {cls}" for cls, count in class_counts.items())
    print(f"  Detected: {detected_summary}\n")

    # Step 2: Generate multiple VLM caption candidates
    print(f"{'=' * 55}")
    print(f"  VLM Caption Candidates")
    print(f"{'=' * 55}")

    captions = captioner.generate_captions_multiple(
        image_path, max_length=max_length,
        num_beams=num_beams, num_return=top_k
    )

    if not captions:
        print("  No captions generated.")
        return None

    # Step 3: Score and rank each candidate against YOLO detections
    optimizer = CaptionOptimizer(alpha=0.6, beta=0.4)

    scored_results = []
    for caption_text in captions:
        words = caption_text.lower().split()
        alignment_score, details = optimizer.score_caption(words, detected_objs)
        scored_results.append({
            "caption": caption_text,
            "words": words,
            "alignment_score": alignment_score,
            "details": details,
        })

    # Sort by alignment score (best match first)
    scored_results.sort(key=lambda x: x["alignment_score"], reverse=True)

    for i, result in enumerate(scored_results):
        star = "\u2605" if i == 0 else " "
        print(f"  [{i+1}] {star} \"{result['caption']}\"")
        print(f"       (alignment: {result['alignment_score']:.2f})")

    # Step 4: Validation report for the best caption
    best = scored_results[0]
    print(f"\n{'=' * 55}")
    print(f"  Validation Report")
    print(f"{'=' * 55}")

    report, confidence = optimizer.validate(best["words"], detected_objs)
    print(report)

    # Step 5: Refine — remove hallucinations, apply object grounding
    refined_caption = optimizer.refine_caption(best["caption"], detected_objs)

    if refined_caption != best["caption"]:
        print(f"\n  Refinement applied:")
        print(f"    Before: \"{best['caption']}\"")
        print(f"    After:  \"{refined_caption}\"")

    return refined_caption


def main():
    parser = argparse.ArgumentParser(
        description="Assistive Vision System — Generate captions with VLM + YOLO grounding."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum caption length (tokens).")
    parser.add_argument("--num_beams", type=int, default=5, help="Beam search width for VLM.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of caption candidates for re-ranking.")
    parser.add_argument("--detect_objects", action="store_true",
                        help="Run YOLOv8 detection to validate and optimize the caption.")
    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found at {image_path}")
        return

    # Load VLM captioner
    try:
        captioner = VLMCaptionGenerator(device=device)
    except ImportError as e:
        print(f"Error: {e}")
        return

    if args.detect_objects:
        # ── Optimized flow: VLM + YOLO grounding ──
        print(f"\nGenerating optimized caption for {image_path.name}...")
        try:
            detector = YoloDetector(device=device)
            caption = generate_optimized_caption(
                image_path, captioner, detector,
                max_length=args.max_length, num_beams=args.num_beams, top_k=args.top_k
            )
        except ImportError as e:
            print(f"YOLO error: {e}")
            print("Falling back to VLM-only caption.")
            caption = generate_vlm_caption(image_path, captioner,
                                           max_length=args.max_length, num_beams=args.num_beams)
        except Exception as e:
            print(f"Error during optimized generation: {e}")
            print("Falling back to VLM-only caption.")
            caption = generate_vlm_caption(image_path, captioner,
                                           max_length=args.max_length, num_beams=args.num_beams)
    else:
        # ── Standard flow: VLM caption only ──
        print(f"\nGenerating caption for {image_path.name}...")
        caption = generate_vlm_caption(image_path, captioner,
                                       max_length=args.max_length, num_beams=args.num_beams)

    if caption is not None:
        print(f"\n{'=' * 55}")
        print(f"  Final Caption: {caption}")
        print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
