# Assistive Vision System

An **Assistive Vision System** that generates rich, accurate image descriptions by combining a **Vision-Language Model (BLIP)** for caption generation with **YOLOv8** for object detection and grounding. The YOLO detections validate and refine the VLM's output, reducing hallucinations and improving caption reliability.

## Architecture

The system uses a dual-model approach for scene understanding:

- **Caption Generation (VLM — BLIP):**
  - Uses a pre-trained **BLIP** model (`Salesforce/blip-image-captioning-large`) to generate natural language captions directly from images.
  - No custom training required — the model is loaded from Hugging Face and runs inference out of the box.

- **Object Detection & Grounding (YOLOv8):**
  - Uses a pre-trained **YOLOv8** nano model (`yolov8n.pt`) to detect and locate objects in the scene.
  - Acts as a "ground truth" validator: detected objects are cross-referenced against the generated caption to catch hallucinations and improve accuracy.

- **Caption Optimizer:**
  - Scores caption candidates against YOLO detections using **synonym-aware matching** (e.g., YOLO's `"person"` matches `"man"`, `"woman"`, `"boy"` in captions).
  - Refines the final caption by removing hallucinated entities not supported by object detection.

```
Input Image
    │
    ├──► BLIP (VLM) ──► Caption: "a dog playing with a ball in the park"
    │
    └──► YOLOv8 ──► Detected: ["dog", "sports ball", "person"]
              │
              ▼
        CaptionOptimizer
        ├─ Validate (match caption words to detections)
        ├─ Refine (remove hallucinations)
        └─ Object-grounded final output
```

## Project Structure

```
Assistive Vision System/
│
├── models/                        # Model definitions & logic
│   ├── vlm.py                     # BLIP Vision-Language Model wrapper
│   ├── yolo_detector.py           # YOLOv8 object detection wrapper
│   ├── caption_optimizer.py       # Caption scoring, validation & refinement
│   ├── cnn_encoder.py             # (Legacy) ResNet-50 feature extractor
│   ├── lstm_decoder.py            # (Legacy) LSTM caption decoder
│   └── image_caption_model.py     # (Legacy) CNN-LSTM wrapper
│
├── preprocessing/                 # (Legacy) Data preprocessing
│   ├── preprocess_images.py       # Image transforms & tensor conversion
│   ├── vocabulary.py              # Token-to-index vocabulary builder
│   └── data_loader.py             # PyTorch dataset & dataloader
│
├── inference.py                   # Main entry point — VLM + YOLO inference
├── train.py                       # (Legacy) CNN-LSTM training script
├── load_captions.py               # (Legacy) Caption parser
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
└── README.md
```

> **Note:** Files marked **(Legacy)** are from the original CNN-LSTM pipeline. They are kept for reference but are no longer part of the active inference pipeline.

## Setup and Requirements

1. **Environment setup:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Required packages: `torch`, `torchvision`, `Pillow`, `ultralytics`, `transformers`

3. **First run:** The BLIP model weights (~1 GB) and YOLOv8 weights (~6 MB) are **automatically downloaded** on the first run via Hugging Face and Ultralytics respectively.

## Usage

### Basic Caption Generation (VLM only)
```bash
python inference.py --image path/to/your/image.jpg
```

### Caption with YOLO Object Grounding (recommended)
```bash
python inference.py --image path/to/your/image.jpg --detect_objects
```

### All Options
| Argument | Default | Description |
|---|---|---|
| `--image` | (required) | Path to the input image |
| `--detect_objects` | `false` | Enable YOLO-based validation and refinement |
| `--max_length` | `50` | Maximum caption length (tokens) |
| `--num_beams` | `5` | Beam search width for VLM |
| `--top_k` | `5` | Number of caption candidates for re-ranking |

### Example Output (with `--detect_objects`)
```
===========================================================
  YOLO Object Detection
===========================================================
  Detected: 1x dog, 1x person, 1x frisbee

===========================================================
  VLM Caption Candidates
===========================================================
  [1] ★ "a man throwing a frisbee to a dog"
       (alignment: 1.00)
  [2]   "a dog playing with a ball in the park"
       (alignment: 0.33)

===========================================================
  Validation Report
===========================================================
  ✓ person → matched "man"
  ✓ dog → matched "dog"
  ✓ frisbee → matched "frisbee"
  Caption Confidence: HIGH (3/3 objects matched)

===========================================================
  Final Caption: a man throwing a frisbee to a dog
===========================================================
```

## Features
- **Pre-trained VLM (BLIP):** No training required — generates high-quality captions out of the box.
- **YOLO Object Grounding:** Cross-validates captions against detected objects for reliability.
- **Synonym-Aware Matching:** Maps YOLO class names to natural language variants (person → man/woman/boy/girl).
- **Hallucination Removal:** Strips fabricated entities and actions not confirmed by object detection.
- **GPU Acceleration:** Uses CUDA automatically when available.
