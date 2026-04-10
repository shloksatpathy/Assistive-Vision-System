# Assistive Vision System

This project is an **Image Captioning System** built using PyTorch. It aims to act as an assistive vision tool by automatically generating descriptive captions for images. The model utilizes a CNN (ResNet-50) architecture as an encoder to extract image features and an LSTM network as a decoder to generate sequential text captions.

## Architecture

The system relies on an Encoder-Decoder architecture:
- **Encoder (CNN):** A pre-trained ResNet-50 model is used to extract fixed-size feature vectors from images. The weights of the CNN backbone are frozen during training, and only a projection layer is learned.
- **Decoder (LSTM):** An LSTM-based Recurrent Neural Network takes the CNN features and generating sequence tokens (captions) word by word, conditioned on the previously generated words.

## Project Structure

```
Assistive Vision System/
│
├── dataset/                    # Data directory (Flickr8K default)
│   └── Flickr8K/
│       ├── images/             # Raw image files (e.g., .jpg)
│       └── captions.txt        # Raw pipe-separated captions
│
├── preprocessing/              # Data parsing and transformation
│   ├── preprocess_images.py    # Resizes (244x244), normalizes & saves image tensors
│   ├── vocabulary.py           # Builds mapping of tokens to numerical indices
│   └── data_loader.py          # PyTorch dataset & data loader implementation
│
├── models/                     # PyTorch neural network definitions
│   ├── cnn_encoder.py          # ResNet-50 feature extractor
│   ├── lstm_decoder.py         # LSTM caption generation model
│   └── image_caption_model.py  # End-to-end Encoder-Decoder wrapper
│
├── checkpoints/                # Saved model weights during training
│
├── load_captions.py            # Parses raw captions text into clean JSON structure
└── train.py                    # Main training loop for the Image Captioning Model
```

## Setup and Requirements

1. **Environment setup:** Ensure you have Python 3.8+ installed. It's recommended to work within a virtual environment.
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   ```

2. **Dependencies:** Make sure you have the following essential packages installed (via `pip`):
   - `torch`, `torchvision`, `torchaudio`
   - `Pillow`
   - `tqdm`

3. **Dataset (Flickr8K):**
   - Download the Flickr8K dataset images and place them inside `dataset/Flickr8K/images/`.
   - Download the corresponding `captions.txt` and place it at `dataset/Flickr8K/captions.txt`.

## Getting Started

### 1. Process the Captions
Raw captions contain metadata (image names, caption number) which are pipe-separated (`|`). Run the following script to create a structured mapping.
```bash
python load_captions.py
```
This generates `clean_captions.txt` (in JSON format) inside the `dataset/Flickr8K/` output folder.

### 2. Preprocess the Images
The dataset requires image normalization and resizing before they are fed into the model. Converting all images beforehand to `.pt` files reduces data-loading overhead heavily during training.
```bash
python preprocessing/preprocess_images.py
```
*Note: This creates a `preprocessed_tensors/` directory.*

### 3. Training the Model
With the data ready, you can start training. The `train.py` script automatically handles loading the dataset, building the vocabulary (if unavailable as `vocab.pth`), initializing the Model + Optimizer + Loss algorithm (CrossEntropy).

```bash
python train.py
```

The script trains for a default 15 epochs, saving model checkpoints like `model_epoch_1.pth` iteratively inside the `checkpoints/` directory.

### 4. Generating Captions (Inference)
Once the model is trained, you can use the `inference.py` script to generate a caption for any new image.

```bash
python inference.py --image path/to/your/image.jpg --checkpoint checkpoints/model_epoch_15.pth --vocab dataset/Flickr8K/vocab.pth
```
Optional arguments:
- `--max_length`: Maximum number of words for the generated caption (default: 20).

## Features
- **GPU Acceleration support:** Will execute using CUDA out of the box if available. 
- **Vocabulary Extraction:** Generates an indexable mapping dropping rare words to limit dimension sizes.
- **Gradient Clipping:** Prevents the exploding gradient problem in LSTM layers.
