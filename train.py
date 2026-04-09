import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from models.image_caption_model import ImageCaptioningModel
from preprocessing.vocabulary import Vocabulary, build_and_save_vocab
from preprocessing.data_loader import get_dataloader

def train():
    # 1. Hyperparameters
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 15
    batch_size = 32
    
    # 2. Paths
    BASE_DIR = Path(__file__).resolve().parent
    TENSORS_DIR = BASE_DIR / "dataset" / "Flickr8K" / "preprocessed_tensors"
    CAPTIONS_FILE = BASE_DIR / "dataset" / "Flickr8K" / "clean_captions.txt"
    VOCAB_FILE = BASE_DIR / "dataset" / "Flickr8K" / "vocab.pth"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 3. Device configuration
    assert torch.cuda.is_available(), "CUDA is requested but not available!"
    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    
    # 4. Vocabulary
    if not VOCAB_FILE.exists():
        print("Vocabulary not found. Building vocabulary...")
        if not CAPTIONS_FILE.exists():
            print(f"Error: Captions file not found at {CAPTIONS_FILE}")
            return
        vocab = build_and_save_vocab(CAPTIONS_FILE, VOCAB_FILE)
    else:
        print("Loading existing vocabulary...")
        vocab = Vocabulary.load_vocabulary(VOCAB_FILE)
        
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # 5. Build Dataloader
    print("Building DataLoader...")
    dataloader = get_dataloader(
        tensors_dir=TENSORS_DIR,
        captions_file=CAPTIONS_FILE,
        vocab=vocab,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 # Keep at 0 for Windows to avoid multiprocess issues unless configured carefully
    )
    
    if dataloader is None:
        print("Error: Could not initialize DataLoader.")
        return
        
    print(f"Total batches per epoch: {len(dataloader)}")
    
    # 6. Initialize Model, Loss, Optimizer
    print("Initializing Model...")
    # train_cnn=False means we freeze the ResNet-50 backbone. We only train the projection layer and the LSTM.
    model = ImageCaptioningModel(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        train_cnn=False 
    ).to(device)
    
    # We ignore the padding index when calculating loss
    pad_idx = vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    # Only update parameters that have requires_grad=True
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate
    )
    
    # 7. Training Loop
    print("\nStarting Training Strategy...")
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, captions) in progress_bar:
            images = images.to(device)
            captions = captions.to(device)
            
            # Forward pass
            outputs = model(images, captions)
            
            # The outputs are of shape (batch_size, seq_length, vocab_size)
            # The expected target is of shape (batch_size, seq_length)
            # For CrossEntropyLoss, we need outputs as (N, C) and targets as (N), so we reshape
            
            # We don't predict the first <start> token, so we align outputs and targets
            # Output shape: (batch_size, seq_length, vocab_size) -> (batch_size * seq_length, vocab_size)
            # Targets shape: (batch_size, seq_length) -> (batch_size * seq_length)
            outputs = outputs.view(-1, outputs.size(2))
            targets = captions.view(-1)
            
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients in RNN/LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = CHECKPOINT_DIR / f"model_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'vocab_size': vocab_size,
            'embed_size': embed_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train()
