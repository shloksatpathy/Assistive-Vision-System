import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

try:
    from preprocessing.vocabulary import Vocabulary
except ImportError:
    from vocabulary import Vocabulary

class PrecomputedTensorDataset(Dataset):
    """
    Dataset for loading precomputed image tensors (.pt files) and their corresponding captions.
    Each image may have multiple captions (typically 5), so we flatten the dataset 
    so that each sample is one (image_tensor, single_caption) pair.
    """
    def __init__(self, tensors_dir, captions_file=None, vocab=None):
        self.tensors_dir = Path(tensors_dir)
        self.vocab = vocab
        
        # Collect all .pt files
        if self.tensors_dir.exists():
            tensor_files = sorted([p for p in self.tensors_dir.iterdir() if p.suffix == '.pt'])
        else:
            tensor_files = []
        
        self.captions = {}
        if captions_file and os.path.exists(captions_file):
            with open(captions_file, 'r', encoding='utf-8') as f:
                self.captions = json.load(f)
                
        # Flatten dataset: pair each caption with its corresponding image tensor
        self.dataset_samples = []
        for tensor_path in tensor_files:
            image_filename = f"{tensor_path.stem}.jpg"
            if self.captions and image_filename in self.captions:
                for caption in self.captions[image_filename]:
                    self.dataset_samples.append((tensor_path, caption))
            else:
                # If no captions, use filename (for testing)
                self.dataset_samples.append((tensor_path, image_filename))
        
    def __len__(self):
        return len(self.dataset_samples)
        
    def __getitem__(self, idx):
        tensor_path, caption = self.dataset_samples[idx]
        
        # Load the precomputed PyTorch tensor
        image_tensor = torch.load(tensor_path, weights_only=True) 
        
        # Tokenize caption if vocab is provided
        if self.vocab and isinstance(caption, str) and self.captions:
            numericalized_caption = self.vocab.numericalize(caption)
            caption = torch.tensor(numericalized_caption)
            
        return image_tensor, caption

class CaptionCollate:
    """
    Collate function to pad variable length captions to the same length in a batch.
    """
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        # Sort batch by caption length (descending) - often useful for RNNs
        batch.sort(key=lambda x: len(x[1]) if isinstance(x[1], torch.Tensor) else 0, reverse=True)
        
        # Separate images and captions
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        
        captions = [item[1] for item in batch]
        
        # If captions are tensors, pad them
        if isinstance(captions[0], torch.Tensor):
            # pad_sequence pads to the max length in the batch along the specified dimension
            # batch_first=True makes output shape (batch_size, max_seq_length)
            captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
            
        return images, captions

def get_dataloader(tensors_dir, captions_file=None, vocab=None, batch_size=32, shuffle=True, num_workers=0):
    """
    Creates and returns a PyTorch DataLoader for the precomputed tensors with padded captions.
    """
    dataset = PrecomputedTensorDataset(tensors_dir, captions_file, vocab)
    
    if len(dataset) == 0:
        print(f"Warning: No valid samples found.")
        return None

    # Use pad_idx 0 if vocab is present
    pad_idx = vocab.stoi["<pad>"] if vocab else 0
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        collate_fn=CaptionCollate(pad_idx=pad_idx)
    )
    return dataloader

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent 
    TENSORS_DIR = BASE_DIR / "dataset" / "Flickr30K" / "preprocessed_tensors"
    CAPTIONS_FILE = BASE_DIR / "dataset" / "Flickr30K" / "clean_captions.txt"
    VOCAB_FILE = BASE_DIR / "dataset" / "Flickr30K" / "vocab.pth"
    
    print(f"Tensors Directory: {TENSORS_DIR}")
    
    # Load vocabulary if it exists
    vocab = None
    if VOCAB_FILE.exists():
        vocab = Vocabulary.load_vocabulary(VOCAB_FILE)
        print(f"Loaded Vocabulary with {len(vocab)} words.")
    else:
        print("Warning: vocab.pth not found. Captions will not be tokenized/padded correctly.")
        
    if TENSORS_DIR.exists():
        dataloader = get_dataloader(
            tensors_dir=TENSORS_DIR, 
            captions_file=CAPTIONS_FILE, 
            vocab=vocab,
            batch_size=4, 
            shuffle=True
        )
        
        if dataloader:
            print(f"\nDataLoader initialized successfully. Total batches: {len(dataloader)}")
            
            for batch_idx, (images, captions) in enumerate(dataloader):
                print(f"\nSample Batch {batch_idx + 1}")
                print(f"Batch Images shape: {images.shape}")
                print(f"Batch Captions shape: {captions.shape}")
                
                # Show first decoded caption
                if vocab and isinstance(captions, torch.Tensor):
                    caption_indices = captions[0].tolist()
                    decoded_words = [vocab.itos[idx] for idx in caption_indices]
                    print(f"Sample decoded caption: {' '.join(decoded_words)}")
                break
    else:
        print(f"Tensors directory does not exist.")
