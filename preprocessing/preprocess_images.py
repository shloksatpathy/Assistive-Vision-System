import os
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

def get_image_transforms():
    """
    Returns a composition of image transforms for preprocessing.
    Filters applied:
    1. Resizes to 244x244 (as specified)
    2. Converts Image to PyTorch Tensor
    3. Normalizes using standard ImageNet mean/std
    """
    transform_pipeline = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # Standard ImageNet mean
            std=[0.229, 0.224, 0.225]   # Standard ImageNet std
        )
    ])
    return transform_pipeline

class Flickr8KImageDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing Flickr8K images on the fly.
    This is generally preferred over saving thousands of .pt tensor files to the disk.
    """
    def __init__(self, images_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform if transform else get_image_transforms()
        
        # Get all image paths
        valid_exts = {'.jpg', '.jpeg', '.png'}
        self.image_paths = [p for p in self.images_dir.iterdir() if p.suffix.lower() in valid_exts]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path.name
        except Exception as e:
            print(f"Error loading image {img_path.name}: {e}")
            # Return None or handle the error as appropriate for your training loop
            return None, img_path.name

def process_and_save_all_images(source_dir, dest_dir):
    """
    Utility function if you explicitly want to process all images and save them as .pt files.
    Note: For large datasets like Flickr8K, this will take up a lot of disk space.
    """
    from tqdm import tqdm
    
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    transform = get_image_transforms()
    
    valid_exts = {'.jpg', '.jpeg', '.png'}
    image_paths = [p for p in source_dir.iterdir() if p.suffix.lower() in valid_exts]
    
    print(f"Found {len(image_paths)} images to process. Saving to {dest_dir}")
    
    for img_path in tqdm(image_paths, desc="Preprocessing images"):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            
            # Save the tensor as a .pt file
            tensor_filename = dest_dir / f"{img_path.stem}.pt"
            torch.save(img_tensor, tensor_filename)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

if __name__ == "__main__":
    # Define paths
    # Resolves to f:\Assistive Vision System
    BASE_DIR = Path(__file__).resolve().parent.parent 
    IMAGES_DIR = BASE_DIR / "dataset" / "Flickr8K" / "images"
    PREPROCESSED_DIR = BASE_DIR / "dataset" / "Flickr8K" / "preprocessed_tensors"
    
    print(f"Image directory set to: {IMAGES_DIR}")
    
    # 1. Option A: Use a Dataset (Recommended for on-the-fly preprocessing)
    print("\n--- Testing Dataset Approach ---")
    if IMAGES_DIR.exists():
        dataset = Flickr8KImageDataset(IMAGES_DIR)
        print(f"Total dataset size: {len(dataset)} images")
        
        if len(dataset) > 0:
            sample_tensor, sample_name = dataset[0]
            print(f"Sample Image Name: {sample_name}")
            print(f"Sample Tensor Shape: {sample_tensor.shape}")
            print(f"Sample Tensor Min Value: {sample_tensor.min():.4f}")
            print(f"Sample Tensor Max Value: {sample_tensor.max():.4f}")
    else:
        print(f"Dataset directory not found at {IMAGES_DIR}")

    # 2. Option B: Save to disk
    # Uncomment the lines below if you want to save preprocessed tensors to disk
    print("\n--- Processing and saving to disk ---")
    process_and_save_all_images(IMAGES_DIR, PREPROCESSED_DIR)
