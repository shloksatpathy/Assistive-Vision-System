import csv
import json
from collections import defaultdict
import os

def load_captions(file_path):
    """
    Loads captions from a pipe-separated file.
    
    Args:
        file_path (str): The path to the captions.txt file.
        
    Returns:
        dict: A dictionary mapping image_name to a list of caption_text.
    """
    captions_dict = defaultdict(list)
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return captions_dict

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        # Skip the header
        header = next(reader, None)
        
        for row in reader:
            if len(row) == 3:
                image_name, _, caption_text = row
                cleaned_caption = f"<start> {caption_text.strip().lower()} <end>"
                captions_dict[image_name].append(cleaned_caption)
                
    return dict(captions_dict)

if __name__ == "__main__":
    # Example usage:
    # Assumes the script is run from the workspace root where the dataset/ folder is relative to it
    captions_path = os.path.join(os.path.dirname(__file__), "dataset", "Flickr8K", "captions.txt")
    
    print(f"Loading captions from {captions_path}...")
    captions = load_captions(captions_path)
    
    print(f"Loaded captions for {len(captions)} images.")
    
    # Print the first item to verify
    if captions:
        first_image = list(captions.keys())[0]
        print(f"\nCaptions for {first_image}:")
        for i, cap in enumerate(captions[first_image]):
            print(f"  {i}: {cap}")
            
    output_path = os.path.join(os.path.dirname(__file__), "dataset", "Flickr8K", "clean_captions.txt")
    print(f"\nSaving clean captions to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(captions, f, indent=4)
    print("Clean captions saved successfully!")
