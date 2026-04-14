"""
VLM (Vision-Language Model) Captioner Module.

Uses a pre-trained BLIP model from Hugging Face to generate 
high-quality image captions without any custom training.
"""

import torch
from PIL import Image
from pathlib import Path

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except ImportError:
    print("WARNING: transformers package not found. Install with: pip install transformers")
    BlipProcessor = None
    BlipForConditionalGeneration = None


class VLMCaptionGenerator:
    """
    Generates image captions using a pre-trained BLIP model.
    Uses 'Salesforce/blip-image-captioning-large' by default (~1GB).
    """

    def __init__(self, model_name="Salesforce/blip-image-captioning-large", device=None):
        """
        Loads the BLIP model and processor.

        Args:
            model_name (str): Hugging Face model identifier.
            device (str or torch.device): Device to run inference on.
        """
        if BlipProcessor is None or BlipForConditionalGeneration is None:
            raise ImportError(
                "Cannot initialize VLMCaptionGenerator because 'transformers' is not installed. "
                "Install it with: pip install transformers"
            )

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        print(f"Loading BLIP model: {model_name}...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"BLIP model loaded successfully on {self.device}.")

    def generate_caption(self, image_path, max_length=50, num_beams=4):
        """
        Generates a caption for a single image.

        Args:
            image_path (str or Path): Path to the image file.
            max_length (int): Maximum number of tokens in the generated caption.
            num_beams (int): Number of beams for beam search decoding.

        Returns:
            str: The generated caption text.
        """
        image_path = Path(image_path)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image '{image_path}': {e}")
            return None

        # Process the image through the BLIP processor
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Generate caption
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
            )

        # Decode the generated token IDs to text
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return caption

    def generate_captions_multiple(self, image_path, max_length=50, num_beams=5, num_return=5):
        """
        Generates multiple caption candidates for re-ranking.

        Args:
            image_path (str or Path): Path to the image file.
            max_length (int): Maximum token length per caption.
            num_beams (int): Number of beams for beam search.
            num_return (int): Number of caption sequences to return.

        Returns:
            list of str: Multiple caption candidates.
        """
        image_path = Path(image_path)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image '{image_path}': {e}")
            return []

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=min(num_return, num_beams),
            )

        captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [cap.strip() for cap in captions]