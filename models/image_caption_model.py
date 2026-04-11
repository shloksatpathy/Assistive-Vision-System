import torch
import torch.nn as nn

# Robust importing to handle running this script directly or from the root directory
try:
    from models.cnn_encoder import CNNEncoder
    from models.lstm_decoder import LSTMDecoder
except ImportError:
    from cnn_encoder import CNNEncoder
    from lstm_decoder import LSTMDecoder

class ImageCaptioningModel(nn.Module):
    """
    End-to-end Image Captioning Model connecting the CNN Encoder and LSTM Decoder.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, max_seq_length=20, train_cnn=False):
        """
        Initializes the comprehensive model.
        
        Args:
            embed_size (int): Shared embedding space dimensionality.
            hidden_size (int): LSTM hidden state dimension size.
            vocab_size (int): Total number of unique vocabulary words.
            num_layers (int): Number of stacked LSTM layers in the decoder.
            max_seq_length (int): Upper limit of words generated during inference.
            train_cnn (bool): Whether to update CNN weights during training.
        """
        super(ImageCaptioningModel, self).__init__()
        
        # 1. Image Encoder
        self.encoder = CNNEncoder(embed_size, train_cnn)
        
        # 2. Text Decoder
        self.decoder = LSTMDecoder(embed_size, hidden_size, vocab_size, num_layers, max_seq_length)
        
    def forward(self, images, captions):
        """
        Standard forward pass for training the model.
        
        Args:
            images (torch.Tensor): Preprocessed images array of shape (batch, channels, H, W).
            captions (torch.Tensor): Tokenized target captions of shape (batch, seq_length).
            
        Returns:
            torch.Tensor: Decoder vocabulary scores of shape (batch, seq_length, vocab_size).
        """
        # Extract features through ResNet
        features = self.encoder(images)
        
        # Process and generate probabilities over words via the LSTM
        outputs = self.decoder(features, captions)
        
        return outputs
        
    def generate_caption(self, image, vocab, max_length=None, beam_size=1):
        """
        Complete inference logic: From raw preprocessed image to predicted word indices.
        
        Args:
            image (torch.Tensor): Single preprocessed image of shape (channels, H, W) or (1, channels, H, W).
            vocab (object): Vocabulary mapping implementation to decode indices.
            max_length (int): Optional override of maximum sequence length to yield.
            beam_size (int): Size of beam search. If 1, uses greedy search.
            
        Returns:
            list: Generated caption word indices.
        """
        # Force a batch dimension on a single input image if necessary
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        with torch.no_grad():
            # Get visual context
            features = self.encoder(image)
            
            # Predict tokens
            if beam_size > 1:
                predicted_indices = self.decoder.generate_caption_beam_search(features, vocab, beam_size, max_length)
            else:
                predicted_indices = self.decoder.generate_caption(features, vocab, max_length)
            
        return predicted_indices

    def generate_captions_top_k(self, image, vocab, max_length=None, beam_size=5, top_k=5):
        """
        Generates multiple caption candidates for re-ranking.
        
        Args:
            image (torch.Tensor): Single preprocessed image.
            vocab (object): Vocabulary mapping.
            max_length (int): Maximum sequence length.
            beam_size (int): Width of beam search.
            top_k (int): Number of top candidates to return.
            
        Returns:
            list of tuple: Each is (word_indices, normalized_score).
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        with torch.no_grad():
            features = self.encoder(image)
            candidates = self.decoder.generate_captions_top_k(
                features, vocab, beam_size, max_length, top_k
            )
            
        return candidates

if __name__ == "__main__":
    # Test the complete End-To-End Image Captioning Model
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    VOCAB_SIZE = 5000
    BATCH_SIZE = 2
    SEQ_LENGTH = 15

    print("Initializing the E2E Image Captioning Model...")
    model = ImageCaptioningModel(
        embed_size=EMBED_SIZE, 
        hidden_size=HIDDEN_SIZE, 
        vocab_size=VOCAB_SIZE
    )
    
    # It's important to set it to eval mode for validation/inference testing
    model.eval()

    print("\nPreparing Dummy Tensors...")
    dummy_images = torch.randn(BATCH_SIZE, 3, 244, 244)
    dummy_captions = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))
    print(f" - Images (Batch Size: {dummy_images.shape[0]}): {dummy_images.shape}")
    print(f" - Captions Sequence: {dummy_captions.shape}")

    print("\nTesting End-To-End Training Forward Pass...")
    with torch.no_grad():
        outputs = model(dummy_images, dummy_captions)
        
    print(f"Final Model Output Shape: {outputs.shape}")
    print(f"Expected Outline: ({BATCH_SIZE}, {SEQ_LENGTH}, {VOCAB_SIZE})")
    
    if outputs.shape == (BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE):
        print("\nSUCCESS: CNN Encoder and LSTM Decoder are perfectly connected!")
    else:
        print("\nERROR: Connectivity mismatch, shape alignment failed.")
