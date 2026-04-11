import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class CNNEncoder(nn.Module):
    """
    CNN Encoder for Image Feature Extraction using Pretrained ResNet-50.
    """
    def __init__(self, embed_size, train_cnn=False):
        """
        Initializes the CNN Encoder.
        
        Args:
            embed_size (int): The dimension of the output feature vector.
            train_cnn (bool): Whether to fine-tune the pretrained CNN.
        """
        super(CNNEncoder, self).__init__()
        
        # Load pretrained ResNet-50
        print("Loading Pretrained ResNet-50 for feature extraction...")
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Remove the final fully connected classification layer
        # By taking all children up to the last one, we get the features before the classification logic
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Freeze or unfreeze the CNN parameters
        # For standard image captioning, we usually freeze the CNN or fine-tune only the last few layers later.
        for param in self.resnet.parameters():
            param.requires_grad = train_cnn
            
        # ResNet-50 outputs 2048-dimensional features before the fully connected layer
        resnet_out_features = resnet.fc.in_features
        
        # Add a linear layer to map to our desired embedding dimension
        self.embed = nn.Linear(resnet_out_features, embed_size)
        
        # Optional: Add batch normalization for better training stability
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """
        Forward pass of the CNN encoder.
        
        Args:
            images (torch.Tensor): A batch of preprocessed images of shape (batch_size, 3, H, W).
            
        Returns:
            torch.Tensor: Extracted features of shape (batch_size, embed_size).
        """
        # Pass images through the ResNet backbone
        # Output shape is (batch_size, 2048, 1, 1) if images are around 224x224
        features = self.resnet(images)
        
        # Flatten the features
        features = features.reshape(features.size(0), -1)
        
        # Pass through linear layer and batch normalization
        features = self.embed(features)
        features = self.bn(features)
        
        return features

if __name__ == "__main__":
    # Test the CNN Encoder
    
    # 1. Initialize the Encoder
    # embedding size is the dimension we'll feed into our language model (RNN/LSTM/Transformer)
    EMBED_SIZE = 256
    encoder = CNNEncoder(embed_size=EMBED_SIZE, train_cnn=False)
    
    # Place encoder in eval mode for testing (especially important for BatchNorm/Dropout)
    encoder.eval()
    
    # 2. Test with dummy tensors based on our previous preprocess_images.py dimension
    # Our preprocessing uses 244x244
    BATCH_SIZE = 4
    dummy_images = torch.randn(BATCH_SIZE, 3, 244, 244)
    print(f"\nDummy Input Images Shape: {dummy_images.shape}")
    
    # 3. Forward Pass
    with torch.no_grad():
        extracted_features = encoder(dummy_images)
        
    print(f"Output Extracted Features Shape: {extracted_features.shape}")
    print(f"Expected shape was: ({BATCH_SIZE}, {EMBED_SIZE})")
    
    if extracted_features.shape == (BATCH_SIZE, EMBED_SIZE):
        print("\nSUCCESS: CNN Encoder is working as expected!")
    else:
        print("\nERROR: Output shape does not match expectation.")
