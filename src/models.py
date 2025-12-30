"""
Model Architectures

Contains CNN encoder, tabular encoder, and fusion model for property valuation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """
    CNN encoder for satellite images using pretrained ResNet-18.
    Outputs a fixed-size embedding vector.
    """
    
    def __init__(self, embedding_dim=512, pretrained=True, freeze_backbone=False):
        """
        Initialize the image encoder.
        
        Args:
            embedding_dim: Output embedding dimension
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        
        # Load pretrained ResNet-18
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            if pretrained:
                self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.backbone = resnet18(weights=None)
        except ImportError:
            # Fallback for older torchvision
            from torchvision.models import resnet18
            self.backbone = resnet18(pretrained=pretrained)
        
        # Get the number of features from the last layer
        backbone_out_features = self.backbone.fc.in_features  # 512 for ResNet-18
        
        # Remove the classification head
        self.backbone.fc = nn.Identity()
        
        # Add projection layer if embedding_dim differs
        if embedding_dim != backbone_out_features:
            self.projection = nn.Sequential(
                nn.Linear(backbone_out_features, embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        else:
            self.projection = nn.Identity()
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Image tensor (B, 3, H, W)
        
        Returns:
            Embedding tensor (B, embedding_dim)
        """
        features = self.backbone(x)
        embeddings = self.projection(features)
        return embeddings


class TabularEncoder(nn.Module):
    """
    MLP encoder for tabular features.
    """
    
    def __init__(self, input_dim, hidden_dims=[64, 32], embedding_dim=32, dropout=0.2):
        """
        Initialize the tabular encoder.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            embedding_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tabular features (B, input_dim)
        
        Returns:
            Embedding tensor (B, embedding_dim)
        """
        return self.encoder(x)


class FusionModel(nn.Module):
    """
    Multimodal fusion model combining image and tabular features.
    
    Architecture:
        Image → ImageEncoder → 512-dim
        Tabular → TabularEncoder → 32-dim
        [Concat] → 544-dim → Dense → 1 (Price)
    """
    
    def __init__(self, 
                 tabular_input_dim,
                 image_embedding_dim=512,
                 tabular_embedding_dim=32,
                 fusion_hidden_dims=[256, 128],
                 dropout=0.3,
                 pretrained=True,
                 freeze_backbone=False):
        """
        Initialize the fusion model.
        
        Args:
            tabular_input_dim: Number of tabular features
            image_embedding_dim: Image encoder output dimension
            tabular_embedding_dim: Tabular encoder output dimension
            fusion_hidden_dims: Hidden dimensions for fusion layers
            dropout: Dropout rate
            pretrained: Use pretrained image encoder
            freeze_backbone: Freeze image encoder backbone
        """
        super().__init__()
        
        # Image encoder
        self.image_encoder = ImageEncoder(
            embedding_dim=image_embedding_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
        
        # Tabular encoder
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_input_dim,
            embedding_dim=tabular_embedding_dim,
            dropout=dropout
        )
        
        # Fusion layers
        fusion_input_dim = image_embedding_dim + tabular_embedding_dim
        
        fusion_layers = []
        prev_dim = fusion_input_dim
        
        for hidden_dim in fusion_hidden_dims:
            fusion_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final regression head
        fusion_layers.append(nn.Linear(prev_dim, 1))
        
        self.fusion = nn.Sequential(*fusion_layers)
    
    def forward(self, image, tabular):
        """
        Forward pass.
        
        Args:
            image: Image tensor (B, 3, H, W)
            tabular: Tabular features (B, num_features)
        
        Returns:
            Predicted price (B, 1)
        """
        # Encode both modalities
        image_embedding = self.image_encoder(image)
        tabular_embedding = self.tabular_encoder(tabular)
        
        # Concatenate embeddings
        combined = torch.cat([image_embedding, tabular_embedding], dim=1)
        
        # Fusion and regression
        output = self.fusion(combined)
        
        return output
    
    def get_image_embedding(self, image):
        """Get image embedding (for Grad-CAM analysis)."""
        return self.image_encoder(image)
    
    def get_tabular_embedding(self, tabular):
        """Get tabular embedding."""
        return self.tabular_encoder(tabular)


class TabularOnlyModel(nn.Module):
    """
    Baseline model using only tabular features (no images).
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        """
        Initialize the tabular-only model.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tabular features (B, input_dim)
        
        Returns:
            Predicted price (B, 1)
        """
        return self.model(x)


class ImageOnlyModel(nn.Module):
    """
    Model using only satellite images (no tabular data).
    """
    
    def __init__(self, hidden_dim=256, pretrained=True):
        super().__init__()
        
        self.image_encoder = ImageEncoder(
            embedding_dim=512,
            pretrained=pretrained
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        embeddings = self.image_encoder(x)
        return self.regressor(embeddings)


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    print("Testing model architectures...\n")
    
    batch_size = 4
    num_features = 17
    
    # Test ImageEncoder
    print("ImageEncoder:")
    img_encoder = ImageEncoder(embedding_dim=512)
    dummy_img = torch.randn(batch_size, 3, 224, 224)
    img_out = img_encoder(dummy_img)
    print(f"  Input: {dummy_img.shape}")
    print(f"  Output: {img_out.shape}")
    print(f"  Parameters: {count_parameters(img_encoder):,}\n")
    
    # Test TabularEncoder
    print("TabularEncoder:")
    tab_encoder = TabularEncoder(input_dim=num_features)
    dummy_tab = torch.randn(batch_size, num_features)
    tab_out = tab_encoder(dummy_tab)
    print(f"  Input: {dummy_tab.shape}")
    print(f"  Output: {tab_out.shape}")
    print(f"  Parameters: {count_parameters(tab_encoder):,}\n")
    
    # Test FusionModel
    print("FusionModel:")
    fusion_model = FusionModel(tabular_input_dim=num_features)
    fusion_out = fusion_model(dummy_img, dummy_tab)
    print(f"  Image Input: {dummy_img.shape}")
    print(f"  Tabular Input: {dummy_tab.shape}")
    print(f"  Output: {fusion_out.shape}")
    print(f"  Parameters: {count_parameters(fusion_model):,}\n")
    
    # Test TabularOnlyModel
    print("TabularOnlyModel:")
    tab_model = TabularOnlyModel(input_dim=num_features)
    tab_only_out = tab_model(dummy_tab)
    print(f"  Input: {dummy_tab.shape}")
    print(f"  Output: {tab_only_out.shape}")
    print(f"  Parameters: {count_parameters(tab_model):,}\n")
    
    print("All model tests passed!")
