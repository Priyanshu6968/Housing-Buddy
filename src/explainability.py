"""
Explainability Module - Grad-CAM

Implements Grad-CAM visualization for understanding model predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Visualizes which regions of an image contribute most to the model's prediction.
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The neural network model
            target_layer: The layer to compute Grad-CAM for (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, tabular_features=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_image: Input image tensor (1, 3, H, W)
            tabular_features: Tabular features tensor (optional, for fusion models)
        
        Returns:
            cam: Grad-CAM heatmap (H, W)
        """
        self.model.eval()
        
        # Forward pass
        if tabular_features is not None:
            output = self.model(input_image, tabular_features)
        else:
            # For image-only or ImageEncoder
            if hasattr(self.model, 'image_encoder'):
                output = self.model.get_image_embedding(input_image)
                output = output.mean()  # Reduce to scalar for gradient
            else:
                output = self.model(input_image)
        
        # Backward pass
        self.model.zero_grad()
        output.backward(torch.ones_like(output))
        
        # Get gradients and activations
        gradients = self.gradients  # (B, C, H, W)
        activations = self.activations  # (B, C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        
        # Weighted sum of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # ReLU to get only positive contributions
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def generate_heatmap_overlay(self, image_tensor, cam, alpha=0.5):
        """
        Generate heatmap overlay on original image.
        
        Args:
            image_tensor: Original image tensor (3, H, W)
            cam: Grad-CAM heatmap
            alpha: Overlay transparency
        
        Returns:
            overlay: RGB image with heatmap overlay
        """
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image_tensor.cpu() * std + mean
        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
        
        # Resize CAM to image size
        cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (image.shape[1], image.shape[0]), Image.BILINEAR
        )) / 255.0
        
        # Apply colormap
        heatmap = cm.jet(cam_resized)[:, :, :3]  # Remove alpha channel
        
        # Blend
        overlay = (1 - alpha) * image + alpha * heatmap
        overlay = np.clip(overlay, 0, 1)
        
        return overlay


def visualize_gradcam(model, image_path, tabular_features=None, 
                      save_path=None, device='cuda'):
    """
    Generate and visualize Grad-CAM for a single image.
    
    Args:
        model: Trained model
        image_path: Path to satellite image
        tabular_features: Tabular features tensor (optional)
        save_path: Path to save visualization
        device: Device to use
    """
    from torchvision import transforms
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    if tabular_features is not None:
        tabular_features = tabular_features.to(device)
    
    # Get target layer (last conv layer of ResNet backbone)
    if hasattr(model, 'image_encoder'):
        target_layer = model.image_encoder.backbone.layer4[-1]
    else:
        target_layer = model.backbone.layer4[-1]
    
    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(img_tensor, tabular_features)
    
    # Generate overlay
    overlay = gradcam.generate_heatmap_overlay(img_tensor.squeeze(0), cam)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    orig_img = np.array(img.resize((224, 224))) / 255.0
    axes[0].imshow(orig_img)
    axes[0].set_title('Original Satellite Image', fontsize=12)
    axes[0].axis('off')
    
    # Grad-CAM heatmap
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Grad-CAM visualization to: {save_path}")
    
    plt.close()
    
    return cam, overlay


def batch_gradcam_analysis(model, dataset, output_dir, 
                           num_samples=20, device='cuda'):
    """
    Generate Grad-CAM visualizations for multiple samples.
    
    Args:
        model: Trained model
        dataset: PropertyDataset
        output_dir: Directory to save visualizations
        num_samples: Number of samples to analyze
        device: Device to use
    """
    from torch.utils.data import DataLoader
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = model.to(device)
    model.eval()
    
    # Get target layer
    if hasattr(model, 'image_encoder'):
        target_layer = model.image_encoder.backbone.layer4[-1]
    else:
        target_layer = model.backbone.layer4[-1]
    
    gradcam = GradCAM(model, target_layer)
    
    # Randomly sample indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    print(f"Generating Grad-CAM for {len(indices)} samples...")
    
    for i, idx in enumerate(indices):
        img_tensor, tab_tensor, price = dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        tab_tensor = tab_tensor.unsqueeze(0).to(device)
        
        # Generate CAM
        cam = gradcam.generate_cam(img_tensor, tab_tensor)
        overlay = gradcam.generate_heatmap_overlay(img_tensor.squeeze(0), cam)
        
        # Get prediction
        with torch.no_grad():
            pred = model(img_tensor, tab_tensor).item()
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Denormalize original image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        orig_img = img_tensor.squeeze(0).cpu() * std + mean
        orig_img = orig_img.permute(1, 2, 0).numpy()
        orig_img = np.clip(orig_img, 0, 1)
        
        axes[0].imshow(orig_img)
        axes[0].set_title('Satellite Image', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM', fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title(f'Predicted: ${pred:,.0f}\nActual: ${price:,.0f}', fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = output_dir / f"gradcam_{idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")
    
    print(f"\nGrad-CAM analysis complete. Visualizations saved to: {output_dir}")


def interpret_gradcam_results(cam, threshold=0.5):
    """
    Interpret Grad-CAM results to identify important regions.
    
    Args:
        cam: Grad-CAM heatmap
        threshold: Threshold for importance
    
    Returns:
        Dict with interpretation
    """
    # Identify hot regions
    hot_mask = cam > threshold
    hot_area = hot_mask.sum() / cam.size
    
    # Analyze spatial distribution
    h, w = cam.shape
    
    # Quadrant analysis
    quadrants = {
        'top_left': cam[:h//2, :w//2].mean(),
        'top_right': cam[:h//2, w//2:].mean(),
        'bottom_left': cam[h//2:, :w//2].mean(),
        'bottom_right': cam[h//2:, w//2:].mean()
    }
    
    # Edge vs center
    center_mask = np.zeros_like(cam, dtype=bool)
    margin = int(0.25 * min(h, w))
    center_mask[margin:-margin, margin:-margin] = True
    
    center_importance = cam[center_mask].mean()
    edge_importance = cam[~center_mask].mean()
    
    return {
        'hot_area_fraction': float(hot_area),
        'max_activation': float(cam.max()),
        'mean_activation': float(cam.mean()),
        'quadrant_importance': quadrants,
        'center_importance': float(center_importance),
        'edge_importance': float(edge_importance),
        'center_vs_edge_ratio': float(center_importance / (edge_importance + 1e-8))
    }


if __name__ == "__main__":
    print("Grad-CAM module loaded successfully.")
    print("\nUsage:")
    print("  from src.explainability import GradCAM, visualize_gradcam")
    print("  cam = visualize_gradcam(model, 'path/to/image.png')")
