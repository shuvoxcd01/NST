"""
Neural Style Transfer using PyTorch

This module implements Neural Style Transfer using VGG19 pretrained model.
Based on the paper "A Neural Algorithm of Artistic Style" by Gatys et al.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import copy


class NeuralStyleTransfer:
    """
    A class for performing Neural Style Transfer using PyTorch.
    
    This implementation uses VGG19 pretrained on ImageNet to extract
    content and style features, then optimizes an image to match both
    the content of one image and the style of another.
    """
    
    def __init__(self, content_layers=None, style_layers=None, device=None):
        """
        Initialize the Neural Style Transfer model.
        
        Args:
            content_layers (list): List of layer names for content extraction.
                                  Default: ['conv_4']
            style_layers (list): List of layer names for style extraction.
                                Default: ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
            device (str): Device to run computations on ('cuda' or 'cpu').
                         Default: Auto-detect CUDA availability
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default layers for content and style extraction
        self.content_layers = content_layers if content_layers else ['conv_4']
        self.style_layers = style_layers if style_layers else ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        # Normalization values for ImageNet
        self.normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        
        # Load VGG19 model
        self.cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(self.device).eval()
        
    def load_image(self, image_path, max_size=512):
        """
        Load and preprocess an image.
        
        Args:
            image_path (str): Path to the image file
            max_size (int): Maximum dimension of the image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        image = Image.open(image_path)
        
        # Resize image maintaining aspect ratio
        size = min(max_size, max(image.size))
        
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])
        
        image = transform(image).unsqueeze(0)
        return image.to(self.device, torch.float)
    
    def image_to_tensor(self, image):
        """
        Convert PIL Image to tensor.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            torch.Tensor: Image tensor
        """
        transform = transforms.ToTensor()
        return transform(image).unsqueeze(0).to(self.device, torch.float)
    
    def tensor_to_image(self, tensor):
        """
        Convert tensor to PIL Image.
        
        Args:
            tensor (torch.Tensor): Input tensor
            
        Returns:
            PIL.Image: Output image
        """
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = transforms.ToPILImage()(image)
        return image
    
    def gram_matrix(self, input_tensor):
        """
        Compute the Gram matrix of an input tensor.
        
        The Gram matrix is used to compute style loss by capturing
        the correlation between different filter responses.
        
        Args:
            input_tensor (torch.Tensor): Input feature maps (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Gram matrix (batch_size, channels, channels)
        """
        batch_size, channels, height, width = input_tensor.size()
        features = input_tensor.view(batch_size * channels, height * width)
        gram = torch.mm(features, features.t())
        
        # Normalize by the number of elements
        return gram.div(batch_size * channels * height * width)
    
    def get_style_model_and_losses(self, style_img, content_img):
        """
        Create a model that computes style and content losses.
        
        Args:
            style_img (torch.Tensor): Style reference image
            content_img (torch.Tensor): Content reference image
            
        Returns:
            tuple: (model, style_losses, content_losses)
        """
        # Normalization module
        normalization = Normalization(self.normalization_mean, self.normalization_std).to(self.device)
        
        content_losses = []
        style_losses = []
        
        model = nn.Sequential(normalization)
        
        i = 0  # increment every time we see a conv
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # Replace inplace ReLU with out-of-place version
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            
            model.add_module(name, layer)
            
            if name in self.content_layers:
                # Add content loss
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)
            
            if name in self.style_layers:
                # Add style loss
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)
        
        # Trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        
        model = model[:(i + 1)]
        
        return model, style_losses, content_losses
    
    def run_style_transfer(self, content_img_path, style_img_path, num_steps=300,
                          style_weight=1000000, content_weight=1, input_img=None):
        """
        Run the style transfer optimization.
        
        Args:
            content_img_path (str): Path to content image
            style_img_path (str): Path to style image
            num_steps (int): Number of optimization steps
            style_weight (float): Weight for style loss
            content_weight (float): Weight for content loss
            input_img (torch.Tensor): Initial image (if None, uses content image)
            
        Returns:
            PIL.Image: Output styled image
        """
        print(f'Building the style transfer model on {self.device}..')
        
        # Load images
        content_img = self.load_image(content_img_path)
        style_img = self.load_image(style_img_path)
        
        # Initialize input image
        if input_img is None:
            input_img = content_img.clone()
        
        # Build the model
        model, style_losses, content_losses = self.get_style_model_and_losses(style_img, content_img)
        
        # We want to optimize the input image, so we need to make it require gradients
        input_img.requires_grad_(True)
        model.requires_grad_(False)
        
        # Use LBFGS optimizer
        optimizer = optim.LBFGS([input_img])
        
        print('Optimizing...')
        run = [0]
        while run[0] <= num_steps:
            
            def closure():
                # Clamp input image to [0, 1]
                with torch.no_grad():
                    input_img.clamp_(0, 1)
                
                optimizer.zero_grad()
                model(input_img)
                
                style_score = 0
                content_score = 0
                
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                
                style_score *= style_weight
                content_score *= content_weight
                
                loss = style_score + content_score
                loss.backward()
                
                run[0] += 1
                if run[0] % 50 == 0:
                    print(f"Step {run[0]}:")
                    print(f'Style Loss: {style_score.item():.4f} Content Loss: {content_score.item():.4f}')
                
                return style_score + content_score
            
            optimizer.step(closure)
        
        # Final clamp
        with torch.no_grad():
            input_img.clamp_(0, 1)
        
        return self.tensor_to_image(input_img)
    
    def transfer_style(self, content_path, style_path, output_path=None, 
                      num_steps=300, style_weight=1000000, content_weight=1):
        """
        High-level method to perform style transfer and optionally save result.
        
        Args:
            content_path (str): Path to content image
            style_path (str): Path to style image
            output_path (str): Path to save output image (optional)
            num_steps (int): Number of optimization steps
            style_weight (float): Weight for style loss
            content_weight (float): Weight for content loss
            
        Returns:
            PIL.Image: Styled output image
        """
        output_img = self.run_style_transfer(
            content_path, 
            style_path,
            num_steps=num_steps,
            style_weight=style_weight,
            content_weight=content_weight
        )
        
        if output_path:
            output_img.save(output_path)
            print(f"Output saved to {output_path}")
        
        return output_img


class Normalization(nn.Module):
    """
    Normalization module to normalize input images.
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # View the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W]
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)
    
    def forward(self, img):
        # Normalize img
        return (img - self.mean) / self.std


class ContentLoss(nn.Module):
    """
    Content loss module.
    """
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # Detach the target content from the tree used to dynamically compute the gradient
        self.target = target.detach()
    
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    """
    Style loss module using Gram matrices.
    """
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self._gram_matrix(target_feature).detach()
    
    def _gram_matrix(self, input_tensor):
        batch_size, channels, height, width = input_tensor.size()
        features = input_tensor.view(batch_size * channels, height * width)
        gram = torch.mm(features, features.t())
        return gram.div(batch_size * channels * height * width)
    
    def forward(self, input):
        gram = self._gram_matrix(input)
        self.loss = nn.functional.mse_loss(gram, self.target)
        return input
