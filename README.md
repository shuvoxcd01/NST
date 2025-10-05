# NST
Neural Style Transfer

## Overview
This repository implements Neural Style Transfer using PyTorch, based on the paper ["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576) by Gatys et al.

Neural Style Transfer is a technique that allows you to apply the artistic style of one image to the content of another image using deep learning.

## Installation

### Requirements
- Python 3.6+
- PyTorch
- torchvision
- Pillow
- (Optional) matplotlib for visualization

### Install Dependencies
```bash
pip install torch torchvision pillow matplotlib
```

## Usage

### Basic Usage

```python
from neural_style_transfer import NeuralStyleTransfer

# Initialize the model
nst = NeuralStyleTransfer()

# Perform style transfer
output_image = nst.transfer_style(
    content_path='path/to/content.jpg',
    style_path='path/to/style.jpg',
    output_path='output.jpg',
    num_steps=300
)
```

### Advanced Usage

You can customize the layers used for content and style extraction:

```python
from neural_style_transfer import NeuralStyleTransfer

# Custom configuration
nst = NeuralStyleTransfer(
    content_layers=['conv_4'],
    style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
)

# Run with custom weights
output = nst.run_style_transfer(
    content_img_path='content.jpg',
    style_img_path='style.jpg',
    num_steps=500,
    style_weight=1000000,  # Higher value = more style
    content_weight=1        # Higher value = more content preservation
)

output.save('styled_output.jpg')
```

## Class API

### NeuralStyleTransfer

Main class for performing neural style transfer.

**Constructor:**
```python
NeuralStyleTransfer(content_layers=None, style_layers=None, device=None)
```
- `content_layers`: List of VGG19 layer names for content extraction (default: ['conv_4'])
- `style_layers`: List of VGG19 layer names for style extraction (default: ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'])
- `device`: Computation device ('cuda' or 'cpu', auto-detected by default)

**Main Methods:**

- `transfer_style(content_path, style_path, output_path, num_steps, style_weight, content_weight)`
  - High-level method to perform style transfer
  - Returns: PIL Image of the styled output

- `run_style_transfer(content_img_path, style_img_path, num_steps, style_weight, content_weight, input_img)`
  - Core optimization method
  - Returns: PIL Image of the styled output

- `load_image(image_path, max_size)`
  - Load and preprocess an image
  - Returns: torch.Tensor

## Examples

See `example_usage.py` for detailed examples.

## Original Implementation

The original TensorFlow/Keras implementation can be found in `Neural_Style_Transfer_with_tf_Keras.ipynb`.

## License

MIT License - see LICENSE file for details.
