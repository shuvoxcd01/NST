# Quick Start Guide

Get started with Neural Style Transfer in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/shuvoxcd01/NST.git
cd NST

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Import the class

```python
from neural_style_transfer import NeuralStyleTransfer
```

### 2. Create an instance

```python
nst = NeuralStyleTransfer()
```

### 3. Apply style transfer

```python
output = nst.transfer_style(
    content_path='path/to/content.jpg',
    style_path='path/to/style.jpg',
    output_path='output.jpg'
)
```

That's it! Your styled image is saved to `output.jpg`.

## Complete Example

```python
from neural_style_transfer import NeuralStyleTransfer
import matplotlib.pyplot as plt

# Initialize
nst = NeuralStyleTransfer()

# Run style transfer
output = nst.transfer_style(
    content_path='my_photo.jpg',
    style_path='picasso_painting.jpg',
    output_path='styled_photo.jpg',
    num_steps=300,          # More steps = better quality
    style_weight=1000000,   # Higher = more style
    content_weight=1        # Higher = more content preservation
)

# Display result
plt.figure(figsize=(10, 10))
plt.imshow(output)
plt.axis('off')
plt.show()
```

## Advanced Configuration

### Custom Layers

```python
nst = NeuralStyleTransfer(
    content_layers=['conv_4'],  # Single deeper layer for content
    style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']  # Multiple layers for style
)
```

### Adjust Weights

- **style_weight**: Controls how much style is applied
  - Higher (e.g., 1e7) = More stylistic
  - Lower (e.g., 1e5) = More subtle

- **content_weight**: Controls content preservation
  - Higher (e.g., 10) = More content preserved
  - Lower (e.g., 0.1) = More transformation

### More Iterations

```python
output = nst.transfer_style(
    content_path='content.jpg',
    style_path='style.jpg',
    num_steps=500,  # More steps for better results
    output_path='output.jpg'
)
```

## Tips for Best Results

1. **Image Size**: Images are automatically resized to max 512px. Larger images take longer.

2. **Style Weight**: 
   - Start with 1000000 (1e6)
   - Increase for more style, decrease for subtlety

3. **Iterations**:
   - 300 steps: Good for testing
   - 500 steps: Better quality
   - 1000+ steps: Best quality (slower)

4. **Content vs Style**:
   - Keep content_weight at 1
   - Adjust style_weight to control the balance

## Troubleshooting

### Out of Memory
- Reduce image size by editing `max_size` parameter in `load_image()`
- Use CPU instead of GPU for smaller memory footprint

### Poor Results
- Increase `num_steps`
- Adjust `style_weight` and `content_weight`
- Try different style images

### Slow Performance
- Reduce `num_steps`
- Use GPU if available
- Reduce image size

## GPU Usage

The class automatically uses GPU if CUDA is available:

```python
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")

# Force CPU usage
nst = NeuralStyleTransfer(device='cpu')

# Force GPU usage (if available)
nst = NeuralStyleTransfer(device='cuda')
```

## Examples

Check out `example_usage.py` for more detailed examples!

## Testing

Run the test suite to verify installation:

```bash
python test_nst.py
```

## Need Help?

- Read the full documentation in `README.md`
- Check the comparison guide in `COMPARISON.md`
- Review the example code in `example_usage.py`
