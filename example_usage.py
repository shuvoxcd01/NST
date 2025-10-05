"""
Example usage of the NeuralStyleTransfer class.

This script demonstrates how to use the NeuralStyleTransfer class
to apply artistic style from one image to the content of another.
"""

from neural_style_transfer import NeuralStyleTransfer


def main():
    """
    Example usage of Neural Style Transfer.
    """
    # Initialize the Neural Style Transfer model
    nst = NeuralStyleTransfer()
    
    # Paths to your images
    content_image_path = 'path/to/your/content_image.jpg'
    style_image_path = 'path/to/your/style_image.jpg'
    output_image_path = 'output_styled_image.jpg'
    
    # Perform style transfer
    # You can adjust num_steps, style_weight, and content_weight for different results
    output_image = nst.transfer_style(
        content_path=content_image_path,
        style_path=style_image_path,
        output_path=output_image_path,
        num_steps=300,          # More steps = better quality but slower
        style_weight=1000000,   # Higher = more style influence
        content_weight=1        # Higher = more content preservation
    )
    
    # Display the result (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(output_image)
        plt.axis('off')
        plt.title('Styled Output')
        plt.show()
    except ImportError:
        print("Install matplotlib to display the image: pip install matplotlib")
        print(f"Image saved to: {output_image_path}")


def advanced_example():
    """
    Advanced example with custom layers and parameters.
    """
    # You can specify custom layers for content and style extraction
    content_layers = ['conv_4']  # Typically use deeper layers for content
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']  # Multiple layers for style
    
    # Initialize with custom layers
    nst = NeuralStyleTransfer(
        content_layers=content_layers,
        style_layers=style_layers
    )
    
    # Load and process images
    content_path = 'path/to/content.jpg'
    style_path = 'path/to/style.jpg'
    
    # Run style transfer with custom parameters
    output = nst.run_style_transfer(
        content_img_path=content_path,
        style_img_path=style_path,
        num_steps=500,
        style_weight=1e6,
        content_weight=1
    )
    
    # Save the output
    output.save('advanced_output.jpg')


if __name__ == '__main__':
    print("Neural Style Transfer Example")
    print("=" * 50)
    print("\nMake sure to update the image paths in the script before running!")
    print("\nBasic usage:")
    print("  nst = NeuralStyleTransfer()")
    print("  output = nst.transfer_style(content_path, style_path, output_path)")
    print("\n" + "=" * 50)
    
    # Uncomment to run the example (after setting correct image paths)
    # main()
