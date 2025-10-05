"""
Test script for the NeuralStyleTransfer class.

This script creates synthetic test images and runs a quick style transfer
to verify that the implementation works correctly.
"""

import os
import sys
from PIL import Image
import numpy as np


def create_test_images(test_dir='/tmp/nst_test'):
    """Create synthetic test images for testing."""
    os.makedirs(test_dir, exist_ok=True)
    
    # Create content image (gradient pattern)
    content_img = Image.new('RGB', (256, 256))
    pixels = content_img.load()
    for i in range(256):
        for j in range(256):
            pixels[i, j] = (i % 256, j % 256, 150)
    
    # Create style image (checkerboard-like pattern)
    style_img = Image.new('RGB', (256, 256))
    pixels = style_img.load()
    for i in range(256):
        for j in range(256):
            val = 255 if ((i // 32) + (j // 32)) % 2 == 0 else 50
            pixels[i, j] = (val, val // 2, val // 3)
    
    content_path = os.path.join(test_dir, 'test_content.jpg')
    style_path = os.path.join(test_dir, 'test_style.jpg')
    
    content_img.save(content_path)
    style_img.save(style_path)
    
    return content_path, style_path


def test_basic_functionality():
    """Test basic instantiation and methods."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Functionality")
    print("=" * 70)
    
    from neural_style_transfer import NeuralStyleTransfer
    
    # Test instantiation
    nst = NeuralStyleTransfer()
    print("✓ NeuralStyleTransfer instantiated")
    print(f"  - Device: {nst.device}")
    print(f"  - Content layers: {nst.content_layers}")
    print(f"  - Style layers: {nst.style_layers}")
    
    # Test with custom layers
    custom_nst = NeuralStyleTransfer(
        content_layers=['conv_3'],
        style_layers=['conv_1', 'conv_2']
    )
    print("✓ Custom layer configuration works")
    
    return True


def test_image_loading():
    """Test image loading and preprocessing."""
    print("\n" + "=" * 70)
    print("TEST 2: Image Loading")
    print("=" * 70)
    
    from neural_style_transfer import NeuralStyleTransfer
    
    content_path, style_path = create_test_images()
    nst = NeuralStyleTransfer()
    
    # Test loading images
    content_tensor = nst.load_image(content_path)
    style_tensor = nst.load_image(style_path)
    
    print(f"✓ Images loaded successfully")
    print(f"  - Content tensor shape: {content_tensor.shape}")
    print(f"  - Style tensor shape: {style_tensor.shape}")
    
    # Test tensor to image conversion
    img = nst.tensor_to_image(content_tensor)
    print(f"✓ Tensor to image conversion works")
    print(f"  - Output image size: {img.size}")
    
    return True


def test_style_transfer():
    """Test the complete style transfer pipeline."""
    print("\n" + "=" * 70)
    print("TEST 3: Style Transfer Pipeline")
    print("=" * 70)
    
    from neural_style_transfer import NeuralStyleTransfer
    
    content_path, style_path = create_test_images()
    output_path = '/tmp/nst_test/test_output.jpg'
    
    print("Running style transfer (5 steps for quick testing)...")
    nst = NeuralStyleTransfer()
    
    output = nst.transfer_style(
        content_path=content_path,
        style_path=style_path,
        output_path=output_path,
        num_steps=5,
        style_weight=1000,
        content_weight=1
    )
    
    print(f"✓ Style transfer completed")
    print(f"  - Output image size: {output.size}")
    print(f"  - Output saved to: {output_path}")
    
    # Verify output file exists
    if os.path.exists(output_path):
        print(f"✓ Output file verified")
        file_size = os.path.getsize(output_path)
        print(f"  - File size: {file_size} bytes")
    else:
        print(f"✗ Output file not found!")
        return False
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("NEURAL STYLE TRANSFER - TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Image Loading", test_image_loading),
        ("Style Transfer Pipeline", test_style_transfer),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name} FAILED with exception:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
