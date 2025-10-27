#!/usr/bin/env python3
"""Test your trained U²-Net clothing background removal model"""
import os
import sys
import glob
from pathlib import Path
from PIL import Image
import torch
import numpy as np

# Try to load U2NET
try:
    from model import U2NET
except ImportError:
    u2net_path = Path("./U-2-Net")
    if not u2net_path.exists():
        print("ERROR: U-2-Net repository not found. Please clone it first:")
        print("git clone https://github.com/xuebinqin/U-2-Net.git")
        sys.exit(1)
    sys.path.insert(0, str(u2net_path / "model"))
    from u2net import U2NET


def preprocess(image, size=320):
    """Preprocess image for model input"""
    orig_w, orig_h = image.size
    img = image.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor, (orig_h, orig_w)


def postprocess(mask_logits, out_hw):
    """Postprocess model output to mask"""
    mask = torch.sigmoid(mask_logits)
    mask = torch.nn.functional.interpolate(
        mask, size=out_hw, mode="bilinear", align_corners=False
    )
    mask = mask[0, 0].clamp(0, 1).cpu().numpy()
    return (mask * 255).astype(np.uint8)


def save_result(image, mask, output_path):
    """Save result as PNG with transparency"""
    # Convert mask to alpha channel
    mask_img = Image.fromarray(mask, mode="L")
    rgba = image.convert("RGBA")
    r, g, b, _ = rgba.split()
    rgba_result = Image.merge("RGBA", (r, g, b, mask_img))
    rgba_result.save(output_path, "PNG")
    print(f"Saved: {output_path}")


def test_model(weights_path="./weights/u2net_clothing.pth", test_dir="./dataset/images", device="cpu"):
    """Test the trained model"""
    
    # Check if weights exist
    if not os.path.exists(weights_path):
        print(f"ERROR: Weights not found at {weights_path}")
        print("Make sure you've trained the model first!")
        sys.exit(1)
    
    # Setup device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {weights_path}...")
    model = U2NET(3, 1)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Find test images
    if os.path.isdir(test_dir):
        test_images = glob.glob(os.path.join(test_dir, "*.jpg"))[:5]  # Test first 5 images
    else:
        test_images = [test_dir]
    
    if not test_images:
        print(f"No images found in {test_dir}")
        sys.exit(1)
    
    print(f"Testing on {len(test_images)} images...")
    
    # Create output directory
    os.makedirs("test_results", exist_ok=True)
    
    # Process each image
    for img_path in test_images:
        print(f"\nProcessing: {img_path}")
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Preprocess
        inp, out_size = preprocess(image)
        inp = inp.to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(inp)
            if isinstance(outputs, (list, tuple)):
                pred = outputs[0]
            else:
                pred = outputs
        
        # Postprocess
        mask = postprocess(pred, out_size)
        
        # Save result
        filename = os.path.basename(img_path).replace(".jpg", ".png")
        output_path = os.path.join("test_results", filename)
        save_result(image, mask, output_path)
    
    print("\n✅ Testing complete! Check the 'test_results' folder for output images.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test U²-Net clothing background removal")
    parser.add_argument("--weights", type=str, default="./weights/u2net_clothing.pth", 
                       help="Path to trained weights")
    parser.add_argument("--test-dir", type=str, default="./dataset/images",
                       help="Directory with test images")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: cuda, mps, cpu, or auto")
    args = parser.parse_args()
    
    test_model(args.weights, args.test_dir, args.device)


