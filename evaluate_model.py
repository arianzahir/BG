#!/usr/bin/env python3
"""Evaluate your trained U²-Net clothing background removal model with metrics"""
import os
import sys
import glob
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm

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
    # Threshold to binary mask
    mask_binary = (mask > 0.5).astype(np.uint8)
    return mask_binary


def load_gt_mask(mask_path):
    """Load ground truth mask and binarize"""
    mask = Image.open(mask_path).convert("L")
    mask_array = np.array(mask) / 255.0
    mask_binary = (mask_array > 0.5).astype(np.uint8)
    return mask_binary


def calculate_iou(pred_mask, gt_mask):
    """Calculate Intersection over Union (IoU)"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0  # Perfect match if both are empty
    return intersection / union


def calculate_dice(pred_mask, gt_mask):
    """Calculate Dice coefficient"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    if total == 0:
        return 1.0  # Perfect match if both are empty
    return 2.0 * intersection / total


def calculate_accuracy(pred_mask, gt_mask):
    """Calculate pixel accuracy"""
    correct = (pred_mask == gt_mask).sum()
    total = pred_mask.size
    return correct / total if total > 0 else 1.0


def calculate_precision(pred_mask, gt_mask):
    """Calculate precision"""
    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
    if (tp + fp) == 0:
        return 1.0
    return tp / (tp + fp)


def calculate_recall(pred_mask, gt_mask):
    """Calculate recall"""
    tp = np.logical_and(pred_mask, gt_mask).sum()
    fn = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()
    if (tp + fn) == 0:
        return 1.0
    return tp / (tp + fn)


def evaluate_model(weights_path="./weights/u2net_clothing.pth", 
                  img_dir="./dataset/images",
                  mask_dir="./dataset/masks",
                  device="auto"):
    """Evaluate the trained model with metrics"""
    
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
    
    # Find test images and corresponding masks
    test_images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not test_images:
        test_images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    
    if not test_images:
        print(f"No images found in {img_dir}")
        sys.exit(1)
    
    print(f"Evaluating on {len(test_images)} images...")
    
    # Metrics storage
    ious = []
    dices = []
    accuracies = []
    precisions = []
    recalls = []
    
    # Process each image
    for img_path in tqdm(test_images, desc="Processing"):
        # Find corresponding mask
        filename = os.path.basename(img_path)
        mask_name = filename.replace(".jpg", ".png").replace(".png", ".png")
        mask_path = os.path.join(mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {filename}, skipping...")
            continue
        
        # Load image and ground truth mask
        image = Image.open(img_path).convert("RGB")
        gt_mask = load_gt_mask(mask_path)
        
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
        pred_mask = postprocess(pred, out_size)
        
        # Resize prediction to match ground truth size
        if pred_mask.shape != gt_mask.shape:
            pred_mask = Image.fromarray(pred_mask)
            pred_mask = pred_mask.resize(gt_mask.shape[::-1], Image.NEAREST)
            pred_mask = np.array(pred_mask)
        
        # Calculate metrics
        iou = calculate_iou(pred_mask, gt_mask)
        dice = calculate_dice(pred_mask, gt_mask)
        accuracy = calculate_accuracy(pred_mask, gt_mask)
        precision = calculate_precision(pred_mask, gt_mask)
        recall = calculate_recall(pred_mask, gt_mask)
        
        ious.append(iou)
        dices.append(dice)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of samples: {len(ious)}")
    print(f"\nIoU (Intersection over Union):     {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"Dice Coefficient:                  {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"Pixel Accuracy:                    {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Precision:                         {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"Recall:                            {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print("="*60)
    print(f"\nBest IoU:  {np.max(ious):.4f}")
    print(f"Worst IoU: {np.min(ious):.4f}")
    print("="*60)
    
    return {
        'iou': np.mean(ious),
        'dice': np.mean(dices),
        'accuracy': np.mean(accuracies),
        'precision': np.mean(precisions),
        'recall': np.mean(recalls)
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate U²-Net clothing background removal")
    parser.add_argument("--weights", type=str, default="./weights/u2net_clothing.pth", 
                       help="Path to trained weights")
    parser.add_argument("--img-dir", type=str, default="./dataset/images",
                       help="Directory with images")
    parser.add_argument("--mask-dir", type=str, default="./dataset/masks",
                       help="Directory with ground truth masks")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: cuda, mps, cpu, or auto")
    args = parser.parse_args()
    
    evaluate_model(args.weights, args.img_dir, args.mask_dir, args.device)
