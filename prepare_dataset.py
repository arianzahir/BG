#!/usr/bin/env python3
"""
Prepare Dataset 1 and Dataset 2 for training by extracting binary masks
from the background-removed images and creating the training structure.
"""
import os
import sys
from pathlib import Path
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm

def extract_alpha_mask(png_path):
    """Extract alpha channel from RGBA PNG as binary mask"""
    img = Image.open(png_path)
    
    # Convert to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Extract alpha channel
    alpha = img.split()[-1]
    
    # Convert to binary: transparent=black (0), opaque=white (255)
    mask = alpha.point(lambda p: 255 if p > 0 else 0, mode='L')
    
    return mask

def main():
    # Setup paths
    original_dir = Path("Dataset 1")
    removed_dir = Path("Dataset 2(removed)")
    output_dir = Path("dataset")
    
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    
    # Create output directories
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all files from both directories
    original_files = sorted(original_dir.glob("*"))
    removed_files = sorted(removed_dir.glob("*"))
    
    # Filter to only image files
    original_files = [f for f in original_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    removed_files = [f for f in removed_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    
    print(f"Found {len(original_files)} original images")
    print(f"Found {len(removed_files)} removed-background images")
    
    if len(original_files) != len(removed_files):
        print(f"WARNING: Mismatch in file counts. Proceeding with {min(len(original_files), len(removed_files))} pairs.")
    
    # Process matching files
    paired_count = 0
    skipped_count = 0
    
    for orig_file, rem_file in tqdm(zip(original_files, removed_files), total=min(len(original_files), len(removed_files)), desc="Processing"):
        try:
            # Copy original image to images folder (keep original extension)
            orig_dest = images_dir / orig_file.name
            shutil.copy2(orig_file, orig_dest)
            
            # Extract mask from removed-background image
            mask = extract_alpha_mask(rem_file)
            
            # Save mask (use same basename as original but as PNG)
            mask_name = orig_file.stem + ".png"
            mask_dest = masks_dir / mask_name
            mask.save(mask_dest)
            
            paired_count += 1
            
        except Exception as e:
            print(f"\nError processing {orig_file.name}: {e}")
            skipped_count += 1
            continue
    
    print(f"\n✓ Successfully processed {paired_count} image-mask pairs")
    if skipped_count > 0:
        print(f"✗ Skipped {skipped_count} files due to errors")
    
    print(f"\nDataset ready at: {output_dir}")
    print(f"  Images: {images_dir} ({paired_count} files)")
    print(f"  Masks:  {masks_dir} ({paired_count} files)")
    
    # Show sample of created files
    print("\nSample files created:")
    sample_imgs = sorted(images_dir.glob("*"))[:5]
    sample_msks = sorted(masks_dir.glob("*"))[:5]
    
    for img, msk in zip(sample_imgs, sample_msks):
        print(f"  {img.name} → {msk.name}")

if __name__ == "__main__":
    main()
