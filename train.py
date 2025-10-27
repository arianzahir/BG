#!/usr/bin/env python3
"""Train U²-Net on clothing dataset"""
import os
import sys
import glob
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Try to load U2NET locally first, then from git clone
try:
    from model import U2NET
except ImportError:
    # Clone U-2-Net if not present
    u2net_path = Path("./U-2-Net")
    if not u2net_path.exists():
        print("Cloning U-2-Net repository...")
        os.system("git clone https://github.com/xuebinqin/U-2-Net.git")
    if (u2net_path / "model" / "u2net.py").exists():
        sys.path.insert(0, str(u2net_path / "model"))
        from u2net import U2NET
    else:
        print("ERROR: Could not find U2NET model definition")
        sys.exit(1)


class ClothingDataset(Dataset):
    """Dataset for clothing images with masks"""
    def __init__(self, img_dir, mask_dir, size=320):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*")))
        
        if len(self.img_paths) == 0:
            raise ValueError(f"No images found in {img_dir}")
        if len(self.img_paths) != len(self.mask_paths):
            raise ValueError(f"Image/Mask count mismatch: {len(self.img_paths)} vs {len(self.mask_paths)}")
        
        self.size = size
        print(f"Loaded {len(self.img_paths)} image-mask pairs")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        
        # Resize to training size
        img = img.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.BILINEAR)
        
        # Convert to numpy and normalize
        img = np.array(img).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = np.expand_dims(mask, 0)  # Add channel dimension
        
        return torch.from_numpy(img), torch.from_numpy(mask)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Training")
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # U²-Net returns multiple outputs, use the first one (main output)
        if isinstance(outputs, (list, tuple)):
            pred = outputs[0]
        else:
            pred = outputs
        
        loss = criterion(pred, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train U²-Net for clothing background removal")
    parser.add_argument("--data", type=str, default="./dataset", help="Dataset root directory")
    parser.add_argument("--images", type=str, default=None, help="Image directory (overrides --data/images)")
    parser.add_argument("--masks", type=str, default=None, help="Mask directory (overrides --data/masks)")
    parser.add_argument("--output", type=str, default="./weights/u2net_clothing.pth", help="Output weights path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--size", type=int, default=320, help="Input image size")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu, auto-detect if not set)")
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():  # MPS for Apple Silicon
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Dataset paths
    if args.images and args.masks:
        img_dir, mask_dir = args.images, args.masks
    else:
        img_dir = os.path.join(args.data, "images")
        mask_dir = os.path.join(args.data, "masks")
    
    # Create dataset
    dataset = ClothingDataset(img_dir, mask_dir, size=args.size)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=2,  # Reduced from 4 for better compatibility with older CPUs
        pin_memory=True if (torch.cuda.is_available() or torch.backends.mps.is_available()) else False
    )
    
    # Model
    model = U2NET(3, 1).to(device)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))
        # You could also save/load optimizer state and epoch number here
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = F.binary_cross_entropy_with_logits
    
    # Output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Average loss: {avg_loss:.4f}")
        
        # Save checkpoint after each epoch
        torch.save(model.state_dict(), args.output)
        print(f"Saved checkpoint to {args.output}")
    
    print("\nTraining completed!")
    print(f"Final model saved to: {args.output}")


if __name__ == "__main__":
    main()
