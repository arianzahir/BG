# Training Guide

This guide walks you through training U²-Net for clothing background removal on your local machine (or in Cursor).

## Prerequisites

- Python 3.8+ 
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM
- Disk space for dataset and model weights

## Step 1: Prepare Your Dataset

You need paired images and masks:

```
dataset/
├── images/
│   ├── shirt_001.jpg
│   ├── shirt_002.jpg
│   └── ...
└── masks/
    ├── shirt_001.png   # Binary mask: white=clothing, black=background
    ├── shirt_002.png
    └── ...
```

**Important:** Image and mask filenames must match (use same extensions or rename to match).

### Creating Masks

You can create masks using:

1. **Manual annotation**: Tools like [GIMP](https://www.gimp.org/), Photoshop
2. **Semi-automatic**: [remove.bg API](https://www.remove.bg/api) → convert to binary mask
3. **Pre-trained models**: Use U²-Net general model to generate initial masks, then refine
4. **Image editors**: Polygonal selection → fill clothing area with white, background with black

## Step 2: Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Step 3: Test Your Setup

```bash
python test_train.py
```

This creates a dummy dataset and verifies everything works.

## Step 4: Start Training

### MacBook M2 Air Users
For optimal performance on your M2 Air, use MPS (Metal Performance Shaders):

```bash
# Recommended settings for M2 Air (8-16GB RAM)
python train.py \
  --data ./dataset \
  --epochs 20 \
  --batch-size 4 \
  --lr 0.001 \
  --size 320 \
  --device mps
```

**Time estimates for M2 Air:**
- **10 epochs, 200 images**: ~2-5 hours
- **20 epochs, 500 images**: ~5-10 hours
- **50 epochs, 1000 images**: ~15-25 hours

**Tip**: If you get memory errors, reduce `--batch-size` to 2. For faster training with more RAM, try `--batch-size 6`.

### Basic Training
```bash
python train.py --data ./dataset
```

### Advanced Training with Custom Parameters
```bash
python train.py \
  --data ./dataset \
  --output ./weights/u2net_clothing.pth \
  --epochs 50 \
  --batch-size 8 \
  --lr 0.0001 \
  --size 320 \
  --device mps  # Use 'cuda' for NVIDIA GPU, 'mps' for Apple Silicon, 'cpu' for CPU
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | `./dataset` | Root directory containing `images/` and `masks/` folders |
| `--images` | None | Override image directory |
| `--masks` | None | Override mask directory |
| `--output` | `./weights/u2net_clothing.pth` | Output path for trained weights |
| `--epochs` | `10` | Number of training epochs |
| `--batch-size` | `4` | Batch size (reduce if OOM, increase if you have GPU memory) |
| `--lr` | `0.001` | Learning rate |
| `--size` | `320` | Input image size (320x320 is standard for U²-Net) |
| `--device` | Auto | `cuda`, `mps` (Apple Silicon), or `cpu` |
| `--resume` | None | Path to checkpoint to resume training from |

### Resuming Training

If training is interrupted, you can resume from the last checkpoint:

```bash
python train.py --data ./dataset --resume ./weights/u2net_clothing.pth --epochs 20
```

## Step 5: Monitor Training

Training output shows:
- Current epoch and batch
- Real-time loss value
- Average loss per epoch

Example output:
```
Epoch 1/10
Training: 100%|████████| 25/25 [00:45<00:00, 1.81s/it, loss=0.1234]
Average loss: 0.1456
Saved checkpoint to ./weights/u2net_clothing.pth
```

## Step 6: Evaluate Your Model

After training, test your model:

```bash
# Start the inference server with your trained weights
uvicorn app.main:app --host 0.0.0.0 --port 8080

# In another terminal, test with an image
curl -X POST "http://localhost:8080/removebg" \
  -F "file=@your_test_image.jpg" \
  --output result.png
```

## Tips for Better Results

### Dataset Quality
- **Quantity**: Start with 100+ image pairs, ideally 500-1000+
- **Diversity**: Include various clothing types, colors, backgrounds
- **Quality**: High-resolution images (at least 512x512) work better
- **Masks**: Accurate masks are crucial - spend time refining them

### Training Parameters
- **GPU Memory**: If you get OOM errors, reduce `--batch-size` to 2 or 1
- **Learning Rate**: Start with 0.001. If loss doesn't decrease, try 0.0001 or 0.0005
- **Epochs**: Start with 10-20 epochs. Overfitting can occur with small datasets
- **Early Stopping**: Monitor validation loss if you split your dataset

### Hardware Recommendations
- **Minimum**: CPU training works but is very slow (hours to days)
- **Recommended**: 
  - NVIDIA GPU with 4GB+ VRAM (GTX 1050 Ti or better) - **~1-3 hours** for 10 epochs
  - Apple Silicon Mac (M1/M2/M3) - **~2-5 hours** for 10 epochs with MPS
- **Ideal**: GPU with 8GB+ VRAM (RTX 3060, RTX 3070, etc.) - **~30-90 minutes** for 10 epochs

## Troubleshooting

### "No images found"
- Check your dataset path is correct
- Verify folder names are exactly `images` and `masks` (case-sensitive)
- Ensure images have valid extensions (.jpg, .png, etc.)

### "CUDA out of memory"
- Reduce `--batch-size` (try 2 or 1)
- Reduce `--size` (try 256 instead of 320)

### Model doesn't converge (loss doesn't decrease)
- Lower learning rate (try 0.0001 or 0.0005)
- Check your masks are correct (white=clothing, black=background)
- Verify images and masks are paired correctly

### Import error for U2NET model
- Run: `git clone https://github.com/xuebinqin/U-2-Net.git`
- The training script should auto-clone, but manual clone works too

## Next Steps

Once training completes:
1. Your weights are saved to `./weights/u2net_clothing.pth`
2. Test inference locally (see Step 6 above)
3. Deploy to Cloud Run for production use
4. Integrate with your mobile app
