## U²-Net Clothing Background Removal (Training + API + Cloud Run)

This project helps you fine-tune U²-Net for clothing background removal and deploy a FastAPI inference service to Google Cloud Run.

### 1) Dataset structure
- Place your pairs as:
  - `dataset/images/` (input photos)
  - `dataset/masks/` (binary masks; clothing=white, background=black)

### 2) Train the Model

#### Option A: Train Locally (Cursor/Your Machine)
```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Prepare your dataset in this structure:
# dataset/
#   images/  # Your clothing photos
#   masks/   # Corresponding masks (white=clothing, black=background)

# Train the model
python train.py --data ./dataset --epochs 20 --batch-size 4 --lr 0.001

# Options:
#   --output      Path to save weights (default: ./weights/u2net_clothing.pth)
#   --epochs      Number of training epochs (default: 10)
#   --batch-size  Batch size (default: 4)
#   --lr          Learning rate (default: 0.001)
#   --size        Input image size (default: 320)
#   --device      Device: cuda or cpu (auto-detect if not set)
#   --resume      Path to checkpoint to resume from
```

#### Option B: Train in Google Colab
Open `colab_u2net_clothing_training.ipynb` in Colab. It will:
- Install deps and clone `U-2-Net`
- Load images/masks, apply transforms, fine-tune `U2NET(3,1)`
- Save weights to Drive as `u2net_clothing.pth`

### 3) Quick Test Training Setup
```bash
# Test that training setup works with dummy data
python test_train.py

# This creates test_dataset/ with dummy images/masks to verify everything loads correctly
```

### 4) Prepare weights
- After training, copy `u2net_clothing.pth` into `weights/u2net_clothing.pth`
- Or host it (e.g., GCS signed URL) and set env `WEIGHTS_URL` when deploying.

### 5) Run Inference Locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Start server (expects weights in weights/u2net_clothing.pth or WEIGHTS_URL set)
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Test:
```bash
curl -s -X POST "http://localhost:8080/removebg" \
  -F "file=@sample.jpg" \
  --output out.png
```

### 6) Build and Deploy to Cloud Run
```bash
# Set your project and region
export PROJECT_ID=your-gcp-project
export REGION=us-central1

gcloud auth login
gcloud config set project $PROJECT_ID

# Build container
gcloud builds submit --tag gcr.io/$PROJECT_ID/u2net-remover:latest .

# Deploy (unauthenticated for demo; restrict in prod)
gcloud run deploy u2net-remover \
  --image gcr.io/$PROJECT_ID/u2net-remover:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --cpu 2 --memory 2Gi --max-instances 3 \
  --set-env-vars WEIGHTS_URL="https://storage.googleapis.com/your-bucket/u2net_clothing.pth"
```

Invoke:
```bash
curl -s -X POST "$(gcloud run services describe u2net-remover --region $REGION --format='value(status.url)')/removebg" \
  -F "file=@sample.jpg" \
  --output out.png
```

### 7) API Documentation
- `GET /healthz` → `{ "status": "ok" }`
- `POST /removebg` (multipart form)
  - field `file`: image file (jpg/png/webp)
  - query `threshold` (optional float 0..1) to hard-mask alpha
  - returns `image/png` with transparent background

### Notes
- The Docker image clones `U-2-Net` to import the `U2NET` model.
- If you prefer baking weights, place them in `weights/` before building.
- CPU inference works; enable GPU variants in Cloud Run if available in your region for speed.
