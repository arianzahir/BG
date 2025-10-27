# Background Removal UI Guide

## 🎉 Your Frontend is Ready!

A simple, beautiful web UI has been created for testing your trained model.

## 🚀 How to Use

### 1. Start the Server
```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Open the UI
Open your browser and go to:
```
http://localhost:8000
```

### 3. Upload an Image
- **Click** the upload area to select an image
- Or **drag and drop** an image onto the upload area
- Supports JPG, PNG, and WebP formats

### 4. Remove Background
- Click the "Remove Background" button
- Wait for processing (usually 2-5 seconds)
- View your result with transparent background!

## ✨ Features

- **Beautiful, modern UI** with gradient design
- **Drag & drop** file upload
- **Preview** of original and result side-by-side
- **Fast processing** using your trained model
- **Mobile responsive** design

## 📁 Files

- **Frontend**: `static/index.html`
- **Backend API**: `app/main.py`
- **Trained Model**: `weights/u2net_clothing.pth`

## 🧪 Testing

### Test via UI
1. Go to `http://localhost:8000`
2. Upload an image
3. Click "Remove Background"

### Test via Command Line
```bash
curl -X POST "http://localhost:8000/removebg" \
  -F "file=@./dataset/images/0071.jpg" \
  --output result.png
```

### Test via Python Script
```bash
python test_model.py --weights ./weights/u2net_clothing.pth --device auto
```

## 📊 API Endpoints

- `GET /` - Frontend UI
- `GET /healthz` - Health check
- `POST /removebg` - Remove background from image
- `GET /docs` - API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

## 🎨 Customization

Edit `static/index.html` to customize:
- Colors
- Layout
- Text
- Styling

The UI uses vanilla JavaScript, so no build process needed!

## 🐛 Troubleshooting

**Server not starting?**
```bash
pkill -f uvicorn
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**Model not loading?**
- Check that `weights/u2net_clothing.pth` exists
- Verify the weights file is valid

**UI not loading?**
- Make sure the server is running
- Check browser console for errors
- Visit `http://localhost:8000/healthz` to verify server is up

## 🎯 Next Steps

1. Test with different clothing images
2. Fine-tune the model if needed (train more epochs)
3. Deploy to production (see `deploy_cloud_run.sh`)
4. Customize the UI to match your brand

## 🚀 Deployment

To deploy to Google Cloud Run:
```bash
bash deploy_cloud_run.sh
```

This will build and deploy your model with the web UI to a public URL!


