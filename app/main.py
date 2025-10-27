from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .inference import BackgroundRemover, load_image_from_upload
import os
from pathlib import Path

app = FastAPI(title="U2Net Clothing Background Removal", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Lazy loader to speed container cold start
remover: BackgroundRemover | None = None


def get_remover() -> BackgroundRemover:
	global remover
	if remover is None:
		# Try local weights first, then Docker path, then URL
		local_weights = Path(__file__).parent.parent / "weights" / "u2net_clothing.pth"
		if local_weights.exists():
			weights_path = str(local_weights)
		else:
			weights_path = os.getenv("WEIGHTS_PATH", "/app/weights/u2net_clothing.pth")
		weights_url = os.getenv("WEIGHTS_URL", "")
		remover = BackgroundRemover(weights_path=weights_path, weights_url=weights_url)
	return remover


@app.get("/")
async def root():
	"""Serve the frontend UI"""
	index_path = Path(__file__).parent.parent / "static" / "index.html"
	if index_path.exists():
		return FileResponse(str(index_path))
	return {"message": "Background Removal API - Visit /docs for API documentation"}

@app.get("/healthz")
async def healthz():
	return JSONResponse({"status": "ok"})


@app.post("/removebg", summary="Remove background from a clothing image")
async def remove_background(
	file: UploadFile = File(..., description="Input image (jpg/png/webp)"),
	threshold: float = 0.5,
):
	try:
		image_pil = await load_image_from_upload(file)
		r = get_remover()
		png_bytes = r.process_pil_to_png_bytes(image_pil, threshold=threshold)
		return Response(content=png_bytes, media_type="image/png")
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))
