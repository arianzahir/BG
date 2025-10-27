from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse
from .inference import BackgroundRemover, load_image_from_upload
import os

app = FastAPI(title="U2Net Clothing Background Removal", version="1.0.0")

# Lazy loader to speed container cold start
remover: BackgroundRemover | None = None


def get_remover() -> BackgroundRemover:
	global remover
	if remover is None:
		weights_path = os.getenv("WEIGHTS_PATH", "/app/weights/u2net_clothing.pth")
		weights_url = os.getenv("WEIGHTS_URL", "")
		remover = BackgroundRemover(weights_path=weights_path, weights_url=weights_url)
	return remover


@app.get("/healthz")
async def healthz():
	return JSONResponse({"status": "ok"})


@app.post("/removebg", summary="Remove background from a clothing image")
async def remove_background(
	file: UploadFile = File(..., description="Input image (jpg/png/webp)"),
	threshold: float = 0.0,
):
	try:
		image_pil = await load_image_from_upload(file)
		r = get_remover()
		png_bytes = r.process_pil_to_png_bytes(image_pil, threshold=threshold)
		return Response(content=png_bytes, media_type="image/png")
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))
