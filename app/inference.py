import io
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2

# Make sure U-2-Net repo is on path (check both local and Docker paths)
U2NET_REPO = None
for path in [Path("./U-2-Net"), Path("/opt/U-2-Net")]:
	if path.exists():
		U2NET_REPO = path
		if str(path) not in sys.path:
			sys.path.insert(0, str(path))
		break

# Try to import U2NET model
U2NET = None
if U2NET_REPO:
	try:
		sys.path.insert(0, str(U2NET_REPO / "model"))
		from u2net import U2NET  # type: ignore
	except Exception as e:
		try:
			# Fallback: try importing from parent directory
			from model.u2net import U2NET  # type: ignore
		except Exception as e2:
			print(f"Warning: Could not import U2NET: {e}")

import requests
from fastapi import UploadFile


def _ensure_weights(weights_path: str, weights_url: str | None) -> None:
	p = Path(weights_path)
	if p.exists():
		return
	if weights_url:
		p.parent.mkdir(parents=True, exist_ok=True)
		resp = requests.get(weights_url, timeout=60)
		resp.raise_for_status()
		p.write_bytes(resp.content)
		return
	raise FileNotFoundError(
		f"Weights not found at {weights_path}. Provide WEIGHTS_URL or bake weights in image."
	)


def _preprocess(image: Image.Image, input_size: int = 320) -> Tuple[torch.Tensor, Tuple[int, int]]:
	orig_w, orig_h = image.size
	img = image.convert("RGB").resize((input_size, input_size), Image.BILINEAR)
	arr = np.asarray(img).astype(np.float32) / 255.0
	# HWC -> CHW
	arr = np.transpose(arr, (0, 1, 2))
	arr = np.transpose(arr, (2, 0, 1))
	tensor = torch.from_numpy(arr).unsqueeze(0)
	return tensor, (orig_h, orig_w)


def _postprocess(mask_logits: torch.Tensor, out_hw: Tuple[int, int], threshold: float = 0.5, image: Image.Image = None) -> Image.Image:
	# mask_logits: [1, 1, H, W]
	mask = torch.sigmoid(mask_logits)
	mask = F.interpolate(mask, size=out_hw, mode="bilinear", align_corners=False)
	mask = mask[0, 0].clamp(0, 1).cpu().numpy()
	
	# Convert to uint8
	mask_u8 = (mask * 255).astype(np.uint8)
	
	# Apply adaptive thresholding for better edge detection
	# Use Otsu's method for automatic threshold selection
	_, mask_binary = cv2.threshold(mask_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	
	# Fine-tune with manual threshold if needed
	if threshold > 0:
		manual_thresh = int(threshold * 255)
		_, manual_mask = cv2.threshold(mask_u8, manual_thresh, 255, cv2.THRESH_BINARY)
		# Combine Otsu and manual threshold
		mask_binary = cv2.bitwise_and(mask_binary, manual_mask)
	
	# Apply more aggressive morphological operations to fill holes
	kernel_close = np.ones((5, 5), np.uint8)
	mask_clean = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
	
	# Remove small noise/holes using contour area filtering
	contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if contours:
		# Get the largest contour (should be the main subject)
		largest_contour = max(contours, key=cv2.contourArea)
		mask_clean = np.zeros_like(mask_clean)
		cv2.fillPoly(mask_clean, [largest_contour], 255)
		
		# Fill remaining small holes
		mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close, iterations=1)
	
	# Apply opening to remove small protrusions
	kernel_open = np.ones((3, 3), np.uint8)
	mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel_open, iterations=1)
	
	# Edge-aware smoothing using bilateral filter on the original prediction
	# This helps with fine details like hair
	if image is not None:
		img_array = np.array(image)
		if img_array.shape[2] == 3:  # RGB
			mask_smooth = cv2.bilateralFilter(mask_u8.astype(np.uint8), 5, 50, 50)
			# Combine with binary mask
			_, mask_smooth_binary = cv2.threshold(mask_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			# Use the cleaner binary version for main subject
			mask_clean = np.maximum(mask_clean, mask_smooth_binary)
	
	# Final closing to ensure smooth edges
	mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
	
	return Image.fromarray(mask_clean, mode="L")


async def load_image_from_upload(file: UploadFile) -> Image.Image:
	content = await file.read()
	return Image.open(io.BytesIO(content)).convert("RGB")


class BackgroundRemover:
	def __init__(self, weights_path: str, weights_url: str | None = None, device: Optional[str] = None) -> None:
		if U2NET is None:
			raise RuntimeError("U-2-Net repo not available. Ensure the U-2-Net repository is cloned (should be in ./U-2-Net)")
		_ensure_weights(weights_path, weights_url)
		if device:
			self.device = torch.device(device)
		elif torch.cuda.is_available():
			self.device = torch.device("cuda")
		elif torch.backends.mps.is_available():
			self.device = torch.device("mps")
		else:
			self.device = torch.device("cpu")
		self.model = U2NET(3, 1)
		state = torch.load(weights_path, map_location="cpu")
		self.model.load_state_dict(state)
		self.model.to(self.device)
		self.model.eval()

	@torch.inference_mode()
	def process_pil_to_png_bytes(self, image: Image.Image, threshold: float = 0.5) -> bytes:
		inp, (orig_h, orig_w) = _preprocess(image)
		inp = inp.to(self.device)
		outputs = self.model(inp)
		# U^2-Net returns multiple side outputs, take d0 (index 0)
		if isinstance(outputs, (list, tuple)):
			logits = outputs[0]
		else:
			logits = outputs
		alpha = _postprocess(logits, (orig_h, orig_w), threshold=threshold, image=image)
		rgba = image.convert("RGBA")
		r, g, b, _ = rgba.split()
		rgba = Image.merge("RGBA", (r, g, b, alpha))
		buf = io.BytesIO()
		rgba.save(buf, format="PNG")
		return buf.getvalue()
