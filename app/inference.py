import io
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Make sure U-2-Net repo is on path when running in Docker
U2NET_REPO = Path("/opt/U-2-Net")
if U2NET_REPO.exists() and str(U2NET_REPO) not in sys.path:
	sys.path.append(str(U2NET_REPO))

try:
	from model import U2NET  # type: ignore
except Exception as e:  # pragma: no cover
	U2NET = None  # type: ignore

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


def _postprocess(mask_logits: torch.Tensor, out_hw: Tuple[int, int]) -> Image.Image:
	# mask_logits: [1, 1, H, W]
	mask = torch.sigmoid(mask_logits)
	mask = F.interpolate(mask, size=out_hw, mode="bilinear", align_corners=False)
	mask = mask[0, 0].clamp(0, 1).cpu().numpy()
	mask_u8 = (mask * 255).astype(np.uint8)
	return Image.fromarray(mask_u8, mode="L")


def load_image_from_upload(file: UploadFile) -> Image.Image:
	content = file.file.read()
	return Image.open(io.BytesIO(content)).convert("RGB")


class BackgroundRemover:
	def __init__(self, weights_path: str, weights_url: str | None = None, device: Optional[str] = None) -> None:
		if U2NET is None:
			raise RuntimeError("U-2-Net repo not available. Ensure it is cloned in Docker build.")
		_ensure_weights(weights_path, weights_url)
		self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = U2NET(3, 1)
		state = torch.load(weights_path, map_location="cpu")
		self.model.load_state_dict(state)
		self.model.to(self.device)
		self.model.eval()

	@torch.inference_mode()
	def process_pil_to_png_bytes(self, image: Image.Image, threshold: float = 0.0) -> bytes:
		inp, (orig_h, orig_w) = _preprocess(image)
		inp = inp.to(self.device)
		outputs = self.model(inp)
		# U^2-Net returns multiple side outputs, take d0 (index 0)
		if isinstance(outputs, (list, tuple)):
			logits = outputs[0]
		else:
			logits = outputs
		alpha = _postprocess(logits, (orig_h, orig_w))
		if threshold > 0:
			alpha = alpha.point(lambda p: 255 if p >= int(threshold * 255) else 0)
		rgba = image.convert("RGBA")
		r, g, b, _ = rgba.split()
		rgba = Image.merge("RGBA", (r, g, b, alpha))
		buf = io.BytesIO()
		rgba.save(buf, format="PNG")
		return buf.getvalue()
