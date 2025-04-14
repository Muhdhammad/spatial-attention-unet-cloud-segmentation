from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import torch
import io
from pathlib import Path
import os
from model import unet

app = FastAPI(title="Cloud Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = unet(in_channels=4, num_classes=1).to(device)

current_dir = Path(__file__).parent

model_path = current_dir / "pytorch_model.bin"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model weights file not found in {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def process_bands(red_bytes, green_bytes, blue_bytes, nir_bytes):
    try:
        # Open each band as an image
        red = Image.open(io.BytesIO(red_bytes))
        green = Image.open(io.BytesIO(green_bytes))
        blue = Image.open(io.BytesIO(blue_bytes))
        nir = Image.open(io.BytesIO(nir_bytes))

        # Stack and normalize as in your training code
        rgb = np.stack([np.array(red), np.array(green), np.array(blue)], axis=2)
        nir = np.expand_dims(np.array(nir), 2)
        combined = np.concatenate([rgb, nir], axis=2)
        normalized = (combined / np.iinfo(combined.dtype).max).astype(np.float32)
        
        return torch.from_numpy(normalized.transpose((2, 0, 1))).unsqueeze(0).to(device)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing images: {str(e)}")


@app.post("/predict", summary="predict cloud segmentation mask")
async def predict(
    red: UploadFile = File(..., description="Red band tiff image"),
    green: UploadFile = File(..., description="Green band tiff image"),
    blue: UploadFile = File(..., description="Blue band tiff image"),
    nir: UploadFile = File(..., description="Nir band tiff image")
):
    try:
        red_content = await red.read()
        green_content = await green.read()
        blue_content = await blue.read()
        nir_content = await nir.read()

        input_tensor = process_bands(red_content, green_content, blue_content, nir_content)

        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.sigmoid(output)
            mask = (pred > 0.5).float().squeeze().cpu().numpy()
        
        # Convert mask to PNG and return
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))

        img_byte_arr = io.BytesIO()
        mask_img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        await red.close()
        await green.close()
        await blue.close()
        await nir.close()


@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "healthy"})
    




