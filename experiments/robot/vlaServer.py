from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io


from openvla_utils import get_processor
from robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


app = FastAPI()

@app.post("/predict")
async def predict(image: UploadFile = File(...), prompt: str = Form(...)):
    # Read image
    img_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # TODO: Process image and prompt with OpenVLA here
    # For demo, return dummy numbers
    action = np.random.rand(7).tolist()
    return JSONResponse(content={"action": action})

#run with: python -m uvicorn vlaServer:app --host 0.0.0.0 --port 8000