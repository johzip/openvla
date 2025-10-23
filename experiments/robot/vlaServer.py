from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import os
import uuid
import time


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


class OpenVLAConfig:
    def __init__(self):
        self.model_family = "openvla"
        self.pretrained_checkpoint = "openvla/openvla-7b"  # <-- set your checkpoint path here!
        self.unnorm_key = "bridge_orig"  # or your dataset key
        self.center_crop = False         # set True if your model was trained with image aug
        self.load_in_8bit = False
        self.load_in_4bit = False

cfg = OpenVLAConfig()
set_seed_everywhere(42)

# ====== LOAD MODEL AND PROCESSOR ONCE ======
model = get_model(cfg)
processor = get_processor(cfg)
resize_size = get_image_resize_size(cfg)

@app.post("/predict")
async def predict(image: UploadFile = File(...), prompt: str = Form(...)):
    # Read image from request
    img_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(pil_image)

    ## Save image to 'out' folder
    #os.makedirs("out", exist_ok=True)
    #timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    #unique_id = uuid.uuid4().hex[:8]
    #image_save_path = os.path.join("out", f"{timestamp}_{unique_id}.png")
    #pil_image.save(image_save_path)

    # Resize image if needed
    if img_np.shape[0] != resize_size or img_np.shape[1] != resize_size:
        pil_image = pil_image.resize((resize_size, resize_size), Image.BILINEAR)
        img_np = np.array(pil_image)

    # Prepare observation dict (like run_libero_eval.py)
    observation = {
        "full_image": img_np
        # Add "state" if your model expects it, otherwise omit
    }

    # Use prompt as task_description
    task_description = prompt

    # Get action from OpenVLA
    try:
        action = get_action(
            cfg,
            model,
            observation,
            task_description,
            processor=processor,
        )
        # Convert to list for JSON serialization
        action_list = action.tolist() if hasattr(action, "tolist") else list(action)
        return JSONResponse(content={"action": action_list})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

#run with: python -m uvicorn vlaServer:app --host 0.0.0.0 --port 8000
#may need: export PYTHONPATH=$PYTHONPATH:/home/zipfelj/openvla/ smth