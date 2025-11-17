from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import os
import uuid
from typing import Optional, List
import time
import json


from experiments.robot.libero.libero_utils import quat2axisangle
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

import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


# possible checkpoints: openvla-7b, openvla-7b-finetuned-libero-spatial, openvla-7b-finetuned-libero-object, openvla-7b-finetuned-libero-goal, openvla-7b-finetuned-libero-10
# possible unnorm_key: bridge_orig, libero_spatial, libero_object, libero_goal, libero_10
class OpenVLAConfig:
    def __init__(self):
        self.model_family = "openvla"
        self.pretrained_checkpoint = "openvla/openvla-7b"  # <-- set your checkpoint path here!
        self.unnorm_key = "bridge_orig"
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
async def predict(image: UploadFile = File(...), prompt: str = Form(...), robot_ee_pos: Optional[str] = Form(None), robot_ee_quat: Optional[str] = Form(None), robot_gripper_pos: Optional[str] = Form(None)):
    try:
        # Read image from request
        img_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(pil_image)

        # Parse robot state from form data
        try:
            ee_pos = None
            ee_quat = None
            gripper_pos = None
            
            if robot_ee_pos:
                ee_pos = json.loads(robot_ee_pos)  # Convert "[x, y, z]" string to list
                
            if robot_ee_quat:
                ee_quat = json.loads(robot_ee_quat)  # Convert "[w, x, y, z]" string to list
                
            if robot_gripper_pos:
                gripper_pos = json.loads(robot_gripper_pos)
        except Exception as e:
            return JSONResponse(content={"error": f"Failed to parse robot state: {str(e)}"}, status_code=400)


        # Resize image if needed
        if img_np.shape[0] != resize_size or img_np.shape[1] != resize_size:
            pil_image = pil_image.resize((resize_size, resize_size), Image.BILINEAR)
            img_np = np.array(pil_image)

        # Prepare observation dict (like run_libero_eval.py)
        observation = {
            "full_image": img_np,
        }

        # Only add state if all robot state components are provided
        if ee_pos is not None and ee_quat is not None and gripper_pos is not None:

            ee_pos_array = np.array(ee_pos)
            ee_quat_array = np.array(ee_quat)
            gripper_pos_array = np.array([gripper_pos])
            
            # Then call quat2axisangle
            axis_angle = quat2axisangle(ee_quat_array)

            observation["state"] = np.concatenate((
                ee_pos_array,                    # robot0_eef_pos equivalent
                axis_angle,   # robot0_eef_quat equivalent (converted to axis-angle)
                gripper_pos_array                # robot0_gripper_qpos equivalent
            ))

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
    except Exception as e:
        # Log the full traceback
        error_msg = traceback.format_exc()
        logger.error(f"Error in predict endpoint: {error_msg}")
        
        # Also print to console (this should work with uvicorn)
        print(f"ERROR: {error_msg}")
        
        return JSONResponse(content={"error": str(e), "traceback": error_msg}, status_code=500)

#needs: export PYTHONPATH=$PYTHONPATH:/home/zipfelj/openvla/LIBERO
#run with: python -m uvicorn vlaServer:app --host 0.0.0.0 --port 8000
#may need: export PYTHONPATH=$PYTHONPATH:/home/zipfelj/openvla/ smth