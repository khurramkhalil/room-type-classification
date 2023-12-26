import base64
import io
import json
import logging
import os
import sys
import time

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, Request, Response
from PIL import Image
from models.room_type_classification import RoomClassifier

# Logging setup
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialise FastAPI APP, model
APP = FastAPI()
MODEL = None

# ===============================================================
# Helpers
# ===============================================================


def cast_to(data, cast_type):
    try:
        return cast_type(data)
    except:
        raise RuntimeError(f"Could not cast {data} to {str(cast_type)}")


def get_image_from_url(image_url):
    response = requests.get(image_url)
    image = Image.open(io.BytesIO(response.content))
    return image


def decode_base64_image(image_string):
    image_binary = base64.b64decode(image_string)
    image = Image.open(io.BytesIO(image_binary))
    return image


def image_to_base64(output_image, to=".jpg"):
    _, encoded_image = cv2.imencode(to, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    image_string = base64.b64encode(encoded_image).decode("utf-8")
    return image_string


# ===============================================================
# App
# ===============================================================


@APP.on_event("startup")
def init():
    global MODEL
    MODEL = RoomClassifier(use_cuda=True)


@APP.get("/health")
def get_health():
    logger.info("Health check successful!")
    return Response(content=json.dumps({"health_status": "ok"}), status_code=200)


def _predict(request_body):
    # Try to parse the JSON string into a Python dictionary
    print("predict function")
    print(type(request_body))
    if isinstance(request_body, str):
        request_body = json.loads(request_body)
    elif not isinstance(request_body, dict):
        raise RuntimeError(f"Invalid request body type: {type(request_body)}")

    print("Extracting instances")
    # Get instances
    if "instances" in request_body.keys():
        instances = request_body["instances"]
    else:
        raise RuntimeError("Instances not found in request")

    if not isinstance(instances, list):
        raise RuntimeError(f"Invalid instances type: {type(instance)}")

    # Only process the first image
    # TODO: add batch processing
    instance = instances[0]
    print("instance", instance)
    # Try to parse the JSON string into a Python dictionary
    if isinstance(instance, str):
        instance = json.loads(instance)
    elif not isinstance(instance, dict):
        raise RuntimeError(f"Invalid instance type: {type(instance)}")

    # Get input arguments
    # Compulsory arguments

    # Get image
    if "image_url" in instance.keys():
        logger.info(f'Retrieving image from {instance["image_url"]}')
        image = get_image_from_url(instance["image_url"])
    elif "image" in instance.keys():
        # If image is provided as base64 string, decode it
        image = decode_base64_image(instance["image"])
    else:
        raise RuntimeError("Either an image url or a base64-encoded image is required")

    print(image.mode)

    if image.mode == "RGBA":
        # Convert RGBA to RGB
        image = image.convert("RGB")

    # Optional arguments
    conf = instance.get("conf_thresh", 0.01)

    # Run model
    outputs = MODEL(image)
    scores = [score for label, score in outputs]
    labels = [label for label, score in outputs]
    print(f"{labels} --> {scores}")

    normalized_score = [format(score * 100, ".1f") for score in scores]

    scores = np.array(scores)
    labels = np.array(labels)
    labels = labels[scores >= conf].tolist()
    scores = scores[scores >= conf].tolist()

    result = {"result": labels, "scores": normalized_score}

    return result


@APP.post("/predict")
async def predict(request: Request):
    tic = time.time()
    try:
        body = await request.json()
        result = _predict(body)

        result["time_taken"] = time.time() - tic

        response = {"predictions": [result]}
        return Response(status_code=200, content=json.dumps(response))
    except Exception as e:
        print(e)
        response = {"predictions": [{"error": str(e), "time_taken": time.time() - tic}]}
        return Response(status_code=500, content=json.dumps(response))
