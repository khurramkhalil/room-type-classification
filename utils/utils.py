import cv2
import numpy as np
from PIL import Image


def resize_image(pil_img, resolution):
    np_img = np.array(pil_img)

    H, W, C = np_img.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    resized_img = cv2.resize(
        np_img, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA
    )

    resized_pil_img = Image.fromarray(resized_img)
    return resized_pil_img
