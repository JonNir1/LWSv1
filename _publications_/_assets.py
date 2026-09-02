"""Shared asset-generation helpers (QR codes) for the poster notebooks."""
import os
from typing import Dict

import numpy as np
import qrcode
from PIL import Image


def make_qr_png(qr_type: str, output_dir: str, url_map: Dict[str, str]) -> Image.Image:
    qr_type = qr_type.lower()
    filepath = os.path.join(output_dir, f"QR_{qr_type}.png")
    try:
        qr_image = Image.open(filepath).convert("RGBA")
    except FileNotFoundError:
        if qr_type not in url_map:
            raise ValueError(f"Invalid qr_type: {qr_type}")
        url = url_map[qr_type]

        qr = qrcode.QRCode(version=None, box_size=10, border=4, error_correction=qrcode.constants.ERROR_CORRECT_H)
        qr.add_data(url)
        qr.make(fit=True)
        qr_image = qr.make_image(fill_color=(0, 0, 0), back_color=(255, 255, 255))
        # make transparent background
        qr_image = qr_image.convert("RGBA")
        img_arr = np.array(qr_image)
        is_white_pixel = (img_arr[:, :, :3] == 255).all(axis=2)
        img_arr[is_white_pixel, 3] = 0   # alpha channel
        qr_image = Image.fromarray(img_arr)
        qr_image.save(filepath)
    return qr_image
