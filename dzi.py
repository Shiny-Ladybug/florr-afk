import numpy as np
import cv2
import os
import math
from PIL import Image
from tqdm import tqdm
import pyvips
import json


def inference_flow(images_path):
    images = os.listdir(".")

    images = [name for name in images if os.path.isdir(
        os.path.join(".", name))]

    images.remove(".git")

    annotated_images_list = []

    for image_name in tqdm(images, desc="Processing images"):
        image_path = os.path.join(
            images_path, f"{image_name}/{image_name}.png")
        image = cv2.imread(image_path)
        cv2.putText(
            image,
            image_name[:5],
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )
        annotated_images_list.append(image)
    if not annotated_images_list:
        print("No annotations were generated to create a gallery.")
        return

    resized_images = []
    max_side = 200
    for img in tqdm(annotated_images_list, desc="Resizing images"):
        try:
            h, w = img.shape[:2]
            scale = max_side / max(h, w)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
            resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
            resized_images.append(resized)
        except Exception as e:
            print(f"Error resizing image: {e}")
            continue

    num_annotations = len(resized_images)
    cols = int(round(math.sqrt(num_annotations)))
    cols = max(cols, 1)
    rows = int(math.ceil(num_annotations / cols))

    tile_h, tile_w = max_side, max_side
    gallery_h = rows * tile_h
    gallery_w = cols * tile_w
    gallery = np.full((gallery_h, gallery_w, 3), 255, dtype=np.uint8)

    for i, ann in tqdm(enumerate(resized_images), desc="Creating gallery"):
        h, w = ann.shape[:2]
        r, c = divmod(i, cols)
        y = r * tile_h + (tile_h - h) // 2
        x = c * tile_w + (tile_w - w) // 2
        gallery[y:y+h, x:x+w] = ann

    print("Saving gallery image...")
    out_path = os.path.join(images_path, "gallery_annotations.png")
    if cv2.imwrite(out_path, gallery):
        print(f"Gallery saved to {out_path}")
    else:
        print("Error saving gallery image.")

    img = pyvips.Image.new_from_file(os.path.join(
        images_path, "gallery_annotations.png"))
    img.dzsave(os.path.join(
        images_path, "gallery_annotations"), tile_size=256)


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    inference_flow(base)
