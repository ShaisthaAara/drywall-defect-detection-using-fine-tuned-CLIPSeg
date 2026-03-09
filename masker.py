import os
import json
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm

# -----------------------------
# INPUT PATHS
# -----------------------------

crack_train_images_path = r"C:\Users\Shaistha\Desktop\origin\cracks\train"
crack_train_coco_path = r"C:\Users\Shaistha\Desktop\origin\cracks\train\_annotations.coco.json"

crack_test_images_path = r"C:\Users\Shaistha\Desktop\origin\cracks\test"
crack_test_coco_path = r"C:\Users\Shaistha\Desktop\origin\cracks\test\_annotations.coco.json"

joins_train_images_path = r"C:\Users\Shaistha\Desktop\origin\joins\train"
joins_train_coco_path = r"C:\Users\Shaistha\Desktop\origin\joins\train\_annotations.coco.json"

joins_test_images_path = r"C:\Users\Shaistha\Desktop\origin\joins\test"
joins_test_coco_path = r"C:\Users\Shaistha\Desktop\origin\joins\test\_annotations.coco.json"

# -----------------------------
# OUTPUT DATASET
# -----------------------------

OUTPUT_DIR = r"C:\Users\Shaistha\Desktop\clipseg_dataset"

images_out = os.path.join(OUTPUT_DIR, "images")
masks_out = os.path.join(OUTPUT_DIR, "masks")

os.makedirs(images_out, exist_ok=True)
os.makedirs(masks_out, exist_ok=True)

metadata = []

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def polygon_to_mask(width, height, segmentation):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for seg in segmentation:
        polygon = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
        draw.polygon(polygon, outline=1, fill=1)

    return np.array(mask)


def bbox_to_mask(width, height, bbox):
    x, y, w, h = bbox
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[int(y):int(y+h), int(x):int(x+w)] = 1
    return mask


# -----------------------------
# PROCESS COCO DATASET
# -----------------------------

def process_dataset(images_path, coco_path, prompt):

    with open(coco_path) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    for img_id, img_info in tqdm(images.items()):

        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        image_path = os.path.join(images_path, file_name)

        if not os.path.exists(image_path):
            continue

        mask = np.zeros((height, width), dtype=np.uint8)

        anns = anns_by_img.get(img_id, [])

        for ann in anns:

            if ann["segmentation"]:
                poly_mask = polygon_to_mask(width, height, ann["segmentation"])
                mask = np.maximum(mask, poly_mask)

            else:
                bbox_mask = bbox_to_mask(width, height, ann["bbox"])
                mask = np.maximum(mask, bbox_mask)

        mask = (mask * 255).astype(np.uint8)

        new_name = f"{prompt.replace(' ', '_')}_{file_name}"

        img_out_path = os.path.join(images_out, new_name)
        mask_out_path = os.path.join(masks_out, new_name.replace(".jpg", ".png"))

        # copy image
        img = cv2.imread(image_path)
        cv2.imwrite(img_out_path, img)

        # save mask
        cv2.imwrite(mask_out_path, mask)

        metadata.append({
            "image": new_name,
            "prompt": prompt
        })


# -----------------------------
# RUN FOR ALL DATASETS
# -----------------------------

print("Processing crack train")
process_dataset(crack_train_images_path, crack_train_coco_path, "crack")

print("Processing crack test")
process_dataset(crack_test_images_path, crack_test_coco_path, "crack")

print("Processing joins train")
process_dataset(joins_train_images_path, joins_train_coco_path, "drywall joint")

print("Processing joins test")
process_dataset(joins_test_images_path, joins_test_coco_path, "drywall joint")


# -----------------------------
# SAVE METADATA
# -----------------------------

df = pd.DataFrame(metadata)
df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

print("Dataset ready!")