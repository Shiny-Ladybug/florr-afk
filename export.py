import argparse
import os
import json
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model")
    parser.add_argument(
        "-i", "--image_path",
        type=str,
        default="./",
        help="Path to the image(s) to be processed",
    )
    parser.add_argument(
        "-o", "--output_path",
        type=str,
        default="./output/",
        help="Path to the output folder",
    )
    args = parser.parse_args()
    folders = [
        f for f in os.listdir(args.image_path)
        if os.path.isdir(os.path.join(args.image_path, f)) and f != ".git" and f != "__pycache__" and f != "output"
    ]
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    for folder in folders:
        folder_path = os.path.join(args.image_path, folder)
        png_path = os.path.join(folder_path, f"{folder}.png")
        json_path = os.path.join(folder_path, f"{folder}.json")
        if os.path.exists(png_path):
            shutil.copy(png_path, os.path.join(
                args.output_path, f"{folder}.png"))
        if os.path.exists(json_path):
            shutil.copy(json_path, os.path.join(
                args.output_path, f"{folder}.json"))
    for folder in folders:
        with open(os.path.join(args.output_path, f"{folder}.json"), "r") as f:
            label = json.load(f)
        label['imagePath'] = f"./{folder}.png"
        with open(os.path.join(args.output_path, f"{folder}.json"), "w") as f:
            json.dump(label, f, indent=4)
