import argparse
import os
from PIL import Image
from models.room_type_classification import RoomClassifier

#code starts here
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input directory containing images")
    parser.add_argument(
        "--cpu",
        action="store_false",
        dest="use_cuda",
        default=True,
        help="Device to run computation. Either CPU or GPU",
    )
    args = parser.parse_args()
    img_files = [args.input]
    # change img_files if args.input a directory
    if os.path.isdir(args.input):
        img_files = [os.path.join(args.input, f) for f in os.listdir(args.input)]
    print(f"Found {len(img_files)} images.")
    model = RoomClassifier(use_cuda=args.use_cuda)
    print("Model has been loaded.")
    print()
    for i, img_path in enumerate(img_files):
        img = Image.open(img_path)
        label_conf = model(img, return_confidence=True)
        print(f"{i+1}. {img_path} --> {label_conf}")
