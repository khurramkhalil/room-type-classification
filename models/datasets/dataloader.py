import sys
import torch

from PIL import Image
from torchvision import datasets


def loader(path):
    return Image.open(path)


def dataset_loader(folder_path, ext_list, transform, BATCH_SIZE=64):
    folder_dataset = datasets.DatasetFolder(
        root=folder_path, loader=loader, extensions=ext_list, transform=transform
    )
    batch_loader = torch.utils.data.DataLoader(
        folder_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    class_labels, labels_to_index = folder_dataset.find_classes(
        folder_path
    )  # Corrected line

    return labels_to_index, batch_loader


if __name__ == "__main__":
    # root_dir_path = sys.argv[1]
    root_dir_path = "/home/proptx/dataset/CLIP/"
    ext_list = (
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
    )
    true_labels, bacth_loader = dataset_loader(root_dir_path, ext_list, transform=None)
    print("True Labels", true_labels)
    print("------------")
