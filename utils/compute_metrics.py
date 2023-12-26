import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn import metrics

from models.room_type_classification import RoomClassifier


def make_pred(input_dir, filename="logs.csv"):
    # input_dir = '/home/usama/usama/data/proptexx_data/room_type_classification/images/usama_added/'
    # input_dir = '/home/usama/usama/data/proptexx_data/room_type_classification/images/train'

    model = RoomClassifier(device="gpu")

    for cls_name in os.listdir(input_dir):
        for img_name in os.listdir(os.path.join(input_dir, cls_name)):
            img_path = os.path.join(input_dir, cls_name, img_name)
            img = Image.open(img_path)
            print(img_path, model(img, k=1))
            label = model(img, k=1)[0]

            print(f"{cls_name},{label},{img_path}")
            with open(filename, "a") as file_obj:
                file_obj.write(f"{cls_name},{label},{img_path}\n")


def make_acc(filename):
    logs = pd.read_csv(filename)
    all_cls = [
        "Balcony",
        "Bar",
        "Basement",
        "Bathroom",
        "Bedroom",
        "Children's Room",
        "Cinema Room",
        "Closet",
        "Dining Room",
        "Empty Room",
        "Game Room",
        "Garage",
        "Garden/Yard",
        "Gym",
        "Kitchen",
        "Laundry Room",
        "Library/Study",
        "Living Room",
        "Loft",
        "Office",
        "Garden w/pool",
        "Storage Room",
        "Terrace",
        "Wine Cellar",
        "Lobby",
        "Gazebo",
        "Mud Room",
        "Workshop",
        "Spa/Sauna Room",
        "Billiard Room",
        "Athletic Court",
        "Pantry",
        "Conservatory",
        "Patio",
        "Hearth Room",
        "Play Room",
        "Porch",
        "Studio",
        "Deck",
        "Computer Room",
        "Amusement Room",
        "Informal Dinning Room",
    ]

    mapping = {
        "conservatory": "Conservatory",
        "deck": "Deck",
        "gazebo": "Gazebo",
        "hearth_room": "Hearth Room",
        "patio": "Patio",
        "Cinema_Room": "Cinema Room",
        "play_room": "Play Room",
        "studio": "Studio",
        "Garden": "Garden/Yard",
        "Laundry": "Laundry Room",
        "Storage_room": "Storage Room",
        "spa_or_sauna_room": "SpaSauna Room",
        "billiard_room": "Billiard Room",
        "Library": "Library Building",
        "Living_Room": "Living Room",
        "Dining": "Dining Room",
        "athletic_court": "Athletic Court",
        "Pool": "Garden w/pool",
        "pantry": "Pantry",
        "Empty_Room": "Empty Room",
        "Wine_Room": "Wine Cellar",
        "Game_Room": "Game Room",
        "Children_Room": "Children's Room",
        "mud_room": "Mud Room",
        "porch": "Porch",
        "computer_room": "Computer Room",
        "amusement_room": "Amusement Room",
        "informal_dining_room": "Informal Dinning Room",
    }

    logs["real"] = logs["real"].replace(mapping)

    # logs['pred'] = logs['pred'].replace(mapping)

    mapping = dict(zip(all_cls, range(len(all_cls))))
    logs["real"] = logs["real"].replace(mapping)
    logs["pred"] = logs["pred"].replace(mapping)

    real = logs["real"].to_list()
    pred = logs["pred"].to_list()

    # Accuracy
    print(((logs["real"] == logs["pred"]).astype(float).sum() / len(logs)))

    # Confusion matrix
    # print(real)
    print(pred)
    cm = metrics.confusion_matrix(real, pred, labels=range(len(all_cls)))

    plt.figure(figsize=(11, 11))
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        xticklabels=all_cls,
        yticklabels=all_cls,
        cbar=False,
        fmt=".0f",
    )
    # plt.subplots_adjust(top=0.98, bottom=0.15, left=0.14, right=0.98)
    # plt.xticks(fontsize=10, rotation=70)
    # plt.yticks(fontsize=10, rotation=30)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.show()
    plt.savefig("plt.png")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "pred":
        make_pred(sys.argv[2], sys.argv[3])
    elif mode == "metric":
        make_acc(sys.argv[2])
