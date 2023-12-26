import argparse
import csv
import os
import os
import time

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from tqdm import tqdm

from models.room_type_base_model import RoomTypeClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input directory containing image classes")
    parser.add_argument(
        "--cpu",
        action="store_false",
        dest="use_cuda",
        default=True,
        help="Device to run computation. Either CPU or GPU",
    )
    args = parser.parse_args()
    args.get_topk_results = 2
    args.use_batch_processing = True
    model_name = "roomtypes"

    model = RoomTypeClassifier(
        use_cuda=True, use_batch_processing=args.use_batch_processing
    )

    if not args.input:
        args.input = "/home/proptx/dataset/CLIP"

    if os.path.isdir(args.input):
        img_classes = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if os.path.isdir(os.path.join(args.input, f))
        ]
    else:
        img_classes = [args.input]

    print(f"Found {len(img_classes)} image classes.")

    csv_file = open("results.csv", "w", newline="")
    writer = csv.writer(csv_file)

    tic = time.time()

    for top_k in [x for x in range(args.get_topk_results + 1) if x != 0]:
        BATCH_SIZE = 5
        predicted_from_clip = []
        true_labels = []

        if args.use_batch_processing:
            predicted_classes, true_labels, labels_mapping = model.batch_predict(
                dir_path=args.input, top_k=top_k, batch_size=BATCH_SIZE
            )
            for class_labels in predicted_classes:
                predicted_from_clip.append(class_labels)

        short_to_full_name = {
            "Balcony": "Balcony",
            "Bar": "Bar",
            "Basement": "Basement",
            "Bathroom": "Bathroom",
            "Bedroom": "Bedroom",
            "Children": "Children Room",
            "Cinema": "Cinema Room",
            "Closet": "Closet",
            "Dining": "Dining",
            "Empty": "Empty Room",
            "Game": "Game Room",
            "Garage": "Garage",
            "Garden": "Garden",
            "Gym": "Gym",
            "Kitchen": "Kitchen",
            "Laundry": "Laundry",
            "Library": "Library",
            "Living": "Living Room",
            "Loft": "Loft",
            "Office": "Office",
            "Pool": "Swimming Pool",
            "Storage": "Storage Room",
            "Terrace": "Terrace",
            "Walk": "Walk in Closet",
            "Wine": "Wine Room",
            "Informal Dinning Room": "Informal Dinning Room",
        }

        if args.use_batch_processing:
            updated_labels_mapping = {
                k.replace("_", " ").title(): v for k, v in labels_mapping.items()
            }
            updated_labels_mapping = {
                k.replace("/", " ").title(): v for k, v in labels_mapping.items()
            }
            # for k, v in updated_labels_mapping.items():

            if "Children Room" in updated_labels_mapping:
                updated_labels_mapping["Children's Room"] = updated_labels_mapping.pop(
                    "Children Room"
                )
            if "Dining" in updated_labels_mapping:
                updated_labels_mapping["Dining Room"] = updated_labels_mapping.pop(
                    "Dining"
                )
            if "Pool" in updated_labels_mapping:
                updated_labels_mapping["Garden/Yard"] = updated_labels_mapping.pop(
                    "Pool"
                )
            if "Library Building" in updated_labels_mapping:
                updated_labels_mapping["Library/Study"] = updated_labels_mapping.pop(
                    "Library Building"
                )
            if "Wineroom" in updated_labels_mapping:
                updated_labels_mapping["Wine Cellar"] = updated_labels_mapping.pop(
                    "Wineroom"
                )

            for top_k_labels in predicted_from_clip:
                for idx in range(len(top_k_labels)):
                    top_k_labels[idx] = updated_labels_mapping[top_k_labels[idx][0]]

            true_labels = [item for label in true_labels for item in label]

        pruned_predicted_labels = []
        for k_lablels, true_label in zip(predicted_from_clip, true_labels):
            if true_label in k_lablels:
                pruned_predicted_labels.append(true_label)
            else:
                pruned_predicted_labels.append(k_lablels[0])

        # Overall Accuracy
        accuracy = accuracy_score(true_labels, pruned_predicted_labels)
        # Precision
        precision = precision_score(
            true_labels, pruned_predicted_labels, average="weighted", zero_division=0
        )
        # Recall
        recall = recall_score(
            true_labels, pruned_predicted_labels, average="weighted", zero_division=0
        )
        # F1_score
        f1_Score = f1_score(true_labels, pruned_predicted_labels, average="weighted")
        # Confusion Matrix
        confusion_mat = confusion_matrix(true_labels, pruned_predicted_labels)
        # Display confusion matrix in some good visualization
        cm_display = ConfusionMatrixDisplay(
            confusion_matrix=confusion_mat,
            display_labels=list(updated_labels_mapping.keys()),
        )
        # Class wise accuracy
        confusion_mat = (
            confusion_mat.astype("float") / confusion_mat.sum(axis=1)[:, np.newaxis]
        )
        # Setting Precision to 3 decimal points
        precised_class_accuracies = [
            "%.3f" % accurcy for accurcy in confusion_mat.diagonal()
        ]
        # Combine label and its corresponding accuracy
        class_wise_accuracy = list(
            zip(list(updated_labels_mapping.keys()), precised_class_accuracies)
        )
        # Sorting class wise accuracy in ascending order
        sorted_class_accuracies = sorted(class_wise_accuracy, key=lambda x: x[1])
        # Plotting Confusion Matrix
        fig, ax = plt.subplots(figsize=(15, 15))
        cm_display.plot(ax=ax, xticks_rotation="vertical")
        # Save confusion matrix plot as png
        plt.savefig(f"k_{top_k}.png")
        # Writing results to csv file
        if top_k == 1:
            headers = (
                ["Top_K"]
                + list(updated_labels_mapping.keys())
                + ["combined accuracy", "precision", "recall", "f1"]
            )
            writer.writerow(headers)
        cumulative_results = [accuracy, precision, recall, f1_Score]
        result_row = (
            [top_k]
            + [accuracy for _, accuracy in class_wise_accuracy]
            + ["%.3f" % result for result in cumulative_results]
        )
        writer.writerow(result_row)

        print("----------------------------Metrics--------------------")
        print(f"k = {top_k}")
        print("-----------------------------------------------")
        print("Accuracy: ", accuracy)
        print("-----------------------------------------------")
        print("Precision: ", precision)
        print("-----------------------------------------------")
        print("Recall: ", recall)
        print("-----------------------------------------------")
        print("f1 Score: ", f1_Score)
        print("-----------------------------------------------")
        print("class_wise_accuracy: ", sorted_class_accuracies)
        print("-----------------------------------------------")

    toc = time.time()
    print(f"Time taken {toc-tic:.2f} secs")
    print("-------------------------------------------------------")
