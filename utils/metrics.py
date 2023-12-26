import argparse
import csv
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from models.room_type_base_model import RoomTypeClassifier


def main(args):
    input_path = args.input_path
    model_name = args.model
    tic = time.time()

    if not args.use_batch_processing:
        if os.path.isdir(input_path):
            folder_paths = [
                os.path.join(input_path, f)
                for f in os.listdir(input_path)
                if os.path.isdir(os.path.join(input_path, f))
            ]
            if not folder_paths:
                folder_paths = [input_path]
            paths = [
                os.path.join(folder_path, f)
                for folder_path in folder_paths
                for f in os.listdir(folder_path)
            ]
    else:
        dir_path = [input_path]

    model = RoomTypeClassifier(
        device="gpu",
        weights="checkpoints/model-v1.0.pt",
        use_batch_processing=args.use_batch_processing,
    )

    csv_file = open("results.csv", "w", newline="")
    writer = csv.writer(csv_file)

    for top_k in [x for x in range(args.get_topk_results + 1) if x != 0]:
        BATCH_SIZE = 256
        predicted_from_clip = []
        true_labels = []

        if args.use_batch_processing:
            predicted_classes, true_labels, labels_mapping = model.batch_predict(
                dir_path, top_k=top_k, batch_size=BATCH_SIZE
            )
            for class_labels in predicted_classes:
                predicted_from_clip.append(class_labels)
        else:
            for img_path in tqdm(paths):
                # True Labels getting from basename
                # pil_image = Image.open(img_path)
                classes = model.predict(img_path, top_k=top_k)
                if model_name == "roomtypes":
                    for class_labels in classes:
                        predicted_from_clip.append(class_labels)
                else:
                    predicted_from_clip.append(classes)
                    image_labels = os.path.basename(img_path).split(".")[0]
                    true_labels.append(image_labels)
            print("Predictions: ", predicted_from_clip)

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
        }

        if args.use_batch_processing:
            updated_labels_mapping = {
                k.replace("_", " ").title(): v for k, v in labels_mapping.items()
            }
            for top_k_labels in predicted_from_clip:
                for idx in range(len(top_k_labels)):
                    top_k_labels[idx] = updated_labels_mapping[top_k_labels[idx]]
            true_labels = [item for label in true_labels for item in label]

        else:
            # Processing  to get True Labels
            for idx in range(len(true_labels)):
                label_zeroth_name = true_labels[idx].split("_")[0]
                true_labels[idx] = short_to_full_name[label_zeroth_name]

        # print("Predicted_Labels", predicted_from_clip)
        # print('------------------')
        # print('True Labels', true_labels)
        # print('------------------')

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="test_data",
        help="For batch processing must be root dir path else image paths ",
    )
    parser.add_argument("--get_topk_results", type=int, default=3)
    parser.add_argument(
        "--use_batch_processing",
        action="store_true",
        dest="use_batch_processing",
        help="Use batch processing",
    )
    args = parser.parse_args()
    # print(args)
    main(args)
