import string

from .room_type_base_model import RoomTypeClassifier


class RoomClassifier:
    def __init__(
        self,
        classifier_checkpoint="trained_weights/room-type-classification-clip-v1.0.0.pt",
        use_cuda=True,
        use_batch_processing=False,
    ):
        # instantiate room type classifier model
        self.room_type_classifier = RoomTypeClassifier(
            use_cuda=use_cuda,
            weights=classifier_checkpoint,
            use_batch_processing=use_batch_processing,
        )

    def __call__(self, pil_img, k=2, return_confidence=True):
        # Get predicted label on images
        predicted_scores = self.room_type_classifier.predict(pil_img, top_k=k)
        predicted_scores = predicted_scores[0]
        # Return
        if return_confidence:
            labels_conf = [
                [string.capwords(label.replace("_", " ")), conf]
                for label, conf in predicted_scores
            ]
        else:
            labels_conf = [label for label, conf in predicted_scores]

        return labels_conf
