import clip
import torch
from tqdm import tqdm

from .prompts_processor import LabelPromptsProcessor
from utils.room_prompts import room_label_and_prompt

from .datasets.dataloader import dataset_loader


class RoomTypeClassifier:
    def __init__(
        self,
        use_cuda=True,
        weights="trained_weights/room-type-classification-clip-v1.0.0.pt",
        use_batch_processing=None,
    ):
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(weights, device=self.device)
        self.classes = LabelPromptsProcessor(room_label_and_prompt)
        self.text = clip.tokenize(self.classes.prompts).to(self.device)
        self.use_batch_processing = use_batch_processing
        self.ext_list = (
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

    def inference(self, pil_image):
        pred_with_class = []
        pil_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        pred_with_class = self.infer_scores(pil_image, self.text, pred_with_class)
        return pred_with_class

    def batch_inference(self, image_batches):
        pred_with_class = []
        true_labels = []
        for batch in tqdm(image_batches, position=0, disable=False):
            pred_with_class = self.infer_scores(
                batch[0].to(self.device), self.text, pred_with_class
            )
            true_labels.append(batch[1].tolist())
        self.true_labels = true_labels
        return pred_with_class

    # For batch prediction
    def batch_predict(self, dir_path, top_k=2, batch_size=64):
        labels_mapping, images = dataset_loader(
            dir_path, self.ext_list, self.preprocess, batch_size
        )
        batch_predictions = self.predict(img_paths=images, top_k=top_k)
        return batch_predictions, self.true_labels, labels_mapping

    def infer_scores(self, images, text, pred_with_class):
        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(images, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        # Making list of lists containing class label and score
        for prob in probs:
            classlabel_with_score = {}
            for label, confidence in zip(self.classes.prompts, prob):
                classlabel_with_score[label] = confidence
            pred_with_class.append(classlabel_with_score)
        return pred_with_class

    def predict_and_sort(self, img_paths):
        if not self.use_batch_processing:
            scores = self.inference(img_paths)
        else:
            scores = self.batch_inference(img_paths)
        all_scores = []
        for score in scores:
            score = sorted(score.items(), key=lambda x: x[1], reverse=True)
            all_scores.append(score)
        return all_scores

    def predict(self, img_paths, top_k=1):
        predictions = self.predict_and_sort(img_paths)
        all_predicted_class = []
        for pred in predictions:
            predicted_classes = []
            for label, confidence in pred:
                predicted_classes.append((self.classes.get_label(label), confidence))
            unique_class_predictions = self.get_unique_class_predictions(
                predicted_classes
            )
            top_k_predictions = unique_class_predictions[:top_k]
            all_predicted_class.append(top_k_predictions)

        return all_predicted_class

    def get_unique_class_predictions(self, pred):
        seen = set()
        result = []
        for tpl in pred:
            if tpl[0] not in seen:
                result.append(tpl)
                seen.add(tpl[0])
        return result
