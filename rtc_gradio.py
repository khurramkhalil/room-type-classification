import gradio as gr
from models.room_type_classification import RoomClassifier
from PIL import Image
import numpy as np

# Initialize the model
MODEL = RoomClassifier(use_cuda=True)


def predict(image: Image, conf_thresh: float = 0.01):
    image = Image.fromarray(np.uint8(image)).convert("RGB")
    if image.mode == "RGBA":
        # Convert RGBA to RGB
        image = image.convert("RGB")

    # Run model
    outputs = MODEL(image)
    scores = [score for label, score in outputs]
    labels = [label for label, score in outputs]

    normalized_score = [format(score * 100, ".1f") for score in scores]

    scores = np.array(scores)
    labels = np.array(labels)
    labels = labels[scores >= conf_thresh].tolist()
    scores = scores[scores >= conf_thresh].tolist()

    result = {"result": labels, "scores": normalized_score}

    return result


# Define Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(),
        # gr.Slider(0.01, 1.0, step=0.01, default=0.01)
    ],
    outputs="json",
)

# Run the app
if __name__ == "__main__":
    iface.launch()
