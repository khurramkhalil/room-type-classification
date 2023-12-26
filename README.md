## Installation
...>
Developed and tested on `python==3.8.13`
1. Install `pytorch` and `torchvision` by
```
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```
2. Install requirements by
```
pip3 install -r requirements.txt
```

### Download trained weights
```
gdown --id 1Qntn0Z4hAp_6XvA8g7nhHXYvrm_AulFi --output trained_weights/room-type-classification-clip-v1.0.0.pt
```

## Run from terminal
```
 python main.py --input test_imgs/v1
```
- `test_imgs` is folder containing images

## Integration
Instantiate `RoomClassifier` class and simply run it on a `PIL.Image` object

```python
from PIL import Image
from models.inference import RoomClassifier

# Initialize the model
model = RoomClassifier() 
print("Model has been loaded.")
print()
# Read image
img = Image.open(IMG_PATH)

# Run model on image
label = model(img)
print(f"{IMG_PATH} --> {label}")

```
`label` will be a list with max two names of predicted class. i.e ['Garden', 'Terrace']
### List of classes
```
Basement
Bathroom
Bedroom
Garage
Garden
Gym
Kitchen
Living Room
Office
Closet
Pool
Game Room
Dining
Cinema Room
Library Building
Balcony          
Laundry
Loft Design
Wine Room
Bar
Children Room
Terrace
Storage Room
Empty Room
Lobby
Conservatory
Patio
Hearth room
Play room
Porch
Studio
Deck
```

### Run docker
```
# build docker
docker build -t room-type-classification .
# run docker container
docker run -it --gpus all -p 8000:8080 room-type-classification
```
### Test docker container
```
curl -X POST \
-H "Content-Type: application/json" \
-d '{"instances":[{"image_url": "https://storage.googleapis.com/proptexx/room_type.jpg"}]}' \                                        
http://localhost:8000/predict
```
