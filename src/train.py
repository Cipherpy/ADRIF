"""Try train the YOLO from scratch."""
import torch
from ultralytics import YOLO

# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset='/home/reshma/marine_mammal/model_data'

model = YOLO("yolo11m.pt")

# Use the model
model.train(epochs=100, data="/home/reshma/marine_mammal/model_data/data.yaml", imgsz=640, batch=8, device=[0,1,2,3,4,5,6,7], workers=0)  # train the model

metrics = model.val()  # evaluate model performance on the validation set
# results = model("/home/reshma/marine_mammal/model_data/test/_DSC0395.JPG")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format

print(path)
