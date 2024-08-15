from ultralytics import YOLO
from PIL import Image


# Load a pretrained YOLOv8n model
model = YOLO("models/best_traffic_med_yolo.pt")
#model = YOLO("best.pt")


results = model(source='image1.jpg', show=True)
