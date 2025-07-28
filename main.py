"""
Tiny YOLOv8 demo
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
• Installs:  pip install -r requirements.txt
• Run:       python main.py
Produces:   test.jpg            (sample image)
            test_detected.jpg   (boxes drawn)
"""
from pathlib import Path
import cv2, requests, numpy as np
from ultralytics import YOLO

# ----------------------------------------------------------------------
# 1) Get a sample image if it isn't already here
sample_url = "https://ultralytics.com/images/bus.jpg"
img_path   = Path("test.jpg")
if not img_path.exists():
    data = requests.get(sample_url).content
    img_array = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    cv2.imwrite(str(img_path), img)
    print("Downloaded sample image ➜ test.jpg")

# 2) Read the image
img = cv2.imread(str(img_path))

# 3) Load a small, pretrained YOLO model (3 MB)
model = YOLO("yolov8n.pt")

# 4) Run inference
results = model(img)

# 5) Draw bounding boxes & save
boxed_img = results[0].plot()          # numpy array with boxes
out_path = "test_detected.jpg"
cv2.imwrite(out_path, boxed_img)

print(f"✅ Detection complete – see {out_path}")
