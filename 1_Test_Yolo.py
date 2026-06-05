from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("dog.jpg")
results[0].show()

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        name = model.names[cls]
        if name == "dog":
            print("dog Found!")
