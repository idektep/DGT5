"""Configuration constants for the object-detection publisher.
Everything you need to tune lives here. Values marked TODO must be set
for your rig before `detect_and_publish.py` will work correctly.
"""

# --- MQTT -------------------------------------------------------------
BROKER = "broker.emqx.io"
PORT = 1883
TOPIC = "idt/object/slot"

# --- Camera / YOLO model ---------------------------------------------
CAM_INDEX = 1              # USB webcam index (top-view camera)
WEIGHTS = r"best.pt"        # TODO: path to your trained YOLOv8 weights
OBJECT_CLASS = 0                # class id of the printed square
CONF = 0.2                      # minimum detection confidence

# --- Work area --------------------------------------------------------
AREA_CM = 20.0           

CORNERS_PX = [(105, 60), (548, 60), (105, 390), (548, 390)]

SLOT_CENTERS_CM = [
    (7.5, 7.5),    # slot 0
    (12.5, 7.5),   # slot 1
    (7.5, 12.5),   # slot 2
    (12.5, 12.5),  # slot 3
]
