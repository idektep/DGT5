"""
Reads the top-view webcam, finds the highest-confidence object, maps its
centroid to one of the 4 slots, and publishes the slot index to MQTT only
when it changes. Publishes -1 when no object is detected.

Press Esc in the window to stop.
"""
import config
import cv2
import paho.mqtt.client as mqtt
from ultralytics import YOLO

def pixel_to_cm(px, py, corners_px, area_cm):
    """Map a pixel coordinate to cm within the work area."""
    xs = [c[0] for c in corners_px]
    ys = [c[1] for c in corners_px]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_cm = (px - x_min) / (x_max - x_min) * area_cm
    y_cm = (py - y_min) / (y_max - y_min) * area_cm
    return x_cm, y_cm


def cm_to_pixel(x_cm, y_cm, corners_px, area_cm):
    """Inverse of pixel_to_cm: map cm back to a pixel coordinate."""
    xs = [c[0] for c in corners_px]
    ys = [c[1] for c in corners_px]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    px = x_min + x_cm / area_cm * (x_max - x_min)
    py = y_min + y_cm / area_cm * (y_max - y_min)
    return px, py

def nearest_slot(x_cm, y_cm, slot_centers_cm):
    """Return the index of the closest slot center to (x_cm, y_cm)."""
    best_i, best_d = -1, float("inf")
    for i, (cx, cy) in enumerate(slot_centers_cm):
        d = (x_cm - cx) ** 2 + (y_cm - cy) ** 2
        if d < best_d:
            best_i, best_d = i, d
    return best_i

def slot_boxes(corners_px, area_cm, slot_centers_cm, cell_cm=2.0):
    """Pixel rectangles (x1,y1,x2,y2) for each 2x2 cm slot cell."""
    h = cell_cm / 2
    boxes = []
    for cx, cy in slot_centers_cm:
        x1, y1 = cm_to_pixel(cx - h, cy - h, corners_px, area_cm)
        x2, y2 = cm_to_pixel(cx + h, cy + h, corners_px, area_cm)
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes

def detect(frame, model):
    """Return (slot, (cx,cy)) for the best object, or (-1, None) if none."""
    result = model(frame, verbose=False)[0]
    best_box, best_conf = None, config.CONF
    for box in result.boxes:
        if int(box.cls) != config.OBJECT_CLASS:
            continue
        conf = float(box.conf)
        if conf >= best_conf:
            best_box, best_conf = box, conf

    if best_box is None:
        return -1, None

    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x_cm, y_cm = pixel_to_cm(cx, cy, config.CORNERS_PX, config.AREA_CM)
    slot = nearest_slot(x_cm, y_cm, config.SLOT_CENTERS_CM)
    return slot, (cx, cy)

def detect_slot(frame, model):
    """Return slot 0..3 for the best object in the frame, or -1 if none."""
    return detect(frame, model)[0]

def draw_overlay(frame, boxes, slot, center):
    """Draw the 4 slot boxes (active one highlighted) and a red cross on the object."""
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        active = (i == slot)
        color = (0, 255, 255) if active else (0, 255, 0)   # yellow if active, else green
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if active else 1)
        cv2.putText(frame, str(i), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if center is not None:
        cx, cy = int(center[0]), int(center[1])
        cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (0, 0, 255), 2)
        cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (0, 0, 255), 2)
# TODO: add main function here.


if __name__ == "__main__":
    main()
