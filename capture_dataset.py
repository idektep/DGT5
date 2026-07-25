"""Capture .jpg frames from the top-view camera for YOLO training.

Keys (with the preview window focused):
    c   = save the current frame
    Esc = quit
"""
import os
import config
import cv2

SAVE_DIR = "captures"

def next_index(folder):
    nums = [
        int(f[4:-4])
        for f in os.listdir(folder)
        if f.startswith("img_") and f.endswith(".jpg") and f[4:-4].isdigit()
    ]
    return max(nums) + 1 if nums else 0
# TODO: add main function here.


if __name__ == "__main__":
    main()
