#!/usr/bin/env python3
"""
Script to download YOLO model during the build process
"""
import os
import sys

print("Downloading YOLO model...")
try:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    print(f"Model downloaded successfully to {model.ckpt_path}")
except Exception as e:
    print(f"Error downloading model: {str(e)}")
    sys.exit(1)