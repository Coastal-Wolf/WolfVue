#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WolfVue Desktop - Official Production Release
Wildlife Video & Image Classifier for the Idaho Gray Wolf Monitoring Program
Created by Nathan Bluto | Facilitated by Dr. Ausband
"""

import sys
import os
from pathlib import Path

# ===================================================================
# CRITICAL: Load PyTorch/ULTRALYTICS FIRST — fixes Windows c10.dll crash
# ===================================================================
try:
    import torch
    import cv2
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("YOLO + PyTorch loaded successfully")
except Exception as e:
    YOLO_AVAILABLE = False
    print(f"YOLO not available (running in demo mode): {e}")
    torch = None
    cv2 = None
    YOLO = None

# ===================================================================
# Now it's completely safe to import PyQt6
# ===================================================================
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

# ===================================================================
# Standard library & third-party imports
# ===================================================================
import json
import yaml
import time
import shutil
import csv
import random
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
