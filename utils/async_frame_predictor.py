import threading
from queue import Queue
import time
import numpy as np
from typing import Optional, Dict, Any
import torch
import cv2

from utils.console_io import dprint

class AsyncPredictor:
    def __init__(self, model, transform, scale: float = 1.0):
        self.model = model
        self.transform = transform
        self.scale = scale
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.predictions_processed = 0
        self.last_prediction_time = time.time()
        self.focal_length = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._prediction_thread, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _prediction_thread(self):
        while self.running:
            try:
                frame = self.input_queue.get_nowait()
                if frame is not None:
                    # Transform and run inference
                    model_input = self.transform(frame)
                    prediction = self.model.infer(model_input)
                    
                    # Process depth prediction
                    depth = prediction["depth"].cpu().numpy() * self.scale
                    
                    # Store focal length
                    if "focallength_px" in prediction:
                        self.focal_length = prediction["focallength_px"].item()
                    
                    # Update metrics
                    current_time = time.time()
                    inference_fps = 1 / (current_time - self.last_prediction_time)
                    self.last_prediction_time = current_time
                    self.predictions_processed += 1

                    dprint(f"Processed {self.predictions_processed} predictions at {inference_fps:.2f} FPS")
                    
                    # Clear previous output and store new one
                    try:
                        self.output_queue.get_nowait()
                    except:
                        pass
                    
                    self.output_queue.put({
                        'depth': depth,
                        'fps': inference_fps
                    })
            except:
                time.sleep(0.1)  # Small sleep when no frame is available

    def process_frame(self, frame):
        """Add a new frame to be processed"""
        try:
            self.input_queue.get_nowait()  # Clear previous input
        except:
            pass
        self.input_queue.put(frame)

    def get_prediction(self) -> Optional[Dict[str, Any]]:
        """Get the latest prediction if available"""
        try:
            return self.output_queue.get_nowait()
        except:
            return None
            
    def get_focal_length(self) -> Optional[float]:
        """Get the last computed focal length"""
        return self.focal_length