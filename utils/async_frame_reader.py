import os
from djitellopy import Tello
import cv2
import numpy as np
import time
import threading
from queue import Queue
from typing import Optional

from dotenv import load_dotenv

from utils.console_io import dprint
from rich import print as rprint

load_dotenv()

FPS = int(os.getenv('FPS'))

class AsyncFrameReader:
    def __init__(self, tello: Tello):
        self.tello = tello
        self.frame_read = None
        self.frame_queue = Queue(maxsize=1)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frames_processed = 0
        self.last_frame_time = time.time()
        dprint("AsyncFrameReader initialized")

    def start(self):
        dprint("Starting AsyncFrameReader...")
        self.running = True
        self.tello.streamoff()
        self.tello.streamon()
        self.frame_read = self.tello.get_frame_read()
        self.thread = threading.Thread(target=self._frame_reader_thread, daemon=True)
        self.thread.start()
        dprint("AsyncFrameReader started successfully")

    def stop(self):
        dprint("Stopping AsyncFrameReader...")
        self.running = False
        if self.thread:
            self.thread.join()
        if self.frame_read:
            self.frame_read.stop()
        dprint(f"AsyncFrameReader stopped. Processed {self.frames_processed} frames total")

    def _frame_reader_thread(self):
        dprint("Frame reader thread started")
        while self.running and not self.frame_read.stopped:
            frame = self.frame_read.frame
            if frame is not None:
                current_time = time.time()
                fps = 1 / (current_time - self.last_frame_time)
                self.last_frame_time = current_time
                
                # Just convert BGR to RGB, no other transformations
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
                
                self.frame_queue.put(frame)
                self.frames_processed += 1
                
                if self.frames_processed % 30 == 0:  # Log every 30 frames
                    dprint(f"Processed frame {self.frames_processed}, Current FPS: {fps:.2f}")
            
            time.sleep(1 / FPS)

    def get_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except:
            return None