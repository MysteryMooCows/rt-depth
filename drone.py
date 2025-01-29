import os
from djitellopy import Tello
import cv2
import numpy as np
import time

from dotenv import load_dotenv
from rich import print as rprint

from utils.async_frame_predictor import AsyncPredictor
from utils.async_frame_reader import AsyncFrameReader

load_dotenv()

DRONE_SPEED = int(os.getenv('DRONE_SPEED'))
FPS = int(os.getenv('FPS'))
RT_SCALE = float(os.getenv('RT_SCALE'))

class Drone:
    def __init__(self, model, transform):
        self.tello = Tello()
        
        # Display settings
        self.window_name = 'Drone View (Press ESC to quit)'
        cv2.namedWindow(self.window_name)
        self.display_width = 960
        self.display_height = 720
        
        # Store last valid frame and depth
        self.last_valid_frame = None
        self.last_valid_depth = None
        
        # Velocities
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        # Track pressed keys
        self.pressed_keys = set()
        
        self.send_rc_control = False
        self.should_stop = False
        
        # Model components
        self.model = model
        self.transform = transform
        self.frame_reader = None
        self.predictor = AsyncPredictor(model, transform, scale=RT_SCALE)

    def run(self):
        self.tello.connect()
        self.tello.set_speed(self.speed)

        # Initialize and start async components
        self.frame_reader = AsyncFrameReader(self.tello)
        self.frame_reader.start()
        self.predictor.start()

        try:
            self._control_loop()
        finally:
            # Cleanup
            self.frame_reader.stop()
            self.predictor.stop()
            self.tello.end()
            cv2.destroyAllWindows()

    def _prepare_display(self, frame, depth):
        """Prepare frame and depth map for display side by side with stable dimensions"""
        # Update last valid frame/depth if new ones are available
        if frame is not None:
            self.last_valid_frame = frame
        if depth is not None:
            self.last_valid_depth = depth
            
        # Create base display canvas with fixed size
        combined = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        # Fixed display dimensions for each half
        half_width = self.display_width // 2
        target_height = self.display_height
        
        # Default dimensions if no valid frame yet
        frame_height = 720  # Standard Tello camera height
        frame_width = 960   # Standard Tello camera width
        
        if self.last_valid_frame is not None:
            frame_height = self.last_valid_frame.shape[0]
            frame_width = self.last_valid_frame.shape[1]
        
        # Calculate scaling once - use same scaling for both frame and depth
        scaling = min(target_height/frame_height, half_width/frame_width)
        new_width = int(frame_width * scaling)
        new_height = int(frame_height * scaling)
        
        # Fixed positions for both frame and depth
        y_offset = (target_height - new_height) // 2
        x_offset = (half_width - new_width) // 2
        
        # Display last valid frame if available
        if self.last_valid_frame is not None:
            # Ensure frame is RGB
            if len(self.last_valid_frame.shape) == 2:
                frame_to_show = cv2.cvtColor(self.last_valid_frame, cv2.COLOR_GRAY2RGB)
            else:
                frame_to_show = self.last_valid_frame
            frame_resized = cv2.resize(frame_to_show, (new_width, new_height))
            # Place frame in left half
            combined[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame_resized
        
        # Display last valid depth if available
        if self.last_valid_depth is not None:
            # Normalize and colorize depth consistently
            depth_vis = cv2.normalize(self.last_valid_depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)
            
            # Ensure depth map has same dimensions as frame area
            depth_vis = cv2.resize(depth_vis, (new_width, new_height))
            
            # Place depth map in right half at same position as frame
            combined[y_offset:y_offset+new_height, 
                    half_width+x_offset:half_width+x_offset+new_width] = depth_vis
        
        # Add static overlay for instructions
        overlay = combined.copy()
        # Larger, more opaque background for better readability
        # cv2.rectangle(overlay, (5, 5), (205, 195), (0, 0, 0), -1)
        # cv2.addWeighted(overlay, 0.4, combined, 0.6, 0, combined)
        
        # Add control instructions with consistent positioning
        instructions = [
            "Controls:",
            "W/S: Forward/Backward",
            "A/D: Left/Right",
            "Q/E: Rotate Left/Right",
            "R/F: Up/Down",
            "T: Takeoff",
            "L: Land",
            "ESC: Quit"
        ]
        
        y = 25
        for instruction in instructions:
            cv2.putText(combined, instruction, (10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
            y += 20
        
        return combined

    def _update_velocities(self, key, pressed):
        """Update velocities based on key press/release"""
        if pressed:
            self.pressed_keys.add(key)
        else:
            self.pressed_keys.discard(key)
            
        # Reset all velocities
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        
        # Set velocities based on currently pressed keys
        for k in self.pressed_keys:
            if k == ord('w'):
                self.for_back_velocity = DRONE_SPEED
            elif k == ord('s'):
                self.for_back_velocity = -DRONE_SPEED
            elif k == ord('a'):
                self.left_right_velocity = -DRONE_SPEED
            elif k == ord('d'):
                self.left_right_velocity = DRONE_SPEED
            elif k == ord('r'):
                self.up_down_velocity = DRONE_SPEED
            elif k == ord('f'):
                self.up_down_velocity = -DRONE_SPEED
            elif k == ord('q'):
                self.yaw_velocity = -DRONE_SPEED
            elif k == ord('e'):
                self.yaw_velocity = DRONE_SPEED

    def _control_loop(self):
        last_update = time.time()
        update_interval = 1.0 / FPS
        
        while not self.should_stop:
            current_time = time.time()
            
            # Get current frame
            frame = self.frame_reader.get_frame()
            if frame is not None:
                self.predictor.process_frame(frame)
            
            # Check for new predictions
            prediction = self.predictor.get_prediction()
            if prediction is not None:
                depth = prediction['depth']
            else:
                depth = None
            
            # Prepare and display the combined view
            display = self._prepare_display(frame, depth)
            if display is not None:
                cv2.imshow(self.window_name, display)
            
            # Process key events
            key = cv2.waitKey(1) & 0xFF
            
            # Handle key press events
            if key != 255:  # Key pressed
                if key == 27:  # ESC
                    self.should_stop = True
                elif key == ord('t'):
                    rprint("[yellow]Taking off...[/yellow]")
                    self.tello.takeoff()
                    self.send_rc_control = True
                elif key == ord('l'):
                    rprint("[yellow]Landing...[/yellow]")
                    self.tello.land()
                    self.send_rc_control = False
                else:
                    # Handle movement key press
                    self._update_velocities(key, True)
            else:
                # Check if any movement keys were released
                for k in list(self.pressed_keys):
                    if cv2.waitKey(1) & 0xFF == 255:  # Key released
                        self._update_velocities(k, False)
            
            # Update drone controls at FPS rate
            if current_time - last_update >= update_interval:
                self.update()
                last_update = current_time
            
            # Small sleep to prevent CPU overuse
            time.sleep(0.001)

    def update(self):
        """Send control commands to the drone"""
        if self.send_rc_control:
            self.tello.send_rc_control(
                self.left_right_velocity,
                self.for_back_velocity,
                self.up_down_velocity,
                self.yaw_velocity
            )