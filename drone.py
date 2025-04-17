import os
from djitellopy import Tello
import cv2
import numpy as np
import time

from dotenv import load_dotenv
from rich import print as rprint

from utils.async_frame_predictor import AsyncPredictor
from utils.async_frame_reader import AsyncFrameReader
from utils.console_io import dprint

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
        
        # Autonomous navigation state
        self.autonomous_mode = False
        self.forward_start_time = None
        self.altitude_adjust_start = None
        self.altitude_adjusting = False
        
        # Yaw control parameters
        self.yaw_start_time = None
        self.yaw_duration = None
        self.yaw_target_angle = None
        
        # Battery tracking
        self.battery_level = 0
        self.last_battery_check = 0
        
        # Model components
        self.model = model
        self.transform = transform
        self.frame_reader = None
        self.predictor = AsyncPredictor(model, transform, scale=RT_SCALE)

    def run(self):
        self.tello.connect()
        self.tello.set_speed(self.speed)
        
        # Get initial battery level
        try:
            self.battery_level = self.tello.get_battery()
            self.last_battery_check = time.time()
            rprint(f"[green]Battery level: {self.battery_level}%[/green]")
        except Exception as e:
            rprint(f"[red]Could not get initial battery level: {e}[/red]")
            self.battery_level = 0
            self.last_battery_check = time.time()

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
            
            # If autonomous mode is active, show the target point
            if self.autonomous_mode and self.last_valid_depth is not None:
                # Find furthest point - HIGHEST depth value
                furthest_idx = np.argmax(self.last_valid_depth)
                furthest_y, furthest_x = np.unravel_index(furthest_idx, self.last_valid_depth.shape)
                
                # Find closest point (for reference)
                closest_idx = np.argmin(self.last_valid_depth)
                closest_y, closest_x = np.unravel_index(closest_idx, self.last_valid_depth.shape)
                
                # Scale coordinates to match display size
                furthest_disp_x = int(furthest_x * new_width / self.last_valid_depth.shape[1])
                furthest_disp_y = int(furthest_y * new_height / self.last_valid_depth.shape[0])
                
                closest_disp_x = int(closest_x * new_width / self.last_valid_depth.shape[1])
                closest_disp_y = int(closest_y * new_height / self.last_valid_depth.shape[0])
                
                # Draw target furthest point crosshairs (green)
                cv2.drawMarker(combined, 
                            (x_offset + furthest_disp_x, y_offset + furthest_disp_y),
                            (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                
                cv2.drawMarker(combined, 
                            (half_width + x_offset + furthest_disp_x, y_offset + furthest_disp_y),
                            (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                
                # Draw closest point reference (red)
                cv2.drawMarker(combined, 
                            (x_offset + closest_disp_x, y_offset + closest_disp_y),
                            (0, 0, 255), cv2.MARKER_DIAMOND, 10, 1)
                
                cv2.drawMarker(combined, 
                            (half_width + x_offset + closest_disp_x, y_offset + closest_disp_y),
                            (0, 0, 255), cv2.MARKER_DIAMOND, 10, 1)
        
        # Add status information (mode and battery)
        # Update battery periodically to avoid too many requests
        current_time = time.time()
        if not hasattr(self, 'last_battery_check') or current_time - self.last_battery_check > 10:
            try:
                self.battery_level = self.tello.get_battery()
                self.last_battery_check = current_time
            except:
                if not hasattr(self, 'battery_level'):
                    self.battery_level = 0
                    self.last_battery_check = current_time
        
        # Battery status with colored indicator
        if self.battery_level > 50:
            bat_color = (0, 255, 0)  # Green for good battery
        elif self.battery_level > 20:
            bat_color = (0, 165, 255)  # Orange for medium battery
        else:
            bat_color = (0, 0, 255)  # Red for low battery
            
            # Make it flash if very low
            if self.battery_level < 10 and int(current_time) % 2 == 0:
                bat_color = (0, 0, 0)
        
        # Draw battery indicator
        battery_width = 120
        battery_height = 25
        battery_x = self.display_width - battery_width - 20
        battery_y = 50
        
        # Battery outline
        cv2.rectangle(combined, (battery_x, battery_y), 
                    (battery_x + battery_width, battery_y + battery_height), 
                    (255, 255, 255), 2)
        
        # Battery fill based on percentage
        fill_width = int((battery_width - 4) * (self.battery_level / 100))
        cv2.rectangle(combined, 
                    (battery_x + 2, battery_y + 2), 
                    (battery_x + 2 + fill_width, battery_y + battery_height - 2), 
                    bat_color, -1)
        
        # Battery percentage text
        cv2.putText(combined, f"{self.battery_level}%", 
                (battery_x + battery_width + 10, battery_y + battery_height - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add autonomous mode status
        if self.autonomous_mode:
            status = "AUTO MODE: ON"
            color = (0, 255, 0)  # Green
        else:
            status = "AUTO MODE: OFF"
            color = (255, 255, 255)  # White
            
        cv2.putText(combined, status, (self.display_width - 200, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        
        # Add yaw status if in autonomous mode
        if self.autonomous_mode:
            # Position below the battery display
            yaw_status_y = 90
            
            if hasattr(self, 'yaw_start_time') and self.yaw_start_time is not None:
                elapsed = time.time() - self.yaw_start_time
                progress = min(elapsed / self.yaw_duration, 1.0) * 100 if self.yaw_duration else 0
                yaw_text = f"YAW: {progress:.0f}% ({self.yaw_target_angle:.1f}°)"
                yaw_color = (0, 165, 255)  # Orange during yaw
            else:
                # Check current yaw error
                if self.last_valid_depth is not None:
                    furthest_idx = np.argmax(self.last_valid_depth)
                    furthest_y, furthest_x = np.unravel_index(furthest_idx, self.last_valid_depth.shape)
                    
                    # Get image center and dimensions
                    h, w = self.last_valid_depth.shape
                    center_y, center_x = h // 2, w // 2
                    
                    # Calculate x offset from center
                    x_offset = furthest_x - center_x
                    
                    # Get focal length from predictor
                    focal_length = self.predictor.get_focal_length()
                    if focal_length is None:
                        focal_length = 700
                        
                    # Calculate yaw angle
                    yaw_angle = np.arctan2(x_offset, focal_length) * (180 / np.pi)
                    yaw_aligned = abs(yaw_angle) < 10
                    
                    if yaw_aligned:
                        yaw_text = "YAW: Aligned"
                        yaw_color = (0, 255, 0)  # Green when aligned
                    else:
                        yaw_text = f"YAW: {yaw_angle:.1f}° needed"
                        yaw_color = (0, 0, 255)  # Red when not aligned
                else:
                    yaw_text = "YAW: No data"
                    yaw_color = (255, 255, 255)  # White when no data
                    
            cv2.putText(combined, yaw_text, 
                    (self.display_width - 200, yaw_status_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, yaw_color, 1, cv2.LINE_AA)
        
        # Add control instructions
        instructions = [
            "Controls:",
            "W/S: Forward/Backward",
            "A/D: Left/Right",
            "Q/E: Rotate Left/Right",
            "R/F: Up/Down",
            "T: Takeoff",
            "L: Land",
            "M: Toggle Autonomous Mode",
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

    def _autonomous_navigation(self, depth):
        """
        Autonomous navigation based on depth data:
        - Finds furthest pixel in depth map
        - Yaws drone to face that direction
        - Moves forward unless obstacle within 1m
        - Adjusts altitude based on target height
        """
        if depth is None or not self.send_rc_control:
            return
        
        # Find coordinates of furthest point (HIGHEST depth value)
        furthest_idx = np.argmax(depth)
        furthest_y, furthest_x = np.unravel_index(furthest_idx, depth.shape)
        furthest_dist = depth[furthest_y, furthest_x]
        
        # Find the closest pixel (for obstacle avoidance)
        closest_idx = np.argmin(depth)
        closest_y, closest_x = np.unravel_index(closest_idx, depth.shape)
        closest_dist = depth[closest_y, closest_x]
        
        # Get image center and dimensions
        h, w = depth.shape
        center_y, center_x = h // 2, w // 2
        
        # Calculate x and y offsets from center
        x_offset = furthest_x - center_x
        y_offset = furthest_y - center_y
        
        # Get focal length from predictor
        focal_length = self.predictor.get_focal_length()
        if focal_length is None:
            focal_length = 700  # Default focal length estimate if not available
        
        # Calculate yaw angle (horizontal)
        yaw_angle = np.arctan2(x_offset, focal_length) * (180 / np.pi)
        
        # Calculate pitch angle (vertical) 
        pitch_angle = np.arctan2(y_offset, focal_length) * (180 / np.pi)
        
        # Consider aligned if within threshold
        yaw_aligned = abs(yaw_angle) < 15  # Slightly more forgiving threshold
        
        # Determine the current state
        currently_yawing = hasattr(self, 'yaw_start_time') and self.yaw_start_time is not None
        currently_adjusting_altitude = hasattr(self, 'altitude_adjust_start') and self.altitude_adjust_start is not None
        currently_moving_forward = hasattr(self, 'forward_start_time') and self.forward_start_time is not None
        
        # ====================== YAW CONTROL ======================
        # Check if we should STOP yawing
        if currently_yawing:
            elapsed_time = time.time() - self.yaw_start_time
            if elapsed_time >= self.yaw_duration:
                # Force stop yawing
                self.yaw_velocity = 0
                self.yaw_start_time = None
                self.yaw_target_angle = None
                dprint("Completed yaw maneuver - STOPPED")
                currently_yawing = False
        
        # Check if we should START yawing (only if not already in a yaw)
        if not yaw_aligned and not currently_yawing:
            # Start a new yaw maneuver
            self.yaw_target_angle = yaw_angle
            self.yaw_start_time = time.time()
            
            # Calculate duration needed to complete the yaw (minimum 0.5s to ensure it registers)
            # Tello rotates at about 100 degrees/second at max speed
            yaw_rate = 100 * (DRONE_SPEED / 100)  # Degrees per second at current speed
            self.yaw_duration = max(abs(yaw_angle) / yaw_rate, 0.5)
            
            # Set yaw velocity
            self.yaw_velocity = int(np.sign(yaw_angle) * DRONE_SPEED)
            
            # Reset forward movement when starting a new yaw
            self.for_back_velocity = 0
            self.forward_start_time = None
            
            dprint(f"Starting yaw: {yaw_angle:.1f}° for {self.yaw_duration:.2f}s")
            currently_yawing = True
        
        # =================== ALTITUDE CONTROL ===================
        # Check if we should STOP altitude adjustment
        if currently_adjusting_altitude:
            elapsed_time = time.time() - self.altitude_adjust_start
            if elapsed_time >= self.altitude_duration:
                # Force stop altitude adjustment
                self.up_down_velocity = 0
                self.altitude_adjust_start = None
                dprint("Completed altitude adjustment - STOPPED")
                currently_adjusting_altitude = False
        
        # Check if we should START altitude adjustment
        pitch_threshold = 15  # degrees
        if abs(pitch_angle) > pitch_threshold and not currently_adjusting_altitude and not currently_yawing:
            # Set a fixed duration for the adjustment
            self.altitude_duration = 0.5  # Half second
            self.altitude_adjust_start = time.time()
            
            # Set a fixed velocity (20% of max speed)
            altitude_speed = int(DRONE_SPEED * 0.2)
            self.up_down_velocity = -int(np.sign(pitch_angle) * altitude_speed)
            
            dprint(f"Starting altitude adjustment: {pitch_angle:.1f}° for {self.altitude_duration:.2f}s")
            currently_adjusting_altitude = True
        
        # ================ FORWARD MOVEMENT CONTROL ================
        # Only consider moving forward if not yawing or adjusting altitude
        can_move_forward = (closest_dist > 1.0 and 
                            yaw_aligned and 
                            not currently_yawing and 
                            not currently_adjusting_altitude)
        
        # Check if we should STOP forward movement
        if currently_moving_forward:
            elapsed_time = time.time() - self.forward_start_time
            if elapsed_time >= 3.0 or not can_move_forward:
                # Force stop forward movement
                self.for_back_velocity = 0
                self.forward_start_time = None
                dprint("Completed forward movement - STOPPED")
                currently_moving_forward = False
        
        # Check if we should START forward movement
        if can_move_forward and not currently_moving_forward:
            self.forward_start_time = time.time()
            self.for_back_velocity = DRONE_SPEED
            dprint(f"Starting forward movement toward target at {furthest_dist:.2f}m")
            currently_moving_forward = True
        
        # ================ SAFETY CHECK ================
        # Make absolutely sure we're not sending commands after we should have stopped
        if not currently_yawing:
            self.yaw_velocity = 0
        
        if not currently_adjusting_altitude:
            self.up_down_velocity = 0
        
        if not currently_moving_forward:
            self.for_back_velocity = 0
        
        # Left-right velocity should always be zero in autonomous mode
        self.left_right_velocity = 0
        
        # Debug current state
        dprint(f"STATE: yaw={currently_yawing}, altitude={currently_adjusting_altitude}, forward={currently_moving_forward}")
        dprint(f"VELOCITY: yaw={self.yaw_velocity}, up_down={self.up_down_velocity}, forward={self.for_back_velocity}")

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
            
            # Run autonomous navigation if enabled
            if self.autonomous_mode and depth is not None:
                self._autonomous_navigation(depth)
            
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
                elif key == ord('m'):  # Toggle autonomous mode
                    self.autonomous_mode = not self.autonomous_mode
                    mode_str = "ON" if self.autonomous_mode else "OFF"
                    rprint(f"[yellow]Autonomous mode: {mode_str}[/yellow]")
                else:
                    # Only update from keys if not in autonomous mode
                    if not self.autonomous_mode:
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