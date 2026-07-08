# input_manager.py
import pygame

class UnifiedInputManager:
    """
    Continuous Input Mapper.
    Exposes gamepad buttons, axes, and d-pads as bipolar continuous axes.
    Preserves continuous sign spaces for advanced phase physics control.
    Saves raw input arrays for real-time telemetry rendering.
    """
    def __init__(self, width, height):
        self.WIDTH = width
        self.HEIGHT = height
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        self.compression_axis = 0.0
        
        # Diagnostic buffers for UI telemetry
        self.raw_axes = []
        self.raw_buttons = []

    def toggle_mouse_lock(self):
        is_grabbed = pygame.event.get_grab()
        pygame.event.set_grab(not is_grabbed)
        pygame.mouse.set_visible(is_grabbed)

    def process_inputs(self, joysticks, dt):
        keys = pygame.key.get_pressed()
        mouse_buttons = pygame.mouse.get_pressed()
        is_real_data = False
        
        # === KEYBOARD MAPPINGS ===
        ctrl_vx = float(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - float(keys[pygame.K_LEFT] or keys[pygame.K_a])
        ctrl_vy = float(keys[pygame.K_DOWN] or keys[pygame.K_s]) - float(keys[pygame.K_UP] or keys[pygame.K_w])
        ctrl_tq = float(keys[pygame.K_e]) - float(keys[pygame.K_q])
        ctrl_comp = float(keys[pygame.K_SPACE]) - float(keys[pygame.K_LSHIFT])

        # Active Phase Semantics (Keyboard)
        ctrl_alch_freq = float(keys[pygame.K_c]) - float(keys[pygame.K_z])
        ctrl_alch_spatial = float(keys[pygame.K_v]) - float(keys[pygame.K_x])

        # Mouse actions
        ctrl_comp += float(mouse_buttons[0]) - float(mouse_buttons[2])
        if pygame.event.get_grab():
            mouse_dx, _ = pygame.mouse.get_rel()
            ctrl_tq += max(-1.0, min(1.0, -mouse_dx * 0.05))

        # Reset diagnostic buffers
        self.raw_axes = []
        self.raw_buttons = []

        # === COMPREHENSIVE GAMEPAD MAPPING WITH DRIVER FALLBACKS ===
        if len(joysticks) > 0:
            joystick = joysticks[0]
            num_axes = joystick.get_numaxes()
            num_buttons = joystick.get_numbuttons()
            
            # Save raw states for GUI diagnostic rendering
            self.raw_axes = [float(joystick.get_axis(i)) for i in range(num_axes)]
            self.raw_buttons = [int(joystick.get_button(i)) for i in range(num_buttons)]
            
            # Left Analog Stick (Continuous movement translation)
            if num_axes >= 2:
                jx = joystick.get_axis(0)
                jy = joystick.get_axis(1)
                ctrl_vx += jx if abs(jx) > 0.15 else 0.0
                ctrl_vy += jy if abs(jy) > 0.15 else 0.0
            
            # Right Analog Stick (Torque and Compression)
            if num_axes >= 4:
                jtq = joystick.get_axis(3)
                jcomp = joystick.get_axis(4) 
                ctrl_tq -= jtq if abs(jtq) > 0.15 else 0.0
                ctrl_comp -= jcomp if abs(jcomp) > 0.15 else 0.0
            
            # Triggers L2/R2 mapped to continuous spatial control (Core vs Shield)
            if num_axes >= 6:
                r2_trigger = (joystick.get_axis(5) + 1.0) / 2.0 if abs(joystick.get_axis(5)) > 0.05 else 0.0
                l2_trigger = (joystick.get_axis(2) + 1.0) / 2.0 if abs(joystick.get_axis(2)) > 0.05 else 0.0
                ctrl_alch_spatial += (r2_trigger - l2_trigger)

            # Face Buttons A/Y and X/B
            if num_buttons >= 4:
                # X/B map to frequency (Gamma vs Theta)
                ctrl_alch_freq += float(joystick.get_button(1)) - float(joystick.get_button(2))
                # Y/A map to spatial geometry
                ctrl_alch_spatial += float(joystick.get_button(3)) - float(joystick.get_button(0))
                
            # Bumpers L1/R1 mapped to frequency tuning shifts
            if num_buttons >= 6:
                ctrl_alch_freq += float(joystick.get_button(5)) - float(joystick.get_button(4))

            # Trigger Fallbacks (if triggers act as raw buttons on cheap controller profiles)
            if num_buttons >= 8:
                ctrl_alch_spatial += float(joystick.get_button(7)) - float(joystick.get_button(6))

            # D-Pad mapping
            try:
                hats = joystick.get_numhats()
                if hats > 0:
                    hx, hy = joystick.get_hat(0)
                    ctrl_vx += float(hx)
                    ctrl_vy -= float(hy)
            except pygame.error:
                pass

        # Normalize outputs to continuous bipolar boundaries
        eeg_vx = max(-1.0, min(1.0, ctrl_vx))
        eeg_vy = max(-1.0, min(1.0, ctrl_vy))
        eeg_tq = max(-1.0, min(1.0, ctrl_tq))
        
        alch_freq = max(-1.0, min(1.0, ctrl_alch_freq))
        alch_spatial = max(-1.0, min(1.0, ctrl_alch_spatial))
        
        # Soft-body deformation decay logic
        if abs(ctrl_comp) > 0.05:
            self.compression_axis = max(-1.0, min(1.0, self.compression_axis + ctrl_comp * dt * 4.0))
        else:
            if self.compression_axis > 0.05: 
                self.compression_axis = max(0.0, self.compression_axis - dt * 3.0)
            elif self.compression_axis < -0.05: 
                self.compression_axis = min(0.0, self.compression_axis + dt * 3.0)
            else: 
                self.compression_axis = 0.0

        # Register active control to prevent timeout
        if (abs(eeg_vx) > 0.05 or abs(eeg_vy) > 0.05 or abs(eeg_tq) > 0.05 or 
            abs(ctrl_comp) > 0.05 or abs(alch_freq) > 0.05 or abs(alch_spatial) > 0.05):
            is_real_data = True

        return is_real_data, eeg_vx, eeg_vy, eeg_tq, self.compression_axis, alch_freq, alch_spatial
