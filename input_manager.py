# input_manager.py
import pygame

class UnifiedInputManager:
    def __init__(self, width, height):
        self.WIDTH = width
        self.HEIGHT = height
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        self.compression_axis = 0.0

    def toggle_mouse_lock(self):
        is_grabbed = pygame.event.get_grab()
        pygame.event.set_grab(not is_grabbed)
        pygame.mouse.set_visible(is_grabbed)

    def process_inputs(self, joysticks, dt):
        keys = pygame.key.get_pressed()
        mouse_buttons = pygame.mouse.get_pressed()
        is_real_data = False
        
        # ДВИЖЕНИЕ И ПОВОРОТЫ
        ctrl_vx = float(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - float(keys[pygame.K_LEFT] or keys[pygame.K_a])
        ctrl_vy = float(keys[pygame.K_DOWN] or keys[pygame.K_s]) - float(keys[pygame.K_UP] or keys[pygame.K_w])
        ctrl_tq = float(keys[pygame.K_e]) - float(keys[pygame.K_q])
        ctrl_comp = float(keys[pygame.K_SPACE]) - float(keys[pygame.K_LSHIFT])

        # АЛХИМИЧЕСКИЕ ОСИ ЭМУЛЯЦИИ COHERENCE
        # 1. Спектральная ось: Z (-1, Theta) vs C (+1, Gamma)
        ctrl_alch_freq = float(keys[pygame.K_c]) - float(keys[pygame.K_z])
        # 2. Пространственная ось: X (-1, Shield) vs V (+1, Core)
        ctrl_alch_spatial = float(keys[pygame.K_v]) - float(keys[pygame.K_x])

        # МЫШЬ
        ctrl_comp += float(mouse_buttons[0]) - float(mouse_buttons[2])
        if pygame.event.get_grab():
            mouse_dx, _ = pygame.mouse.get_rel()
            ctrl_tq += max(-1.0, min(1.0, -mouse_dx * 0.05))

        # ГЕЙМПАД
        if len(joysticks) > 0:
            joystick = joysticks[0]
            num_axes = joystick.get_numaxes()
            num_buttons = joystick.get_numbuttons()
            
            if num_axes >= 2:
                jx = joystick.get_axis(0)
                jy = joystick.get_axis(1)
                ctrl_vx += jx if abs(jx) > 0.15 else 0.0
                ctrl_vy += jy if abs(jy) > 0.15 else 0.0
            
            if num_axes >= 4:
                jtq = joystick.get_axis(3)
                ctrl_tq -= jtq if abs(jtq) > 0.15 else 0.0
            
            if num_axes >= 6:
                r2 = (joystick.get_axis(5) + 1.0) / 2.0
                l2 = (joystick.get_axis(2) + 1.0) / 2.0
                ctrl_comp += (r2 - l2)

            # Эмуляция частоты на кнопках геймпада (кнопки X/B и Y/A)
            if num_buttons >= 4:
                # Спектральная ось на кнопках: B (кнопка 1, +1) и X (кнопка 2, -1)
                ctrl_alch_freq += float(joystick.get_button(1)) - float(joystick.get_button(2))
                # Пространственная ось на кнопках: Y (кнопка 3, +1) и A (кнопка 0, -1)
                ctrl_alch_spatial += float(joystick.get_button(3)) - float(joystick.get_button(0))

        eeg_vx = max(-1.0, min(1.0, ctrl_vx))
        eeg_vy = max(-1.0, min(1.0, ctrl_vy))
        eeg_tq = max(-1.0, min(1.0, ctrl_tq))
        
        # Сглаживание осей алхимии геймпада
        alch_freq = max(-1.0, min(1.0, ctrl_alch_freq))
        alch_spatial = max(-1.0, min(1.0, ctrl_alch_spatial))
        
        if abs(ctrl_comp) > 0.05:
            self.compression_axis = max(-1.0, min(1.0, self.compression_axis + ctrl_comp * dt * 4.0))
        else:
            if self.compression_axis > 0.05: self.compression_axis = max(0.0, self.compression_axis - dt * 3.0)
            elif self.compression_axis < -0.05: self.compression_axis = min(0.0, self.compression_axis + dt * 3.0)
            else: self.compression_axis = 0.0

        if abs(eeg_vx) > 0.05 or abs(eeg_vy) > 0.05 or abs(eeg_tq) > 0.05 or abs(ctrl_comp) > 0.05 or abs(alch_freq) > 0.05 or abs(alch_spatial) > 0.05:
            is_real_data = True

        return is_real_data, eeg_vx, eeg_vy, eeg_tq, self.compression_axis, alch_freq, alch_spatial
