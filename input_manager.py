# input_manager.py
import pygame
import math

class UnifiedInputManager:
    def __init__(self, width, height):
        self.WIDTH = width
        self.HEIGHT = height
        # Скрываем курсор и захватываем окно ввода для бесконечного считывания дельты мыши
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        # Нейтральное состояние оси сжатия: 0.0 (слайм среднего размера)
        self.compression_axis = 0.0
        
        # Постоянная фиксация (latch) калибровки триггеров
        # Как только один раз увидим сигнал ниже -0.5, флаг защелкнется в True навсегда
        self.rt_rests_at_minus_one = False
        self.lt_rests_at_minus_one = False

    def toggle_mouse_lock(self):
        """Освобождает или захватывает мышь в системе (по кнопке ESC)"""
        is_grabbed = pygame.event.get_grab()
        pygame.event.set_grab(not is_grabbed)
        pygame.mouse.set_visible(is_grabbed)

    def process_inputs(self, joysticks, dt):
        keys = pygame.key.get_pressed()
        mouse_buttons = pygame.mouse.get_pressed()
        
        is_real_data = False
        
        # --- 1. КЛАВИАТУРА (Попарное вычитание полуосей кнопок -> биполярные оси -1..1) ---
        kb_vx = float(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - float(keys[pygame.K_LEFT] or keys[pygame.K_a])
        kb_vy = float(keys[pygame.K_DOWN] or keys[pygame.K_s]) - float(keys[pygame.K_UP] or keys[pygame.K_w])
        kb_tq = float(keys[pygame.K_e]) - float(keys[pygame.K_q])
        
        # Разность кнопок сжатия: Пробел (+1) сжимает, Левый Shift (-1) расширяет
        kb_comp = float(keys[pygame.K_SPACE]) - float(keys[pygame.K_LSHIFT])
        
        ctrl_vx = kb_vx
        ctrl_vy = kb_vy
        ctrl_tq = kb_tq
        ctrl_comp = kb_comp

        # --- 2. МЫШЬ (Разность кликов -> биполярная ось сжатия + дельта поворота) ---
        # ЛКМ (+1) сжимает, ПКМ (-1) расширяет слайм
        mouse_comp = float(mouse_buttons[0]) - float(mouse_buttons[2])
        ctrl_comp += mouse_comp
        
        # Считываем непрерывную ось вращения через дельту мыши (огеймпадивание)
        if pygame.event.get_grab():
            mouse_dx, mouse_dy = pygame.mouse.get_rel()
            # Знак инвертирован для корректного неевклидова вращения
            mouse_tq = -mouse_dx * 0.05
            mouse_tq = max(-1.0, min(1.0, mouse_tq))
            ctrl_tq += mouse_tq

        # --- 3. ГЕЙМПАД (Считывание и перевод в чистые оси -1..1 с автокалибровкой триггеров) ---
        if len(joysticks) > 0:
            joystick = joysticks[0]
            num_axes = joystick.get_numaxes()
            
            # Левый стик (Движение)
            if num_axes >= 2:
                ctrl_vx += joystick.get_axis(0)
                ctrl_vy += joystick.get_axis(1)
            
            # Предварительно считываем физические значения триггеров для калибровки
            axis_2_val = joystick.get_axis(2) if num_axes >= 3 else 0.0
            axis_5_val = joystick.get_axis(5) if num_axes >= 6 else 0.0
            
            # КАЛИБРОВКА (LATCH): Если курок в покое выдал отрицательное значение — фиксируем его полярность
            if axis_5_val < -0.5:
                self.rt_rests_at_minus_one = True
            if axis_2_val < -0.5:
                self.lt_rests_at_minus_one = True

            # Правый стик (Поворот)
            if num_axes >= 4:
                axis_3_val = joystick.get_axis(3)
                
                # Если мы жестко зафиксировали, что ось 2 — это левый курок L2,
                # то правый стик X — это строго ось 3. Никаких динамических сбросов при нажатии!
                if self.lt_rests_at_minus_one:
                    jtq = axis_3_val
                else:
                    jtq = axis_2_val
                ctrl_tq -= jtq
            
            # Разность триггеров: R2 (+1) сжимает, L2 (-1) расширяет слайм
            if num_axes >= 6:
                r2_normalized = (axis_5_val + 1.0) / 2.0 if self.rt_rests_at_minus_one else axis_5_val
                l2_normalized = (axis_2_val + 1.0) / 2.0 if self.lt_rests_at_minus_one else 0.0
                
                gp_comp = r2_normalized - l2_normalized
                ctrl_comp += gp_comp

        # Нормализуем результирующие оси в диапазон [-1.0, 1.0]
        eeg_vx = max(-1.0, min(1.0, ctrl_vx))
        eeg_vy = max(-1.0, min(1.0, ctrl_vy))
        eeg_tq = max(-1.0, min(1.0, ctrl_tq))
        
        # Интегрируем дельту сжатия/разжатия во времени
        if abs(ctrl_comp) > 0.05:
            self.compression_axis = max(-1.0, min(1.0, self.compression_axis + ctrl_comp * dt * 3.5))
        else:
            # Плавный физический возврат к нейтральному состоянию (0.0) при отсутствии инпута
            if self.compression_axis > 0.05:
                self.compression_axis = max(0.0, self.compression_axis - dt * 2.5)
            elif self.compression_axis < -0.05:
                self.compression_axis = min(0.0, self.compression_axis + dt * 2.5)
            else:
                self.compression_axis = 0.0

        # Возвращаем чистую биполярную ось [-1.0, 1.0] для физического движка
        ui_compression = self.compression_axis

        if abs(eeg_vx) > 0.05 or abs(eeg_vy) > 0.05 or abs(eeg_tq) > 0.05 or abs(ctrl_comp) > 0.05:
            is_real_data = True

        return is_real_data, eeg_vx, eeg_vy, eeg_tq, ui_compression
