# phase_vortex_labyrinth.py
import pygame
import torch
import torch.nn.functional as F
import math
import sys
import numpy as np
import random
import warnings
import traceback

warnings.filterwarnings("ignore", category=UserWarning)

# Попытка импортировать реальные модули ЭЭГ
try:
    from neuro_driver import RealNeuroDriver
    from symbiotic_engine import SymbioticEngineGPU
    from implicit_config import COORDS_16_X, COORDS_16_Y
    HAS_NEURO = True
except ImportError:
    HAS_NEURO = False
    COORDS_16_X = [10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14]
    COORDS_16_Y = [-2.72, -7.43, -4.77, -10.15, -10.14, -4.77, -7.42, -2.73, 2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71]

WIDTH, HEIGHT = 800, 800
COMPUTE_RES = 128  # Сетка 128x128 для численного решения уравнений Навье-Стокса (Stable Fluids)

class PurePhaseVortexLabyrinth:
    def __init__(self, device):
        self.device = device
        self.res = COMPUTE_RES
        
        # Конфигурационные параметры физики и чувствительности
        self.cfg = {
            'vorticity_sensitivity': 0.15,  # Чувствительность вращения камеры от вихрей жидкости
            'manual_rotation_speed': 1.8,   # Базовая скорость ручного вращения (клавиатура)
            'fluid_damping': 0.80,          # Вязкое трение (затухание) скорости слайма
        }
        
        # Переменная для сглаживания вихревого вращения (low-pass filter)
        self.smooth_vorticity = 0.0
        
        # Физические сетки симуляции (в мировом пространстве)
        self.u = torch.zeros((1, 1, self.res, self.res), dtype=torch.float32, device=self.device) # Горизонтальная скорость
        self.v = torch.zeros((1, 1, self.res, self.res), dtype=torch.float32, device=self.device) # Вертикальная скорость
        self.density = torch.zeros((1, 3, self.res, self.res), dtype=torch.float32, device=self.device) # RGB плотность цвета
        
        # Физическое тело игрока — мягкий деформируемый слайм (плотность)
        self.player_density = torch.zeros((1, 1, self.res, self.res), dtype=torch.float32, device=self.device)
        
        # Координатная сетка для полу-Лагранжевой адвекции
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.res, device=self.device),
            torch.linspace(-1.0, 1.0, self.res, device=self.device),
            indexing='ij'
        )
        self.grid_x = x_grid # Форма (res, res)
        self.grid_y = y_grid # Форма (res, res)
        
        # Геометрия FreeEEG16
        self.pin_x = torch.tensor(COORDS_16_X, device=self.device)
        self.pin_y = torch.tensor(COORDS_16_Y, device=self.device)
        
        # Динамические масштабы упругого сжатия цитоскелета для каждого из 16 ядер
        self.current_pin_scales = torch.ones(16, device=self.device)
        
        # Статический каркас лабиринта и динамическое поле неньютоновского геля стен
        self.orig_obstacles = self.init_obstacles()
        self.wall_density = self.orig_obstacles.clone() # Очень прочные, но деформируемые давлением стены
        
        # Аватар игрока (Lander) — позиция и угол хранятся в мировом пространстве
        self.player_pos = torch.tensor([WIDTH / 2.0, HEIGHT / 2.0], dtype=torch.float32, device=self.device)
        self.player_angle = 0.0 # Угол поворота аватара в радианах
        
    def init_obstacles(self):
        obstacles = torch.zeros((1, 1, self.res, self.res), device=self.device)
        r = torch.sqrt(self.grid_x**2 + self.grid_y**2)
        theta = torch.atan2(self.grid_y, self.grid_x)
        
        # 3 очень плотных концентрических барьера-кольца толщиной 0.10 (~40 пикселей)
        rings_cfg = [
            {"r_min": 0.18, "r_max": 0.28, "gap_start": -0.5, "gap_end": 0.5}, # Толщина 0.10
            {"r_min": 0.48, "r_max": 0.58, "gap_start": 1.2,  "gap_end": 2.2},  # Толщина 0.10
            {"r_min": 0.78, "r_max": 0.88, "gap_start": -2.3, "gap_end": -1.3}  # Толщина 0.10
        ]
        
        for ring in rings_cfg:
            ring_mask = (r >= ring["r_min"]) & (r <= ring["r_max"])
            gap_mask = (theta >= ring["gap_start"]) & (theta <= ring["gap_end"])
            obstacles[0, 0, ring_mask & ~gap_mask] = 1.0
            
        return obstacles

    def reset_world(self):
        self.player_pos.copy_(torch.tensor([WIDTH / 2.0, HEIGHT / 2.0], device=self.device))
        self.player_angle = 0.0
        self.u.zero_()
        self.v.zero_()
        self.density.zero_()
        self.player_density.zero_()
        self.wall_density.copy_(self.orig_obstacles)
        self.smooth_vorticity = 0.0

    def advect(self, field, dt):
        # Полу-Лагранжева адвекция с аппаратным билинейным сэмплингом PyTorch
        dx = self.u[0, 0] * (dt * 2.0 / self.res)
        dy = self.v[0, 0] * (dt * 2.0 / self.res)
        
        sampling_grid = torch.stack([self.grid_x - dx, self.grid_y - dy], dim=-1).unsqueeze(0)
        sampling_grid = torch.clamp(sampling_grid, -1.0, 1.0)
        
        return F.grid_sample(field, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)

    def project(self):
        # Строгая непроницаемая маска твердых барьеров у стен
        block_mask = (self.wall_density > 0.15).float()
        
        # Обнуляем скорость внутри стен для предотвращения любого численного просачивания жидкости
        self.u = self.u * (1.0 - block_mask)
        self.v = self.v * (1.0 - block_mask)
        
        u_pad = F.pad(self.u, (1, 1, 0, 0), mode='replicate')
        v_pad = F.pad(self.v, (0, 0, 1, 1), mode='replicate')
        div = 0.5 * (u_pad[:, :, :, 2:] - u_pad[:, :, :, :-2] + v_pad[:, :, 2:, :] - v_pad[:, :, :-2, :])
        
        # Итерации Якоби для давления
        p = torch.zeros_like(self.u)
        for _ in range(40):
            p_pad = F.pad(p, (1, 1, 1, 1), mode='replicate')
            p = 0.25 * (p_pad[:, :, 1:-1, 2:] + p_pad[:, :, 1:-1, :-2] + 
                        p_pad[:, :, 2:, 1:-1] + p_pad[:, :, :-2, 1:-1] - div)
            # Давление полностью блокируется твердыми границами стен
            p = p * (1.0 - block_mask)
            
        p_pad = F.pad(p, (1, 1, 1, 1), mode='replicate')
        self.u -= 0.5 * (p_pad[:, :, 1:-1, 2:] - p_pad[:, :, 1:-1, :-2])
        self.v -= 0.5 * (p_pad[:, :, 2:, 1:-1] - p_pad[:, :, :-2, 1:-1])
        
        # Повторно фиксируем нулевую скорость на границах (герметизация)
        self.u = self.u * (1.0 - block_mask)
        self.v = self.v * (1.0 - block_mask)

    def step(self, dt, time_sec, eeg_c0, eeg_vx, eeg_vy, eeg_tq, eeg_phases, is_real_data, compression, scale):
        try:
            # Предохранители от численного взрыва при сбоях симуляции (NaN/Inf recovery)
            self.u = torch.nan_to_num(self.u, nan=0.0, posinf=0.0, neginf=0.0)
            self.v = torch.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)
            self.density = torch.nan_to_num(self.density, nan=0.0, posinf=0.0, neginf=0.0)
            self.wall_density = torch.nan_to_num(self.wall_density, nan=0.0, posinf=0.0, neginf=0.0)
            self.player_density = torch.nan_to_num(self.player_density, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ограничение максимальной скорости течений (CFL clamp)
            self.u = torch.clamp(self.u, -200.0, 200.0)
            self.v = torch.clamp(self.v, -200.0, 200.0)
            
            # Медленное рассеивание (dissipation)
            self.density *= 0.985
            self.u *= 0.99
            self.v *= 0.99
            
            # Медленное адвективное течение самих стен (эффект пластичной эрозии/сдвига геля под сильным током)
            self.wall_density = self.advect(self.wall_density, dt * 0.08)
            
            # ФИЗИЧЕСКАЯ ЭРОЗИЯ ПОТОКОМ: Сильные течения и реактивные струи постепенно размывают гель стен!
            fluid_speed = torch.sqrt(self.u**2 + self.v**2)
            erosion = fluid_speed * 0.04 * dt
            self.wall_density = torch.clamp(self.wall_density - erosion, 0.0, 1.0)
            
            # Эластичное восстановление формы стен лабиринта (Memory Foam Effect)
            self.wall_density += (self.orig_obstacles - self.wall_density) * 0.35 * dt
            self.wall_density = torch.clamp(self.wall_density, 0.0, 1.0)
            
            # Поворот аватара (камеры) рассчитывается ниже через физический вихревой ротор течения!
            cos_p = math.cos(self.player_angle)
            sin_p = math.sin(self.player_angle)
            
            y_indices, x_indices = torch.meshgrid(
                torch.arange(self.res, device=self.device, dtype=torch.float32),
                torch.arange(self.res, device=self.device, dtype=torch.float32),
                indexing='ij'
            )
            
            # === ДИНАМИЧЕСКИЙ ЦИТОСКЕЛЕТ СЛАЙМА ===
            self.current_pin_scales = torch.ones(16, device=self.device) * scale
            for i in range(16):
                # Проверяем упругую деформацию луча скелета к центру
                for step_idx in range(6):
                    curr_scale = scale * (1.0 - step_idx * 0.15)
                    dx = self.pin_x[i].item() * curr_scale * cos_p - self.pin_y[i].item() * curr_scale * sin_p
                    dy = self.pin_x[i].item() * curr_scale * sin_p + self.pin_y[i].item() * curr_scale * cos_p
                    
                    wx = self.player_pos[0].item() + dx
                    wy = self.player_pos[1].item() + dy
                    
                    # Фиксируем нативное CPU-ограничение диапазона координат
                    gx_idx = int(max(0, min(self.res - 1, (wx / WIDTH) * self.res)))
                    gy_idx = int(max(0, min(self.res - 1, (wy / HEIGHT) * self.res)))
                    
                    if self.wall_density[0, 0, gy_idx, gx_idx] < 0.25:
                        self.current_pin_scales[i] = curr_scale
                        break
                    else:
                        self.current_pin_scales[i] = scale * 0.2 # Предельное сжатие при жестком сдавливании
            
            # Вычисляем мировые координаты всех 16 пинов в виде тензора (16, 2)
            dx_pins = self.pin_x * self.current_pin_scales * cos_p - self.pin_y * self.current_pin_scales * sin_p
            dy_pins = self.pin_x * self.current_pin_scales * sin_p + self.pin_y * self.current_pin_scales * cos_p
            
            pin_world_coords = torch.stack([
                self.player_pos[0] + dx_pins,
                self.player_pos[1] + dy_pins
            ], dim=1) # Форма (16, 2)
            
            # Вычисляем попарные расстояния и направления между всеми узлами
            diffs = pin_world_coords.unsqueeze(0) - pin_world_coords.unsqueeze(1) # (16, 16, 2)
            distances = torch.norm(diffs, dim=2, keepdim=True) # (16, 16, 1)
            directions = diffs / (distances + 1e-5) # (16, 16, 2)
            
            # 1. Впрыск сил из 16 электродов (ядер цитоскелета)
            resting_player_density = torch.zeros_like(self.wall_density)
            
            for i in range(16):
                pin_world_x = pin_world_coords[i, 0].item()
                pin_world_y = pin_world_coords[i, 1].item()
                
                # Переводим в координаты сетки
                pin_x_grid = (pin_world_x / WIDTH) * self.res
                pin_y_grid = (pin_world_y / HEIGHT) * self.res
                
                # Накапливаем опорную форму слайма сжатого скелета (Rest Template)
                d2_rest = (x_indices - pin_x_grid)**2 + (y_indices - pin_y_grid)**2
                resting_density_node = torch.exp(-d2_rest / 16.0) # Мягкий радиус ядер
                resting_player_density[0, 0] = torch.max(resting_player_density[0, 0], resting_density_node)
                
                # Генерация истинных вращающихся вихрей вокруг 16 ядер когерентностей ЭЭГ!
                if is_real_data and eeg_phases is not None:
                    dx_grid = x_indices - pin_x_grid
                    dy_grid = y_indices - pin_y_grid
                    d2_dist = dx_grid**2 + dy_grid**2
                    
                    # Локальный пространственный конверт вихря
                    envelope = torch.exp(-d2_dist / 12.0).unsqueeze(0).unsqueeze(0)
                    
                    phase_angle = eeg_phases[i].item()
                    # Скорость вращения вихря когерентности модулируется фазовой активностью ЭЭГ
                    vortex_strength = math.sin(phase_angle) * 15.0 * (1.0 + compression * 2.0)
                    
                    # Математический расчет завихрения (spinning vortex velocity field)
                    d2_reg = d2_dist + 1.0
                    vortex_u = -dy_grid / d2_reg * vortex_strength
                    vortex_v = dx_grid / d2_reg * vortex_strength
                    
                    self.u += envelope * vortex_u.unsqueeze(0).unsqueeze(0)
                    self.v += envelope * vortex_v.unsqueeze(0).unsqueeze(0)
                    
                    # Впрыск цвета по фазе
                    self.density[:, 0, :, :] += envelope[0, 0] * abs(math.cos(phase_angle))
                    self.density[:, 1, :, :] += envelope[0, 0] * abs(math.sin(phase_angle))
                    self.density[:, 2, :, :] += envelope[0, 0] * abs(math.sin(phase_angle * 1.5))

            # === РАСПРЕДЕЛЕННЫЕ СИЛЫ КОГЕРЕНТНОСТЕЙ (ДЕСЕНТРАЛИЗОВАННЫЙ ДВИГАТЕЛЬ) ===
            # Если реальные данные не активны (например, эмуляция), генерируем когерентность искусственно
            if eeg_c0 is None or not is_real_data:
                # Вектор движения в локальной системе координат
                v_local = torch.tensor([eeg_vx, -eeg_vy], dtype=torch.float32, device=self.device)
                
                # Проецируем вектор на попарные направления между ядрами
                # Сохраняем исходный знак (Preserve full bipolar -1 to 1 space)
                eeg_c0 = torch.sum(directions * v_local.view(1, 1, 2), dim=2) * 0.15
                
                if abs(eeg_tq) > 0.01:
                    # Добавляем тангенциальные силы для создания вращательной когерентности
                    cx_pin = self.pin_x * self.current_pin_scales
                    cy_pin = self.pin_y * self.current_pin_scales
                    tangents = torch.stack([-cy_pin, cx_pin], dim=1) # (16, 2)
                    tangents = tangents / (torch.norm(tangents, dim=1, keepdim=True) + 1e-5)
                    
                    eeg_c0 += torch.sum(directions * tangents.unsqueeze(1), dim=2) * eeg_tq * 0.1
                    
                # Клонируем, чтобы безопасно применить in-place операцию fill_diagonal_
                eeg_c0 = eeg_c0.clone()
                eeg_c0.fill_diagonal_(0.0)

            # Переносим c0 на GPU и преобразуем к 16x16
            c0_gpu = eeg_c0.to(self.device).float()[:16, :16]
            
            # Сила притяжения/отталкивания между каждой парой узлов на основе их когерентности
            # diffs[i, j] указывает от i к j. Если c0[i, j] > 0, притягиваем i к j.
            forces = directions * c0_gpu.unsqueeze(2) # Форма (16, 16, 2)
            net_forces = torch.sum(forces, dim=1) # Суммарный вектор силы для каждого из 16 узлов. Форма (16, 2)

            # Впрыскиваем распределенные реактивные джеты для каждого из 16 узлов отдельно!
            if is_real_data:
                for i in range(16):
                    pin_x_grid = (pin_world_coords[i, 0] / WIDTH) * self.res
                    pin_y_grid = (pin_world_coords[i, 1] / HEIGHT) * self.res
                    
                    d2_node = (x_indices - pin_x_grid)**2 + (y_indices - pin_y_grid)**2
                    node_footprint = torch.exp(-d2_node / 8.0).unsqueeze(0).unsqueeze(0)
                    
                    # Применяем локальный реактивный импульс на основе суммарных сил когерентности
                    # Узел выбрасывает жидкость в направлении силы, толкая соответствующий край слайма
                    force_x = net_forces[i, 0] * 120.0 * (1.0 + compression * 3.0)
                    force_y = net_forces[i, 1] * 120.0 * (1.0 + compression * 3.0)
                    
                    self.u += node_footprint * force_x
                    self.v += node_footprint * force_y
                    
                    # Отрисовка золотистых реактивных следов пропорционально силе
                    force_mag = torch.sqrt(force_x**2 + force_y**2).item()
                    if force_mag > 0.1:
                        self.density[:, 0:1, :, :] += node_footprint * min(1.0, force_mag / 50.0)
                        self.density[:, 1:2, :, :] += node_footprint * min(0.85, force_mag / 60.0)
                        self.density[:, 2:3, :, :] += node_footprint * min(0.1, force_mag / 200.0)

            # 3. Шаг численного решения Навье-Стокса
            self.u = self.advect(self.u, dt)
            self.v = self.advect(self.v, dt)
            self.density = self.advect(self.density, dt)
            self.project()

            # 4. МЯГКОЕ ТЕЛО СЛАЙМА (Advection & Cohesion):
            # Инициализация тела слайма при старте
            if torch.sum(self.player_density) < 1e-4:
                self.player_density.copy_(resting_player_density)
                
            # Тело слайма перемещается и деформируется ТОЛЬКО за счет адвекции полем скоростей Навье-Стокса!
            self.player_density = self.advect(self.player_density, dt * 1.5)
            
            # Сила упругого стягивания слайма обратно к его ядрам когерентностей (электродам)
            self.player_density += (resting_player_density - self.player_density) * 4.5 * dt
            
            # Герметичное ограничение: плотность слайма мгновенно зануляется внутри твердых границ геля стен
            block_mask = (self.wall_density > 0.15).float()
            self.player_density = self.player_density * (1.0 - block_mask)
            self.player_density = torch.clamp(self.player_density, 0.0, 1.0)

            # 5. КОРТИКАЛЬНЫЙ ЦЕНТР МАСС (COM): 
            # Камера и физический якорь игрока ВСЕГДА строго и абсолютно центрируются на центр масс его слайм-тела!
            d_sum = torch.sum(self.player_density) + 1e-6
            com_x = torch.sum(x_indices * self.player_density[0, 0]) / d_sum
            com_y = torch.sum(y_indices * self.player_density[0, 0]) / d_sum
            
            # Переводим центр масс в пиксельные мировые координаты
            com_world_x = (com_x / self.res) * WIDTH
            com_world_y = (com_y / self.res) * HEIGHT
            
            # Абсолютное центрирование без лагов (камера жестко прикована к массе амебы)
            self.player_pos[0] = com_world_x
            self.player_pos[1] = com_world_y

            self.player_pos[0] = torch.clamp(self.player_pos[0], 20.0, WIDTH - 20.0)
            self.player_pos[1] = torch.clamp(self.player_pos[1], 20.0, HEIGHT - 20.0)

            # === РАСЧЕТ ФИЗИЧЕСКОГО ВИХРЕВОГО МОМЕНТА (VORTICITY) ===
            # Ротор течения вращает тело слайма, что плавно разворачивает камеру на угол player_angle!
            u_pad = F.pad(self.u, (1, 1, 1, 1), mode='replicate')
            v_pad = F.pad(self.v, (1, 1, 1, 1), mode='replicate')
            dv_dx = 0.5 * (v_pad[:, :, 1:-1, 2:] - v_pad[:, :, 1:-1, :-2])
            du_dy = 0.5 * (u_pad[:, :, 2:, 1:-1] - u_pad[:, :, :-2, 1:-1])
            vorticity = dv_dx - du_dy # (1, 1, res, res)
            
            # Средняя завихренность течений, воздействующая на массу слайма
            avg_vorticity = torch.sum(vorticity * self.player_density) / d_sum
            
            # Сглаживаем вихревое вращение с помощью экспоненциального фильтра (low-pass filter)
            self.smooth_vorticity = self.smooth_vorticity * 0.92 + avg_vorticity.item() * 0.08
            
            # Динамически меняем угол курса на основе сглаженного вращения жидкости и ручного управления
            if is_real_data:
                self.player_angle += eeg_tq * dt * self.cfg['manual_rotation_speed']
                self.player_angle += self.smooth_vorticity * dt * self.cfg['vorticity_sensitivity']
                self.player_angle = (self.player_angle + math.pi) % (2.0 * math.pi) - math.pi

            # === ПОБЕГ ЗА ПРЕДЕЛЫ ЛАБИРИНТА ===
            dx_c = self.player_pos[0] - WIDTH / 2.0
            dy_c = self.player_pos[1] - HEIGHT / 2.0
            r_norm = torch.sqrt(dx_c**2 + dy_c**2) / 400.0 # Нормализованное расстояние от центра
            
            if r_norm.item() > 0.89: # Успешный выход за пределы самого внешнего кольца (r_max = 0.88)!
                self.reset_world()
        except Exception as e:
            print("[CRITICAL EXCEPTION IN ARENA.STEP]:")
            traceback.print_exc()
            pygame.quit()
            sys.exit()

    def draw_electrode_sensors(self, surface):
        """Отрисовка позиций 16 сухих электродов FreeEEG16, зафиксированных на экране и динамически сжимающихся"""
        for i in range(16):
            pin_scale = self.current_pin_scales[i].item()
            sx = WIDTH / 2.0 + self.pin_x[i].item() * pin_scale
            sy = HEIGHT / 2.0 + self.pin_y[i].item() * pin_scale
            pygame.draw.circle(surface, (0, 255, 255, 100), (int(sx), int(sy)), 8, 1)
            pygame.draw.circle(surface, (0, 255, 255), (int(sx), int(sy)), 2)

    def draw_tension_lines(self, surface, compression):
        """Проекция векторов течений жидкости в системе отсчета камеры"""
        step_x = WIDTH // 24
        step_y = HEIGHT // 24
        
        # Получаем аффинную сетку вида камеры
        theta = -self.player_angle
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        px_norm = (self.player_pos[0] / WIDTH) * 2.0 - 1.0
        py_norm = (self.player_pos[1] / HEIGHT) * 2.0 - 1.0
        
        M = torch.tensor([[
            [cos_t, -sin_t, px_norm],
            [sin_t,  cos_t, py_norm]
        ]], dtype=torch.float32, device=self.device)
        
        grid = F.affine_grid(M, size=(1, 1, self.res, self.res), align_corners=True)
        cam_u = F.grid_sample(self.u, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        cam_v = F.grid_sample(self.v, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        u_cpu = cam_u[0, 0].cpu().numpy()
        v_cpu = cam_v[0, 0].cpu().numpy()
        
        # Поворачиваем сами вектора скоростей в систему координат камеры
        cos_r = math.cos(-self.player_angle)
        sin_r = math.sin(-self.player_angle)
        
        for i in range(1, 23):
            for j in range(1, 23):
                grid_y = int(i * (self.res / 24))
                grid_x = int(j * (self.res / 24))
                
                vx_world = float(u_cpu[grid_y, grid_x])
                vy_world = float(v_cpu[grid_y, grid_x])
                
                # Локальный вектор скорости на экране
                vx = vx_world * cos_r - vy_world * sin_r
                vy = vx_world * sin_r + vy_world * cos_r
                
                speed = math.hypot(vx, vy)
                if speed > 0.5:
                    max_draw_len = 15.0
                    if speed > max_draw_len:
                        draw_vx = (vx / speed) * max_draw_len
                        draw_vy = (vy / speed) * max_draw_len
                    else:
                        draw_vx = vx
                        draw_vy = vy
                    
                    start_x = j * step_x
                    start_y = i * step_y
                    end_x = start_x + draw_vx * 1.5
                    end_y = start_y + draw_vy * 1.5
                    
                    # Игнорируем невалидные/взорвавшиеся координаты отрисовки
                    if not (math.isfinite(end_x) and math.isfinite(end_y)):
                        continue
                    
                    col_factor = min(1.0, speed / 40.0)
                    color = (0, int(150 + col_factor * 105), int(255 - col_factor * 100))
                    pygame.draw.line(surface, color, (start_x, start_y), (int(end_x), int(end_y)), 1)

    def draw_phase_dials(self, surface):
        """РЕНДЕР ПРИБОРНОЙ ПАНЕЛИ ФАЗОВЫХ СТРЕЛОК ДЛЯ СЕНСОРНОГО РЕЗОНАНСА"""
        start_x = WIDTH - 180
        start_y = HEIGHT - 180
        dial_size = 35
        gap = 6
        
        u_cpu = self.u[0, 0].cpu().numpy()
        v_cpu = self.v[0, 0].cpu().numpy()
        
        pygame.draw.rect(surface, (10, 15, 30, 210), (start_x - 10, start_y - 25, 175, 185), border_radius=5)
        pygame.draw.rect(surface, (0, 255, 255, 80), (start_x - 10, start_y - 25, 175, 185), 1, border_radius=5)
        
        font_small = pygame.font.SysFont("Consolas", 10, bold=True)
        surface.blit(font_small.render("CORTICAL PHASE DIALS", True, (0, 255, 255)), (start_x, start_y - 20))
        
        # Получаем фазы течения в области игрока для сравнения
        p_gx_idx = int(max(0, min(self.res - 1, (self.player_pos[0].item() / WIDTH) * self.res)))
        p_gy_idx = int(max(0, min(self.res - 1, (self.player_pos[1].item() / HEIGHT) * self.res)))
        flow_phase = math.atan2(v_cpu[p_gy_idx, p_gx_idx], u_cpu[p_gy_idx, p_gx_idx] + 1e-5)
        
        for i in range(16):
            row = i // 4
            col = i % 4
            cx = start_x + col * (dial_size + gap) + dial_size // 2
            cy = start_y + row * (dial_size + gap) + dial_size // 2
            
            pygame.draw.circle(surface, (20, 30, 50), (cx, cy), dial_size // 2)
            pygame.draw.circle(surface, (0, 255, 255, 50), (cx, cy), dial_size // 2, 1)
            
            u_phase = (i * 0.4) % (2 * math.pi)
            ux = cx + math.cos(u_phase) * (dial_size // 2 - 2)
            uy = cy + math.sin(u_phase) * (dial_size // 2 - 2)
            pygame.draw.line(surface, (0, 255, 255), (cx, cy), (int(ux), int(uy)), 2)
            
            tx = cx + math.cos(flow_phase) * (dial_size // 2 - 2)
            ty = cy + math.sin(flow_phase) * (dial_size // 2 - 2)
            pygame.draw.line(surface, (255, 215, 0), (cx, cy), (int(tx), int(ty)), 1)
            
            surface.blit(font_small.render(str(i), True, (130, 150, 180)), (cx - 4, cy - 5))

    def render_field(self):
        """Рендер поля плотности цвета и препятствий с переносом в координаты камеры"""
        # Угол и смещение для вида от лица игрока
        theta = -self.player_angle
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        px_norm = (self.player_pos[0] / WIDTH) * 2.0 - 1.0
        py_norm = (self.player_pos[1] / HEIGHT) * 2.0 - 1.0
        
        # Аффинная матрица камеры [2, 3]
        M = torch.tensor([[
            [cos_t, -sin_t, px_norm],
            [sin_t,  cos_t, py_norm]
        ]], dtype=torch.float32, device=self.device)
        
        grid = F.affine_grid(M, size=(1, 3, self.res, self.res), align_corners=True)
        
        # Переносим плотность и маску препятствий в координаты камеры
        cam_density = F.grid_sample(self.density, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        cam_walls = F.grid_sample(self.wall_density, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        cam_player = F.grid_sample(self.player_density, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        vis = cam_density[0].permute(1, 2, 0) # Перевод в (H, W, 3)
        vis = torch.clamp(vis, 0.0, 1.0)
        
        # Отрисовка стен: смешиваем цвета течений жидкости со светящимся неоново-розовым гелем стен в камере
        wall_val = cam_walls[0, 0].unsqueeze(-1) # (H, W, 1)
        wall_color = torch.tensor([0.9, 0.15, 0.6], device=self.device).view(1, 1, 3) # Насыщенный розово-малиновый неон геля
        vis = vis * (1.0 - wall_val * 0.94) + wall_color * wall_val * 0.85
        
        # === ВЫСОКОКАЧЕСТВЕННАЯ СЛИЗИСТАЯ ВИЗУАЛИЗАЦИЯ (ТЗ) ===
        player_val = cam_player[0, 0].unsqueeze(-1) # (H, W, 1)
        
        # 1. Мягкая внутренняя цитоплазма (светящееся бирюзовое желе)
        jelly_color = torch.tensor([0.0, 0.45, 0.65], device=self.device).view(1, 1, 3)
        
        # 2. Яркая неоновая клеточная мембрана (резкий, сильный циановый ободок на границе)
        membrane_color = torch.tensor([0.2, 1.0, 0.95], device=self.device).view(1, 1, 3)
        membrane_mask = torch.clamp(1.0 - torch.abs(player_val - 0.18) / 0.08, 0.0, 1.0)
        membrane_mask = membrane_mask ** 3.0 # Резкая форма границы
        
        # Смешиваем слои на фоне цветного поля течений
        vis = vis * (1.0 - player_val * 0.6) + jelly_color * player_val * 0.6
        vis = vis * (1.0 - membrane_mask * 0.8) + membrane_color * membrane_mask * 0.95
        
        # Дополнительная отрисовка оригинального (целевого) каркаса лабиринта в виде тусклой фиолетовой голограммы
        cam_orig = F.grid_sample(self.orig_obstacles, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        orig_val = cam_orig[0, 0].unsqueeze(-1)
        vis = torch.where(orig_val > 0.5, vis * 0.45 + torch.tensor([0.22, 0.05, 0.5], device=self.device) * 0.55, vis)
        
        rgb = (vis * 255).to(torch.uint8).cpu().numpy()
        
        # Генерация pygame surface
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        surf = pygame.transform.smoothscale(surf, (WIDTH, HEIGHT))
        return surf

def main():
    try:
        pygame.init()
        pygame.joystick.init()
        
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Fluid Vortex Labyrinth")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Consolas", 12, bold=True)
        
        joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
        for j in joysticks: j.init()

        if HAS_NEURO:
            driver = RealNeuroDriver()
            driver.start_lsl_scanning_thread()
            driver.start_ble_scanning_thread()
            neuro_engine = SymbioticEngineGPU(device_name='cuda')
            device = neuro_engine.device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        arena = PurePhaseVortexLabyrinth(device)
        ui_compression = 0.0

        running = True
        while running:
            dt = clock.tick(60) / 1000.0
            dt = min(0.032, dt)  # Предохранитель (CFL) от скачков кадра на старте и лагов окна
            time_sec = pygame.time.get_ticks() / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    # Регулировка чувствительности вихревого вращения на клавиши 1 и 2
                    if event.key == pygame.K_1:
                        arena.cfg['vorticity_sensitivity'] = max(0.0, arena.cfg['vorticity_sensitivity'] - 0.05)
                    if event.key == pygame.K_2:
                        arena.cfg['vorticity_sensitivity'] = min(2.0, arena.cfg['vorticity_sensitivity'] + 0.05)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                ui_compression = min(1.0, ui_compression + dt * 2.0)
            else:
                ui_compression = max(0.0, ui_compression - dt * 2.0)

            # Вычисляем активность входных сигналов
            is_real_data = False
            eeg_vx, eeg_vy, eeg_tq = 0.0, 0.0, 0.0
            eeg_phases = None
            eeg_c0 = None
            
            # Проверяем эмуляцию по умолчанию (WASD / Стрелки)
            emul_active = any(keys[k] for k in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN, pygame.K_a, pygame.K_d, pygame.K_w, pygame.K_s, pygame.K_q, pygame.K_e])
            if emul_active:
                is_real_data = True
                if keys[pygame.K_LEFT] or keys[pygame.K_a]: eeg_vx -= 1.0
                if keys[pygame.K_RIGHT] or keys[pygame.K_d]: eeg_vx += 1.0
                if keys[pygame.K_UP] or keys[pygame.K_w]: eeg_vy -= 1.0
                if keys[pygame.K_DOWN] or keys[pygame.K_s]: eeg_vy += 1.0
                if keys[pygame.K_q]: eeg_tq -= 1.0
                if keys[pygame.K_e]: eeg_tq += 1.0
                
            eeg_phases = (torch.arange(16, device=device) * 0.4 + time_sec * 24.0 * 2 * math.pi) % (2 * math.pi)

            # Если нейроинтерфейс импортирован, проверяем подключение оборудования
            if HAS_NEURO:
                active_slots = [i for i in range(5) if driver.workers[i].is_connected or any(v == i for v in driver.lsl_inlets.values())]
                has_hardware = len(active_slots) > 0
                
                if has_hardware:
                    is_real_data = True # Приоритет отдаем аппаратному сигналу, если он есть
                    C = len(active_slots) * 16
                    for slot_idx in active_slots:
                        q = driver.queues[slot_idx]
                        while len(q) > 0:
                            sample = q.popleft()
                            start, end = slot_idx * 16, (slot_idx + 1) * 16
                            neuro_engine.pinned_cpu_buffer[start:end, :-1] = neuro_engine.pinned_cpu_buffer[start:end, 1:].clone()
                            neuro_engine.pinned_cpu_buffer[start:end, -1] = torch.tensor(sample)

                    c0_gpu, _, phases_gpu, vx, vy, tq, _, _ = neuro_engine.get_predictive_ciplv(C)
                    eeg_c0 = c0_gpu[:16, :16] # Используем матрицу когерентности 16x16
                    eeg_phases = phases_gpu[:16]
                    eeg_vx = vx.item() * 0.012
                    eeg_vy = -vy.item() * 0.012
                    eeg_tq = tq.item() * 0.012

            if len(joysticks) > 0 and joysticks[0].get_numaxes() >= 5:
                ui_compression = (joysticks[0].get_axis(4) + 1.0) / 2.0
                gp_active = any(abs(joysticks[0].get_axis(a)) > 0.05 for a in range(joysticks[0].get_numaxes()))
                is_real_data = is_real_data or gp_active

            compression = ui_compression
            scale = 1.5 + (1.0 - compression) * 5.0

            # Численный шаг физики и газодинамики жидкости
            arena.step(dt, time_sec, eeg_c0, eeg_vx, eeg_vy, eeg_tq, eeg_phases, is_real_data, compression, scale)
            
            # Рендеринг заднего плана и течений (сдвинутых и повернутых на GPU)
            bg_surface = arena.render_field()
            screen.blit(bg_surface, (0, 0))

            # Оверлей векторов сил (трансформированных под камеру)
            arena.draw_tension_lines(screen, compression)

            # Оверлей электродов (зафиксирован на экране и упруго деформируемый)
            arena.draw_electrode_sensors(screen)

            # Панель фазовых приборов
            arena.draw_phase_dials(screen)

            # Отрисовка аватара игрока точно в центре экрана, всегда развернутого ВВЕРХ
            px_val = WIDTH // 2
            py_val = HEIGHT // 2
            
            # Силуэт корабля (направлен вверх к верхней кромке монитора)
            pygame.draw.circle(screen, (255, 255, 255), (px_val, py_val), 14, 2)
            pygame.draw.line(screen, (0, 255, 255), (px_val, py_val), (px_val, py_val - 18), 3) # Курсовой указатель вперед (вверх)
            pygame.draw.circle(screen, (0, 255, 255), (px_val, py_val), 6)

            # UI
            ui_surf = pygame.Surface((450, 95), pygame.SRCALPHA)
            ui_surf.fill((10, 15, 30, 140)) 
            pygame.draw.rect(ui_surf, (0, 255, 255, 60), (0, 0, 450, 95), 1) 
            screen.blit(ui_surf, (10, 10))
            
            screen.blit(font.render(f"FLUID INJECTION COMPRESSION: {(compression*100):.0f}%", True, (255, 255, 255)), (20, 20))
            screen.blit(font.render("ЦЕЛЬ: Пробиться сквозь течения и совершить побег за внешнее кольцо лабиринта", True, (255, 200, 0)), (20, 40))
            screen.blit(font.render(f"[1 / 2] ЧУВСТВИТЕЛЬНОСТЬ ВИХРЕЙ (Vorticity Sens): {arena.cfg['vorticity_sensitivity']:.2f}", True, (0, 255, 200)), (20, 60))
            screen.blit(font.render(f"Связь ЭЭГ: {'АКТИВНА' if is_real_data else 'ПОКОЙ (Используйте WASD/Стрелки)'}", True, (0, 255, 0) if is_real_data else (150, 150, 150)), (20, 75))

            pygame.display.flip()

        if HAS_NEURO:
            driver.scanner_running = False
        pygame.quit()
        sys.exit()
    except Exception as e:
        print("[CRITICAL EXCEPTION IN MAIN LOOP]:")
        traceback.print_exc()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()
