import pygame
import torch
import torch.nn.functional as F
import math
import sys
import numpy as np
import random

# Попытка импортировать твои реальные модули ЭЭГ
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
MAZE_DIM = 11  
CELL_SIZE = WIDTH // MAZE_DIM
COMPUTE_RES = 144  # Оптимальное разрешение для Tensor Cores вычислений (144x144)

class MazeStructure:
    """Генератор каркаса лабиринта"""
    def __init__(self, dim):
        self.dim = dim
        self.grid = np.ones((dim, dim), dtype=np.float32)
        self.generate(1, 1)
        self.exit_x, self.exit_y = dim - 2, dim - 2
        self.grid[self.exit_y][self.exit_x] = 0.0

    def generate(self, cx, cy):
        self.grid[cy][cx] = 0
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 < nx < self.dim - 1 and 0 < ny < self.dim - 1 and self.grid[ny][nx] == 1:
                self.grid[cy + dy // 2][cx + dx // 2] = 0
                self.generate(nx, ny)

class PurePhaseVortexLabyrinth:
    def __init__(self, device):
        self.device = device
        self.maze = MazeStructure(MAZE_DIM)
        
        self.res = COMPUTE_RES
        
        self.cfg = {
            'intent_gain': 750.0,   # Базовая сила ЭЭГ тяги
            'torque_gain': 0.15,    # Чувствительность разворота ЭЭГ
            'eeg_smooth': 0.96,     # Фильтрация шума ЭЭГ
            'node_strength': 4500.0,# Сила отталкивания несинхронных вихрей
            'max_vel': 350.0,       
            'damping': 0.88         
        }
        
        self.core_delta2 = 25.0 ** 2
        
        self.smooth_vx = 0.0
        self.smooth_vy = 0.0
        self.smooth_tq = 0.0
        
        # Геометрия FreeEEG16
        self.pin_x = torch.tensor(COORDS_16_X, device=self.device)
        self.pin_y = torch.tensor(COORDS_16_Y, device=self.device)
        
        # Координаты Аватара ( player_pos — мировые, на экране он ВСЕГДА в центре )
        self.player_pos = torch.tensor([WIDTH / 2.0, HEIGHT - 100.0], dtype=torch.float32, device=self.device)
        self.player_vel = torch.zeros(2, dtype=torch.float32, device=self.device)
        self.player_angle = 0.0 
        
        # --- ОРГАНИЧЕСКИЙ ЛАБИРИНТ: 400 ВИХРЕЙ ПО СПИРАЛИ ФИБОНАЧЧИ ---
        self.num_world_nodes = 400
        self.num_world_elec = self.num_world_nodes
        self.total_vortices = 16 + self.num_world_elec
        
        self.positions = torch.zeros((self.total_vortices, 2), dtype=torch.float32, device=self.device)
        self.gammas = torch.zeros(self.total_vortices, dtype=torch.float32, device=self.device)
        self.chiralities = torch.ones(self.total_vortices, dtype=torch.float32, device=self.device)
        self.phases = torch.zeros(self.total_vortices, dtype=torch.float32, device=self.device)
        
        phi_gold = (1.0 + math.sqrt(5.0)) / 2.0
        golden_angle = phi_gold * 2.0 * math.pi
        
        world_x = []
        world_y = []
        world_freqs = []
        world_chirality = []
        
        for i in range(self.num_world_nodes):
            r = math.sqrt(i) / math.sqrt(self.num_world_nodes) * 360.0
            theta = i * golden_angle
            
            x = WIDTH / 2.0 + math.cos(theta) * r
            y = HEIGHT / 2.0 + math.sin(theta) * r
            
            world_x.append(x)
            world_y.append(y)
            world_chirality.append(-1.0 if i % 2 == 0 else 1.0)
            
        self.world_centers = torch.tensor(list(zip(world_x, world_y)), dtype=torch.float32, device=self.device)
        self.chiralities[16:] = torch.tensor(world_chirality, dtype=torch.float32, device=self.device)
        
        # Определение Стен и Пола на базе синусоидальной волны
        scale_x, scale_y = 0.015, 0.015
        wave_val = torch.sin(self.world_centers[:, 0] * scale_x) * torch.cos(self.world_centers[:, 1] * scale_y)
        self.is_barrier = wave_val > 0.12 
        
        freqs = []
        for i in range(self.num_world_nodes):
            if self.is_barrier[i]:
                freqs.append(random.uniform(-4.0, -2.0) if i % 2 == 0 else random.uniform(2.0, 4.0))
            else:
                freqs.append(1.0 if i % 2 == 0 else -1.0)
                
        self.node_freqs = torch.tensor(freqs, dtype=torch.float32, device=self.device)
        self.node_captured = torch.zeros(self.num_world_nodes, dtype=torch.bool, device=self.device)
        self.chiralities[:16] = 1.0 
        
        # Сетка координат экрана (Tensor Cores)
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, HEIGHT, self.res, device=self.device), 
            torch.linspace(0, WIDTH, self.res, device=self.device),
            indexing='ij'
        )
        self.P_grid = torch.stack([x_grid.flatten(), y_grid.float().flatten()], dim=1) 
        self.P_sq = torch.sum(self.P_grid**2, dim=1, keepdim=True) 

    def reset_world(self):
        self.player_pos = torch.tensor([WIDTH / 2.0, HEIGHT - 100.0], dtype=torch.float32, device=self.device)
        self.player_vel.zero_()
        self.player_angle = 0.0
        self.node_captured.zero_()
        self.phases.zero_()

    def step(self, ciplv_tensor, compression, dt, time_sec, eeg_vx, eeg_vy, eeg_tq, eeg_phases):
        chaos = 1.0 - compression
        
        # Сглаживание ЭЭГ сигналов
        sm = self.cfg['eeg_smooth']
        self.smooth_vx = self.smooth_vx * sm + eeg_vx * (1.0 - sm)
        self.smooth_vy = self.smooth_vy * sm + eeg_vy * (1.0 - sm)
        self.smooth_tq = self.smooth_tq * sm + eeg_tq * (1.0 - sm)
        
        # Вращение аватара по сглаженному углу
        self.player_angle += self.smooth_tq * dt * self.cfg['torque_gain'] * 10.0
        self.player_angle = (self.player_angle + math.pi) % (2 * math.pi) - math.pi
        
        # === ИСПРАВЛЕНО (ЛОКАЛЬНОСТЬ АВАТАРА): Аватар больше НЕ вращается относительно экрана! ===
        # Его пины на экране всегда стоят жестко (Верх — это верх, Лево — лево)
        # Это обеспечивает прямую деконструкцию геймпада и идеальную читаемость направлений
        scale = 1.5 + chaos * 5.0
        self.positions[:16, 0] = WIDTH / 2.0 + self.pin_x * scale
        self.positions[:16, 1] = HEIGHT / 2.0 + self.pin_y * scale

        # Вращаем мир вокруг игрока
        cos_cam = math.cos(-self.player_angle)
        sin_cam = math.sin(-self.player_angle)
        
        dx_w = self.world_centers[:, 0] - self.player_pos[0]
        dy_w = self.world_centers[:, 1] - self.player_pos[1]
        
        rx_w = dx_w * cos_cam - dy_w * sin_cam
        ry_w = dx_w * sin_cam + dy_w * cos_cam
        
        self.positions[16:, 0] = WIDTH / 2.0 + rx_w
        self.positions[16:, 1] = HEIGHT / 2.0 + ry_w

        # Обновление зарядов
        if ciplv_tensor is not None:
            user_gammas = torch.sum(ciplv_tensor, dim=1) * 350.0 
        else:
            user_gammas = torch.ones(16, device=self.device) * 100.0
            
        self.gammas[:16] = user_gammas
        self.phases[:16] = eeg_phases
        
        world_base_phases = (torch.arange(self.num_world_nodes, device=self.device) * 0.1 + time_sec * self.node_freqs * (0.2 + 0.8 * compression)) % (2 * math.pi)
        captured_mask = self.node_captured
        
        self.phases[16:] = torch.where(captured_mask, self.phases[0], world_base_phases)
        self.gammas[16:] = torch.where(captured_mask, torch.tensor(350.0, device=self.device), torch.lerp(torch.tensor(100.0, device=self.device), torch.tensor(self.cfg['node_strength'], device=self.device), compression))

        # ====================================================================
        # ЭТАП 3: РАСЧЕТ ИНДУЦИРОВАННЫХ СКОРОСТЕЙ И ОТТАЛКИВАНИЙ НА GPU
        # ====================================================================
        coords = self.positions.unsqueeze(1) - self.positions.unsqueeze(0) 
        dx = coords[:, :, 0]
        dy = coords[:, :, 1]
        r2 = dx**2 + dy**2 + self.core_delta2
        
        ind_u = -self.gammas.unsqueeze(0) * dy / (2 * math.pi * r2)
        ind_v = self.gammas.unsqueeze(0) * dx / (2 * math.pi * r2)
        
        ind_u.fill_diagonal_(0.0)
        ind_v.fill_diagonal_(0.0)
        
        vortex_vel_u = torch.sum(ind_u, dim=1)
        vortex_vel_v = torch.sum(ind_v, dim=1)

        avg_flow_u = torch.mean(vortex_vel_u[:16]).item()
        avg_flow_v = torch.mean(vortex_vel_v[:16]).item()

        # Движение сонаправлено повороту аватара
        cos_a = math.cos(self.player_angle)
        sin_a = math.sin(self.player_angle)
        
        # Движение из эфира: затухание в пустоте
        dists = torch.sqrt((self.player_pos[0] - self.world_centers[:, 0])**2 + (self.player_pos[1] - self.world_centers[:, 1])**2)
        local_vortex_density = torch.clamp(torch.sum(torch.exp(-dists / 100.0)) / 5.0, 0.0, 1.0).item()
        current_damping = self.cfg['damping'] * local_vortex_density

        intent_u = (cos_a * self.smooth_vx - sin_a * self.smooth_vy) * self.cfg['intent_gain'] * local_vortex_density
        intent_v = (sin_a * self.smooth_vx + cos_a * self.smooth_vy) * self.cfg['intent_gain'] * local_vortex_density
        
        self.player_vel[0] += (intent_u + avg_flow_u * local_vortex_density) * dt
        self.player_vel[1] += (intent_v + avg_flow_v * local_vortex_density) * dt
        self.player_vel *= current_damping
        
        v_mag = torch.sqrt(self.player_vel[0]**2 + self.player_vel[1]**2)
        if v_mag > self.cfg['max_vel']:
            self.player_vel = (self.player_vel / v_mag) * self.cfg['max_vel']
            
        self.player_pos += self.player_vel * dt

        # === ИСПРАВЛЕНО (ЛОКАЛЬНЫЙ ФАЗОВЫЙ ЗАХВАТ ИЛИ ОТТАЛКИВАНИЕ): ===
        # Мы проверяем близость каждого ИНДИВИДУАЛЬНОГО ПИНА аватара к вихрям среды на экране!
        # Считаем попарные расстояния между твоими 16 пинами и 400 вихрями среды в экранных координатах
        dx_inter = self.positions[:16, 0].unsqueeze(1) - self.positions[16:, 0].unsqueeze(0) # (16, 400)
        dy_inter = self.positions[:16, 1].unsqueeze(1) - self.positions[16:, 1].unsqueeze(0)
        dists_inter = torch.sqrt(dx_inter**2 + dy_inter**2 + 1e-5)
        
        # Область локального влияния пина ЭЭГ на вихрь среды (50 пикселей)
        close_mask = (dists_inter < 50.0)
        
        # Разность фаз каждого пина с каждым близким вихрем
        p_diff = self.phases[:16].unsqueeze(1) - self.phases[16:].unsqueeze(0) # (16, 400)
        coherence = torch.cos(p_diff) # Косинусная близость (-1.0 ... 1.0)
        
        # Если фаза локального пина совпадает с фазой вихря среды — вихрь захвачен (резонанс)
        aligned_mask = close_mask & (coherence > 0.70)
        captured_this_frame = torch.any(aligned_mask, dim=0) # (400)
        self.node_captured = self.node_captured | captured_this_frame
        
        # Если вихрь близко к твоему пину, но НЕ синхронизирован — он упруго отталкивает этот пин,
        # создавая разворачивающий и выталкивающий момент сил для всего аватара!
        repel_mask = close_mask & (~self.node_captured.unsqueeze(0)) & (coherence < 0.40)
        
        if torch.any(repel_mask):
            repel_power = (1.0 - coherence) * 2000.0 * (0.5 + compression)
            # Направление толчка: от вихря среды j к пину аватара i (dx_inter/dy_inter направлены от i к j, поэтому берем минус)
            push_u = -dx_inter / dists_inter * repel_power * repel_mask.float()
            push_v = -dy_inter / dists_inter * repel_power * repel_mask.float()
            
            self.player_vel[0] += torch.sum(push_u) * dt
            self.player_vel[1] += torch.sum(push_v) * dt

        # Ограничение мира
        self.player_pos[0] = torch.clamp(self.player_pos[0], 100, WIDTH*2)
        self.player_pos[1] = torch.clamp(self.player_pos[1], 100, HEIGHT*2)
        
        if torch.sum(self.node_captured).item() > (self.num_world_nodes * 0.7):
            self.reset_world()

        return scale

    def render_field(self, compression, time_sec):
        """ПОЛНЫЙ ОПТИМИЗИРОВАННЫЙ РЕНДЕР ЧЕРЕЗ СВЕРХБЫСТРЫЙ MATMUL"""
        rgb = torch.zeros((self.res, self.res, 3), dtype=torch.uint8, device=self.device)
        
        pos_x = self.positions[:, 0].view(-1, 1, 1)
        pos_y = self.positions[:, 1].view(-1, 1, 1)
        phases = self.phases.view(1, -1)
        chiralities = self.chiralities.view(1, -1)
        
        radii = torch.ones(self.total_vortices, device=self.device) * 24.0
        radii[:16] = 30.0 + (1.0 - compression) * 15.0
        # Выделяем Опорный Пин 0 (Нос аватара) — он всегда горит золотом на севере платы
        radii[0] = 55.0 
        radii_g = radii.view(1, -1)
        
        # Fast GEMM на тензорных ядрах
        V_sq = torch.sum(self.positions**2, dim=1, keepdim=True).T
        D2 = self.P_sq + V_sq - 2.0 * torch.matmul(self.P_grid, self.positions.T)
        dist = torch.sqrt(torch.clamp(D2, 1e-5))
        
        w = 1.0 / (D2 + 0.1)
        
        # Чтение координат из P_grid
        dy = self.P_grid[:, 1].unsqueeze(1) - self.positions[:, 1].unsqueeze(0)
        dx = self.P_grid[:, 0].unsqueeze(1) - self.positions[:, 0].unsqueeze(0)
        angle = torch.atan2(dy, dx)
        
        vortex_phases = chiralities * angle + phases
        
        sum_x = torch.sum(torch.cos(vortex_phases) * w * torch.exp(-dist / radii_g), dim=1)
        sum_y = torch.sum(torch.sin(vortex_phases) * w * torch.exp(-dist / radii_g), dim=1)
        sum_weight = torch.sum(w, dim=1)
        
        avg_x = sum_x / sum_weight
        avg_y = sum_y / sum_weight
        
        final_phase = torch.atan2(avg_y, avg_x)
        phase_norm = (final_phase + math.pi) / (2.0 * math.pi)
        magnitude = torch.sqrt(avg_x**2 + avg_y**2)
        
        # СЛОИ ПРОЗРАЧНОСТИ (ПОЛУПРОЗРАЧНЫЙ ПОЛ)
        cx = phase_norm
        cy = 1.0 
        cz = torch.clamp(magnitude * 0.12, 0.0, 1.0) # Прозрачный пол
        
        t_r = torch.remainder(cx * 6.0 + 0.0, 6.0)
        t_g = torch.remainder(cx * 6.0 + 4.0, 6.0)
        t_b = torch.remainder(cx * 6.0 + 2.0, 6.0)
        
        r = torch.clamp(torch.abs(t_r - 3.0) - 1.0, 0.0, 1.0)
        g = torch.clamp(torch.abs(t_g - 3.0) - 1.0, 0.0, 1.0)
        b = torch.clamp(torch.abs(t_b - 3.0) - 1.0, 0.0, 1.0)
        
        factor = 1.0 - torch.abs(2.0 * cz - 1.0)
        r_final = cz + cy * (r - 0.5) * factor
        g_final = cz + cy * (g - 0.5) * factor
        b_final = cz + cy * (b - 0.5) * factor
        
        # Бегущие волны фазы пола
        contour_phase = phase_norm * 8.0 - time_sec * 4.0
        contour = contour_phase - torch.floor(contour_phase)
        t_a = torch.clamp((contour - 0.85) / 0.15, 0.0, 1.0)
        t_b = torch.clamp((0.15 - contour) / 0.15, 0.0, 1.0)
        contour_line = t_a * t_a * (3.0 - 2.0 * t_a) + t_b * t_b * (3.0 - 2.0 * t_b)
        
        r_final += contour_line * 0.15 * magnitude
        g_final += contour_line * 0.15 * magnitude
        b_final += contour_line * 0.15 * magnitude
        
        # Свечение сингулярностей
        t_s = torch.clamp((0.18 - magnitude) / 0.18, 0.0, 1.0)
        singularity = t_s * t_s * (3.0 - 2.0 * t_s)
        
        vortex_energy_mask = torch.clamp(sum_weight / 5.0, 0.0, 1.0)
        r_final += singularity * 0.8 * vortex_energy_mask

        # === СОЛИД СЛОЙ СТЕН И АВАТАРА ===
        wall_dist = dist[:, 16:]
        min_wall_dist = torch.min(wall_dist, dim=1)[0]
        wall_core_mask = torch.clamp(1.0 - min_wall_dist / (CELL_SIZE * 0.45), 0.0, 1.0)
        
        r_final = torch.max(r_final, wall_core_mask * 0.70)
        b_final = torch.max(b_final, wall_core_mask * 0.95)
        g_final = torch.max(g_final, wall_core_mask * compression * 0.3)
        
        player_dist = dist[:, :16]
        min_player_dist = torch.min(player_dist, dim=1)[0]
        player_core_mask = torch.clamp(1.0 - min_player_dist / 16.0, 0.0, 1.0)
        
        g_final = torch.max(g_final, player_core_mask * 0.80)
        b_final = torch.max(b_final, player_core_mask * 0.95)

        ref_dist = dist[:, 0]
        ref_core_mask = torch.clamp(1.0 - ref_dist / 22.0, 0.0, 1.0)
        r_final = torch.max(r_final, ref_core_mask * 1.0)
        g_final = torch.max(g_final, ref_core_mask * 0.75)
        
        # Маска захваченных золотых вихрей
        captured_field = torch.zeros(self.res * self.res, device=self.device)
        for i in range(self.num_world_nodes):
            if self.node_captured[i]:
                dx_c = self.P_grid[:, 0] - self.positions[16+i, 0]
                dy_c = self.P_grid[:, 1] - self.positions[16+i, 1]
                dist_c = torch.sqrt(dx_c**2 + dy_c**2)
                mask_c = torch.clamp(1.0 - dist_c / 20.0, 0.0, 1.0)
                captured_field = torch.max(captured_field, mask_c)
                
        r_final += captured_field * 0.6
        g_final += captured_field * 0.5
        
        r_img = (torch.clamp(r_final, 0.0, 1.0) * 255).to(torch.uint8).view(self.res, self.res)
        g_img = (torch.clamp(g_final, 0.0, 1.0) * 255).to(torch.uint8).view(self.res, self.res)
        b_img = (torch.clamp(b_final, 0.0, 1.0) * 255).to(torch.uint8).view(self.res, self.res)
        
        rgb = torch.stack([r_img, g_img, b_img], dim=-1)
        
        surf = pygame.surfarray.make_surface(np.transpose(rgb.cpu().numpy(), (1, 0, 2)))
        surf = pygame.transform.smoothscale(surf, (WIDTH, HEIGHT))
        return surf

def main():
    pygame.init()
    pygame.joystick.init()
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Spontaneous Phase Vortex Labyrinth")
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
    ui_compression = 1.0

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        time_sec = pygame.time.get_ticks() / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1: arena.cfg['intent_gain'] = max(100, arena.cfg['intent_gain'] - 100)
                if event.key == pygame.K_2: arena.cfg['intent_gain'] += 100
                if event.key == pygame.K_3: arena.cfg['torque_gain'] = max(0.01, arena.cfg['torque_gain'] - 0.02)
                if event.key == pygame.K_4: arena.cfg['torque_gain'] += 0.02
                if event.key == pygame.K_5: arena.cfg['eeg_smooth'] = max(0.80, arena.cfg['eeg_smooth'] - 0.01)
                if event.key == pygame.K_6: arena.cfg['eeg_smooth'] = min(0.99, arena.cfg['eeg_smooth'] + 0.01)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            ui_compression = min(1.0, ui_compression + dt * 2.0)
        else:
            ui_compression = max(0.0, ui_compression - dt * 2.0)

        if len(joysticks) > 0 and joysticks[0].get_numaxes() >= 5:
            ui_compression = (joysticks[0].get_axis(4) + 1.0) / 2.0

        compression = ui_compression

        eeg_vx, eeg_vy, eeg_tq = 0.0, 0.0, 0.0
        eeg_phases = None
        
        if HAS_NEURO:
            active_slots = [i for i in range(5) if driver.workers[i].is_connected or any(v == i for v in driver.lsl_inlets.values())]
            is_real = len(active_slots) > 0
            C = len(active_slots) * 16 if is_real else 16

            if is_real:
                for slot_idx in active_slots:
                    q = driver.queues[slot_idx]
                    while len(q) > 0:
                        sample = q.popleft()
                        start, end = slot_idx * 16, (slot_idx + 1) * 16
                        neuro_engine.pinned_cpu_buffer[start:end, :-1] = neuro_engine.pinned_cpu_buffer[start:end, 1:].clone()
                        neuro_engine.pinned_cpu_buffer[start:end, -1] = torch.tensor(sample)

            _, _, phases_gpu, vx, vy, tq, _, _ = neuro_engine.get_predictive_ciplv(C)
            eeg_phases = phases_gpu[:16]
            eeg_vx = vx.item()
            eeg_vy = -vy.item() 
            eeg_tq = tq.item()
        else:
            if keys[pygame.K_LEFT] or keys[pygame.K_a]: eeg_vx -= 1.0
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]: eeg_vx += 1.0
            if keys[pygame.K_UP] or keys[pygame.K_w]: eeg_vy -= 1.0
            if keys[pygame.K_DOWN] or keys[pygame.K_s]: eeg_vy += 1.0
            if keys[pygame.K_q]: eeg_tq -= 1.0
            if keys[pygame.K_e]: eeg_tq += 1.0
            
            eeg_phases = (torch.arange(16, device=device) * 0.4 + time_sec * 6.0) % (2 * math.pi)

        # Вычисляем относительную физику по направлению взгляда
        scale = arena.step(None, compression, dt, time_sec, eeg_vx, eeg_vy, eeg_tq, eeg_phases)
        
        # === ГЕНЕРАЦИЯ ПОЛНОГО ПОЛЯ НА GPU ===
        bg_surface = arena.render_field(compression, time_sec)
        
        # Вывод на экран (НИ ОДНОГО КРУЖОЧКА, НИ ОДНОЙ ЛИНИИ)
        screen.blit(bg_surface, (0, 0))

        # Отрисовка пульсирующей ауры вокруг твоего аватара в центре экрана
        pygame.draw.circle(screen, (0, 255, 255), (WIDTH // 2, HEIGHT // 2), 80, 1)

        # === ПОЛУПРОЗРАЧНЫЙ СТЕКЛЯННЫЙ UI ===
        ui_surf = pygame.Surface((450, 95), pygame.SRCALPHA)
        ui_surf.fill((10, 15, 30, 140)) 
        pygame.draw.rect(ui_surf, (0, 255, 255, 60), (0, 0, 450, 95), 1) 
        screen.blit(ui_surf, (10, 10))
        
        captured_count = torch.sum(arena.node_captured).item()
        
        screen.blit(font.render(f"COMPRESSION (Левый Триггер): {(compression*100):.0f}%", True, (255, 255, 255)), (20, 20))
        screen.blit(font.render(f"ЗАХВАЧЕНО ВИХРЕЙ ИЗ 1000: {captured_count:.0f} / 1000 (Цель: 700)", True, (255, 200, 0)), (20, 40))
        screen.blit(font.render(f"[1 / 2] Чувствительность ЭЭГ: {arena.cfg['intent_gain']:.0f}", True, (0, 255, 200)), (20, 60))
        screen.blit(font.render(f"ЭЭГ Сигнал: {'АКТИВЕН (BLE/LSL)' if HAS_NEURO else 'ЭМУЛЯТОР (WASD + Q/E)'}", True, (0, 255, 0) if HAS_NEURO else (255, 100, 100)), (20, 75))

        pygame.display.flip()

    if HAS_NEURO:
        driver.scanner_running = False
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
