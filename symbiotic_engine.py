# symbiotic_engine.py
import math
import torch
import numpy as np
from implicit_config import COORDS_16_X, COORDS_16_Y, REF_16_X, REF_16_Y


def sign(val):
    """Вспомогательная функция знака числа"""
    return 1.0 if val > 0 else (-1.0 if val < 0 else 0.0)


class SymbioticEngineGPU:
    def __init__(self, device_name='cuda', sample_rate_eeg=250, sample_rate_audio=48000):
        self.device = torch.device(device_name) if device_name == 'cuda' and torch.cuda.is_available() else torch.device('cpu')
        self.fs_eeg, self.fs_audio, self.lag_compensation_sec = sample_rate_eeg, sample_rate_audio, 0.020
        base_f = 110.0
        freqs = [base_f * (2.0 ** ((i % 16) / 16.0)) * (1.0 + (i // 16) * 0.5) for i in range(80)]
        self.all_audio_freqs = torch.tensor(freqs, device=self.device).unsqueeze(1)
        self.all_audio_phase_accumulators = torch.zeros((80, 1), device=self.device)
        self.pinned_cpu_buffer = torch.zeros((80, 500), dtype=torch.float32).pin_memory()
        self.gpu_eeg_tensor = torch.zeros((80, 500), dtype=torch.float32, device=self.device)
        
        cx, cy = torch.tensor(COORDS_16_X, dtype=torch.float32, device=self.device), torch.tensor(COORDS_16_Y, dtype=torch.float32, device=self.device)
        dist_ref = torch.sqrt((cx - REF_16_X)**2 + (cy - REF_16_Y)**2)
        self.ref_debias_weight = 1.0 + (dist_ref.unsqueeze(1) + dist_ref.unsqueeze(0)) / 40.0
        
        coords_x, coords_y = torch.cat([cx for _ in range(5)]), torch.cat([cy for _ in range(5)])
        self.dX_raw = coords_x.unsqueeze(0) - coords_x.unsqueeze(1)
        self.dY_raw = coords_y.unsqueeze(0) - coords_y.unsqueeze(1)
        self.dTQ_raw = coords_x.unsqueeze(0) * coords_y.unsqueeze(1) - coords_y.unsqueeze(0) * coords_x.unsqueeze(1)
        self.dTQ_raw /= (torch.max(torch.abs(self.dTQ_raw)) + 1e-8)
        
        debias = torch.cat([self.ref_debias_weight for _ in range(5)], dim=0)
        self.debias_matrix = torch.cat([debias for _ in range(5)], dim=1)
        self.dX, self.dY, self.dTQ = self.dX_raw * self.debias_matrix, self.dY_raw * self.debias_matrix, self.dTQ_raw * self.debias_matrix

        # Инициализируем переменные физического состояния игрока прямо на GPU
        self.gpu_player_pos = torch.tensor([1.5, 1.5, 0.0], dtype=torch.float32, device=self.device)
        self.ctrl_move_x = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.ctrl_move_y = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.ctrl_torque = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.ctrl_drill = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.ctrl_oracle = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.player_speed = 3.5
        self.maze_grid_gpu = None

    def set_maze(self, maze):
        self.maze = maze
        self.maze_grid_gpu = torch.tensor(maze.grid, dtype=torch.int32, device=self.device)

    def get_predictive_ciplv(self, C):
        self.gpu_eeg_tensor[:C, :].copy_(self.pinned_cpu_buffer[:C, :], non_blocking=True)
        active_eeg_gpu = self.gpu_eeg_tensor[:C, :]
        T = active_eeg_gpu.shape[1]
        Xf = torch.fft.fft(active_eeg_gpu, dim=1)
        freqs = torch.fft.fftfreq(T, d=1.0 / self.fs_eeg, device=self.device)
        notch_mask = torch.ones(T, device=self.device)
        notch_mask[(torch.abs(freqs) >= 49.0) & (torch.abs(freqs) <= 51.0)] = 0.0
        notch_mask[(torch.abs(freqs) >= 99.0) & (torch.abs(freqs) <= 101.0)] = 0.0
        Xf *= notch_mask.unsqueeze(0)
        
        power_spec = torch.abs(Xf)**2
        log_freqs, log_power = torch.log(torch.abs(freqs[4:90])), torch.log(power_spec[:, 4:90] + 1e-8)
        mean_x, mean_y = torch.mean(log_freqs), torch.mean(log_power, dim=1, keepdim=True)
        num = torch.sum((log_freqs - mean_x) * (log_power - mean_y), dim=1)
        den = torch.sum((log_freqs - mean_x)**2)
        mean_beta_gpu = torch.mean(-(num / (den + 1e-8)))
        
        h = torch.zeros(T, device=self.device); h[0] = 1; h[1:T//2] = 2
        phases = torch.angle(torch.fft.ifft(Xf * h.unsqueeze(0), dim=1))
        inst_freq = phases[:, -1] - phases[:, -2]
        future_phases = phases[:, -1] + inst_freq * int(self.lag_compensation_sec * self.fs_eeg)
        
        phase_diff = torch.exp(-1j * future_phases).unsqueeze(1) @ torch.exp(-1j * future_phases).unsqueeze(1).conj().T
        real, imag = torch.real(phase_diff), torch.imag(phase_diff)
        ciplv_dynamic = (imag * (torch.abs(real) < 0.96).float()) / torch.sqrt(1.0 - (torch.clamp(real, -0.95, 0.95)**2))
        ciplv_dynamic.fill_diagonal_(0.0)
        
        c0 = ciplv_dynamic[0:16, 0:16] if C >= 16 else ciplv_dynamic
        c0_size = c0.shape[0]
        vx = torch.sum(c0 * self.dX[0:c0_size, 0:c0_size]) * 15.0
        vy = torch.sum(c0 * self.dY[0:c0_size, 0:c0_size]) * 15.0
        torque = torch.sum(c0 * self.dTQ[0:c0_size, 0:c0_size]) * 3.0
        
        drill_axis = torch.sum(ciplv_dynamic[16:32, 16:32] * self.dY[0:16, 0:16]) * 15.0 if C >= 32 else torch.tensor(0.0, device=self.device)
        oracle_axis = torch.sum(ciplv_dynamic[32:48, 32:48] * self.dX[0:16, 0:16]) * 15.0 if C >= 48 else torch.tensor(0.0, device=self.device)
        return ciplv_dynamic, mean_beta_gpu, phases[:, -1].clone(), vx, vy, torque, drill_axis, oracle_axis

    def check_collision_axis_gpu(self, tx, ty):
        gx, gy = tx.int(), ty.int()
        in_bounds = (gx >= 0) & (gx < self.maze_dim) & (gy >= 0) & (gy < self.maze_dim)
        is_wall = torch.tensor(False, device=self.device)
        if in_bounds:
            is_wall = self.maze_grid_gpu[gy, gx] == 1
        return in_bounds & (~is_wall)

    def move_player_3d_paradigm_gpu(self, move_x, move_y, torque, dt):
        # Поворот и тригонометрия на GPU
        self.gpu_player_pos[2] += torque * dt * 2.0
        self.gpu_player_pos[2] = (self.gpu_player_pos[2] + math.pi) % (2.0 * math.pi) - math.pi
        
        forward_speed = -move_y * self.player_speed * 0.2
        strafe_speed = move_x * self.player_speed * 0.2
        
        sin_a = torch.sin(self.gpu_player_pos[2])
        cos_a = torch.cos(self.gpu_player_pos[2])
        
        dx = sin_a * forward_speed + cos_a * strafe_speed
        dy = -cos_a * forward_speed + sin_a * strafe_speed
        
        target_dx = dx * dt
        target_dy = dy * dt
        
        # Фиксированные шаги для субстеппинга на GPU во избежание вызовов .item()
        sdx = target_dx / 4.0
        sdy = target_dy / 4.0
        
        for _ in range(4):
            next_x = self.gpu_player_pos[0] + sdx
            sign_x = torch.sign(sdx)
            check_x = next_x + sign_x * 0.2
            
            can_move_x = self.check_collision_axis_gpu(check_x, self.gpu_player_pos[1])
            self.gpu_player_pos[0] = torch.where(can_move_x, next_x, self.gpu_player_pos[0])
            
            next_y = self.gpu_player_pos[1] + sdy
            sign_y = torch.sign(sdy)
            check_y = next_y + sign_y * 0.2
            
            can_move_y = self.check_collision_axis_gpu(self.gpu_player_pos[0], check_y)
            self.gpu_player_pos[1] = torch.where(can_move_y, next_y, self.gpu_player_pos[1])

    def synthesize_crossmodulated_audio(self, ciplv_dynamic, C, frames):
        if not hasattr(self, '_audio_t_cache') or self._audio_t_cache.shape[1] != frames:
            self._audio_t_cache = torch.arange(frames, device=self.device).unsqueeze(0) / self.fs_audio
        freqs, accum = self.all_audio_freqs[:C, :], self.all_audio_phase_accumulators[:C, :]
        final_waves = (torch.mean(torch.abs(ciplv_dynamic), dim=1, keepdim=True) + 0.1) * torch.sin(2 * math.pi * freqs * self._audio_t_cache + accum + torch.matmul(ciplv_dynamic, torch.sin(2 * math.pi * freqs * self._audio_t_cache + accum)) * 2.5)
        self.all_audio_phase_accumulators[:C, :] = (accum + 2 * math.pi * freqs * (frames / self.fs_audio)) % (2 * math.pi)
        mid = max(1, C // 2)
        return torch.clamp(torch.stack((torch.sum(final_waves[:mid, :], dim=0) / float(mid), torch.sum(final_waves[mid:, :], dim=0) / float(C - mid + 1e-5)), dim=1), -1.0, 1.0).cpu().numpy()
