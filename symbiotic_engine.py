# symbiotic_engine.py
import math
import torch
import numpy as np
from implicit_config import COORDS_16_X, COORDS_16_Y, REF_16_X, REF_16_Y


def sign(val):
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

        self.gpu_player_pos = torch.tensor([1.5, 1.5, 0.0], dtype=torch.float32, device=self.device)
        self.player_speed = 3.5
        self.maze_grid_gpu = None
        
        # Окно Хенна для STFT, убирающее спектральную утечку частот
        self.stft_window = torch.hann_window(128, device=self.device)

    def set_maze(self, maze):
        self.maze = maze
        self.maze_grid_gpu = torch.tensor(maze.grid, dtype=torch.int32, device=self.device)

    def get_predictive_ciplv(self, C):
        self.gpu_eeg_tensor[:C, :].copy_(self.pinned_cpu_buffer[:C, :], non_blocking=True)
        active_eeg_gpu = self.gpu_eeg_tensor[:C, :]
        
        # 1. STFT с окном Хенна для чистейшего спектрометра
        X = torch.stft(active_eeg_gpu, n_fft=128, hop_length=16, window=self.stft_window, return_complex=True) 
        stft_freqs = torch.linspace(0, self.fs_eeg / 2.0, X.shape[1], device=self.device)
        
        # 2. Выбираем ВЕСЬ спектр от 3 Гц до 100 Гц, жестко вырезая сетевой шум 50 Гц (48-52)
        valid_mask = (stft_freqs >= 3.0) & (stft_freqs <= 100.0) & ~((stft_freqs >= 48.0) & (stft_freqs <= 52.0))
        X_valid = X[:, valid_mask, :] # [C, ~48 частот, Time]
        freqs = stft_freqs[valid_mask] # [~48 частот]
        
        num_frames = X_valid.shape[2]
        
        # 3. ciPLV по матричному алгоритму
        Z = X_valid / (torch.abs(X_valid) + 1e-8) # [C, F, T]
        Z = Z.permute(1, 0, 2) # [F, C, T] 
        
        PLV = torch.bmm(Z, Z.conj().transpose(1, 2)) / num_frames # [F, C, C]
        
        Real_PLV = torch.real(PLV)
        Imag_PLV = torch.imag(PLV)
        ciPLV = (Imag_PLV * (torch.abs(Real_PLV) < 0.99).float()) / torch.sqrt(1.0 - torch.clamp(Real_PLV, -0.98, 0.98)**2)
        
        ciPLV = ciPLV.permute(1, 2, 0) # [C, C, F]
        
        # 4. Взвешиваем ciPLV на мощность
        power = torch.mean(torch.abs(X_valid), dim=2) # [C, F]
        pair_power = torch.sqrt(power.unsqueeze(1) * power.unsqueeze(0)) # [C, C, F]
        eeg_c0_spectrum = ciPLV * pair_power * 2.0
        
        c0_global = torch.sum(eeg_c0_spectrum, dim=2)
        c0_size = min(C, 16)
        c0_sub = c0_global[:c0_size, :c0_size]
        vx = torch.sum(c0_sub * self.dX[0:c0_size, 0:c0_size]) * 15.0
        vy = torch.sum(c0_sub * self.dY[0:c0_size, 0:c0_size]) * 15.0
        torque = torch.sum(c0_sub * self.dTQ[0:c0_size, 0:c0_size]) * 3.0
        
        return eeg_c0_spectrum, freqs, vx, vy, torque

    def check_collision_axis_gpu(self, tx, ty):
        gx, gy = tx.int(), ty.int()
        in_bounds = (gx >= 0) & (gx < self.maze_dim) & (gy >= 0) & (gy < self.maze_dim)
        is_wall = torch.tensor(False, device=self.device)
        if in_bounds:
            is_wall = self.maze_grid_gpu[gy, gx] == 1
        return in_bounds & (~is_wall)
