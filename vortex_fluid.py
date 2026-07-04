# vortex_fluid.py
import torch
import torch.nn.functional as F

class FluidSolver:
    def __init__(self, res, device):
        self.res = res
        self.device = device
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1.0, 1.0, res, device=device),
            torch.linspace(-1.0, 1.0, res, device=device),
            indexing='ij'
        )
        self.grid_x = x_grid
        self.grid_y = y_grid

    def advect(self, field, u, v, dt):
        dx = u[0, 0] * (dt * 2.0 / self.res)
        dy = v[0, 0] * (dt * 2.0 / self.res)
        sampling_grid = torch.stack([self.grid_x - dx, self.grid_y - dy], dim=-1).unsqueeze(0)
        
        # ТОПОЛОГИЯ ТОРУСА: Бесконечное заворачивание координат сетки
        sampling_grid = torch.remainder(sampling_grid + 1.0, 2.0) - 1.0
        
        return F.grid_sample(field, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)

    def project(self, u, v, wall_density, target_div=None):
        # ФИЗИКА: Снизили порог блокировки до 0.05. Теперь даже тонкая стена полностью останавливает жидкость.
        block_mask = (wall_density > 0.05).float()
        u = u * (1.0 - block_mask)
        v = v * (1.0 - block_mask)
        
        # МЕНЯЕМ REPLICATE НА CIRCULAR: Давление проходит сквозь края вселенной
        u_pad = F.pad(u, (1, 1, 0, 0), mode='circular')
        v_pad = F.pad(v, (0, 0, 1, 1), mode='circular')
        div = 0.5 * (u_pad[:, :, :, 2:] - u_pad[:, :, :, :-2] + v_pad[:, :, 2:, :] - v_pad[:, :, :-2, :])
        
        if target_div is not None:
            # Внедряем источниковый/стоковый член в уравнение дивергенции для создания течений сжатия и расширения
            div = div - target_div
            
        p = torch.zeros_like(u)
        for _ in range(40):
            p_pad = F.pad(p, (1, 1, 1, 1), mode='circular')
            p = 0.25 * (p_pad[:, :, 1:-1, 2:] + p_pad[:, :, 1:-1, :-2] + 
                        p_pad[:, :, 2:, 1:-1] + p_pad[:, :, :-2, 1:-1] - div)
            
        p_pad = F.pad(p, (1, 1, 1, 1), mode='circular')
        u -= 0.5 * (p_pad[:, :, 1:-1, 2:] - p_pad[:, :, 1:-1, :-2])
        v -= 0.5 * (p_pad[:, :, 2:, 1:-1] - p_pad[:, :, :-2, 1:-1])
        
        u = u * (1.0 - block_mask)
        v = v * (1.0 - block_mask)
        return u, v
