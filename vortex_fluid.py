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
        
        # TORUS TOPOLOGY: wrapping coordinates infinitely
        sampling_grid = torch.remainder(sampling_grid + 1.0, 2.0) - 1.0
        
        return F.grid_sample(field, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)

    def project(self, u, v, wall_density, target_div=None):
        # PHYSICS: Partial-slip boundaries inside obstacles
        block_mask = (wall_density > 0.4).float()
        u = u * (1.0 - block_mask)
        v = v * (1.0 - block_mask)
        
        # CRITICAL FIX: Symmetrically pad both velocity components circularly in both dimensions.
        # This completely eliminates asymmetric boundary gradients, resolving coordinate-axis funnelling.
        u_pad = F.pad(u, (1, 1, 1, 1), mode='circular')
        v_pad = F.pad(v, (1, 1, 1, 1), mode='circular')
        
        # Symmetric central differences using fully padded tensors
        u_diff = u_pad[:, :, 1:-1, 2:] - u_pad[:, :, 1:-1, :-2]
        v_diff = v_pad[:, :, 2:, 1:-1] - v_pad[:, :, :-2, 1:-1]
        div = 0.5 * (u_diff + v_diff)
        
        if target_div is not None:
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
