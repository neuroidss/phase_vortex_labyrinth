# ginzburg_landau_fluid_solver.py
import torch
import torch.nn.functional as F

class GinzburgLandauFluidSolver:
    """
    Quantum-inspired Hydrodynamic Wave Solver.
    Integrates Ginzburg-Landau complex-valued reaction-diffusion systems
    directly with non-linear viscosity Navier-Stokes projection.
    """
    def __init__(self, res, device):
        self.res = res
        self.device = device
        
        # Spatial coordinate maps for Laplacian computations
        dy, dx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, res, device=device),
            torch.linspace(-1.0, 1.0, res, device=device), indexing='ij'
        )
        self.dx = dx
        self.dy = dy

    def compute_laplacian(self, field):
        """ Fully replicate padded symmetric central difference Laplacian to prevent torus artifacts """
        field_pad = F.pad(field, (1, 1, 1, 1), mode='replicate')
        laplacian = (field_pad[:, :, 1:-1, 2:] + field_pad[:, :, 1:-1, :-2] +
                     field_pad[:, :, 2:, 1:-1] + field_pad[:, :, :-2, 1:-1] - 4.0 * field)
        return laplacian

    def step_non_linear_matter(self, density_complex, u, v, viscosity, surface_tension, buoyancy, dt):
        """
        Updates the physical state of the alchemical fields by evaluating
        local shear stress, surface tension pressures, and convective buoyancy.
        """
        # Complex diffusion step (Ginzburg-Landau Dispersion)
        # R, G, B are stored as pairs of Real/Imaginary components
        for c in range(3):
            real_part = density_complex[:, c*2 : c*2+1]
            imag_part = density_complex[:, c*2+1 : c*2+2]
            
            lap_real = self.compute_laplacian(real_part)
            lap_imag = self.compute_laplacian(imag_part)
            
            # Dispersion coupling factor: Yin (Water) disperses slower, Yang (Fire) propagates rapidly
            dispersion_factor = 0.08 if c == 2 else (0.25 if c == 0 else 0.15)
            
            # Complex Ginzburg-Landau update
            density_complex[:, c*2 : c*2+1] += (0.1 * lap_real - dispersion_factor * lap_imag) * dt
            density_complex[:, c*2+1 : c*2+2] += (0.1 * lap_imag + dispersion_factor * lap_real) * dt

        # Apply Surface Tension force (Laplace pressure gradient)
        # Force points towards the gradient of the surface tension field
        st_pad = F.pad(surface_tension, (1, 1, 1, 1), mode='replicate')
        grad_st_x = 0.5 * (st_pad[:, :, 1:-1, 2:] - st_pad[:, :, 1:-1, :-2])
        grad_st_y = 0.5 * (st_pad[:, :, 2:, 1:-1] - st_pad[:, :, :-2, 1:-1])
        
        # Apply Viscous Shear Stress damping on Navier-Stokes velocity
        u_lap = self.compute_laplacian(u)
        v_lap = self.compute_laplacian(v)
        
        # Velocity update incorporating dynamic viscosity, buoyancy, and Laplace pressure
        u += (viscosity * u_lap + grad_st_x + buoyancy[:, 0:1]) * dt
        v += (viscosity * v_lap + grad_st_y + buoyancy[:, 1:2]) * dt
        
        density_complex = torch.clamp(density_complex, -2.5, 2.5)
        u = torch.clamp(u, -65.0, 65.0)
        v = torch.clamp(v, -65.0, 65.0)
        
        return density_complex, u, v
