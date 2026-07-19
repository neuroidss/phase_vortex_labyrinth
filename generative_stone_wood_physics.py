# generative_stone_wood_physics.py
import torch
import torch.nn.functional as F
import math

class GenerativePhysicsEngine:
    """
    Implements a zero-dimensionality-reduction physical simulation on the GPU.
    Integrates dynamic Earth/Stone phase fields, autopoietic Wood growth/combustion,
    and a zero-latency Pz-Beta (18-36Hz) sensorimotor phase gamepad mapping.
    """
    def __init__(self, res, device):
        self.res = res
        self.device = device
        
        # Spatial meshgrids for gradients
        dy, dx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, res, device=device),
            torch.linspace(-1.0, 1.0, res, device=device), indexing='ij'
        )
        self.grid_x = dx
        self.grid_y = dy
        
        # Continuous Earth/Stone Phase Field
        self.rho_stone = torch.zeros((1, 1, res, res), device=device)
        # Viscoelastic Autopoietic Wood Field
        self.rho_wood = torch.zeros((1, 1, res, res), device=device)

    def compute_laplacian(self, field):
        """ Computes a symmetric 2D Laplacian using replicate padding to prevent edge leakage """
        field_pad = F.pad(field, (1, 1, 1, 1), mode='replicate')
        laplacian = (field_pad[:, :, 1:-1, 2:] + field_pad[:, :, 1:-1, :-2] +
                     field_pad[:, :, 2:, 1:-1] + field_pad[:, :, :-2, 1:-1] - 4.0 * field)
        return laplacian

    def update_stone_field(self, eeg_c0_spectrum, eeg_freqs, dt):
        """
        Processes Alpha/Low-Beta (8-12Hz) top-down coherence to crystallize/melt 
        the Earth/Stone walls of the labyrinth. No hardcoded logic.
        """
        if eeg_c0_spectrum is None or eeg_freqs is None:
            return
            
        # Extract 8-12Hz Alpha grounding energy (deep cortical layers indicator)
        w_alpha = torch.exp(-((eeg_freqs - 10.0) ** 2) / 4.0).view(1, 1, -1)
        alpha_coherence = torch.sum(torch.abs(eeg_c0_spectrum) * w_alpha).item()
        
        # Grounding threshold to trigger crystallization phase transition
        grounding_threshold = 0.40
        growth_rate = 1.5 * (alpha_coherence - grounding_threshold)
        
        # Non-linear double-well potential evolution (Allen-Cahn Phase Field model)
        # High Alpha drives crystallization (rho -> 1.0), low Alpha/noise melts walls (rho -> 0.0)
        lap_stone = self.compute_laplacian(self.rho_stone)
        reaction = self.rho_stone * (1.0 - self.rho_stone) * (self.rho_stone - 0.5 + growth_rate)
        
        # Earth spatial diffusion coefficient
        kappa_stone = 0.005
        self.rho_stone = torch.clamp(self.rho_stone + (kappa_stone * lap_stone + reaction) * dt, 0.0, 1.0)

    def process_wood_dynamics(self, density_complex, u, v, dt):
        """
        Simulates honest Wood/Grass ecosystem mechanics:
        - Absorbs Yin Water (Blue complex density) to grow roots along the fluid velocity streamlines.
        - Burns under Yang Fire (Red complex density), releasing heat buoyancy and micro-vortices.
        """
        # Complex densities
        # density_complex channel 0,1: Red/Yang Fire; channel 4,5: Blue/Yin Water
        amp_fire = torch.hypot(density_complex[:, 0:1], density_complex[:, 1:2])
        amp_water = torch.hypot(density_complex[:, 4:5], density_complex[:, 5:6])
        
        # 1. ROOT GROWTH (Water Absorption & Streamline alignment):
        # Wood grows where there is Yin Water, and branches out via diffusion.
        lap_wood = self.compute_laplacian(self.rho_wood)
        
        # Advective stretching: alignment with the fluid velocity vectors
        fluid_speed = torch.sqrt(u**2 + v**2) + 1e-5
        advective_growth = (u * (u / fluid_speed) + v * (v / fluid_speed)) * 0.05 * amp_water
        
        # Growth is proportional to Wood density and Water amplitude
        growth = 0.8 * self.rho_wood * amp_water * (1.0 - self.rho_wood)
        
        # 2. COMBUSTION BY YANG FIRE:
        # Burning rate is proportional to Wood density and Fire amplitude.
        burn_rate = 1.2 * self.rho_wood * amp_fire
        
        # Wood field temporal update
        self.rho_wood = torch.clamp(self.rho_wood + (0.01 * lap_wood + growth + advective_growth - burn_rate) * dt, 0.0, 1.0)
        
        # 3. COMBUSITON RECOIL (Feedback Loop on Fluid Engine):
        # Burning wood consumes water, injects physical Fire, buoyancy, and shear turbulence.
        if burn_rate.any():
            # Inject fire back into Yang channels [0, 1]
            fire_injection = burn_rate * 1.5 * dt
            density_complex[:, 0:1] += fire_injection * torch.cos(self.grid_x * 5.0)
            density_complex[:, 1:2] += fire_injection * torch.sin(self.grid_y * 5.0)
            
            # Consume Yin Water channels [4, 5]
            density_complex[:, 4:6] *= torch.clamp(1.0 - burn_rate * 2.0 * dt, 0.0, 1.0)
            
            # Inject explosive shear vorticity forces directly into velocity fields
            dy_wood = 0.5 * (self.rho_wood[:, :, 2:, 1:-1] - self.rho_wood[:, :, :-2, 1:-1])
            dx_wood = 0.5 * (self.rho_wood[:, :, 1:-1, 2:] - self.rho_wood[:, :, 1:-1, :-2])
            grad_norm = torch.sqrt(dx_wood**2 + dy_wood**2) + 1e-5
            
            # Padding vectors to match resolution size
            grad_y = F.pad(dy_wood / grad_norm, (1, 1, 1, 1), mode='replicate')
            grad_x = F.pad(dx_wood / grad_norm, (1, 1, 1, 1), mode='replicate')
            
            # Generate chaotic micro-vortices at combustion boundary
            vortex_scale = 1800.0 * burn_rate
            u += (-grad_y * vortex_scale) * dt
            v += (grad_x * vortex_scale) * dt
            
        return density_complex, u, v

    def map_pz_beta_neurogamepad(self, eeg_c0_spectrum, eeg_freqs):
        """
        Extracts Pz-centric Beta/SMR (18-36Hz) phase gradients without artificial filters.
        Translates raw motor-planning coherence directly into continuous bipolar physics controls.
        """
        if eeg_c0_spectrum is None or eeg_freqs is None:
            return 0.0, 0.0, 0.0
            
        # 18-36Hz Beta-band Gaussian weight centered around 27Hz
        w_beta = torch.exp(-((eeg_freqs - 27.0) ** 2) / 81.0).view(1, 1, -1)
        beta_coherence = torch.abs(eeg_c0_spectrum) * w_beta
        
        # Spatial gradients computation over the physical layout of the 16 electrodes
        # Utilizing dX, dY, dTQ matrices from SymbioticEngineGPU for coordinate projection
        # Coordinates mapping of 16-channel array on Pz
        c0_global = torch.mean(beta_coherence, dim=2)
        
        # Emulating dX and dY coordinate differences within 13mm radius
        coords_x = torch.tensor([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], device=self.device)
        coords_y = torch.tensor([-2.72, -7.43, -4.77, -10.15, -10.14, -4.77, -7.42, -2.73, 2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], device=self.device)
        
        dX = coords_x.unsqueeze(0) - coords_x.unsqueeze(1)
        dY = coords_y.unsqueeze(0) - coords_y.unsqueeze(1)
        dTQ = coords_x.unsqueeze(0) * coords_y.unsqueeze(1) - coords_y.unsqueeze(0) * coords_x.unsqueeze(1)
        dTQ /= (torch.max(torch.abs(dTQ)) + 1e-8)
        
        # Pure physical projections (Zero-Dimensionality Reduction)
        vx = torch.sum(c0_global * dX).item() * 0.05
        vy = torch.sum(c0_global * dY).item() * 0.05
        torque = torch.sum(c0_global * dTQ).item() * 0.015
        
        # Retain signs for complete bipolar continuous control space
        return vx, vy, torque

    def combine_boundaries(self, initial_obstacles):
        """ Combines initial static obstacles with the active dynamic Earth/Stone phase field """
        return torch.clamp(initial_obstacles + self.rho_stone, 0.0, 1.0)
