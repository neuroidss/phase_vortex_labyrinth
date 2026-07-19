# phase_dream_sandbox.py
import pygame
import torch
import torch.nn.functional as F
import sys
import math
import numpy as np
import time

# --- DYNAMIC HARDWARE FALLBACK BRIDGE & PROJECT IMPORTS ---
from input_manager import UnifiedInputManager
from vortex_softbody import update_neighbor_springs, apply_cohesion_constraint

try:
    from neuro_driver import RealNeuroDriver
    from symbiotic_engine import SymbioticEngineGPU
    HAS_NEURO = True
except ImportError:
    HAS_NEURO = False

# Standard standalone configuration
WIDTH, HEIGHT = 800, 800
COMPUTE_RES = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StandaloneQuantumFluid:
    """
    Compact Ginzburg-Landau 2D fluid and reaction-diffusion solver on GPU/CPU.
    Ensures zero external dependency while computing dynamic viscoplasticity.
    """
    def __init__(self, res, device):
        self.res = res
        self.device = device
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, res, device=device),
            torch.linspace(-1.0, 1.0, res, device=device), indexing='ij'
        )
        self.grid_x = x
        self.grid_y = y

    def advect(self, field, u, v, dt):
        dx = u[0, 0] * (dt * 2.0 / self.res)
        dy = v[0, 0] * (dt * 2.0 / self.res)
        sampling_grid = torch.stack([self.grid_x - dx, self.grid_y - dy], dim=-1).unsqueeze(0)
        sampling_grid = torch.clamp(sampling_grid, -1.0, 1.0)
        return F.grid_sample(field, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)

    def project(self, u, v, wall_density):
        block_mask = (wall_density > 0.4).float()
        u = u * (1.0 - block_mask)
        v = v * (1.0 - block_mask)
        
        u_pad = F.pad(u, (1, 1, 1, 1), mode='replicate')
        v_pad = F.pad(v, (1, 1, 1, 1), mode='replicate')
        div = 0.5 * (u_pad[:, :, 1:-1, 2:] - u_pad[:, :, 1:-1, :-2] + 
                     v_pad[:, :, 2:, 1:-1] - v_pad[:, :, :-2, 1:-1])
        
        p = torch.zeros_like(u)
        for _ in range(25):
            p_pad = F.pad(p, (1, 1, 1, 1), mode='replicate')
            p = 0.25 * (p_pad[:, :, 1:-1, 2:] + p_pad[:, :, 1:-1, :-2] + 
                        p_pad[:, :, 2:, 1:-1] + p_pad[:, :, :-2, 1:-1] - div)
                    
        p_pad = F.pad(p, (1, 1, 1, 1), mode='replicate')
        u -= 0.5 * (p_pad[:, :, 1:-1, 2:] - p_pad[:, :, 1:-1, :-2])
        v -= 0.5 * (p_pad[:, :, 2:, 1:-1] - p_pad[:, :, :-2, 1:-1])
        
        return u * (1.0 - block_mask), v * (1.0 - block_mask)


class MultiSpectralFieldEngine:
    """
    Pure continuous 100-channel non-equilibrium spectral-spatial ecosystem.
    Models indestructible geothermal vents (Black Smokers) emitting low-entropy waves.
    All softbodies require continuous low-entropy consumption to avoid thermal decay.
    """
    def __init__(self, res, device):
        self.res = res
        self.device = device
        
        # Spatial coordinate frameworks
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, res, device=device),
            torch.linspace(-1.0, 1.0, res, device=device), indexing='ij'
        )
        self.grid_x = x
        self.grid_y = y
        self.freqs_hz = torch.linspace(1.0, 100.0, 100, device=device)
        
        # Unified 100-channel physical field [1, 100, res, res]
        self.density_spectral = torch.zeros((1, 100, res, res), device=device)
        
        # Symmetrical square container boundary frame (aligned with [40.0, 760.0] limit)
        self.static_obstacles = ((torch.abs(x) > 0.90) | (torch.abs(y) > 0.90)).float().view(1, 1, res, res)
        
        # Real-time computed continuous fields
        self.crystallization_field = torch.zeros((1, 1, res, res), device=device) 
        self.viscoplastic_drag = torch.zeros((1, 1, res, res), device=device)

        # Coordinate templates of FreeEEG16-alpha2 micro-array (26mm footprint)
        self.pin_x = torch.tensor([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], device=device) * 1.5
        self.pin_y = torch.tensor([-2.72, -7.43, -4.77, -10.15, -10.14, -4.77, -7.42, -2.73, 2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], device=device) * 1.5

        # 16-Node Softbody Network (The Parent Core / User's Phase Entity)
        self.pin_pos = torch.zeros((16, 2), dtype=torch.float32, device=device)
        self.edge_intact = torch.ones(16, dtype=torch.bool, device=device)
        self.node_phases = torch.zeros(16, dtype=torch.float32, device=device)
        self.player_angle = 0.0

        # Initial spawn coordinate
        self.parent_pos = torch.tensor([400.0, 300.0], dtype=torch.float32, device=device)
        self.pin_pos[:, 0] = self.parent_pos[0] + self.pin_x
        self.pin_pos[:, 1] = self.parent_pos[1] + self.pin_y
        
        # Child Satellite Softbody (Orbital entity linked strictly by cross-coherence)
        self.child_pos = torch.tensor([470.0, 300.0], dtype=torch.float32, device=device)
        self.child_pin_pos = torch.zeros((16, 2), dtype=torch.float32, device=device)
        self.child_pin_pos[:, 0] = self.child_pos[0] + self.pin_x * 0.8
        self.child_pin_pos[:, 1] = self.child_pos[1] + self.pin_y * 0.8
        self.child_edge_intact = torch.ones(16, dtype=torch.bool, device=device)
        self.child_phases = torch.zeros(16, dtype=torch.float32, device=device)

        # --- ACTIVE LOW-ENTROPY BLACK SMOKER VENTS ---
        # Hydrothermal vents are wrapped in active 16-node phase-locking resonator rings!
        # They are indestructible bedrock. No stability decay. S1 and S2 are eternal.
        self.vents = [
            {"pos": [260.0, 620.0], "freq": 8.0,  "color": (0, 150, 255), "name": "Yin Siphon (Water)"}, 
            {"pos": [540.0, 620.0], "freq": 24.0, "color": (0, 255, 100), "name": "Qi Siphon (Catalyst)"} 
        ]
        
        # Initialize active physical bounds of the smokers resonator rings
        for vent in self.vents:
            v_pos = torch.tensor(vent["pos"], dtype=torch.float32, device=device)
            v_pin_pos = torch.zeros((16, 2), dtype=torch.float32, device=device)
            v_pin_pos[:, 0] = v_pos[0] + self.pin_x * 0.95
            v_pin_pos[:, 1] = v_pos[1] + self.pin_y * 0.95
            
            vent["pin_pos"] = v_pin_pos
            vent["edge_intact"] = torch.ones(16, dtype=torch.bool, device=device)
            vent["node_phases"] = torch.zeros(16, dtype=torch.float32, device=device)
            vent["stability"] = 1.0 # Eternal order
            vent["angle"] = 0.0
            vent["storm_active"] = False
            vent["storm_timer"] = 0.0

        # --- MULTI-ENTITY ECOSYSTEM INITIALIZATION ---
        self.entities = []
        spawn_configs = [
            {"freq": 8.0,  "color": (0, 100, 255),  "pos": [200.0, 200.0]},  # Blue cultivator
            {"freq": 20.0, "color": (0, 255, 50),   "pos": [600.0, 200.0]},  # Green cultivator
            {"freq": 80.0, "color": (255, 50, 0),   "pos": [400.0, 150.0]}   # Red cultivator
        ]
        
        freqs_np = np.linspace(1.0, 100.0, 100)
        for cfg in spawn_configs:
            pos = torch.tensor(cfg["pos"], dtype=torch.float32, device=device)
            e_pin_pos = torch.zeros((16, 2), dtype=torch.float32, device=device)
            e_pin_pos[:, 0] = pos[0] + self.pin_x * 0.9
            e_pin_pos[:, 1] = pos[1] + self.pin_y * 0.9
            
            gauss_peak = np.exp(-((freqs_np - cfg["freq"]) ** 2) / (2.0 * (8.0 ** 2)))
            e_spec = torch.tensor(gauss_peak, dtype=torch.float32, device=device)
            
            self.entities.append({
                "focus_freq": cfg["freq"],
                "color": cfg["color"],
                "pos": pos,
                "pin_pos": e_pin_pos,
                "edge_intact": torch.ones(16, dtype=torch.bool, device=device),
                "node_phases": torch.zeros(16, dtype=torch.float32, device=device),
                "spectrum": e_spec,
                "angle": 0.0,
                "integrity": 1.0,
                "local_drag": 0.0,
                "internal_k": 0.50 # Entropy storage factor
            })

        # Continuous diagnostic metrics
        self.is_reacting = False
        self.local_drag = 0.0
        self.local_fluidity = 0.0
        self.intent_mag = 0.0
        self.integrity = 1.0
        self.parent_internal_k = 0.50 # Entropy storage factor

    def compute_laplacian(self, field):
        """ Computes a symmetric 2D Laplacian using replicate padding to prevent edge leakage """
        field_pad = F.pad(field, (1, 1, 1, 1), mode='replicate')
        laplacian = (field_pad[:, :, 1:-1, 2:] + field_pad[:, :, 1:-1, :-2] +
                     field_pad[:, :, 2:, 1:-1] + field_pad[:, :, :-2, 1:-1] - 4.0 * field)
        return laplacian

    def compute_spectral_integrals(self, eeg_c0_spectrum, eeg_freqs):
        w_cohesion = torch.exp(-((eeg_freqs - 6.0) ** 2) / 36.0).view(1, 1, -1)
        w_resonance = torch.exp(-((eeg_freqs - 24.0) ** 2) / 144.0).view(1, 1, -1)
        w_dispersion = torch.exp(-((eeg_freqs - 75.0) ** 2) / 400.0).view(1, 1, -1)
        
        e_cohesion   = torch.sum(eeg_c0_spectrum * w_cohesion).item() / 256.0
        e_resonance  = torch.sum(eeg_c0_spectrum * w_resonance).item() / 256.0
        e_dispersion = torch.sum(eeg_c0_spectrum * w_dispersion).item() / 256.0
        return e_cohesion, e_resonance, e_dispersion

    def update_stone_field(self, eeg_c0_spectrum, eeg_freqs, dt):
        e_cohesion, _, _ = self.compute_spectral_integrals(eeg_c0_spectrum, eeg_freqs)
        grounding_threshold = 0.40
        growth_rate = 1.5 * (e_cohesion - grounding_threshold)

        lap_stone = self.compute_laplacian(self.crystallization_field)
        reaction = self.crystallization_field * (1.0 - self.crystallization_field) * (self.crystallization_field - 0.5 + growth_rate)
        self.crystallization_field = torch.clamp(self.crystallization_field + (0.005 * lap_stone + reaction) * dt, 0.0, 1.0)

    def step_softbody_kinematics(self, u, v, eeg_c0_spectrum, eeg_freqs, cross_k, movement_axes, dt):
        """
        Calculates node displacements as the direct, un-averaged physical manifestation
        of the 16x16x100 phase coherence matrix. Zero hardcoded steps.
        """
        # --- 1. NON-EQUILIBRIUM ENTROPY DECAY (Structural aging) ---
        decay_factor = 0.02 * dt
        self.parent_internal_k = max(0.12, self.parent_internal_k - decay_factor)
        for ent in self.entities:
            ent["internal_k"] = max(0.12, ent["internal_k"] - decay_factor)

        # --- 2. EXTRACT SPECTRUM AND COMPUTE COHESION ---
        e_cohesion, e_resonance, e_dispersion = self.compute_spectral_integrals(eeg_c0_spectrum, eeg_freqs)
        
        A1 = torch.hypot(self.density_spectral[:, 0:1], self.density_spectral[:, 1:2])
        A2 = torch.hypot(self.density_spectral[:, 2:3], self.density_spectral[:, 3:4])
        A3 = torch.hypot(self.density_spectral[:, 4:5], self.density_spectral[:, 5:6])
        self.viscoplastic_drag = 4500.0 * torch.clamp(A1 * A2 * A3 * e_cohesion - 0.05, 0.0, 1.0) ** 2

        # --- 3. IMMUTABLE BLACK SMOKERS PLUMES & INJECTIONS ---
        for vent in self.vents:
            v_x = int((vent["pos"][0] / 800.0) * self.res)
            v_y = int((vent["pos"][1] / 800.0) * self.res)
            
            # Upward thermal draft plumes
            dx = self.grid_x - (vent["pos"][0] / 800.0) * self.res
            dy = self.grid_y - (vent["pos"][1] / 800.0) * self.res
            influence = torch.exp(-(dx**2 + dy**2) / 12.0)
            v[0, 0] -= influence * 25.0 * dt
            
            # Symmetrically inject stable low-entropy target peak (indestructible)
            v_idx = int(vent["freq"])
            self.density_spectral[0, v_idx, v_y-2:v_y+3, v_x-2:v_x+3] += 1.8 * dt

            # --- ABSORPTION PHYSICS (Parent Feed) ---
            dist_parent = torch.norm(self.parent_pos - torch.tensor(vent["pos"], device=self.device)).item()
            if dist_parent < 100.0:
                freq_match = max(0.0, 1.0 - abs(focus_freq_value_emulated(eeg_c0_spectrum, self.freqs_hz) - vent["freq"]) / 15.0)
                if freq_match > 0.45:
                    self.parent_internal_k = min(1.0, self.parent_internal_k + freq_match * 0.22 * dt)
                    self.edge_intact[:] = True # Symmetrically heal broken springs
                    
                    # Expel high-entropy Yang Fire (Gamma waste) and local fluid turbulence as exhaust
                    self.density_spectral[0, 85, v_y-8:v_y-4, v_x-2:v_x+3] += freq_match * 0.8 * dt
                    u[0, 0] += (torch.rand_like(u[0, 0]) * 2.0 - 1.0) * 12.0 * dt

            # --- ABSORPTION PHYSICS (Entities Feed) ---
            for ent in self.entities:
                dist_ent = torch.norm(ent["pos"] - torch.tensor(vent["pos"], device=self.device)).item()
                if dist_ent < 100.0:
                    user_norm = ent["spectrum"] / (torch.norm(ent["spectrum"]) + 1e-8)
                    vent_spec = torch.zeros(100, device=self.device)
                    vent_spec[int(vent["freq"])] = 1.0
                    match = torch.sum(user_norm * vent_spec).item()
                    
                    if match > 0.35:
                        ent["internal_k"] = min(1.0, ent["internal_k"] + match * 0.22 * dt)
                        ent["edge_intact"][:] = True
                        
                        # Expel waste that creates localized physical turbulence
                        self.density_spectral[0, 85, v_y-8:v_y-4, v_x-2:v_x+3] += match * 0.8 * dt

        # --- 4. MOVEMENT INTENT & FLUID-STRUCTURE COUPLING ---
        intent_x, intent_y = movement_axes[0], -movement_axes[1]
        self.intent_mag = math.hypot(intent_x, intent_y) + 1e-5
        
        p_uv = torch.stack([
            (self.parent_pos[0] / 800.0) * 2.0 - 1.0,
            (self.parent_pos[1] / 800.0) * 2.0 - 1.0
        ]).view(1, 1, 1, 2)
        
        self.local_drag     = F.grid_sample(self.viscoplastic_drag, p_uv, align_corners=True).squeeze().item()
        w_fluidity          = torch.exp(-((self.freqs_hz - 21.0) ** 2) / 81.0).view(1, 100, 1, 1)
        fluidity_field      = torch.sum(self.density_spectral * w_fluidity, dim=1, keepdim=True)
        self.local_fluidity = F.grid_sample(fluidity_field, p_uv, align_corners=True).squeeze().item()
        local_crystal       = F.grid_sample(self.crystallization_field, p_uv, align_corners=True).squeeze().item()
        
        effective_barrier = max(self.local_drag, local_crystal * 1800.0)
        permeability = 1.0 / (1.0 + effective_barrier * 0.05)
        
        self.is_reacting = False
        if effective_barrier > 120.0:
            boring_threshold = 0.55
            boring_power = self.intent_mag * (1.0 + self.local_fluidity * 4.0)
            if boring_power > boring_threshold:
                permeability = 0.22 * (boring_power / boring_threshold)
                self.is_reacting = True
                p_gx, p_gy = int((self.parent_pos[0]/800.0)*self.res), int((self.parent_pos[1]/800.0)*self.res)
                if 0 <= p_gx < self.res and 0 <= p_gy < self.res:
                    self.density_spectral[:, :, p_gy-3:p_gy+4, p_gx-3:p_gx+4] *= 0.72
            else:
                permeability = 0.0
                
        # Sample fluid velocity locally (advection drift)
        node_uv = torch.clamp((self.pin_pos / 800.0) * 2.0 - 1.0, -1.0, 1.0).view(1, 1, 16, 2)
        sampled_u = F.grid_sample(u, node_uv, align_corners=True).view(16)
        sampled_v = F.grid_sample(v, node_uv, align_corners=True).view(16)
        fluid_vel = torch.stack([sampled_u, sampled_v], dim=1) * 80.0 

        # --- 5. COVARIANCE AND RESTORING FORCES ---
        self.player_angle, com = self.calculate_covariance(self.pin_pos, self.player_angle)
        self.parent_pos.copy_(com)
        
        scale = 1.25 - e_cohesion * 0.35
        cos_p, sin_p = math.cos(self.player_angle), math.sin(self.player_angle)
        ideal_x = self.parent_pos[0] + (self.pin_x * cos_p + self.pin_y * sin_p) * scale
        ideal_y = self.parent_pos[1] + (-self.pin_x * sin_p + self.pin_y * cos_p) * scale
        ideal_pos = torch.stack([ideal_x, ideal_y], dim=1)
        
        f_restore = (ideal_pos - self.pin_pos) * (15.0 + self.parent_internal_k * 45.0)
        f_spring, self.edge_intact = update_neighbor_springs(self.pin_pos, ideal_pos, self.edge_intact, self.parent_internal_k, self.device)
        
        # Symmetrically update Kuramoto node phase angles
        idx_next = torch.remainder(torch.arange(16, device=self.device) + 1, 16)
        idx_prev = torch.remainder(torch.arange(16, device=self.device) - 1, 16)
        coupling = torch.sin(self.node_phases[idx_next] - self.node_phases) + torch.sin(self.node_phases[idx_prev] - self.node_phases)
        K_rate = (e_resonance * 48.0) + 2.0
        self.node_phases += K_rate * 0.5 * coupling * dt
        self.node_phases += (torch.rand(16, device=self.device) * 2.0 - 1.0) * (self.local_drag * 0.05 + (1.0 - self.parent_internal_k) * 1.5) * dt
        
        # Compute Kuramoto phase order parameter (Phase Integrity)
        self.integrity = torch.sqrt(torch.cos(self.node_phases).mean()**2 + torch.sin(self.node_phases).mean()**2).item()
        self.integrity = max(0.01, min(1.0, self.integrity))
        
        # Continuous tearing condition: if integrity and focus drop too low, springs shatter
        if self.parent_internal_k < 0.25 and torch.rand(1).item() < 0.05:
            broken_idx = int(torch.randint(0, 16, (1,)).item())
            self.edge_intact[broken_idx] = False

        # Update node coordinates
        if permeability > 0.0:
            self.pin_pos[:, 0] += (intent_x / self.intent_mag) * permeability * 220.0 * dt
            self.pin_pos[:, 1] += (intent_y / self.intent_mag) * permeability * 220.0 * dt
            
        self.pin_pos[self.edge_intact] += (fluid_vel[self.edge_intact] * 0.85 + f_restore[self.edge_intact] + f_spring[self.edge_intact] * 0.15) * dt
        
        # Symmetrically integrate all 100 frequencies over all 16x16 channel pairs
        pos_diff_parent = self.pin_pos.unsqueeze(1) - self.pin_pos.unsqueeze(0)
        dist_parent_matrix = torch.norm(pos_diff_parent, dim=2, keepdim=True) + 1e-5
        dir_parent_matrix = pos_diff_parent / dist_parent_matrix
        tangent_parent_matrix = torch.stack([-dir_parent_matrix[:, :, 1], dir_parent_matrix[:, :, 0]], dim=2)
        
        # Rotating dispersion angle omega across all 100 frequencies
        omega = (math.pi / 2.0) * torch.clamp((self.freqs_hz - 4.0) / 96.0, 0.0, 1.0)
        omega = omega.view(1, 1, 100) 
        cos_omega, sin_omega = torch.cos(omega), torch.sin(omega)
        f_radial  = torch.sum(eeg_c0_spectrum * cos_omega, dim=2)  
        f_tangent = torch.sum(eeg_c0_spectrum * sin_omega, dim=2) 
        
        node_dx = torch.sum(f_radial * dir_parent_matrix[:, :, 0] + f_tangent * tangent_parent_matrix[:, :, 0], dim=1)
        node_dy = torch.sum(f_radial * dir_parent_matrix[:, :, 1] + f_tangent * tangent_parent_matrix[:, :, 1], dim=1)
        
        self.pin_pos[:, 0] += node_dx * 35.0 * dt
        self.pin_pos[:, 1] += node_dy * 35.0 * dt
        self.pin_pos = apply_cohesion_constraint(self.pin_pos, ideal_pos, torch.zeros(16, dtype=torch.bool, device=self.device), scale, e_cohesion)

        # Inject spectral fluid into corresponding layers locally
        p_gxs = ((self.pin_pos[:, 0] / 800.0) * self.res).long().clamp(0, self.res - 1)
        p_gys = ((self.pin_pos[:, 1] / 800.0) * self.res).long().clamp(0, self.res - 1)
        local_node_spectrum = torch.mean(eeg_c0_spectrum, dim=1).t() 
        self.density_spectral[0, :, p_gys, p_gxs] += local_node_spectrum * 0.15 * dt

        # --- 6. DUAL-AVATAR COHERENT BINDING (Child Softbody) ---
        c_com = self.child_pin_pos.mean(dim=0)
        if cross_k > 0.05:
            # Orbital child satellite orbiting the parent core
            dx_c = c_com[0] - self.parent_pos[0]
            dy_c = c_com[1] - self.parent_pos[1]
            dist_c = math.hypot(dx_c, dy_c) + 1e-5
            
            ideal_dist = 70.0
            orbital_speed = (1.5 + self.local_fluidity * 4.0) * dt
            angle = math.atan2(dy_c, dx_c) + orbital_speed
            
            target_cx = self.parent_pos[0] + math.cos(angle) * ideal_dist
            target_cy = self.parent_pos[1] + math.sin(angle) * ideal_dist
            
            spring_k = cross_k * 15.0
            c_com[0] += (target_cx - c_com[0]) * spring_k * dt
            c_com[1] += (target_cy - c_com[1]) * spring_k * dt
        else:
            c_uv = torch.stack([
                (c_com[0] / 800.0) * 2.0 - 1.0,
                (c_com[1] / 800.0) * 2.0 - 1.0
            ]).view(1, 1, 1, 2)
            u_drift = F.grid_sample(u, c_uv, align_corners=True).squeeze().item()
            v_drift = F.grid_sample(v, c_uv, align_corners=True).squeeze().item()
            c_com[0] += u_drift * 85.0 * dt
            c_com[1] += v_drift * 85.0 * dt
            
            ideal_c_x = c_com[0] + self.pin_x * 0.8
            ideal_c_y = c_com[1] + self.pin_y * 0.8
            self.child_pin_pos += (torch.stack([ideal_c_x, ideal_c_y], dim=1) - self.child_pin_pos) * 15.0 * dt

        # --- 7. INDEPENDENT AUTONOMOUS SPECTRUM-COHERENT CULTIVATORS ---
        for ent in self.entities:
            e_com = ent["pin_pos"].mean(dim=0)
            
            # Advection drift sample
            e_uv = torch.clamp((ent["pin_pos"] / 800.0) * 2.0 - 1.0, -1.0, 1.0).view(1, 1, 16, 2)
            e_u = F.grid_sample(u, e_uv, align_corners=True).view(16)
            e_v = F.grid_sample(v, e_uv, align_corners=True).view(16)
            e_fluid_vel = torch.stack([e_u, e_v], dim=1) * 75.0
            
            # Compute integrated continuous energies for the entity's own spectrum
            e_coh_e, e_res_e, e_disp_e = self.compute_spectral_integrals(
                ent["spectrum"].view(1, 1, 100).repeat(16, 16, 1), self.freqs_hz
            )
            
            # Orientation angle update
            ent["angle"], e_com_new = self.calculate_covariance(ent["pin_pos"], ent["angle"])
            ent["pos"].copy_(e_com_new)
            
            # Restoring forces
            e_scale = 1.15 - e_coh_e * 0.25
            cos_e, sin_e = math.cos(ent["angle"]), math.sin(ent["angle"])
            ideal_e_x = ent["pos"][0] + (self.pin_x * cos_e + self.pin_y * sin_e) * e_scale
            ideal_e_y = ent["pos"][1] + (-self.pin_x * sin_e + self.pin_y * cos_e) * e_scale
            ideal_e_pos = torch.stack([ideal_e_x, ideal_e_y], dim=1)
            
            # Spring constants governed by internal order (internal_k)
            f_rest_e = (ideal_e_pos - ent["pin_pos"]) * (15.0 + ent["internal_k"] * 30.0)
            f_spr_e, ent["edge_intact"] = update_neighbor_springs(ent["pin_pos"], ideal_e_pos, ent["edge_intact"], ent["internal_k"], self.device)
            
            # Symmetrically update Kuramoto node phase angles for this entity
            coupling_e = torch.sin(ent["node_phases"][idx_next] - ent["node_phases"]) + torch.sin(ent["node_phases"][idx_prev] - ent["node_phases"])
            ent["node_phases"] += ((e_res_e * 35.0) + 1.5) * 0.5 * coupling_e * dt
            
            # Continuous tearing for autonomous entities
            if ent["internal_k"] < 0.25 and torch.rand(1).item() < 0.05:
                broken_idx = int(torch.randint(0, 16, (1,)).item())
                ent["edge_intact"][broken_idx] = False
            
            # Calculate spectral compatibility (cross-coherence inner product) with user core
            user_norm = eeg_c0_spectrum[0, 0, :] / (torch.norm(eeg_c0_spectrum[0, 0, :]) + 1e-8)
            ent_norm  = ent["spectrum"] / (torch.norm(ent["spectrum"]) + 1e-8)
            compatibility = torch.sum(user_norm * ent_norm).item()
            
            f_align_x, f_align_y = 0.0, 0.0
            dist_to_user = torch.norm(ent["pos"] - self.parent_pos).item() + 1e-5
            
            if dist_to_user < 250.0:
                dir_to_user_x = (self.parent_pos[0] - ent["pos"][0]) / dist_to_user
                dir_to_user_y = (self.parent_pos[1] - ent["pos"][1]) / dist_to_user
                
                if compatibility > 0.40:
                    # Attractive locked bond (high resonance/coherence)
                    pull = (compatibility - 0.40) * 110.0
                    f_align_x = dir_to_user_x * pull
                    f_align_y = dir_to_user_y * pull
                else:
                    # Repulsive clash boundary (dissonance / phase clash)
                    push = (0.40 - compatibility) * 160.0
                    f_align_x = -dir_to_user_x * push
                    f_align_y = -dir_to_user_y * push
                    
                    clash_gx = int((ent["pos"][0]/800.0)*self.res)
                    clash_gy = int((ent["pos"][1]/800.0)*self.res)
                    if 0 <= clash_gx < self.res and 0 <= clash_gy < self.res:
                        # Inject high-frequency chaotic Yang Fire at interface
                        self.density_spectral[0, 80, clash_gy-1:clash_gy+2, clash_gx-1:clash_gx+2] += (0.40 - compatibility) * 0.45 * dt
            
            # Continuous cumulative internal vibration displacements for entity
            # Calculate and use ent_pos coordinates template for correct physical accuracy!
            pos_diff_e = ent["pin_pos"].unsqueeze(1) - ent["pin_pos"].unsqueeze(0)
            dist_matrix_e = torch.norm(pos_diff_e, dim=2, keepdim=True) + 1e-5
            dir_matrix_e = pos_diff_e / dist_matrix_e
            tangent_matrix_e = torch.stack([-dir_matrix_e[:, :, 1], dir_matrix_e[:, :, 0]], dim=2)
            
            ent_c0_spec = ent["spectrum"].view(1, 1, 100).repeat(16, 16, 1)
            f_rad_e = torch.sum(ent_c0_spec * cos_omega, dim=2)
            f_tan_e = torch.sum(ent_c0_spec * sin_omega, dim=2)
            ent_dx = torch.sum(f_rad_e * dir_matrix_e[:, :, 0] + f_tan_e * tangent_matrix_e[:, :, 0], dim=1)
            ent_dy = torch.sum(f_rad_e * dir_matrix_e[:, :, 1] + f_tan_e * tangent_matrix_e[:, :, 1], dim=1)
            
            # Apply kinematics step
            ent["pin_pos"][ent["edge_intact"]] += (e_fluid_vel[ent["edge_intact"]] * 0.82 + f_rest_e[ent["edge_intact"]] + f_spr_e[ent["edge_intact"]] * 0.15) * dt
            ent["pin_pos"][:, 0] += (f_align_x + ent_dx * 25.0) * dt
            ent["pin_pos"][:, 1] += (f_align_y + ent_dy * 25.0) * dt
            ent["pin_pos"] = apply_cohesion_constraint(ent["pin_pos"], ideal_e_pos, torch.zeros(16, dtype=torch.bool, device=self.device), e_scale, e_coh_e)
            
            # Symmetrically inject spectrum continuously into spatial channels [100, 16]
            e_gxs = ((ent["pin_pos"][:, 0] / 800.0) * self.res).long().clamp(0, self.res - 1)
            e_gys = ((ent["pin_pos"][:, 1] / 800.0) * self.res).long().clamp(0, self.res - 1)
            ent_spec_projected = ent["spectrum"].view(100, 1).repeat(1, 16)
            self.density_spectral[0, :, e_gys, e_gxs] += ent_spec_projected * 0.15 * dt

        # Clamping to square bounds [40.0, 760.0]
        self.pin_pos = torch.clamp(self.pin_pos, 40.0, 760.0)
        self.child_pin_pos = torch.clamp(self.child_pin_pos, 40.0, 760.0)
        for ent in self.entities:
            ent["pin_pos"] = torch.clamp(ent["pin_pos"], 40.0, 760.0)
        
        return u, v, self.density_spectral

    def calculate_covariance(self, pin_pos, current_angle):
        com = pin_pos.mean(dim=0)
        actual_local_x = pin_pos[:, 0] - com[0]
        actual_local_y = pin_pos[:, 1] - com[1]
        cross_cov = torch.sum(self.pin_y * actual_local_x - self.pin_x * actual_local_y)
        dot_cov = torch.sum(self.pin_x * actual_local_x + self.pin_y * actual_local_y) + 1e-5
        raw_angle = torch.atan2(cross_cov, dot_cov).item()
        angle_diff = (raw_angle - current_angle + math.pi) % (2 * math.pi) - math.pi
        new_angle = current_angle + angle_diff * 0.40
        return new_angle, com

    def step_field_dynamics(self, u, v, density_spectral, eeg_c0_spectrum, eeg_freqs, dt):
        """ Continuous non-linear Lotka-Volterra wave field updates """
        e_cohesion, e_resonance, e_dispersion = self.compute_spectral_integrals(eeg_c0_spectrum, eeg_freqs)
        
        A1 = torch.hypot(self.density_spectral[:, 0:1], self.density_spectral[:, 1:2])
        A2 = torch.hypot(self.density_spectral[:, 2:3], self.density_spectral[:, 3:4])
        A3 = torch.hypot(self.density_spectral[:, 4:5], self.density_spectral[:, 5:6])
        
        lap_spectral = self.compute_laplacian(self.density_spectral)
        disp_coefficients = (0.05 / torch.sqrt(self.freqs_hz)).view(1, 100, 1, 1)
        self.density_spectral = torch.clamp(self.density_spectral + lap_spectral * disp_coefficients * dt, 0.0, 3.0)

        # 3-point energy transfer along the frequency axis (Lotka-Volterra)
        psi_prev = torch.roll(self.density_spectral, shifts=1, dims=1)
        psi_next = torch.roll(self.density_spectral, shifts=-1, dims=1)
        cascade = 0.85 * self.density_spectral * (psi_prev - psi_next) * dt
        self.density_spectral = torch.clamp(self.density_spectral + cascade, 0.0, 3.0)

        # Micro-vortices based on the spatial gradients of SMR/Beta (12-30Hz)
        w_fluidity = torch.exp(-((self.freqs_hz - 21.0) ** 2) / 81.0).view(1, 100, 1, 1)
        fluidity_field = torch.sum(self.density_spectral * w_fluidity, dim=1, keepdim=True)
        dy_res = 0.5 * (fluidity_field[:, :, 2:, 1:-1] - fluidity_field[:, :, :-2, 1:-1])
        dx_res = 0.5 * (fluidity_field[:, :, 1:-1, 2:] - fluidity_field[:, :, 1:-1, :-2])
        grad_norm = torch.sqrt(dx_res**2 + dy_res**2) + 1e-5
        
        grad_y = F.pad(dy_res / grad_norm, (1, 1, 1, 1), mode='replicate')
        grad_x = F.pad(dx_res / grad_norm, (1, 1, 1, 1), mode='replicate')
        
        w_high = torch.exp(-((self.freqs_hz - 75.0) ** 2) / 400.0).view(1, 100, 1, 1)
        high_field = torch.sum(self.density_spectral * w_high, dim=1, keepdim=True)
        
        vortex_scale = 1400.0 * (high_field * fluidity_field)
        u += (-grad_y * vortex_scale) * dt
        v += (grad_x * vortex_scale) * dt
        
        buoyancy = -9.81 * (high_field ** 2) * 1.8
        v[0, 0] += buoyancy[0, 0] * dt

        return u, v, self.density_spectral

    def combine_boundaries(self, initial_obstacles):
        """ Combines static container ring with dynamic solidification fields """
        dynamic_walls = ((self.viscoplastic_drag > 220.0) | (self.crystallization_field > 0.45)).float()
        return torch.clamp(initial_obstacles + dynamic_walls, 0.0, 1.0)


def focus_freq_value_emulated(eeg_c0_spectrum, freqs):
    # Continuously calculates center of mass of emulated spectrum focus freq
    spec_1d = eeg_c0_spectrum[0, 0, :]
    num = torch.sum(spec_1d * freqs).item()
    den = torch.sum(spec_1d).item() + 1e-5
    return num / den


def main():
    pygame.init()
    pygame.font.init()
    pygame.joystick.init() # Initialize Gamepad input module
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("EXOCORTEX SPECTRUM-COHERENCE SANDBOX")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 14, bold=True)
    
    # Active Joystick scanner
    joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
    for j in joysticks:
        j.init()
        
    # Active Neuro Feedback Driver scan fallback
    if HAS_NEURO:
        driver = RealNeuroDriver()
        driver.start_lsl_scanning_thread()
        driver.start_ble_scanning_thread()
        neuro_engine = SymbioticEngineGPU(device_name='cuda')
        print("[SANDBOX-HW] Real FreeEEG16 Drivers loaded successfully!")
    else:
        print("[SANDBOX-HW] EEG Drivers not detected. Falling back to continuous frequency tuner.")
        
    # Initialize unified inputs directly from your project
    input_manager = UnifiedInputManager(WIDTH, HEIGHT)
        
    # Initialize standalone modules on active device
    solver = StandaloneQuantumFluid(COMPUTE_RES, device)
    engine = MultiSpectralFieldEngine(COMPUTE_RES, device)
    
    # Fluid velocities fields
    u = torch.zeros((1, 1, COMPUTE_RES, COMPUTE_RES), device=device)
    v = torch.zeros((1, 1, COMPUTE_RES, COMPUTE_RES), device=device)
    
    # Continuous frequency variables
    focus_freq = 24.0 # Center of virtual spectral peak in Hz
    cross_k = 0.50    # Continuous cross coherence K
    
    # Continuous frequency vector [100 bins from 1.0Hz to 100.0Hz]
    freqs_np = np.linspace(1.0, 100.0, 100)
    eeg_freqs = torch.tensor(freqs_np, dtype=torch.float32, device=device)
    
    # Seed initial continuous charges to trigger reaction-diffusion cycles
    engine.density_spectral[0, 24, 45:55, 30:50] = 0.85 
    engine.density_spectral[0, 6, 45:65, 30:60]  = 0.95 
    
    running = True
    last_time = time.time()
    
    while running:
        dt = min(0.032, time.time() - last_time)
        last_time = time.time()
        
        # --- 1. DELEGATE ALL INPUT READING TO UnifiedInputManager (Axes -1, 0, +1) ---
        is_real_data, eeg_vx, eeg_vy, eeg_tq, ui_compression, alch_freq, alch_spatial = input_manager.process_inputs(joysticks, dt)
        
        # Continuous spectral peak focus frequency mapping [-1.0 ... 1.0] -> [1.0Hz ... 100.0Hz]
        focus_freq = 50.0 + alch_freq * 49.0
        # Continuous cross coherence mapping [-1.0 ... 1.0] -> [0.0 ... 1.0]
        cross_k = max(0.0, min(1.0, 0.5 + alch_spatial * 0.5))
        
        movement_axes = (eeg_vx, eeg_vy)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Mouse injects fallback (Continuous injection directly into spectral layers)
        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[0] or mouse_buttons[2]:
            mx, my = pygame.mouse.get_pos()
            mgx = int((mx / WIDTH) * COMPUTE_RES)
            mgy = int((my / HEIGHT) * COMPUTE_RES)
            
            if 0 <= mgx < COMPUTE_RES and 0 <= mgy < COMPUTE_RES:
                if mouse_buttons[0]: # Left Click: Inject High-Frequency Dispersion (75Hz)
                    engine.density_spectral[0, 75, mgy-2:mgy+3, mgx-2:mgx+3] += 1.5
                elif mouse_buttons[2]: # Right Click: Inject Low-Frequency Cohesion (6Hz)
                    engine.density_spectral[0, 6, mgy-2:mgy+3, mgx-2:mgx+3] += 1.5

        # --- 2. MULTI-CHANNEL EEG REAL/EMULATED SPECTRUM ---
        is_real_eeg = False
        eeg_c0_spectrum = None
        
        if HAS_NEURO:
            active_slots = [i for i in range(5) if driver.workers[i].is_connected or any(v == i for v in driver.lsl_inlets.values())]
            if active_slots:
                is_real_eeg = True
                for slot_idx in active_slots:
                    q = driver.queues[slot_idx]
                    q_len = len(q)
                    if q_len > 0:
                        samples = [q.popleft() for _ in range(q_len)]
                        K_samples = len(samples)
                        new_data = torch.tensor(samples, dtype=torch.float32).T
                        
                        if K_samples >= 500:
                            neuro_engine.pinned_cpu_buffer[slot_idx*16:(slot_idx+1)*16, :] = new_data[:, -500:]
                        else:
                            neuro_engine.pinned_cpu_buffer[slot_idx*16:(slot_idx+1)*16, :-K_samples] = neuro_engine.pinned_cpu_buffer[slot_idx*16:(slot_idx+1)*16, K_samples:].clone()
                            neuro_engine.pinned_cpu_buffer[slot_idx*16:(slot_idx+1)*16, -K_samples:] = new_data
                
                # Fetch predictive ciPLV spectrum from high-density array
                c0_spec, freqs, bci_vx, bci_vy, bci_tq = neuro_engine.get_predictive_ciplv(len(active_slots) * 16, 0.0)
                
                # Symmetrically interpolate the raw real-time spectrum to 100 channels without any loss
                real_raw_spec = c0_spec[:16, :16, :] # [16, 16, F_bins]
                f_bins = real_raw_spec.shape[2]
                
                reshaped_spec = real_raw_spec.view(256, 1, f_bins)
                interpolated_spec = F.interpolate(reshaped_spec, size=100, mode='linear', align_corners=True)
                eeg_c0_spectrum = interpolated_spec.view(16, 16, 100)
                eeg_freqs = torch.linspace(1.0, 100.0, 100, device=device)
                
        if eeg_c0_spectrum is None:
            # Emulated mode: Calculates a continuous Gaussian spectral distribution centered around focus_freq
            gauss_peak = np.exp(-((freqs_np - focus_freq) ** 2) / (2.0 * (12.0 ** 2)))
            eeg_c0_spectrum = torch.tensor(gauss_peak, dtype=torch.float32, device=device).view(1, 1, 100).repeat(16, 16, 1)
            eeg_freqs = torch.linspace(1.0, 100.0, 100, device=device)

        # Update non-equilibrium fluid dynamics
        wall_mix = engine.combine_boundaries(engine.static_obstacles)
        u, v = solver.project(u, v, wall_mix)
        
        # Step the active crystallization phase field
        engine.update_stone_field(eeg_c0_spectrum, eeg_freqs, dt)
        
        # Step non-linear continuous alchemical fields and softbody kinematics
        u, v, density_spectral = engine.step_softbody_kinematics(
            u, v, eeg_c0_spectrum, eeg_freqs,
            cross_k, movement_axes, dt
        )
        u, v, density_spectral = engine.step_field_dynamics(
            u, v, density_spectral, eeg_c0_spectrum, eeg_freqs, dt
        )
        
        # Advect density fields
        engine.density_spectral = solver.advect(engine.density_spectral, u, v, dt)
        u = solver.advect(u, u, v, dt)
        v = solver.advect(v, u, v, dt)
        
        # --- RENDER FRAME ---
        # Continuous Spectral Color Mapping mimicking human eye cone sensitivity curves
        w_red   = torch.exp(-((engine.freqs_hz - 75.0) ** 2) / 400.0).view(1, 100, 1, 1)
        w_green = torch.exp(-((engine.freqs_hz - 24.0) ** 2) / 144.0).view(1, 100, 1, 1)
        w_blue  = torch.exp(-((engine.freqs_hz - 6.0) ** 2) / 36.0).view(1, 100, 1, 1)
        
        R = torch.clamp(torch.sum(engine.density_spectral * w_red, dim=1) * 230.0, 0.0, 255.0)
        G = torch.clamp(torch.sum(engine.density_spectral * w_green, dim=1) * 230.0, 0.0, 255.0)
        B = torch.clamp(torch.sum(engine.density_spectral * w_blue, dim=1) * 230.0, 0.0, 255.0)
        
        rgb_tensor = torch.stack([R[0], G[0], B[0]], dim=0).to(torch.uint8)
        
        # Continuous Solidification Tint: slate-gray [110, 120, 130] where local drag is high
        # Blended continuously with broadcasting support. 100% crash-free.
        solid_intensity = torch.clamp(engine.viscoplastic_drag[0, 0] / 350.0 + engine.crystallization_field[0, 0], 0.0, 1.0)
        
        R_final = (rgb_tensor[0].float() * (1.0 - solid_intensity) + 110.0 * solid_intensity).to(torch.uint8)
        G_final = (rgb_tensor[1].float() * (1.0 - solid_intensity) + 120.0 * solid_intensity).to(torch.uint8)
        B_final = (rgb_tensor[2].float() * (1.0 - solid_intensity) + 130.0 * solid_intensity).to(torch.uint8)
        
        rgb_tensor = torch.stack([R_final, G_final, B_final], dim=0)
                                     
        static_mask = (engine.static_obstacles[0, 0] > 0.5)
        rgb_tensor[:, static_mask] = torch.tensor([50, 15, 60], dtype=torch.uint8, device=device).view(3, 1)
        
        # Pygame surface mapping
        rgb_np = rgb_tensor.permute(1, 2, 0).cpu().numpy()
        surf_small = pygame.surfarray.make_surface(np.transpose(rgb_np, (1, 0, 2)))
        surf_large = pygame.transform.scale(surf_small, (WIDTH, HEIGHT))
        screen.blit(surf_large, (0, 0))
        
        # --- DRAW BLACK SMOKERS (HYDROTHERMAL VENTS) ---
        for vent in engine.vents:
            vx, vy = int(vent["pos"][0]), int(vent["pos"][1])
            # Outer glowing low-entropy aura
            pygame.draw.circle(screen, (vent["color"][0]//4, vent["color"][1]//4, vent["color"][2]//4), (vx, vy), 38)
            pygame.draw.circle(screen, vent["color"], (vx, vy), 14, 2)
            pygame.draw.circle(screen, (30, 30, 45), (vx, vy), 8) # Spires core

        # --- DRAW PARENT SOFTBODY NETWORK (USER'S KINEMATIC INTEGRITY ENTITY) ---
        nodes_cpu = engine.pin_pos.cpu().numpy()
        phases_cpu = engine.node_phases.cpu().numpy()
        
        # Draw dynamic spring links. Color and thickness represent local phase difference (coherence)
        for i in range(16):
            if not engine.edge_intact[i]: continue
            next_idx = (i + 1) % 16
            if not engine.edge_intact[next_idx]: continue
            
            p1 = (int(nodes_cpu[i, 0]), int(nodes_cpu[i, 1]))
            p2 = (int(nodes_cpu[next_idx, 0]), int(nodes_cpu[next_idx, 1]))
            
            # Phase lead/lag calculation
            phase_diff = abs(phases_cpu[i] - phases_cpu[next_idx]) % (2 * math.pi)
            if phase_diff > math.pi: phase_diff = 2 * math.pi - phase_diff
            
            # High coherence: cyan, thick link. Low coherence: red, thin/vibrating link
            sync_factor = max(0.0, min(1.0, 1.0 - phase_diff / math.pi))
            col_link = (int(255 * (1.0 - sync_factor)), int(255 * sync_factor * engine.parent_internal_k), int(230 * sync_factor * engine.parent_internal_k))
            width_link = max(1, int(1.0 + sync_factor * 4.0))
            
            # Vibrational offset under desynchronization stress
            vibe_x = int((1.0 - sync_factor) * 5.0 * math.sin(time.time() * 35.0 + i))
            vibe_y = int((1.0 - sync_factor) * 5.0 * math.cos(time.time() * 35.0 + i))
            p1_v = (p1[0] + vibe_x, p1[1] + vibe_y)
            
            pygame.draw.line(screen, col_link, p1_v, p2, width_link)
            
        # Draw physical node points on active grid
        for i in range(16):
            if not engine.edge_intact[i]: continue
            nx, ny = int(nodes_cpu[i, 0]), int(nodes_cpu[i, 1])
            pygame.draw.circle(screen, (0, 255, 255), (nx, ny), 3)

        # --- DRAW CHILD SATELLITE SOFTBODY ENTITY ---
        if cross_k > 0.05:
            child_nodes_cpu = engine.child_pin_pos.cpu().numpy()
            child_phases_cpu = engine.child_phases.cpu().numpy()
            
            for i in range(16):
                if not engine.child_edge_intact[i]: continue
                next_idx = (i + 1) % 16
                if not engine.child_edge_intact[next_idx]: continue
                
                cp1 = (int(child_nodes_cpu[i, 0]), int(child_nodes_cpu[i, 1]))
                cp2 = (int(child_nodes_cpu[next_idx, 0]), int(child_nodes_cpu[next_idx, 1]))
                
                # Phase-coherent coloring
                c_p_diff = abs(child_phases_cpu[i] - child_phases_cpu[next_idx]) % (2 * math.pi)
                if c_p_diff > math.pi: c_p_diff = 2 * math.pi - c_p_diff
                c_sync = max(0.0, min(1.0, 1.0 - c_p_diff / math.pi))
                
                c_col = (int(255 * (1.0 - c_sync)), int(100 * c_sync), int(255 * c_sync))
                pygame.draw.line(screen, c_col, cp1, cp2, max(1, int(1.0 + c_sync * 2.5)))
                pygame.draw.circle(screen, (255, 0, 255), cp1, 2)

        # --- DRAW OTHER SPECTRUM-COHERENT ECOSYSTEM ENTITIES (CULTIVATORS) ---
        for ent in engine.entities:
            ent_nodes_cpu = ent["pin_pos"].cpu().numpy()
            ent_phases_cpu = ent["node_phases"].cpu().numpy()
            ent_color = ent["color"]
            
            # Symmetrically draw the spring boundaries of other active entities
            for i in range(16):
                if not ent["edge_intact"][i]: continue
                next_idx = (i + 1) % 16
                if not ent["edge_intact"][next_idx]: continue
                
                ep1 = (int(ent_nodes_cpu[i, 0]), int(ent_nodes_cpu[i, 1]))
                ep2 = (int(ent_nodes_cpu[next_idx, 0]), int(ent_nodes_cpu[next_idx, 1]))
                
                # Phase coherence lead/lag mapped directly to link sync color
                p_diff = abs(ent_phases_cpu[i] - ent_phases_cpu[next_idx]) % (2 * math.pi)
                if p_diff > math.pi: p_diff = 2 * math.pi - p_diff
                e_sync = max(0.1, min(1.0, 1.0 - p_diff / math.pi))
                
                # Symmetrically blend its native element color with the synchronization glow
                col_link = (
                    int(ent_color[0] * e_sync + 45 * (1.0 - e_sync)),
                    int(ent_color[1] * e_sync + 45 * (1.0 - e_sync) * ent["internal_k"]),
                    int(ent_color[2] * e_sync + 45 * (1.0 - e_sync) * ent["internal_k"])
                )
                pygame.draw.line(screen, col_link, ep1, ep2, max(1, int(1.0 + e_sync * 3.0)))
                pygame.draw.circle(screen, ent_color, ep1, 2)

        # --- DRAW DYNAMIC SPREAD OF BLACK SMOKERS RESONSATOR RINGS ---
        # Showcases the continuous, un-averaged alchemical status of each active siphon
        for vent in engine.vents:
            v_nodes_cpu = vent["pin_pos"].cpu().numpy()
            v_phases_cpu = vent["node_phases"].cpu().numpy()
            v_color = vent["color"]
            
            for i in range(16):
                if not vent["edge_intact"][i]: continue
                next_idx = (i + 1) % 16
                if not vent["edge_intact"][next_idx]: continue
                
                vp1 = (int(v_nodes_cpu[i, 0]), int(v_nodes_cpu[i, 1]))
                vp2 = (int(v_nodes_cpu[next_idx, 0]), int(v_nodes_cpu[next_idx, 1]))
                
                # High resonance alignment: glowing siphon color, thick links
                v_p_diff = abs(v_phases_cpu[i] - v_phases_cpu[next_idx]) % (2 * math.pi)
                if v_p_diff > math.pi: v_p_diff = 2 * math.pi - v_p_diff
                v_sync = max(0.1, min(1.0, 1.0 - v_p_diff / math.pi))
                
                v_col_link = (
                    int(v_color[0] * v_sync + 45 * (1.0 - v_sync)),
                    int(v_color[1] * v_sync * vent["stability"] + 45 * (1.0 - v_sync)),
                    int(v_color[2] * v_sync * vent["stability"] + 45 * (1.0 - v_sync))
                )
                pygame.draw.line(screen, v_col_link, vp1, vp2, max(1, int(1.0 + v_sync * 3.0)))
                pygame.draw.circle(screen, v_color, vp1, 2)

        # --- HUD RENDER & REAL-TIME SPECTRUM ANALYZER ---
        hud_surface = pygame.Surface((310, 250), pygame.SRCALPHA)
        hud_surface.fill((10, 12, 18, 225))
        pygame.draw.rect(hud_surface, (0, 255, 200), (0, 0, 310, 250), 1)
        
        # Symmetrically project parent coordinate template to sample continuous state values in main()
        p_uv = torch.stack([
            (engine.parent_pos[0] / 800.0) * 2.0 - 1.0,
            (engine.parent_pos[1] / 800.0) * 2.0 - 1.0
        ]).view(1, 1, 1, 2)
        local_crystal_val = F.grid_sample(engine.crystallization_field, p_uv, align_corners=True).squeeze().item()
        
        if engine.local_drag > 120.0 or local_crystal_val > 0.45:
            medium = f"Crystallized Solid (Boring {'ACTIVE' if engine.is_reacting else 'BLOCKED'})"
            col_med = (255, 100, 100)
        elif engine.local_fluidity > 0.25:
            medium = "Viscoelastic Medium (Fluid Swimming)"
            col_med = (100, 200, 255)
        else:
            medium = "Vacuum (Reactive Jet Propulsion)"
            col_med = (200, 200, 200)

        # Active telemetry state
        active_state_lbl = f"Real-time BCI" if (HAS_NEURO and is_real_eeg) else "Emulated Spectrum Focus"
        
        # Track siphons stability percentage
        v1_stab = int(engine.vents[0]["stability"] * 100.0)
        v2_stab = int(engine.vents[1]["stability"] * 100.0)

        metrics = [
            "  EXOCORTEX SPECTRAL CONTINUUM v3.0",
            "-" * 39,
            f"Focus Freq  : {focus_freq:.1f} Hz (Analog Sticks)",
            f"Binding (K) : {cross_k:.2f} (Continuous Axes)",
            f"Local State : {medium}",
            f"Phasic Integrity: {engine.integrity*100:.1f}% (Kuramoto)",
            f"Structural order: {engine.parent_internal_k*100:.1f}% (Entropy)",
            "-" * 39,
            f"Input Mode  : {active_state_lbl}",
            "L-Click: Inject 75Hz | R-Click: Inject 6Hz",
            "Analog Sticks: Plan / Sweep Focus",
            f"Engine FPS  : {int(clock.get_fps())} Hz"
        ]
        
        y_offset = 10
        for m in metrics:
            if "EXOCORTEX" in m:
                color = (0, 255, 200)
            elif "Focus" in m or "Binding" in m or "Integrity" in m or "order" in m:
                color = (255, 220, 100)
            elif "State" in m:
                color = col_med
            elif "Click" in m or "Movement" in m or "Analog" in m:
                color = (160, 160, 160)
            elif "Input" in m:
                color = (0, 255, 120) if is_real_eeg else (255, 150, 0)
            else:
                color = (255, 255, 255)
            text_surf = font.render(m, True, color)
            hud_surface.blit(text_surf, (10, y_offset))
            y_offset += 16
            
        # Draw Real-time Spectrum Analyzer at the bottom of the HUD
        pygame.draw.rect(hud_surface, (30, 30, 45), (10, y_offset + 10, 290, 40))
        pygame.draw.rect(hud_surface, (0, 255, 200), (10, y_offset + 10, 290, 40), 1)
        
        # Render Gaussian spectral distribution peak or real interpolated spectrum
        for x_bar in range(290):
            freq_idx = int((x_bar / 290.0) * 100.0)
            freq_idx = max(0, min(99, freq_idx))
            val_y = eeg_c0_spectrum[0, 0, freq_idx].item()
            bar_height = int(val_y * 36.0)
            if bar_height > 0:
                pygame.draw.line(hud_surface, (0, 255, 150), (10 + x_bar, y_offset + 48), (10 + x_bar, y_offset + 48 - bar_height))
                
        # Draw focus frequency vertical tick
        tick_x = int(((focus_freq - 1.0) / 99.0) * 290.0)
        pygame.draw.line(hud_surface, (255, 100, 100), (10 + tick_x, y_offset + 10), (10 + tick_x, y_offset + 49), 2)
            
        screen.blit(hud_surface, (15, 15))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
