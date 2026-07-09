# vortex_physics.py
import torch
import torch.nn.functional as F
import math
import numpy as np
import random
from vortex_fluid import FluidSolver
from vortex_maze import PythonMaze
from vortex_obstacles import init_arena_obstacles
from implicit_config import COORDS_16_X, COORDS_16_Y, ALCHEMY_ENTITIES_CONFIG, SEMANTIC_PILLS_DB
from vortex_telemetry import update_rune_zones
from vortex_unified_physics import calculate_covariance_angle, apply_unified_actor_forces, update_unified_slime_kinematics

class PhaseVortexArena:
    """
    Cauldron Smelting and Labyrinth Navigation Engine.
    Implements multi-frequency cross-coherence coupling equations.
    Allows high-density FreeEEG16 operators to generate order out of fluid chaos
    through real-time Phase-Amplitude Coupling and non-linear wave synchronization.
    """
    def __init__(self, device, width, height, res, seed=202607):
        self.device = device
        self.WIDTH, self.HEIGHT = width, height
        self.res = res
        self.current_seed = seed
        self.solver = FluidSolver(res, device)
        
        self.cfg = {
            'bci_mode': '120_jets', 
            'coherence_relative_to_physical': True,  
            'sticky_walls': False,   
            'vorticity_sensitivity': 0.15, 
            'fluid_damping': 0.80,
            'inner_wall_penetration_limit': 0.04,   
            'outer_wall_repulsion_scale': 150000.0, 
            'outer_wall_penetration_limit': 0.001,  
            'hide_tiled_labyrinths': True,    
            'show_debug_window': True,       
            'torque_sensitivity_multiplier': 0.0008, 
            'base_scale': 1.25,              
            'compress_scale_mult': 0.45,     
            'expand_scale_mult': 1.25,       
            'base_node_radius': 8.0,         
            'compress_radius_sub': 4.0,      
            'expand_radius_add': 4.0,        
            'fluid_cohesion_force': 45.0,    
            'fluid_cohesion_gravity': 120.0, 
        }
        self.smooth_vorticity = 0.0
        
        self.u = torch.zeros((1, 1, res, res), device=device)
        self.v = torch.zeros((1, 1, res, res), device=device)
        
        # Complex density representing unified C^3 vector space
        self.density_complex = torch.zeros((1, 6, res, res), device=device)
        self.density = torch.zeros((1, 3, res, res), device=device) 
        
        self.player_density = torch.zeros((1, 1, res, res), device=device)
        self.wall_density = torch.zeros((1, 1, res, res), device=device)
        self.inner_obstacles = torch.zeros((1, 1, res, res), device=device)
        self.outer_obstacles = torch.zeros((1, 1, res, res), device=device)
        self.inner_wall_density = torch.zeros((1, 1, res, res), device=device)
        
        self.pin_x = torch.tensor(COORDS_16_X, dtype=torch.float32, device=device)
        self.pin_y = torch.tensor(COORDS_16_Y, dtype=torch.float32, device=device)
        self.pair_i, self.pair_j = torch.triu_indices(16, 16, offset=1, device=device)
        
        dx_base = self.pin_x.unsqueeze(0) - self.pin_x.unsqueeze(1)
        dy_base = self.pin_y.unsqueeze(0) - self.pin_y.unsqueeze(1)
        self.base_dist = torch.sqrt(dx_base**2 + dy_base**2)
        
        self.pin_pos = torch.zeros((16, 2), dtype=torch.float32, device=device)
        self.pin_captured = torch.zeros(16, dtype=torch.bool, device=device)
        self.edge_intact = torch.ones(16, dtype=torch.bool, device=device)
        
        self.player_pos = torch.tensor([width/2, height/2], dtype=torch.float32, device=device)
        self.portal_pos = torch.zeros(2, dtype=torch.float32, device=device)
        self.player_angle = 0.0
        self.player_angular_vel = 0.0
        self.screen_size = torch.tensor([width, height], dtype=torch.float32, device=device)
        self.eeg_c0_matrix = torch.zeros((16, 16), device=device)
        
        self.y_indices, self.x_indices = torch.meshgrid(
            torch.arange(res, device=device, dtype=torch.float32),
            torch.arange(res, device=device, dtype=torch.float32), indexing='ij'
        )
        self.rune_zones = []
        
        # Dynamic alchemical matrices
        self.cauldron_temp = 300.0
        self.mixture_entropy = 1.0
        self.pill_quality = 100.0
        self.smelting_progress = 0.0
        self.target_freq_desc = "None"
        self.target_spat_desc = "None"
        self.emergent_pill_name = "None"
        self.emergent_pill_similarity = 0.0
        
        self.score_resonance = 0.0
        self.score_containment = 0.0
        self.score_temp = 0.0
        self.score_vortex = 0.0
        
        self.reset_world()
        
    def init_obstacles(self):
        return init_arena_obstacles(self.solver.grid_x, self.solver.grid_y, self.maze.grid, self.res, self.device)

    def reset_world(self):
        self.maze = PythonMaze(11, seed=self.current_seed)
        dim = self.maze.dim
        self.goal_cell = self.maze.cauldron_cell
        
        self.inner_obstacles, self.outer_obstacles = self.init_obstacles()
        self.orig_obstacles = self.inner_obstacles + self.outer_obstacles
        self.wall_density.copy_(self.orig_obstacles)
        self.cell_w = (self.WIDTH * 0.8) / self.maze.dim
        
        self.cauldron_pos = torch.tensor([
            (self.WIDTH * 0.1) + (self.maze.cauldron_cell[0] + 0.5) * self.cell_w,
            (self.HEIGHT * 0.1) + (self.maze.cauldron_cell[1] + 0.5) * self.cell_w
        ], dtype=torch.float32, device=self.device)
        
        self.pin_pos = torch.zeros((16, 2), dtype=torch.float32, device=self.device)
        
        # DEBUG CHECKBOX SKIP: Player spawns directly near Cauldron with entities pre-collected
        self.SKIP_TO_CAULDRON = True
        
        if self.SKIP_TO_CAULDRON:
            # Spawn 65px below cauldron center to avoid instant teleports while allowing auto-smelting
            self.player_pos.copy_(torch.tensor([self.cauldron_pos[0].item(), self.cauldron_pos[1].item() + 65.0], dtype=torch.float32, device=self.device))
        else:
            spawn_x = (self.WIDTH * 0.1) + (self.maze.spawn_cell[0] + 0.5) * self.cell_w
            spawn_y = (self.HEIGHT * 0.1) + (self.maze.spawn_cell[1] + 0.5) * self.cell_w
            self.player_pos.copy_(torch.tensor([spawn_x, spawn_y], dtype=torch.float32, device=self.device))
            
        self.player_angle = 0.0
        self.portal_pos = torch.tensor([-1000.0, -1000.0], dtype=torch.float32, device=self.device)
        self.pin_captured.zero_()
        self.edge_intact.fill_(True)
        self.pin_pos[:, 0] = self.player_pos[0] + self.pin_x * 1.5
        self.pin_pos[:, 1] = self.player_pos[1] + self.pin_y * 1.5
        
        self.u.zero_()
        self.v.zero_()
        self.density_complex.zero_()
        self.density.zero_()
        self.player_density.zero_()
        self.inner_wall_density.copy_(self.inner_obstacles)
        self.wall_density.copy_(self.orig_obstacles)
        self.smooth_vorticity = 0.0
        self.eeg_c0_matrix.zero_()
        self.rune_zones = []

        # Spawn alchemical entities in designated labyrinth sectors (or inside Cauldron if skipping)
        self.alchemy_entities = []
        for ent_cfg in ALCHEMY_ENTITIES_CONFIG:
            if self.SKIP_TO_CAULDRON:
                e_pos = torch.tensor([
                    self.cauldron_pos[0].item(),
                    self.cauldron_pos[1].item()
                ], dtype=torch.float32, device=self.device)
                initial_state = 'cauldron'
            else:
                e_pos = torch.tensor([
                    self.cauldron_pos[0].item() + ent_cfg['offset'][0],
                    self.cauldron_pos[1].item() + ent_cfg['offset'][1]
                ], dtype=torch.float32, device=self.device)
                initial_state = 'cauldron'
            
            self.alchemy_entities.append({
                'type': ent_cfg['type'], 
                'pos': e_pos, 
                'state': initial_state, 
                'phase': 0.0, 
                'freq': ent_cfg['freq'], 
                'tq': ent_cfg['tq'],
                'vector': ent_cfg['vector']
            })
                
        self.smelting_progress = 0.0
        self.pill_created = False
        self.pill_quality = 100.0
        self.emergent_pill_name = "Unknown"
        self.emergent_pill_similarity = 0.0
        
        self.score_resonance = 0.0
        self.score_containment = 0.0
        self.score_temp = 0.0
        self.score_vortex = 0.0

    def get_emergent_target(self):
        active_ents = [e for e in self.alchemy_entities if e['type'] != 'pill']
        if not active_ents:
            return 14.0, 0.0
            
        net_freq = sum(e['freq'] for e in active_ents) / len(active_ents)
        net_tq = sum(e['tq'] for e in active_ents) / len(active_ents)
        return net_freq, net_tq

    def decode_cauldron_state(self):
        cgx, cgy = int(self.res//2), int(self.res//2)
        r_re = self.density_complex[0, 0, cgy, cgx].item()
        r_im = self.density_complex[0, 1, cgy, cgx].item()
        g_re = self.density_complex[0, 2, cgy, cgx].item()
        g_im = self.density_complex[0, 3, cgy, cgx].item()
        b_re = self.density_complex[0, 4, cgy, cgx].item()
        b_im = self.density_complex[0, 5, cgy, cgx].item()
        
        amp_r = math.hypot(r_re, r_im)
        amp_g = math.hypot(g_re, g_im)
        amp_b = math.hypot(b_re, b_im)
        
        vec = torch.tensor([amp_r, amp_g, amp_b], dtype=torch.float32, device=self.device)
        norm = torch.norm(vec) + 1e-8
        vec = vec / norm
        
        best_name = "Unknown Slag"
        best_sim = -1.0
        
        for name, data in SEMANTIC_PILLS_DB.items():
            target_vec = torch.tensor(data["vector"], dtype=torch.float32, device=self.device)
            target_vec = target_vec / (torch.norm(target_vec) + 1e-8)
            sim = torch.sum(vec * target_vec).item()
            if sim > best_sim:
                best_sim = sim
                best_name = name
                
        return best_name, best_sim

    def step(self, dt, time_sec, eeg_c0_spectrum, eeg_vx, eeg_vy, eeg_tq, is_real_data, compression, scale_factor, eeg_freqs=None, alch_freq=0.0, alch_spatial=0.0):
        if torch.is_tensor(eeg_vx): eeg_vx = eeg_vx.item()
        if torch.is_tensor(eeg_vy): eeg_vy = eeg_vy.item()
        if torch.is_tensor(eeg_tq): eeg_tq = eeg_tq.item()

        blend = max(-1.0, min(1.0, compression))
        
        if blend < 0.0:
            scale = self.cfg['base_scale'] - blend * self.cfg['expand_scale_mult']
            node_radius = self.cfg['base_node_radius'] - blend * self.cfg['expand_radius_add']
        else:
            scale = self.cfg['base_scale'] - blend * self.cfg['compress_scale_mult']
            node_radius = self.cfg['base_node_radius'] - blend * self.cfg['compress_radius_sub']

        damping_factor = 0.96 - abs(blend) * 0.02
        self.u = torch.nan_to_num(self.u, nan=0.0) * damping_factor
        self.v = torch.nan_to_num(self.v, nan=0.0) * damping_factor
        self.density_complex = torch.nan_to_num(self.density_complex, nan=0.0) * 0.975
        
        self.u = torch.clamp(self.u, -55.0, 55.0)
        self.v = torch.clamp(self.v, -55.0, 55.0)
        
        erosion = torch.sqrt(self.u**2 + self.v**2) * 0.005 * dt
        self.inner_wall_density = torch.clamp(self.inner_obstacles - erosion, 0.0, 1.0)
        self.inner_wall_density += (self.inner_obstacles - self.inner_wall_density) * 1.8 * dt
        self.inner_wall_density = torch.clamp(self.inner_wall_density, 0.0, 1.0)
        self.wall_density = torch.clamp(self.inner_wall_density + self.outer_obstacles, 0.0, 1.0)

        # Calculate Angle via unified physics helper
        self.player_angle, com = calculate_covariance_angle(
            self.pin_pos, self.pin_x, self.pin_y, self.player_angle, eeg_tq, dt, self.device
        )

        # Apply movement forces to fluid grid via unified module
        self.u, self.v = apply_unified_actor_forces(
            self.device, self.res, self.WIDTH, self.HEIGHT, self.pin_pos, self.pin_x, self.pin_y, self.player_angle,
            self.edge_intact, self.u, self.v, self.wall_density, eeg_vx, eeg_vy, eeg_tq, dt, blend,
            node_radius, self.x_indices, self.y_indices, self.player_density,
            self.cfg['fluid_cohesion_force'], self.cfg['fluid_cohesion_gravity']
        )

        # Apply slime boundary coordinate updates from unified module
        self.pin_pos, self.edge_intact, new_com = update_unified_slime_kinematics(
            self.device, self.res, self.WIDTH, self.HEIGHT, self.pin_pos, self.pin_x, self.pin_y, self.player_angle,
            self.edge_intact, self.u, self.v, self.wall_density, eeg_tq, dt, scale, blend, self.cfg, self.cell_w, self.pin_captured
        )
        self.player_pos[0], self.player_pos[1] = new_com[0], new_com[1]

        # --- SPECTRAL EXTRACTION OF COHERENCE LAYERS ---
        theta_coh_val = 0.0
        smr_coh_val = 0.0
        gamma_coh_val = 0.0

        if is_real_data and eeg_freqs is not None and eeg_c0_spectrum is not None:
            theta_mask = (eeg_freqs >= 4.0) & (eeg_freqs <= 8.0)
            gamma_mask = (eeg_freqs >= 30.0) & (eeg_freqs <= 100.0)
            theta_coh = torch.sum(eeg_c0_spectrum[:, :, theta_mask]).item()
            gamma_coh = torch.sum(eeg_c0_spectrum[:, :, gamma_mask]).item()
            alch_freq = max(-1.0, min(1.0, (gamma_coh - theta_coh) / (theta_coh + gamma_coh + 1e-5) * 2.0))
            inner_idx = [2, 5, 10, 13]
            outer_idx = [0, 1, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15]
            c0_inner = torch.sum(self.eeg_c0_matrix[inner_idx][:, inner_idx]).item()
            c0_outer = torch.sum(self.eeg_c0_matrix[outer_idx][:, outer_idx]).item()
            alch_spatial = max(-1.0, min(1.0, (c0_inner - c0_outer) / (c0_inner + c0_outer + 1e-5) * 2.0))

            # Full multi-frequency resolution extraction
            num_bins = eeg_c0_spectrum.shape[2]
            t_slice = eeg_c0_spectrum[:, :, 2:5]
            s_slice = eeg_c0_spectrum[:, :, 6:10]
            g_slice = eeg_c0_spectrum[:, :, 15:51]
            
            theta_coh_val = torch.mean(torch.abs(t_slice)).item()
            smr_coh_val   = torch.mean(torch.abs(s_slice)).item()
            gamma_coh_val = torch.mean(torch.abs(g_slice)).item()

        # Pure, un-hacked cauldron temperature calculation based on User Actions
        self.cauldron_temp = 300.0 + (eeg_tq**2) * 1500.0 + (gamma_coh_val * 2200.0)
        self.cauldron_temp = max(300.0, min(4500.0, self.cauldron_temp))

        # --- MULTI-FREQUENCY NON-LINEAR WAVE COUPLING (ORDER FROM CHAOS) ---
        R_re, R_im = self.density_complex[0, 0], self.density_complex[0, 1]
        G_re, G_im = self.density_complex[0, 2], self.density_complex[0, 3]
        B_re, B_im = self.density_complex[0, 4], self.density_complex[0, 5]
        
        amp_R = torch.hypot(R_re, R_im) + 1e-8
        amp_G = torch.hypot(G_re, G_im) + 1e-8
        amp_B = torch.hypot(B_re, B_im) + 1e-8
        
        phase_R = torch.atan2(R_im, R_re)
        phase_B = torch.atan2(B_im, B_re)
        
        # 1. Phase-Locking: SMR Catalyst (Qi / Green) locks Yin and Yang into synchronization
        phase_diff = phase_R - phase_B
        lock_in_mult = 1.0 + smr_coh_val * 4.5 if is_real_data else 1.0
        lock_in_force = amp_G * torch.sin(phase_diff) * 25.0 * lock_in_mult * dt
        
        # 2. Phase-Amplitude Coupling: Theta phase modulates Gamma amplitude
        pac_strength = (theta_coh_val * gamma_coh_val) * 6.5 if is_real_data else 0.25
        amp_R_new = amp_R * (1.0 + pac_strength * amp_B * torch.cos(phase_B) * dt * 2.0)
        
        phase_R_new = phase_R - lock_in_force
        phase_B_new = phase_B + lock_in_force
        
        # Write back multi-frequency non-linear updates to the C^3 space
        self.density_complex[0, 0] = amp_R_new * torch.cos(phase_R_new)
        self.density_complex[0, 1] = amp_R_new * torch.sin(phase_R_new)
        self.density_complex[0, 4] = amp_B * torch.cos(phase_B_new)
        self.density_complex[0, 5] = amp_B * torch.sin(phase_B_new)

        # 3. Dynamic viscous damping of chaotic fluid velocities if Theta coherence is high
        if is_real_data:
            viscous_absorption = theta_coh_val * 0.40
            self.u *= (1.0 - viscous_absorption * dt)
            self.v *= (1.0 - viscous_absorption * dt)

        # Update scoring metrics based on pure un-hacked user alignments
        target_f_val, target_tq_val = self.get_emergent_target()
        target_f = max(-1.0, min(1.0, (target_f_val - 14.0) / 66.0)) 
        target_s = max(-1.0, min(1.0, target_tq_val / 45.0))
        
        self.target_freq_desc = f"EMERGENT: {target_f_val:.0f}Hz"
        self.target_spat_desc = f"EMERGENT TQ: {target_tq_val:.0f}"

        self.score_resonance = max(0.0, min(1.0, 1.0 - abs(alch_freq - target_f)/2.0))
        self.score_containment = max(0.0, min(1.0, 1.0 - abs(alch_spatial - target_s)/2.0))
        
        if 1500.0 <= self.cauldron_temp <= 2500.0:
            self.score_temp = 1.0 - abs(self.cauldron_temp - 2000.0)/500.0
        else:
            self.score_temp = max(0.0, 1.0 - abs(self.cauldron_temp - 2000.0)/1500.0) * 0.3

        self.score_vortex = min(1.0, abs(eeg_tq) * 2.0)
        self.emergent_pill_name, self.emergent_pill_similarity = self.decode_cauldron_state()

        active_ents = [e for e in self.alchemy_entities if e['type'] != 'pill']
        if len(active_ents) == len(ALCHEMY_ENTITIES_CONFIG) and all(e['state'] == 'cauldron' for e in active_ents):
            dist_player = torch.norm(self.player_pos - self.cauldron_pos).item()
            if dist_player < self.cell_w * 2.0:
                resonance_mult = self.score_resonance * self.score_containment * self.score_temp
                
                # === BIFURCATION PUMP (ENTROPY EXPORT) ===
                if self.score_temp > 0.5 and (self.score_resonance * self.score_containment) > 0.5:
                    R_re_pad = F.pad(self.density_complex[:, 0], (1, 1, 1, 1), mode='circular')
                    R_im_pad = F.pad(self.density_complex[:, 1], (1, 1, 1, 1), mode='circular')
                    G_re_pad = F.pad(self.density_complex[:, 2], (1, 1, 1, 1), mode='circular')
                    G_im_pad = F.pad(self.density_complex[:, 3], (1, 1, 1, 1), mode='circular')
                    B_re_pad = F.pad(self.density_complex[:, 4], (1, 1, 1, 1), mode='circular')
                    B_im_pad = F.pad(self.density_complex[:, 5], (1, 1, 1, 1), mode='circular')
                    
                    noise_field = torch.abs(R_re_pad[:, 1:-1, 2:] - R_re_pad[:, 1:-1, :-2]) + \
                                  torch.abs(R_im_pad[:, 2:, 1:-1] - R_im_pad[:, :-2, 1:-1]) + \
                                  torch.abs(G_re_pad[:, 1:-1, 2:] - G_re_pad[:, 1:-1, :-2]) + \
                                  torch.abs(G_im_pad[:, 2:, 1:-1] - G_im_pad[:, :-2, 1:-1]) + \
                                  torch.abs(B_re_pad[:, 1:-1, 2:] - B_re_pad[:, 1:-1, :-2]) + \
                                  torch.abs(B_im_pad[:, 2:, 1:-1] - B_im_pad[:, :-2, 1:-1])
                                  
                    FEIGENBAUM_DELTA = 4.6692016
                    
                    # Real-time multi-frequency coherence synchronization accelerates pump capacity up to 5x
                    coherence_synergy = (theta_coh_val + smr_coh_val + gamma_coh_val) / 3.0 if is_real_data else 0.0
                    pump_multiplier = 1.0 + (coherence_synergy * 4.0)
                    
                    pump_power = (self.score_resonance * self.score_temp) * FEIGENBAUM_DELTA * pump_multiplier
                    
                    cx_grid, cy_grid = (self.cauldron_pos[0]/self.WIDTH)*self.res, (self.cauldron_pos[1]/self.HEIGHT)*self.res
                    dist_c = torch.sqrt((self.x_indices - cx_grid)**2 + (self.y_indices - cy_grid)**2) + 1e-5
                    radial_x = (self.x_indices - cx_grid) / dist_c
                    radial_y = (self.y_indices - cy_grid) / dist_c
                    
                    inside_cauldron = (dist_c < 12.0).float()
                    noise_field_sq = noise_field.squeeze(0)
                    
                    # Physical action of the pump: spits chaos out, crystallizes core inside
                    entropy_export = noise_field_sq * inside_cauldron * pump_power * 60.0 * dt
                    self.u[0, 0] += radial_x * entropy_export
                    self.v[0, 0] += radial_y * entropy_export
                    
                    cleanse_factor = torch.clamp(1.0 - (noise_field * inside_cauldron * pump_power * 2.0 * dt), 0.0, 1.0)
                    self.density_complex *= cleanse_factor.unsqueeze(1)
                    
                    # Authentic alchemical-physical progress accumulation (no auto-progress!)
                    self.smelting_progress = min(1.0, self.smelting_progress + torch.sum(entropy_export).item() * 0.0001)
                    self.pill_quality = max(0.0, min(100.0, self.pill_quality + dt * 5.0))
                else:
                    self.smelting_progress = max(0.0, self.smelting_progress - dt * 0.06)
                    penalty = ((1.0 - self.score_resonance)*1.5 + (1.0 - self.score_containment)*1.0 + (1.0 - self.score_temp)*2.0) * dt * 3.0
                    self.pill_quality = max(0.0, self.pill_quality - penalty)
            else:
                self.smelting_progress = max(0.0, self.smelting_progress - dt * 0.10)

            if self.smelting_progress >= 1.0 and not self.pill_created:
                pill_data = SEMANTIC_PILLS_DB.get(self.emergent_pill_name, SEMANTIC_PILLS_DB["Foundation Pill"])
                self.alchemy_entities = [{'type': 'pill', 'pos': self.cauldron_pos.clone(), 'state': 'done', 'freq': 14.0, 'phase': 0.0, 'tq': 20.0, 'vector': pill_data['vector']}]
                self.pill_created = True
                
                # Spawn Goal Portal EXACTLY at cauldron position (0 offset) as originally requested
                self.portal_pos = self.cauldron_pos.clone()

        for ent in self.alchemy_entities:
            ent['phase'] = (ent['phase'] + ent['freq'] * dt) % (2 * math.pi)
            phase_cos = math.cos(ent['phase'])
            phase_sin = math.sin(ent['phase'])

            uv = (ent['pos'] / self.screen_size) * 2.0 - 1.0
            if ent['state'] == 'idle':
                u_val = F.grid_sample(self.u, uv.view(1,1,1,2), align_corners=True).squeeze()
                v_val = F.grid_sample(self.v, uv.view(1,1,1,2), align_corners=True).squeeze()
                ent['pos'][0] += torch.clamp(u_val, -12.0, 12.0) * 8.0 * dt
                ent['pos'][1] += torch.clamp(v_val, -12.0, 12.0) * 8.0 * dt
                
                dist_p = torch.norm(ent['pos'] - self.player_pos).item()
                if dist_p < self.cell_w * 2.0:
                    dir_p = self.player_pos - ent['pos']
                    ent['pos'] += (dir_p / (dist_p + 1e-5)) * 55.0 * dt 

                wval = F.grid_sample(self.wall_density, uv.view(1,1,1,2), align_corners=True).squeeze()
                if wval > 0.05:
                    dir_c = self.cauldron_pos - ent['pos']
                    ent['pos'] += (dir_c / (torch.norm(dir_c) + 1e-5)) * 120.0 * dt

                ent['pos'][0] = torch.clamp(ent['pos'][0], 30.0, self.WIDTH - 30.0)
                ent['pos'][1] = torch.clamp(ent['pos'][1], 30.0, self.HEIGHT - 30.0)

                if torch.norm(ent['pos'] - self.cauldron_pos).item() < self.cell_w * 1.2:
                    ent['state'] = 'cauldron'

            elif ent['state'] == 'cauldron':
                dir_c = self.cauldron_pos - ent['pos']
                dist_c = torch.norm(dir_c).item()
                if dist_c > 2.0:
                    ent['pos'] += (dir_c / dist_c) * 45.0 * dt

            egx = (ent['pos'][0] / self.WIDTH) * self.res
            egy = (ent['pos'][1] / self.HEIGHT) * self.res
            dx_e = torch.remainder(self.x_indices - egx + self.res/2, self.res) - self.res/2
            dy_e = torch.remainder(self.y_indices - egy + self.res/2, self.res) - self.res/2
            mask_e = torch.exp(-(dx_e**2 + dy_e**2) / (5.0**2))
            
            rate = 30.0 * dt
            vec = ent['vector']
            
            for c in range(3):
                projection = vec[c]
                if projection > 0:
                    self.density_complex[0, c*2]   += mask_e * phase_cos * rate * projection
                    self.density_complex[0, c*2+1] += mask_e * phase_sin * rate * projection
                    
            self.u[0, 0] += mask_e * (-dy_e) * ent['tq'] * dt 
            self.v[0, 0] += mask_e * (dx_e) * ent['tq'] * dt

        # Unified Clamping Guard values to avoid NaNs
        self.density_complex = torch.clamp(self.density_complex, -2.5, 2.5)

        # Detect Goal Portal Capture (Only active after pill is created, preventing premature wraps)
        if self.pill_created:
            dx_p = torch.remainder(self.pin_pos[:, 0] - self.portal_pos[0] + self.WIDTH/2, self.WIDTH) - self.WIDTH/2
            dy_p = torch.remainder(self.pin_pos[:, 1] - self.portal_pos[1] + self.HEIGHT/2, self.HEIGHT) - self.HEIGHT/2
            self.pin_captured = self.pin_captured | ((torch.sqrt(dx_p**2 + dy_p**2)) < self.cell_w * 0.35)
        else:
            self.pin_captured.zero_()

        self.player_density = self.solver.advect(self.player_density, self.u, self.v, dt * 1.5)
        self.player_density += (self.player_density * 0.0 - self.player_density) * 4.5 * dt
        self.player_density = torch.clamp(self.player_density, 0.0, 1.0)
        self.player_density = torch.clamp(self.player_density * (1.0 - (self.wall_density > 0.1).float()), 0.0, 1.0)

        self.density_complex = self.solver.advect(self.density_complex, self.u, self.v, dt)
        
        R = torch.sqrt(self.density_complex[:, 0]**2 + self.density_complex[:, 1]**2)
        G = torch.sqrt(self.density_complex[:, 2]**2 + self.density_complex[:, 3]**2)
        B = torch.sqrt(self.density_complex[:, 4]**2 + self.density_complex[:, 5]**2)
        self.density = torch.stack([R, G, B], dim=1) 
        self.density = torch.clamp(self.density, 0.0, 1.0)

        self.u = self.solver.advect(self.u, self.u, self.v, dt)
        self.v = self.solver.vvect(self.v, self.u, self.v, dt) if hasattr(self.solver, 'vvect') else self.solver.advect(self.v, self.u, self.v, dt)
        self.u, self.v = self.solver.project(self.u, self.v, self.wall_density, None)
