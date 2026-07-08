# vortex_combat.py
import torch
import torch.nn.functional as F
import math
import numpy as np
import random
from vortex_fluid import FluidSolver
from implicit_config import COORDS_16_X, COORDS_16_Y, ALCHEMY_ENTITIES_CONFIG, SEMANTIC_PILLS_DB
from vortex_softbody import update_neighbor_springs, apply_cohesion_constraint

class PhaseVortexCombat:
    """
    Arena Module for Domain Clashes.
    Integrates a rigorous Kuramoto Coupled Oscillator Network to define health.
    Damage is defined as the phase decoherence (Xin Mo) of the soft-body connectome.
    Integrates 120-jet BCI/Gamepad inputs and dynamic domain rule imposition.
    """
    def __init__(self, device, width, height, res, player_pill_data, difficulty=1):
        self.device = device
        self.WIDTH, self.HEIGHT = width, height
        self.res = res
        self.solver = FluidSolver(res, device)
        
        # Player attributes inherited from the Cauldron smelting process
        self.player_pill_name = player_pill_data.get('name', 'Foundation Pill')
        self.player_quality = player_pill_data.get('quality', 50.0)
        self.player_vector = torch.tensor(player_pill_data['vector'], dtype=torch.float32, device=device)
        self.player_vector /= (torch.norm(self.player_vector) + 1e-8)
        self.player_domain_charge = 0.0
        
        # Determine elemental archetype
        self.is_player_yang = self.player_vector[0] > self.player_vector[2]
        self.is_player_yin = self.player_vector[2] > self.player_vector[0] and self.player_vector[2] > self.player_vector[1]
        self.is_player_slag = (self.player_quality < 40.0)
        
        # Kuramoto Coupled Oscillator States [16 internal phases in radians]
        self.player_node_phases = torch.zeros(16, dtype=torch.float32, device=device)
        self.bot_node_phases = torch.zeros(16, dtype=torch.float32, device=device)
        
        # Coupling strength K scales with pill quality (Self-Cultivation capacity)
        self.player_K = (self.player_quality / 100.0) * 15.0 + 2.0
        if self.is_player_slag:
            self.player_K = 0.5 
            
        self.player_integrity = 1.0 # Kuramoto Order Parameter [0.0 ... 1.0]
        self.bot_integrity = 1.0
        
        # Active control metrics
        self.player_freq_val = 0.0
        self.player_spatial_val = 0.0
        self.energy_absorbed = 0.0
        self.last_snap_event = False
        self.last_heal_event = False
        
        # Raw Gamepad diagnostics buffer
        self.raw_axes = []
        self.raw_buttons = []
        
        # Physical debug telemetry for real-time spectroscopy
        self.clash_intensity = 0.0
        self.player_shear_stress = 0.0  
        self.bot_shear_stress = 0.0     
        self.player_density_val = 0.0   
        self.bot_density_val = 0.0      
        self.player_jitter_avg = 0.0    
        self.bot_jitter_avg = 0.0       
        
        # Difficulty scaling
        self.difficulty = difficulty
        self.bot_quality = min(99.0, 30.0 + 15.0 * difficulty)
        self.bot_K = (self.bot_quality / 100.0) * 15.0 + 2.0
        
        # Generate Bot
        bot_keys = list(SEMANTIC_PILLS_DB.keys())
        self.bot_pill_name = np.random.choice(bot_keys)
        self.bot_vector = torch.tensor(SEMANTIC_PILLS_DB[self.bot_pill_name]['vector'], dtype=torch.float32, device=device)
        self.bot_vector /= (torch.norm(self.bot_vector) + 1e-8)
        self.bot_domain_charge = 0.0
        
        self.is_bot_yang = self.bot_vector[0] > self.bot_vector[2]
        self.is_bot_yin = self.bot_vector[2] > self.bot_vector[0] and self.bot_vector[2] > self.bot_vector[1]
        
        # Fluid Dynamics State
        self.u = torch.zeros((1, 1, res, res), device=device)
        self.v = torch.zeros((1, 1, res, res), device=device)
        self.density_complex = torch.zeros((1, 6, res, res), device=device)
        self.density = torch.zeros((1, 3, res, res), device=device)
        
        # Circular boundary
        y_grid, x_grid = torch.meshgrid(torch.linspace(-1, 1, res, device=device), torch.linspace(-1, 1, res, device=device), indexing='ij')
        dist_from_center = torch.sqrt(x_grid**2 + y_grid**2)
        self.wall_density = (dist_from_center > 0.85).float().view(1, 1, res, res)
        
        self.player_density = torch.zeros((1, 1, res, res), device=device)
        self.bot_density = torch.zeros((1, 1, res, res), device=device)
        self.orig_obstacles = self.wall_density.clone() 
        
        self.pin_x = torch.tensor(COORDS_16_X, dtype=torch.float32, device=device)
        self.pin_y = torch.tensor(COORDS_16_Y, dtype=torch.float32, device=device)
        self.pair_i, self.pair_j = torch.triu_indices(16, 16, offset=1, device=device)
        
        # Player Slime State
        self.player_pos = torch.tensor([width * 0.3, height * 0.5], dtype=torch.float32, device=device)
        self.player_pin_pos = torch.zeros((16, 2), dtype=torch.float32, device=device)
        self.player_pin_pos[:, 0] = self.player_pos[0] + self.pin_x * 1.5
        self.player_pin_pos[:, 1] = self.player_pos[1] + self.pin_y * 1.5
        self.player_edge_intact = torch.ones(16, dtype=torch.bool, device=device)
        self.player_angle = 0.0
        
        # Bot Slime State
        self.bot_pos = torch.tensor([width * 0.7, height * 0.5], dtype=torch.float32, device=device)
        self.bot_pin_pos = torch.zeros((16, 2), dtype=torch.float32, device=device)
        self.bot_pin_pos[:, 0] = self.bot_pos[0] + self.pin_x * 1.5
        self.bot_pin_pos[:, 1] = self.bot_pos[1] + self.pin_y * 1.5
        self.bot_edge_intact = torch.ones(16, dtype=torch.bool, device=device)
        self.bot_angle = math.pi
        
        self.y_indices, self.x_indices = torch.meshgrid(
            torch.arange(res, device=device, dtype=torch.float32),
            torch.arange(res, device=device, dtype=torch.float32), indexing='ij'
        )
        
        self.combat_time = 0.0
        self.winner = None

    def _inject_domain_waves(self, pos, vector, tq, dt):
        """ Injects the entity's phase vector into the complex fluid to establish Domain """
        px_g = (pos[0] / self.WIDTH) * self.res
        py_g = (pos[1] / self.HEIGHT) * self.res
        dx = torch.remainder(self.x_indices - px_g + self.res/2, self.res) - self.res/2
        dy = torch.remainder(self.y_indices - py_g + self.res/2, self.res) - self.res/2
        mask = torch.exp(-(dx**2 + dy**2) / (12.0**2))
        
        rate = 45.0 * dt
        phase_ang = (self.combat_time * 14.0) % (2 * math.pi) 
        phase_cos = math.cos(phase_ang)
        phase_sin = math.sin(phase_ang)
        
        for c in range(3):
            if vector[c] > 0.01:
                self.density_complex[0, c*2]   += mask * phase_cos * rate * vector[c]
                self.density_complex[0, c*2+1] += mask * phase_sin * rate * vector[c]
                
        self.u[0, 0] += mask * (-dy) * tq * dt
        self.v[0, 0] += mask * (dx) * tq * dt

    def _inject_domain_explosion(self, pos, vector, scale, is_yang):
        """ Generates a violent physical shockwave pushing or pulling local fluid """
        px_g = (pos[0] / self.WIDTH) * self.res
        py_g = (pos[1] / self.HEIGHT) * self.res
        dx = torch.remainder(self.x_indices - px_g + self.res/2, self.res) - self.res/2
        dy = torch.remainder(self.y_indices - py_g + self.res/2, self.res) - self.res/2
        
        dist = torch.sqrt(dx**2 + dy**2) + 1e-5
        mask = torch.exp(-(dist**2) / (25.0**2))
        
        for c in range(3):
            if vector[c] > 0.01:
                self.density_complex[0, c*2]   += mask * 120.0 * vector[c]
                
        force_dir = 1.0 if is_yang else -1.0
        self.u[0, 0] += (dx / dist) * mask * 1100.0 * scale * force_dir
        self.v[0, 0] += (dy / dist) * mask * 1100.0 * scale * force_dir

    def _apply_120_jet_eeg_physics(self, eeg_c0_spectrum, pin_pos, dt):
        """ Translates BCI signals into continuous micro-turbulent protection jets """
        if eeg_c0_spectrum is None:
            return
            
        c0_total = torch.mean(eeg_c0_spectrum[:16, :16, :], dim=2) 
        c0_120 = c0_total[self.pair_i, self.pair_j]
        
        mid_x = (pin_pos[self.pair_i, 0] + pin_pos[self.pair_j, 0]) / 2.0
        mid_y = (pin_pos[self.pair_i, 1] + pin_pos[self.pair_j, 1]) / 2.0
        dx_ideal = self.pin_x[self.pair_i] - self.pin_x[self.pair_j]
        dy_ideal = self.pin_y[self.pair_i] - self.pin_y[self.pair_j]
        
        jet_force_x = c0_120 * dx_ideal * 30.0
        jet_force_y = c0_120 * dy_ideal * 30.0
        
        pin_gx_120 = torch.remainder((mid_x / self.WIDTH) * self.res, self.res)
        pin_gy_120 = torch.remainder((mid_y / self.HEIGHT) * self.res, self.res)
        dx_shape_120 = torch.remainder(self.x_indices.unsqueeze(0) - pin_gx_120.reshape(120, 1, 1) + self.res/2, self.res) - self.res/2
        dy_shape_120 = torch.remainder(self.y_indices.unsqueeze(0) - pin_gy_120.reshape(120, 1, 1) + self.res/2, self.res) - self.res/2
        
        node_influence_120 = torch.exp(-(dx_shape_120**2 + dy_shape_120**2) / (5.0))
        node_inf_120_norm = node_influence_120 / (torch.sum(node_influence_120, dim=(1, 2), keepdim=True) + 1e-8)
        
        self.u[0, 0] += torch.sum(node_inf_120_norm * jet_force_x.reshape(120, 1, 1), dim=0) * dt
        self.v[0, 0] += torch.sum(node_inf_120_norm * jet_force_y.reshape(120, 1, 1), dim=0) * dt

    def _apply_domain_rule_imposition(self, dt):
        """ 
        Tug-Of-War Domain Physics. 
        Boundary zones form dynamic Feigenbaum fractal barriers.
        Imposes specific physics parameters locally based on domain dominance.
        """
        r_re_g = self.density_complex[0, 0]
        r_im_g = self.density_complex[0, 1]
        g_re_g = self.density_complex[0, 2]
        g_im_g = self.density_complex[0, 3]
        b_re_g = self.density_complex[0, 4]
        b_im_g = self.density_complex[0, 5]
        
        amp_r_g = torch.hypot(r_re_g, r_im_g)
        amp_g_g = torch.hypot(g_re_g, g_im_g)
        amp_b_g = torch.hypot(b_re_g, b_im_g)
        
        grid_vec = torch.stack([amp_r_g, amp_g_g, amp_b_g], dim=0) 
        grid_norm = torch.norm(grid_vec, dim=0, keepdim=True) + 1e-8
        grid_vec_norm = grid_vec / grid_norm 
        
        S_p = torch.clamp(torch.sum(grid_vec_norm * self.player_vector.view(3, 1, 1), dim=0), 0.0, 1.0)
        S_b = torch.clamp(torch.sum(grid_vec_norm * self.bot_vector.view(3, 1, 1), dim=0), 0.0, 1.0)
        
        # 1. CHAOTIC FRACTAL CLASH INTERFACE
        clash_mask = torch.clamp(4.0 * S_p * S_b, 0.0, 1.0)
        FEIGENBAUM_DELTA = 4.6692016
        self.clash_intensity = torch.mean(clash_mask).item()
        
        bifurcation_wave_u = torch.sin(self.density_complex[0, 0] * 8.0) * torch.cos(self.density_complex[0, 4] * 5.0)
        bifurcation_wave_v = torch.cos(self.density_complex[0, 2] * 8.0) * torch.sin(self.density_complex[0, 0] * 5.0)
        
        self.u[0, 0] += FEIGENBAUM_DELTA * clash_mask * bifurcation_wave_u * 45.0 * dt
        self.v[0, 0] += FEIGENBAUM_DELTA * clash_mask * bifurcation_wave_v * 45.0 * dt
        
        # 2. SECTOR DOMINANCE PHYSICS
        player_dominates = (S_p > S_b).float()
        bot_dominates = (S_b >= S_p).float()
        
        if self.is_player_yang:
            self.u[0, 0] += player_dominates * (-self.y_indices + self.res/2) * 0.25 * dt
            self.v[0, 0] += player_dominates * (self.x_indices - self.res/2) * 0.25 * dt
        elif self.is_player_yin:
            self.u[0, 0] *= (1.0 - player_dominates * 0.35 * dt)
            self.v[0, 0] *= (1.0 - player_dominates * 0.35 * dt)
        else:
            self.u[0, 0] *= (1.0 - player_dominates * 0.12 * dt)
            self.v[0, 0] *= (1.0 - player_dominates * 0.12 * dt)

        if self.is_bot_yang:
            self.u[0, 0] += bot_dominates * (-self.y_indices + self.res/2) * 0.25 * dt
            self.v[0, 0] += bot_dominates * (self.x_indices - self.res/2) * 0.25 * dt
        elif self.is_bot_yin:
            self.u[0, 0] *= (1.0 - bot_dominates * 0.35 * dt)
            self.v[0, 0] *= (1.0 - bot_dominates * 0.35 * dt)

    def _evaluate_kuramoto_neural_decoherence(self, pin_pos, identity_vector, is_player, dt):
        """ 
        Models connectome health using a Kuramoto Coupled Oscillator Network.
        Hostile phase density scrambles internal node synchronization.
        If player matches protective configurations, they absorb chaos to heal broken springs.
        """
        px_norm = torch.clamp((pin_pos[:, 0] / self.WIDTH) * 2.0 - 1.0, -1.0, 1.0)
        py_norm = torch.clamp((pin_pos[:, 1] / self.HEIGHT) * 2.0 - 1.0, -1.0, 1.0)
        grid_uv = torch.stack([px_norm, py_norm], dim=1).view(1, 1, 16, 2)
        
        # Fetch local complex phase matrices
        r_re = F.grid_sample(self.density_complex[:, 0:1], grid_uv, align_corners=True).squeeze()
        r_im = F.grid_sample(self.density_complex[:, 1:2], grid_uv, align_corners=True).squeeze()
        g_re = F.grid_sample(self.density_complex[:, 2:3], grid_uv, align_corners=True).squeeze()
        g_im = F.grid_sample(self.density_complex[:, 3:4], grid_uv, align_corners=True).squeeze()
        b_re = F.grid_sample(self.density_complex[:, 4:5], grid_uv, align_corners=True).squeeze()
        b_im = F.grid_sample(self.density_complex[:, 5:6], grid_uv, align_corners=True).squeeze()
        
        local_m = torch.sqrt(r_re**2 + r_im**2 + g_re**2 + g_im**2 + b_re**2 + b_im**2)
        local_vec = torch.stack([torch.hypot(r_re, r_im), torch.hypot(g_re, g_im), torch.hypot(b_re, b_im)], dim=1)
        local_vec_norm = local_vec / (torch.norm(local_vec, dim=1, keepdim=True) + 1e-8)
        similarity = torch.sum(local_vec_norm * identity_vector.unsqueeze(0), dim=1)
        
        # High dissonance indicates exposure to a hostile phase domain
        dissonance = torch.clamp(0.5 - similarity, 0.0, 1.0)
        
        # Disruption jitter is generated directly by hostile phase density magnitude
        disruption_force = dissonance * local_m * 450.0
        
        # Slag receives extreme, unshielded stress
        quality = self.player_quality if is_player else self.bot_quality
        quality_mod = 1.0 if not (is_player and self.is_player_slag) else 0.1
        jitter_force = disruption_force * (100.0 / (quality * quality_mod + 1e-5)) * 0.5
        
        # Save metrics for GUI spectroscopy
        if is_player:
            self.player_shear_stress = torch.mean(dissonance).item() 
            self.player_density_val = torch.mean(local_m).item()
            self.player_jitter_avg = torch.mean(jitter_force).item()
            edge_intact = self.player_edge_intact
        else:
            self.bot_shear_stress = torch.mean(dissonance).item()
            self.bot_density_val = torch.mean(local_m).item()
            self.bot_jitter_avg = torch.mean(jitter_force).item()
            edge_intact = self.bot_edge_intact

        # --- RECEPTIVE NEUTRAL ABSORPTION / PARRIES ---
        is_absorbing = False
        if is_player and (self.is_player_yin or not self.is_player_yang) and not self.is_player_slag:
            if self.player_spatial_val < -0.3: # Shield expansion is needed to absorb waves
                is_absorbing = True
                
        if is_absorbing:
            self.energy_absorbed += float(torch.mean(jitter_force).item()) * 0.05
            if self.energy_absorbed >= 12.0:
                if (~edge_intact).any():
                    broken_nodes = torch.nonzero(~edge_intact).squeeze(1)
                    target_idx = broken_nodes[0].item()
                    edge_intact[target_idx] = True # Regrow connected spring
                    self.last_heal_event = True
                self.energy_absorbed = 0.0
            return torch.zeros((16, 2), device=self.device) # Absorbed, zero disruption force

        # --- KURAMOTO PHASE UPDATES ---
        # 1. Base oscillator movement (14Hz SMR frequency)
        phases = self.player_node_phases if is_player else self.bot_node_phases
        phases += 14.0 * 2 * math.pi * dt
        
        # 2. Kuramoto internal coupling step
        idx_next = torch.remainder(torch.arange(16, device=self.device) + 1, 16)
        phase_diffs = phases[idx_next] - phases
        K = self.player_K if is_player else self.bot_K
        phases += K * torch.sin(phase_diffs) * dt
        
        # 3. Dynamic phase scrambling (Xin Mo phase drift)
        scramble_rate = dissonance * local_m * 18.0
        phase_noise = (torch.rand(16, device=self.device) * 2.0 - 1.0) * scramble_rate
        phases += phase_noise * dt
        
        # Clamp phases within (-pi, pi) for rendering and stability
        phases_clamped = torch.remainder(phases + math.pi, 2 * math.pi) - math.pi
        if is_player:
            self.player_node_phases.copy_(phases_clamped)
        else:
            self.bot_node_phases.copy_(phases_clamped)
            
        # 4. Calculate final order parameter (cohesion value H)
        cos_sum = torch.cos(phases_clamped).mean()
        sin_sum = torch.sin(phases_clamped).mean()
        integrity = torch.sqrt(cos_sum**2 + sin_sum**2).item()
        integrity = max(0.1, min(1.0, integrity))
        
        if is_player:
            self.player_integrity = integrity
        else:
            self.bot_integrity = integrity

        # Generate physical jitter offset vectors based on Kuramoto phase instability
        angles = torch.rand(16, device=self.device) * 2.0 * math.pi
        offset_x = torch.cos(angles) * jitter_force
        offset_y = torch.sin(angles) * jitter_force
        
        # Self-damage backlash (Xin Mo)
        if is_player and self.player_domain_charge > 0.1 and self.player_spatial_val < -0.4:
            backlash = self.player_domain_charge * 120.0
            offset_x += (random.random() * 2.0 - 1.0) * backlash
            offset_y += (random.random() * 2.0 - 1.0) * backlash
            self.last_snap_event = True
        
        return torch.stack([offset_x, offset_y], dim=1)

    def _update_slime_kinematics(self, com, pin_pos, edge_intact, angle, tq, alch_spatial, is_player, dt):
        """ 
        Morphs slime scale and connected spring stiffness based on direct axis input.
        Detects spring tension and permanently breaks connections if threshold is breached.
        """
        spatial_blend = float(alch_spatial) if is_player else (0.4 if self.is_bot_yang else -0.3)
        ideal_scale = 1.25 - spatial_blend * 0.35 
        
        cos_p, sin_p = math.cos(angle), math.sin(angle)
        ideal_x = self.pin_x * cos_p + self.pin_y * sin_p
        ideal_y = -self.pin_x * sin_p + self.pin_y * cos_p
        ideal_pos = com.unsqueeze(0) + torch.stack([ideal_x * ideal_scale, ideal_y * ideal_scale], dim=1)
        
        pin_uv = torch.stack([(pin_pos[:, 0]/self.WIDTH)*2-1, (pin_pos[:, 1]/self.HEIGHT)*2-1], dim=1).view(1,1,16,2)
        su = F.grid_sample(self.u, pin_uv, align_corners=True).squeeze()
        sv = F.grid_sample(self.v, pin_uv, align_corners=True).squeeze()
        fluid_vel = torch.stack([su, sv], dim=1) * 80.0
        
        local_x, local_y = pin_pos[:, 0] - com[0], pin_pos[:, 1] - com[1]
        dist_l = torch.sqrt(local_x**2 + local_y**2) + 1e-5
        tx, ty = -local_y / dist_l, local_x / dist_l
        
        # Calculate dynamic spring tension using the Kuramoto Order Parameter (Phase Integrity)
        integrity_mult = self.player_integrity if is_player else self.bot_integrity
        stiffness_mult = (1.0 + spatial_blend * 1.2) * integrity_mult
        
        f_spring, edge_intact = update_neighbor_springs(pin_pos, ideal_pos, edge_intact, cohesion=0.8, device=self.device)
        f_spring *= stiffness_mult
        
        # Evaluate physical elongation strain limits
        idx_I = torch.arange(16, device=self.device)
        idx_J = torch.remainder(idx_I + 1, 16)
        d_curr = torch.norm(pin_pos[idx_I] - pin_pos[idx_J], dim=1)
        
        quality_factor = self.player_quality if is_player else self.bot_quality
        if is_player and self.is_player_slag:
            quality_factor = 5.0 # Slag springs have zero yield strength
            
        elongation_limit = (20.0 + quality_factor * 0.25) * max(0.2, integrity_mult)
        has_snapped = (d_curr > elongation_limit) & edge_intact
        edge_intact[has_snapped] = False
        
        if is_player and has_snapped.any():
            self.last_snap_event = True
            
        f_restore = (ideal_pos - pin_pos) * 15.0
        pin_vel = fluid_vel * 0.9 + f_spring * 0.2 + f_restore
        pin_vel[:, 0] += tx * tq * 100.0
        pin_vel[:, 1] += ty * tq * 100.0
        
        wall_val = F.grid_sample(self.wall_density, pin_uv, align_corners=True).squeeze()
        pushed = wall_val > 0.1
        if pushed.any():
            center_x, center_y = self.WIDTH/2, self.HEIGHT/2
            dir_to_center_x = center_x - pin_pos[pushed, 0]
            dir_to_center_y = center_y - pin_pos[pushed, 1]
            norms = torch.sqrt(dir_to_center_x**2 + dir_to_center_y**2) + 1e-5
            pin_vel[pushed, 0] += (dir_to_center_x / norms) * 800.0
            pin_vel[pushed, 1] += (dir_to_center_y / norms) * 800.0
            
        pin_vel = torch.clamp(pin_vel, -300.0, 300.0)
        pin_pos[edge_intact] += pin_vel[edge_intact] * dt
        pin_pos[~edge_intact] += pin_vel[~edge_intact] * dt * 0.1 
        
        # Snapped/Ruptured nodes bleed grey high-entropy phase noise (impedes flow control)
        if (~edge_intact).any():
            broken_idx = torch.nonzero(~edge_intact).squeeze(1)
            for idx in broken_idx:
                gx_b, gy_b = int((pin_pos[idx, 0]/self.WIDTH)*self.res), int((pin_pos[idx, 1]/self.HEIGHT)*self.res)
                if 0 < gx_b < self.res and 0 < gy_b < self.res:
                    self.density_complex[0, 0:6, gy_b-1:gy_b+2, gx_b-1:gx_b+2] += (random.random() * 2.0 - 1.0) * 20.0 * dt
        
        dummy_captured = torch.zeros(16, dtype=torch.bool, device=self.device)
        pin_pos = apply_cohesion_constraint(pin_pos, ideal_pos, dummy_captured, ideal_scale, cohesion_level=0.8)
        
        return pin_pos.mean(dim=0), pin_pos, edge_intact

    def _bot_ai_logic(self, dt):
        bot_dir = self.player_pos - self.bot_pos
        bot_dist = torch.norm(bot_dir) + 1e-5
        bot_move_x, bot_move_y, bot_tq = 0.0, 0.0, 0.0
        
        if self.difficulty <= 2:
            bot_move_x = (bot_dir[0] / bot_dist) * 400.0 * dt
            bot_move_y = (bot_dir[1] / bot_dist) * 400.0 * dt
            self.bot_domain_charge += dt * 0.4
            if self.bot_domain_charge > 1.0:
                self._inject_domain_explosion(self.bot_pos, self.bot_vector, 1.0, self.is_bot_yang)
                self.bot_domain_charge = 0.0
                
        elif self.difficulty <= 5:
            bot_tq = math.sin(self.combat_time * 3.0) * 0.8
            if self.is_bot_yang:
                bot_move_x = (bot_dir[0] / bot_dist) * 600.0 * dt
                bot_move_y = (bot_dir[1] / bot_dist) * 600.0 * dt
            else:
                if bot_dist < 200.0:
                    bot_move_x = -(bot_dir[0] / bot_dist) * 500.0 * dt
                    bot_move_y = -(bot_dir[1] / bot_dist) * 500.0 * dt
                    
            self.bot_domain_charge += dt * 0.7
            if self.bot_domain_charge > 1.0 and bot_dist < 300.0:
                self._inject_domain_explosion(self.bot_pos, self.bot_vector, 1.4, self.is_bot_yang)
                self.bot_domain_charge = 0.0
                
        else:
            bot_tq = (random.random() * 2.0 - 1.0) * 1.5
            synthetic_eeg = torch.rand((16, 16, 1), device=self.device) * 0.9
            self._apply_120_jet_eeg_physics(synthetic_eeg, self.bot_pin_pos, dt)
            
            bot_move_x = (bot_dir[0] / bot_dist + math.sin(self.combat_time * 5.0)) * 800.0 * dt
            bot_move_y = (bot_dir[1] / bot_dist + math.cos(self.combat_time * 7.0)) * 800.0 * dt
            
            self.bot_domain_charge += dt * 1.2
            if self.bot_domain_charge > 1.0:
                self._inject_domain_explosion(self.bot_pos, self.bot_vector, 2.0, self.is_bot_yang)
                self.bot_domain_charge = 0.0

        bx_g, by_g = int((self.bot_pos[0]/self.WIDTH)*self.res), int((self.bot_pos[1]/self.HEIGHT)*self.res)
        if 0 < bx_g < self.res and 0 < by_g < self.res:
            self.u[0, 0, by_g-2:by_g+3, bx_g-2:bx_g+3] += bot_move_x
            self.v[0, 0, by_g-2:by_g+3, bx_g-2:bx_g+3] += bot_move_y
            
        self.bot_angle -= bot_tq * 3.5 * dt
        
        px_g = (self.bot_pos[0] / self.WIDTH) * self.res
        py_g = (self.bot_pos[1] / self.HEIGHT) * self.res
        dx = torch.remainder(self.x_indices - px_g + self.res/2, self.res) - self.res/2
        dy = torch.remainder(self.y_indices - py_g + self.res/2, self.res) - self.res/2
        mask = torch.exp(-(dx**2 + dy**2) / (8.0**2))
        for c in range(3):
            if self.bot_vector[c] > 0.01:
                self.density_complex[0, c*2] += mask * 15.0 * dt * self.bot_vector[c]

        return bot_tq

    def step(self, dt, eeg_c0_spectrum, eeg_vx, eeg_vy, eeg_tq, alch_freq, alch_spatial, is_real_data):
        self.combat_time += dt
        if self.winner is not None:
            return 
            
        self.player_freq_val = float(alch_freq)
        self.player_spatial_val = float(alch_spatial)
            
        self.u = torch.nan_to_num(self.u, nan=0.0) * 0.90 
        self.v = torch.nan_to_num(self.v, nan=0.0) * 0.90
        self.density_complex = torch.nan_to_num(self.density_complex, nan=0.0) * 0.95
        
        # 1. Player Input & 120-Jet Physics
        if is_real_data and eeg_c0_spectrum is not None:
            self._apply_120_jet_eeg_physics(eeg_c0_spectrum, self.player_pin_pos, dt)
        
        # 2. Player Movement Input
        self.player_angle -= eeg_tq * 3.5 * dt
        bci_force_x = (-math.sin(self.player_angle) * -eeg_vy + math.cos(self.player_angle) * eeg_vx) * 1200.0 * dt
        bci_force_y = (-math.cos(self.player_angle) * -eeg_vy - math.sin(self.player_angle) * eeg_vx) * 1200.0 * dt
        
        px_g, py_g = int((self.player_pos[0]/self.WIDTH)*self.res), int((self.player_pos[1]/self.HEIGHT)*self.res)
        if 0 < px_g < self.res and 0 < py_g < self.res:
            self.u[0, 0, py_g-2:py_g+3, px_g-2:px_g+3] += bci_force_x
            self.v[0, 0, py_g-2:py_g+3, px_g-2:px_g+3] += bci_force_y

        # 3. Active Domain Pumping & Explosions
        target_f = 1.0 if self.player_vector[0] > self.player_vector[2] else -1.0 
        target_s = 1.0 if self.player_vector[0] > 0.5 else -1.0
        
        # Only allows active pumping if the player has synthesized a valid non-slag core
        if not self.is_player_slag:
            if abs(self.player_freq_val - target_f) < 0.6 and abs(self.player_spatial_val - target_s) < 0.6:
                self.player_domain_charge += dt * 1.5
                if self.player_domain_charge >= 1.0:
                    self._inject_domain_explosion(self.player_pos, self.player_vector, 2.0, self.is_player_yang)
                    self.player_domain_charge = 0.0
            else:
                self.player_domain_charge = max(0.0, self.player_domain_charge - dt)

        # Passive Aura
        dx = torch.remainder(self.x_indices - px_g + self.res/2, self.res) - self.res/2
        dy = torch.remainder(self.y_indices - py_g + self.res/2, self.res) - self.res/2
        mask = torch.exp(-(dx**2 + dy**2) / (8.0**2))
        for c in range(3):
            if self.player_vector[c] > 0.01:
                self.density_complex[0, c*2] += mask * 15.0 * dt * self.player_vector[c]

        # 4. Domain Clashes & Bifurcations
        self._apply_domain_rule_imposition(dt)

        # 5. Rogue Cultivator AI Step
        bot_tq = self._bot_ai_logic(dt)

        # 6. Physical Stress & Tension Jitter forces
        player_disruption = self._evaluate_kuramoto_neural_decoherence(self.player_pin_pos, self.player_vector, is_player=True, dt=dt)
        self.player_pin_pos[self.player_edge_intact] += player_disruption[self.player_edge_intact] * dt
        
        bot_disruption = self._evaluate_kuramoto_neural_decoherence(self.bot_pin_pos, self.bot_vector, is_player=False, dt=dt)
        self.bot_pin_pos[self.bot_edge_intact] += bot_disruption[self.bot_edge_intact] * dt

        # 7. Kuramoto Cohesion Win/Loss Condition (Shattered Connectome)
        if self.player_integrity < 0.25:
            self.winner = "Rogue Cultivator"
        elif self.bot_integrity < 0.25:
            self.winner = "Player"

        # 8. Kinematics Updates
        self.player_pos, self.player_pin_pos, self.player_edge_intact = self._update_slime_kinematics(
            self.player_pos, self.player_pin_pos, self.player_edge_intact, self.player_angle, eeg_tq, self.player_spatial_val, True, dt)
            
        self.bot_pos, self.bot_pin_pos, self.bot_edge_intact = self._update_slime_kinematics(
            self.bot_pos, self.bot_pin_pos, self.bot_edge_intact, self.bot_angle, bot_tq, 0.0, False, dt)

        # 9. Advection & Projection
        self.density_complex = self.solver.advect(self.density_complex, self.u, self.v, dt)
        
        R = torch.sqrt(self.density_complex[:, 0]**2 + self.density_complex[:, 1]**2)
        G = torch.sqrt(self.density_complex[:, 2]**2 + self.density_complex[:, 3]**2)
        B = torch.sqrt(self.density_complex[:, 4]**2 + self.density_complex[:, 5]**2)
        self.density = torch.stack([R, G, B], dim=1) 
        self.density = torch.clamp(self.density, 0.0, 1.0)
        
        self.u = self.solver.advect(self.u, self.u, self.v, dt)
        self.v = self.solver.vvect(self.v, self.u, self.v, dt) if hasattr(self.solver, 'vvect') else self.solver.advect(self.v, self.u, self.v, dt)
        self.u, self.v = self.solver.project(self.u, self.v, self.wall_density, None)
        
        self._update_render_density(self.player_pin_pos, self.player_density, self.player_edge_intact)
        self._update_render_density(self.bot_pin_pos, self.bot_density, self.bot_edge_intact)

    def _update_render_density(self, pin_pos, density_tensor, edge_intact):
        active_nodes = pin_pos[edge_intact]
        if active_nodes.numel() == 0:
            density_tensor.zero_()
            return
        gx = torch.remainder((active_nodes[:, 0] / self.WIDTH) * self.res, self.res)
        gy = torch.remainder((active_nodes[:, 1] / self.HEIGHT) * self.res, self.res)
        n_nodes = active_nodes.shape[0]
        dx = torch.remainder(self.x_indices.unsqueeze(0) - gx.view(n_nodes, 1, 1) + self.res/2, self.res) - self.res/2
        dy = torch.remainder(self.y_indices.unsqueeze(0) - gy.view(n_nodes, 1, 1) + self.res/2, self.res) - self.res/2
        influence = torch.exp(-(dx**2 + dy**2) / 10.0)
        density_tensor.copy_(torch.max(influence, dim=0, keepdim=True)[0].unsqueeze(0))
