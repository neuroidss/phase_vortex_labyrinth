# vortex_combat.py
import torch
import torch.nn.functional as F
import math
import numpy as np
import random
from vortex_fluid import FluidSolver
from implicit_config import COORDS_16_X, COORDS_16_Y, ALCHEMY_ENTITIES_CONFIG, SEMANTIC_PILLS_DB
from vortex_softbody import update_neighbor_springs, apply_cohesion_constraint
from vortex_unified_physics import calculate_covariance_angle, apply_unified_actor_forces, update_unified_slime_kinematics
import combat_config

class PhaseVortexCombat:
    """
    Arena Module for Domain Clashes.
    Symmetrically simulates an array of multiple actors (Cultivators) under identical physical laws.
    Supports extensible 2v2/5v5 setups. AI players act as input emulators mapping directly to Kuramoto node phases.
    """
    def __init__(self, device, width, height, res, player_pill_data, difficulty=1, actor0_data=None, actor1_data=None):
        self.device = device
        self.WIDTH, self.HEIGHT = width, height
        self.res = res
        self.solver = FluidSolver(res, device)
        
        self.y_indices, self.x_indices = torch.meshgrid(
            torch.arange(res, device=device, dtype=torch.float32),
            torch.arange(res, device=device, dtype=torch.float32), indexing='ij'
        )
        
        self.pin_x = torch.tensor(COORDS_16_X, dtype=torch.float32, device=device)
        self.pin_y = torch.tensor(COORDS_16_Y, dtype=torch.float32, device=device)
        self.pair_i, self.pair_j = torch.triu_indices(16, 16, offset=1, device=device)
        
        # Fluid Dynamics State
        self.u = torch.zeros((1, 1, res, res), device=device)
        self.v = torch.zeros((1, 1, res, res), device=device)
        self.density_complex = torch.zeros((1, 6, res, res), device=device)
        self.density = torch.zeros((1, 3, res, res), device=device)
        
        # Circular boundary
        y_grid, x_grid = torch.meshgrid(torch.linspace(-1, 1, res, device=device), torch.linspace(-1, 1, res, device=device), indexing='ij')
        dist_from_center = torch.sqrt(x_grid**2 + y_grid**2)
        self.wall_density = (dist_from_center > 0.85).float().view(1, 1, res, res)
        self.orig_obstacles = self.wall_density.clone() 
        
        self.player_density = torch.zeros((1, 1, res, res), device=device)
        self.bot_density = torch.zeros((1, 1, res, res), device=device)
        
        self.combat_time = 0.0
        self.winner = None
        self.difficulty = difficulty
        self.clash_intensity = 0.0

        # --- SYMMETRIC MULTIPLAYER ACTORS REGISTRY ---
        self.actors = []
        
        # Setup Actor 0: Human or Custom Testing Bot
        if actor0_data is not None:
            p_vec = torch.tensor(actor0_data['vector'], dtype=torch.float32, device=device)
            p_vec /= (torch.norm(p_vec) + 1e-8)
            self._add_actor(
                actor_id=0,
                actor_type=actor0_data.get('type', 'bot'),
                pill_name=actor0_data.get('pill_name', 'Foundation Pill'),
                quality=actor0_data.get('quality', 100.0),
                vector=p_vec,
                spawn_pos=torch.tensor([width * 0.3, height * 0.5], dtype=torch.float32, device=device),
                custom_name=actor0_data.get('custom_name', 'Bot Alpha'),
                style=actor0_data.get('style', 'Balanced Triad'),
                desc=actor0_data.get('desc', 'Benchmark challenger.')
            )
        else:
            p_vector = torch.tensor(player_pill_data['vector'], dtype=torch.float32, device=device)
            p_vector /= (torch.norm(p_vector) + 1e-8)
            self._add_actor(
                actor_id=0,
                actor_type="player",
                pill_name=player_pill_data.get('name', 'Foundation Pill'),
                quality=player_pill_data.get('quality', 50.0),
                vector=p_vector,
                spawn_pos=torch.tensor([width * 0.3, height * 0.5], dtype=torch.float32, device=device)
            )
        
        # Setup Actor 1: Symmetrical Adversary loaded from Config Archetypes
        if actor1_data is not None:
            b_vec = torch.tensor(actor1_data['vector'], dtype=torch.float32, device=device)
            b_vec /= (torch.norm(b_vec) + 1e-8)
            self._add_actor(
                actor_id=1,
                actor_type="bot",
                pill_name=actor1_data.get('pill_name', 'Foundation Pill'),
                quality=actor1_data.get('quality', 100.0),
                vector=b_vec,
                spawn_pos=torch.tensor([width * 0.7, height * 0.5], dtype=torch.float32, device=device),
                custom_name=actor1_data.get('custom_name', 'Bot Beta'),
                style=actor1_data.get('style', 'Balanced Triad'),
                desc=actor1_data.get('desc', 'Benchmark opponent.')
            )
        else:
            bot_keys = list(SEMANTIC_PILLS_DB.keys())
            bot_pill_name = np.random.choice(bot_keys)
            b_vector = torch.tensor(SEMANTIC_PILLS_DB[bot_pill_name]['vector'], dtype=torch.float32, device=device)
            b_vector /= (torch.norm(b_vector) + 1e-8)
            bot_quality = min(99.0, 30.0 + 15.0 * difficulty)
            
            archetype_list = combat_config.BOT_ARCHETYPES.get(bot_pill_name, [{"name": "Rogue Cultivator", "style": "Standard", "desc": "An enigmatic phase-locked adversary."}])
            arch = random.choice(archetype_list)
            
            self._add_actor(
                actor_id=1,
                actor_type="bot",
                pill_name=bot_pill_name,
                quality=bot_quality,
                vector=b_vector,
                spawn_pos=torch.tensor([width * 0.7, height * 0.5], dtype=torch.float32, device=device),
                custom_name=arch["name"],
                style=arch["style"],
                desc=arch["desc"]
            )

        self._sync_legacy_properties()

    def _add_actor(self, actor_id, actor_type, pill_name, quality, vector, spawn_pos, custom_name=None, style=None, desc=None):
        """ Instantiates an actor state dictionary with perfectly symmetric physical properties """
        start_angle = 0.0 if actor_type == "player" else math.pi
        cos_a = math.cos(start_angle)
        sin_a = math.sin(start_angle)
        rot_x = self.pin_x * cos_a + self.pin_y * sin_a
        rot_y = -self.pin_x * sin_a + self.pin_y * cos_a

        pin_pos = torch.zeros((16, 2), dtype=torch.float32, device=self.device)
        pin_pos[:, 0] = spawn_pos[0] + rot_x * 1.5
        pin_pos[:, 1] = spawn_pos[1] + rot_y * 1.5
        
        base_K = (quality / 100.0) * 15.0 + 2.0
        is_slag = (quality < 40.0)
        if is_slag:
            base_K = 0.5
            
        # Symmetrically distribute physical element types along the 16 nodes (Fractally mixed slime!) [1.2.8]
        v_norm = vector / (torch.sum(vector) + 1e-8)
        num_yang = int(round(float(v_norm[0]) * 16))
        num_cat = int(round(float(v_norm[1]) * 16))
        num_yin = 16 - num_yang - num_cat
        
        node_vecs = []
        for _ in range(num_yang): node_vecs.append([1.0, 0.0, 0.0])
        for _ in range(num_cat): node_vecs.append([0.0, 1.0, 0.0])
        for _ in range(num_yin): node_vecs.append([0.0, 0.0, 1.0])
        
        while len(node_vecs) < 16: node_vecs.append([0.577, 0.577, 0.577])
        node_vecs = node_vecs[:16]
        node_vectors_tensor = torch.tensor(node_vecs, dtype=torch.float32, device=self.device)
            
        self.actors.append({
            'id': actor_id,
            'type': actor_type,
            'pill_name': pill_name,
            'custom_name': custom_name if custom_name else ("Player" if actor_type == "player" else "Rogue Cultivator"),
            'style': style if style else "Standard",
            'desc': desc if desc else "A practitioner of phase locking.",
            'quality': quality,
            'vector': vector,
            'node_vectors': node_vectors_tensor,
            'is_yang': vector[0] > vector[2],
            'is_yin': vector[2] > vector[0] and vector[2] > vector[1],
            'is_slag': is_slag,
            'is_pure_mastery': (pill_name == "Void Core" or quality >= 100.0),
            
            # Kuramoto coupled oscillator phases
            'node_phases': torch.zeros(16, dtype=torch.float32, device=self.device),
            'K': base_K,
            'K_active': base_K,
            'integrity': 1.0,
            
            # Position & deformation kinematics
            'pos': spawn_pos.clone(),
            'pin_pos': pin_pos,
            'edge_intact': torch.ones(16, dtype=torch.bool, device=self.device),
            'angle': start_angle,
            
            # Live input control registers (continuously mapped)
            'vx': 0.0,
            'vy': 0.0,
            'tq': 0.0,
            'freq_val': 0.0,
            'spatial_val': 0.0,
            'domain_charge': 0.0,
            
            # Live combat telemetry
            'shear_stress': 0.0,
            'density_val': 0.0,
            'jitter_avg': 0.0,
            'energy_absorbed': 0.0,
            'beam_active': False,
            'beam_intensity': 0.0,
            'stabilization_factor': 1.0,
            'last_snap_event': False,
            'last_heal_event': False,
            'explosions_triggered': 0
        })

    def _sync_legacy_properties(self):
        """ Maps symmetric actor dictionaries to class properties for legacy renderer compatibility """
        p_act = self.actors[0]
        b_act = self.actors[1]
        
        self.player_pill_name = p_act['pill_name']
        self.bot_pill_name = b_act['pill_name']
        self.bot_custom_name = b_act['custom_name']
        self.bot_style = b_act['style']
        self.bot_desc = b_act['desc']
        
        self.player_quality = p_act['quality']
        self.player_vector = p_act['vector']
        self.player_domain_charge = p_act['domain_charge']
        self.bot_domain_charge = b_act['domain_charge']
        
        self.is_player_yang = p_act['is_yang']
        self.is_player_yin = p_act['is_yin']
        self.is_player_slag = p_act['is_slag']
        self.is_pure_mastery = p_act['is_pure_mastery']
        
        self.player_node_phases = p_act['node_phases']
        self.bot_node_phases = b_act['node_phases']
        
        self.player_K = p_act['K']
        self.player_K_active = p_act['K_active']
        self.bot_K = b_act['K']
        self.bot_K_active = b_act['K_active']
        
        self.player_integrity = p_act['integrity']
        self.bot_integrity = b_act['integrity']
        
        self.player_freq_val = p_act['freq_val']
        self.player_spatial_val = p_act['spatial_val']
        
        self.player_shear_stress = p_act['shear_stress']
        self.bot_shear_stress = b_act['shear_stress']
        
        self.player_density_val = p_act['density_val']
        self.bot_density_val = b_act['density_val']
        
        self.player_jitter_avg = p_act['jitter_avg']
        self.bot_jitter_avg = b_act['jitter_avg']
        
        self.energy_absorbed = p_act['energy_absorbed']
        self.player_beam_active = p_act['beam_active']
        self.player_beam_intensity = p_act['beam_intensity']
        self.pill_stabilization_factor = p_act['stabilization_factor']
        
        self.player_pos = p_act['pos']
        self.bot_pos = b_act['pos']
        
        self.player_pin_pos = p_act['pin_pos']
        self.bot_pin_pos = b_act['pin_pos']
        
        self.player_edge_intact = p_act['edge_intact']
        self.bot_edge_intact = b_act['edge_intact']
        
        self.player_angle = p_act['angle']
        self.bot_angle = b_act['angle']

    def _execute_bot_ai(self, actor):
        """ AI Controller acts as input emulator, outputting strategic continuous movement based on its style """
        closest_opponent = None
        min_dist = 99999.0
        for other in self.actors:
            if other is not actor:
                d = torch.norm(other['pos'] - actor['pos']).item()
                if d < min_dist:
                    min_dist = d
                    closest_opponent = other
                    
        if closest_opponent is None:
            actor['vx'], actor['vy'], actor['tq'] = 0.0, 0.0, 0.0
            actor['freq_val'], actor['spatial_val'] = 0.0, 0.0
            return
            
        bot_dir = closest_opponent['pos'] - actor['pos']
        bot_dist = torch.norm(bot_dir).item() + 1e-5
        dir_x = bot_dir[0] / bot_dist
        dir_y = bot_dir[1] / bot_dist
        
        # Determine movement and strategy parameters based on custom bot archetype
        style = actor['style']
        difficulty_mult = 0.4 + 0.15 * self.difficulty
        difficulty_mult = min(1.2, difficulty_mult)
        
        if style == "Aggressive Yang":
            # Direct charge style to trigger frequent Yang explosions
            actor['vx'] = dir_x * 0.9 * difficulty_mult
            actor['vy'] = dir_y * 0.9 * difficulty_mult
            actor['tq'] = math.sin(self.combat_time * 5.0) * 0.3
            actor['freq_val'] = 1.0
            actor['spatial_val'] = 1.0
            
        elif style == "Speed Yang":
            # Circular orbiting patterns
            tangent_x = -dir_y
            tangent_y = dir_x
            actor['vx'] = (dir_x * 0.4 + tangent_x * 0.8) * difficulty_mult
            actor['vy'] = (dir_y * 0.4 + tangent_y * 0.8) * difficulty_mult
            actor['tq'] = 0.5 * difficulty_mult
            actor['freq_val'] = 0.8
            actor['spatial_val'] = 0.6
            
        elif style == "Defensive Yin":
            # Maintain distance, activate Shield stance to parry
            if bot_dist < 220.0:
                actor['vx'] = -dir_x * 0.6 * difficulty_mult
                actor['vy'] = -dir_y * 0.6 * difficulty_mult
                actor['spatial_val'] = -0.8
            else:
                actor['vx'] = dir_x * 0.2 * difficulty_mult
                actor['vy'] = dir_y * 0.2 * difficulty_mult
                actor['spatial_val'] = -0.3
            actor['tq'] = 0.0
            actor['freq_val'] = -1.0
            
        elif style == "Elusive Yin":
            # Actively runs away from approaching player
            if bot_dist < 300.0:
                actor['vx'] = -dir_x * 0.9 * difficulty_mult
                actor['vy'] = -dir_y * 0.9 * difficulty_mult
                actor['spatial_val'] = -0.5
            else:
                actor['vx'] = (random.random() * 2.0 - 1.0) * 0.2
                actor['vy'] = (random.random() * 2.0 - 1.0) * 0.2
                actor['spatial_val'] = -0.3
            actor['tq'] = math.sin(self.combat_time * 2.0) * 0.4
            actor['freq_val'] = -1.0
            
        elif style in ["Balanced Triad", "Steady Triad"]:
            # SMR Catalyst / Grass - locks in at 0.0 frequency
            if bot_dist < 180.0:
                actor['vx'] = -dir_x * 0.5 * difficulty_mult
                actor['vy'] = -dir_y * 0.5 * difficulty_mult
                actor['spatial_val'] = 0.0
            else:
                actor['vx'] = dir_x * 0.6 * difficulty_mult
                actor['vy'] = dir_y * 0.6 * difficulty_mult
                actor['spatial_val'] = 0.0
            actor['tq'] = math.sin(self.combat_time) * 0.3
            actor['freq_val'] = 0.0
            
        elif style == "Vortex Spinner":
            # Extreme rotational current strategy
            actor['vx'] = dir_x * 0.3 * difficulty_mult
            actor['vy'] = dir_y * 0.3 * difficulty_mult
            actor['tq'] = 1.0 * difficulty_mult
            actor['freq_val'] = 0.5
            actor['spatial_val'] = -0.2
            
        elif style == "Chaotic Warp":
            # Rapid random warp adjustments
            actor['vx'] = (random.random() * 2.0 - 1.0) * 0.95 * difficulty_mult
            actor['vy'] = (random.random() * 2.0 - 1.0) * 0.95 * difficulty_mult
            actor['tq'] = (random.random() * 2.0 - 1.0) * 0.8 * difficulty_mult
            actor['freq_val'] = math.sin(self.combat_time * 12.0)
            actor['spatial_val'] = math.cos(self.combat_time * 8.0)
            
        else:
            actor['vx'] = dir_x * 0.5 * difficulty_mult
            actor['vy'] = dir_y * 0.5 * difficulty_mult
            actor['tq'] = 0.0
            actor['freq_val'] = 0.0
            actor['spatial_val'] = 0.0

    def _inject_domain_waves(self, pos, vector, tq, dt):
        """ Injects the entity's phase vector into the complex fluid to establish Domain """
        px_g = (pos[0] / self.WIDTH) * self.res
        py_g = (pos[1] / self.HEIGHT) * self.res
        dx = torch.remainder(self.x_indices - px_g + self.res/2, self.res) - self.res/2
        dy = torch.remainder(self.y_indices - py_g + self.res/2, self.res) - self.res/2
        mask = torch.exp(-(dx**2 + dy**2) / (12.0**2))
        
        rate = 85.0 * dt  
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
                # Deposit massive non-linear phase spikes to desynchronize the opponent
                self.density_complex[0, c*2]   += mask * 1.5 * scale * vector[c]
                self.density_complex[0, c*2+1] += mask * 1.5 * scale * vector[c]
                
        force_dir = 1.0 if is_yang else -1.0
        self.u[0, 0] += (dx / dist) * mask * 1200.0 * scale * force_dir
        self.v[0, 0] += (dy / dist) * mask * 1200.0 * scale * force_dir

    def _apply_120_jet_eeg_physics(self, eeg_c0_spectrum, pin_pos, node_phases, dt):
        """ 
        Translates raw continuous BCI matrices into 16 localized boundary thruster jets.
        DECENTRALIZED NODE-GRADIENT FLUID PROJECTOR (TRUE HETERARCHY):
        - Radial node thrust proportional to phase-locking: Sum_j C_ij cos(Phi_i - Phi_j).
        - Tangential boundary shear current proportional to local phase gradient: Sum_j C_ij sin(Phi_i - Phi_j).
        """
        if eeg_c0_spectrum is None:
            return
            
        c0_total = torch.mean(eeg_c0_spectrum[:16, :16, :], dim=2) 
        com = pin_pos.mean(dim=0)
        
        # Radial vector from COM directly to each of the 16 nodes (boundary thrusters)
        radial_x = pin_pos[:, 0] - com[0]
        radial_y = pin_pos[:, 1] - com[1]
        dist_nodes = torch.sqrt(radial_x**2 + radial_y**2) + 1e-5
        
        normal_x = radial_x / dist_nodes
        normal_y = radial_y / dist_nodes
        
        # Tangential vector for circular boundary currents (Whirlpools)
        tangent_x = -normal_y
        tangent_y = normal_x
        
        # Reconstruct real-time 16x16 cross-node phase difference matrices
        phase_diff_matrix = node_phases.unsqueeze(1) - node_phases.unsqueeze(0)
        
        # Non-linear wave-field weights
        radial_weight = c0_total * torch.cos(phase_diff_matrix)
        tangent_weight = c0_total * torch.sin(phase_diff_matrix)
        
        # Sum connections per node to get integrated boundary thrust amplitudes
        node_radial_amp = torch.sum(radial_weight, dim=1)
        node_tangent_amp = torch.sum(tangent_weight, dim=1)
        
        # Compute boundary forces: in-phase creates expansion, phase-gradients drive boundary torque
        f_x = (node_radial_amp * normal_x + node_tangent_amp * tangent_x) * 85.0
        f_y = (node_radial_amp * normal_y + node_tangent_amp * tangent_y) * 85.0
        
        # Project 16 boundary forces onto the 2D solver grid
        pin_gx = torch.remainder((pin_pos[:, 0] / self.WIDTH) * self.res, self.res)
        pin_gy = torch.remainder((pin_pos[:, 1] / self.HEIGHT) * self.res, self.res)
        
        dx_shape = torch.remainder(self.x_indices.unsqueeze(0) - pin_gx.reshape(16, 1, 1) + self.res/2, self.res) - self.res/2
        dy_shape = torch.remainder(self.y_indices.unsqueeze(0) - pin_gy.reshape(16, 1, 1) + self.res/2, self.res) - self.res/2
        
        node_influence = torch.exp(-(dx_shape**2 + dy_shape**2) / (5.0))
        node_inf_norm = node_influence / (torch.sum(node_influence, dim=(1, 2), keepdim=True) + 1e-8)
        
        self.u[0, 0] += torch.sum(node_inf_norm * f_x.reshape(16, 1, 1), dim=0) * dt
        self.v[0, 0] += torch.sum(node_inf_norm * f_y.reshape(16, 1, 1), dim=0) * dt

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
        
        S_p = torch.clamp(torch.sum(grid_vec_norm * self.actors[0]['vector'].view(3, 1, 1), dim=0), 0.0, 1.0)
        S_b = torch.clamp(torch.sum(grid_vec_norm * self.actors[1]['vector'].view(3, 1, 1), dim=0), 0.0, 1.0)
        
        # 1. CHAOTIC FRACTAL CLASH INTERFACE
        clash_mask = torch.clamp(4.0 * S_p * S_b, 0.0, 1.0)
        self.clash_intensity = torch.mean(clash_mask).item()
        
        bifurcation_wave_u = torch.sin(self.density_complex[0, 0] * 8.0) * torch.cos(self.density_complex[0, 4] * 5.0)
        bifurcation_wave_v = torch.cos(self.density_complex[0, 2] * 8.0) * torch.sin(self.density_complex[0, 0] * 5.0)
        
        self.u[0, 0] += combat_config.SCRAMBLE_BROKEN_NODE_SCALE * 10.0 * clash_mask * bifurcation_wave_u * 45.0 * dt
        self.v[0, 0] += combat_config.SCRAMBLE_BROKEN_NODE_SCALE * clash_mask * bifurcation_wave_v * 45.0 * dt
        
        # 2. SECTOR DOMINANCE PHYSICS
        player_dominates = (S_p > S_b).float()
        bot_dominates = (S_b >= S_p).float()
        
        if self.actors[0]['is_yang']:
            self.u[0, 0] += player_dominates * (-self.y_indices + self.res/2) * 0.25 * dt
            self.v[0, 0] += player_dominates * (self.x_indices - self.res/2) * 0.25 * dt
        elif self.actors[0]['is_yin']:
            self.u[0, 0] *= (1.0 - player_dominates * 0.35 * dt)
            self.v[0, 0] *= (1.0 - player_dominates * 0.35 * dt)
        else:
            self.u[0, 0] *= (1.0 - player_dominates * 0.12 * dt)
            self.v[0, 0] *= (1.0 - player_dominates * 0.12 * dt)

        if self.actors[1]['is_yang']:
            self.u[0, 0] += bot_dominates * (-self.y_indices + self.res/2) * 0.25 * dt
            self.v[0, 0] += bot_dominates * (self.x_indices - self.res/2) * 0.25 * dt
        elif self.actors[1]['is_yin']:
            self.u[0, 0] *= (1.0 - bot_dominates * 0.35 * dt)
            self.v[0, 0] *= (1.0 - bot_dominates * 0.35 * dt)

    def _evaluate_cognitive_esports_loop_for_actor(self, act, eeg_c0_spectrum, is_real_data, dt):
        """ Symmetrically evaluates cognitive stabilizer loops and assist states for each actor """
        if act['type'] == "bot" or not is_real_data or eeg_c0_spectrum is None:
            # Gamepad/AI assist path: Synthesize stable spatial-spectral layouts
            act['assist_mode_active'] = True
            synthetic_matrix = torch.zeros((16, 16, 1), device=self.device)
            
            if act['spatial_val'] < -0.3:
                # Synthesize defensive Shield Assist matrix
                for i in range(16):
                    synthetic_matrix[i, (i+1)%16, 0] = 0.8
                    synthetic_matrix[(i+1)%16, i, 0] = 0.8
                act['assist_profile'] = "Shield Assist"
            elif act['spatial_val'] > 0.3:
                # Synthesize aggressive Core Assist matrix
                for i in range(16):
                    synthetic_matrix[i, (i+8)%16, 0] = 0.75
                act['assist_profile'] = "Core Assist"
            else:
                act['assist_profile'] = "Mesh Assist"
                for i in range(16):
                    idx_n = (i + 1) % 16
                    synthetic_matrix[i, idx_n, 0] = 0.45
                    synthetic_matrix[idx_n, i, 0] = 0.45

            act['K_active'] = act['K']
            act['beam_active'] = False
            act['beam_intensity'] = 0.0
            act['stabilization_factor'] = 0.50
            return synthetic_matrix

        # FreeEEG16 path: multi-frequency spectral evaluation
        act['assist_mode_active'] = False
        act['assist_profile'] = "FreeEEG16 Native"
        
        theta_coh = torch.mean(eeg_c0_spectrum[:, :, 2:5], dim=2)
        smr_coh   = torch.mean(eeg_c0_spectrum[:, :, 6:10], dim=2)
        gamma_coh = torch.mean(eeg_c0_spectrum[:, :, 15:51], dim=2)
        
        t_amp = torch.mean(torch.abs(theta_coh)).item()
        s_amp = torch.mean(torch.abs(smr_coh)).item()
        g_amp = torch.mean(torch.abs(gamma_coh)).item()
        
        real_time_vector = torch.tensor([g_amp, s_amp, t_amp], dtype=torch.float32, device=self.device)
        norm_real = torch.norm(real_time_vector) + 1e-8
        real_time_vector = real_time_vector / norm_real
        
        lens_focus_similarity = torch.sum(real_time_vector * act['vector']).item()
        lens_focus_similarity = max(0.01, min(1.0, lens_focus_similarity))
        
        act['K_active'] = act['K'] + (lens_focus_similarity * 48.0)
        act['stabilization_factor'] = lens_focus_similarity
        
        if lens_focus_similarity > 0.65 and g_amp > 0.04:
            beam_multiplier = 2.0 if act['is_pure_mastery'] else 1.0
            act['beam_active'] = True
            act['beam_intensity'] = (lens_focus_similarity - 0.65) * 18.0 * beam_multiplier
            
            # Focused beam target: locate the closest opponent and apply phase noise
            closest_opp = None
            min_dist = 99999.0
            for other in self.actors:
                if other is not act:
                    d = torch.norm(other['pos'] - act['pos']).item()
                    if d < min_dist:
                        min_dist = d
                        closest_opp = other
            if closest_opp:
                closest_opp['domain_charge'] = max(0.0, closest_opp['domain_charge'] - dt * act['beam_intensity'] * 1.5)
                # Reduced focused beam phase scrambling to keep battles tactically engaging
                scramble_drift = (torch.rand(16, device=self.device) * 2.0 - 1.0) * act['beam_intensity'] * combat_config.BEAM_SCRAMBLE_SCALE
                closest_opp['node_phases'] += scramble_drift * dt
        else:
            act['beam_active'] = False
            act['beam_intensity'] = 0.0
            
        return eeg_c0_spectrum

    def _evaluate_kuramoto_neural_decoherence(self, act, dt):
        """ Models connectome health symmetrically for any actor using Kuramoto Coupling """
        px_norm = torch.clamp((act['pin_pos'][:, 0] / self.WIDTH) * 2.0 - 1.0, -1.0, 1.0)
        py_norm = torch.clamp((act['pin_pos'][:, 1] / self.HEIGHT) * 2.0 - 1.0, -1.0, 1.0)
        grid_uv = torch.stack([px_norm, py_norm], dim=1).view(1, 1, 16, 2)
        
        r_re = F.grid_sample(self.density_complex[:, 0:1], grid_uv, align_corners=True).squeeze()
        r_im = F.grid_sample(self.density_complex[:, 1:2], grid_uv, align_corners=True).squeeze()
        g_re = F.grid_sample(self.density_complex[:, 2:3], grid_uv, align_corners=True).squeeze()
        g_im = F.grid_sample(self.density_complex[:, 3:4], grid_uv, align_corners=True).squeeze()
        b_re = F.grid_sample(self.density_complex[:, 4:5], grid_uv, align_corners=True).squeeze()
        b_im = F.grid_sample(self.density_complex[:, 5:6], grid_uv, align_corners=True).squeeze()
        
        local_m = torch.sqrt(r_re**2 + r_im**2 + g_re**2 + g_im**2 + b_re**2 + b_im**2)
        local_vec = torch.stack([torch.hypot(r_re, r_im), torch.hypot(g_re, g_im), torch.hypot(b_re, b_im)], dim=1)
        local_vec_norm = local_vec / (torch.norm(local_vec, dim=1, keepdim=True) + 1e-8)
        
        # Symmetrical node-wise element similarity based on individual node elements (Fractally mixed nodes!) [1.2.8]
        similarity = torch.sum(local_vec_norm * act['node_vectors'], dim=1)
        
        # Pure cosine-distance based dissonance mapping for continuous, organic wave interaction [Bruña et al., 2018]
        dissonance = torch.clamp(1.0 - similarity, 0.0, 1.0)
        
        # --- GPU-NATIVE SPATIAL CLASH GRADIENT OF REAL WAVE INTERFERENCE FIELDS ---
        # Calculate instantaneous real propagating wave values [1, 1, res, res] for all 3 channels
        time_factor = self.combat_time * 12.0
        wave_R = self.density_complex[:, 0:1] * math.cos(time_factor) - self.density_complex[:, 1:2] * math.sin(time_factor)
        wave_G = self.density_complex[:, 2:3] * math.cos(time_factor * 0.5) - self.density_complex[:, 3:4] * math.sin(time_factor * 0.5)
        wave_B = self.density_complex[:, 4:5] * math.cos(time_factor * 0.25) - self.density_complex[:, 5:6] * math.sin(time_factor * 0.25)
        
        wave_R_pad = F.pad(wave_R, (1, 1, 1, 1), mode='circular')
        wave_G_pad = F.pad(wave_G, (1, 1, 1, 1), mode='circular')
        wave_B_pad = F.pad(wave_B, (1, 1, 1, 1), mode='circular')
        
        # Compute horizontal/vertical spatial derivatives of propagating waves for all 3 channels
        grad_x_R = 0.5 * (wave_R_pad[:, :, 1:-1, 2:] - wave_R_pad[:, :, 1:-1, :-2])
        grad_y_R = 0.5 * (wave_R_pad[:, :, 2:, 1:-1] - wave_R_pad[:, :, :-2, 1:-1])
        grad_x_G = 0.5 * (wave_G_pad[:, :, 1:-1, 2:] - wave_G_pad[:, :, 1:-1, :-2])
        grad_y_G = 0.5 * (wave_G_pad[:, :, 2:, 1:-1] - wave_G_pad[:, :, :-2, 1:-1])
        grad_x_B = 0.5 * (wave_B_pad[:, :, 1:-1, 2:] - wave_B_pad[:, :, 1:-1, :-2])
        grad_y_B = 0.5 * (wave_B_pad[:, :, 2:, 1:-1] - wave_B_pad[:, :, :-2, 1:-1])
        
        R_grad = torch.sqrt(grad_x_R**2 + grad_y_R**2)
        G_grad = torch.sqrt(grad_x_G**2 + grad_y_G**2)
        B_grad = torch.sqrt(grad_x_B**2 + grad_y_B**2)
        
        # Symmetrically calculate cross-channel clashing based on sharp inter-element spatial overlaps (fractal boundaries) [1.1.1, 1.2.3]
        clash_field = (
            R_grad * torch.abs(wave_B_pad[:, :, 1:-1, 1:-1]) + B_grad * torch.abs(wave_R_pad[:, :, 1:-1, 1:-1]) +
            R_grad * torch.abs(wave_G_pad[:, :, 1:-1, 1:-1]) + G_grad * torch.abs(wave_R_pad[:, :, 1:-1, 1:-1]) +
            G_grad * torch.abs(wave_B_pad[:, :, 1:-1, 1:-1]) + B_grad * torch.abs(wave_G_pad[:, :, 1:-1, 1:-1])
        )
        clash_gradient = F.grid_sample(clash_field, grid_uv, align_corners=True).squeeze()

        # Calculate proximity damage
        proximity_damage = 0.0
        
        # Symmetrical Rock-Paper-Scissors (RPS) element advantage check (Fully Unified from Config)
        # 0 = Yang (Red/Fire), 1 = Catalyst (Green/Grass), 2 = Yin (Blue/Water)
        elem_act = torch.argmax(act['node_vectors'], dim=1) # [16]
        rps_mult = torch.ones(16, device=self.device)
        
        for other in self.actors:
            if other is not act:
                dist_opp = torch.norm(act['pos'] - other['pos']).item()
                clash_f = max(0.0, (220.0 - dist_opp) / 220.0)
                proximity_damage += clash_f * combat_config.PROXIMITY_DAMAGE_SCALE
                
                opp_core = torch.argmax(other['vector']).item()
                
                # Advantage condition: Act node beats Opp Core element
                has_advantage = (elem_act == 0) & (opp_core == 1) | \
                                (elem_act == 1) & (opp_core == 2) | \
                                (elem_act == 2) & (opp_core == 0)
                                
                has_disadvantage = (elem_act == 1) & (opp_core == 0) | \
                                   (elem_act == 2) & (opp_core == 1) | \
                                   (elem_act == 0) & (opp_core == 2)
                                      
                rps_mult = torch.where(has_advantage, torch.tensor(0.50, device=self.device), rps_mult)
                rps_mult = torch.where(has_disadvantage, torch.tensor(1.65, device=self.device), rps_mult)
                
        # Symmetrically project spatial clash gradient into disruption force
        disruption_force = ((dissonance * local_m * 4.0) + clash_gradient * 38.0 + proximity_damage) * rps_mult
        
        mean_sim = torch.mean(similarity).item()
        mean_sim = max(0.0, min(1.0, mean_sim))
        
        if act['type'] == "bot":
            # Symmetrically boost bot's coupling based on difficulty, quality, and domain proximity from Config
            bot_assist = combat_config.K_BOT_ASSIST_BASE + (act['quality'] / 100.0) * combat_config.K_BOT_ASSIST_QUALITY_SCALE
            act['K_active'] = act['K'] + bot_assist + mean_sim * combat_config.K_BOT_ASSIST_SIMILARITY_SCALE
            act['stabilization_factor'] = 0.5 + (act['quality'] / 100.0) * 0.3 + mean_sim * 0.2
            act['stabilization_factor'] = max(0.1, min(1.0, act['stabilization_factor']))
            stab_factor = act['stabilization_factor']
        else:
            if act.get('assist_mode_active', False):
                # Gamepad assist gets a symmetrical baseline K_active boost
                act['K_active'] = act['K'] + 35.0
                act['stabilization_factor'] = 0.75
            stab_factor = act['stabilization_factor']
            if act['is_pure_mastery']:
                stab_factor = 0.05
                
        quality = act['quality']
        quality_mod = 1.0 if not act['is_slag'] else 0.05
        
        # Calculate dynamic physical jitter force, clamped using Config clamp to prevent boolean evaluations on vectors
        jitter_force = (disruption_force * (100.0 / (quality * quality_mod + 1e-5)) * combat_config.JITTER_FORCE_MULTIPLIER) * (1.0 - stab_factor * 0.70)
        jitter_force = torch.clamp(jitter_force, 0.0, combat_config.JITTER_FORCE_MAX_CLAMP)
        
        act['shear_stress'] = torch.mean(dissonance).item()
        act['density_val'] = torch.mean(local_m).item()
        act['jitter_avg'] = torch.mean(jitter_force).item()
        
        # --- RECEPTIVE NEUTRAL ABSORPTION / PARRIES (FULLY SYMMETRIC) ---
        is_absorbing = False
        if (act['is_yin'] or not act['is_yang']) and not act['is_slag']:
            if act['spatial_val'] < -0.3:
                is_absorbing = True
                
        if is_absorbing:
            act['energy_absorbed'] += float(torch.mean(jitter_force).item()) * 0.12 * (act['stabilization_factor'] + 0.1)
            if act['energy_absorbed'] >= 12.0:
                if (~act['edge_intact']).any():
                    broken_nodes = torch.nonzero(~act['edge_intact']).squeeze(1)
                    target_idx = broken_nodes[0].item()
                    act['edge_intact'][target_idx] = True
                    act['last_heal_event'] = True
                act['energy_absorbed'] = 0.0
            
            # Attenuate physical displacement force during parry state to secure structure
            jitter_force = jitter_force * 0.08

        # --- KURAMOTO PHASE UPDATES ---
        phases = act['node_phases']
        phases += 14.0 * 2 * math.pi * dt
        
        # Symmetrical Bidirectional Circular Ring Coupling for superior physical stability (prevents numerical overshoots)
        idx_next = torch.remainder(torch.arange(16, device=self.device) + 1, 16)
        idx_prev = torch.remainder(torch.arange(16, device=self.device) - 1, 16)
        coupling = torch.sin(phases[idx_next] - phases) + torch.sin(phases[idx_prev] - phases)
        
        # Clamp coupling coefficient to mathematically secure Euler integration stability (K * dt <= 0.40)
        K = min(35.0, act['K_active'])
        phases += K * 0.5 * coupling * dt
        
        # Symmetrically scaled phase noise with RPS advantage applied for progressive 15-35s encounters
        num_broken = (~act['edge_intact']).sum().item()
        scramble_rate = ((dissonance * local_m * 1.5) + (num_broken * 0.15) + clash_gradient * 6.5) * rps_mult
        
        # Parry filter absorbs 75% of phase noise, preventing immediate desynchronization
        if is_absorbing:
            scramble_rate = scramble_rate * 0.25
            
        phase_noise = (torch.rand(16, device=self.device) * 2.0 - 1.0) * scramble_rate
        phases += phase_noise * dt
        
        phases_clamped = torch.remainder(phases + math.pi, 2 * math.pi) - math.pi
        act['node_phases'].copy_(phases_clamped)
        
        cos_sum = torch.cos(phases_clamped).mean()
        sin_sum = torch.sin(phases_clamped).mean()
        integrity = torch.sqrt(cos_sum**2 + sin_sum**2).item()
        act['integrity'] = max(0.01, min(1.0, integrity))
        
        angles = torch.rand(16, device=self.device) * 2.0 * math.pi
        offset_x = torch.cos(angles) * jitter_force
        offset_y = torch.sin(angles) * jitter_force
        
        if act['type'] == "player" and act['domain_charge'] > 0.1 and act['spatial_val'] < -0.4:
            backlash = act['domain_charge'] * 120.0
            offset_x += (random.random() * 2.0 - 1.0) * backlash
            offset_y += (random.random() * 2.0 - 1.0) * backlash
            act['last_snap_event'] = True
            
        return torch.stack([offset_x, offset_y], dim=1)

    def step(self, dt, time_sec, eeg_c0_spectrum, eeg_vx, eeg_vy, eeg_tq, is_real_data, compression, scale_factor, eeg_freqs=None, alch_freq=0.0, alch_spatial=0.0):
        self.combat_time += dt
        if self.winner is not None:
            return 

        p_blend = max(-1.0, min(1.0, compression))
        p_scale = scale_factor
        p_node_radius = 8.0 - p_blend * 4.0

        b_blend = 0.0
        b_scale = 1.25
        b_node_radius = 8.0

        # --- 1. STATELESS COVARIANCE ORIENTATION CALCULATIONS FOR ALL ACTORS ---
        for act in self.actors:
            act['angle'], com = calculate_covariance_angle(
                act['pin_pos'], self.pin_x, self.pin_y, act['angle'], act['tq'], dt, self.device
            )
            act['pos'].copy_(com)

        # --- 2. EXTRACT & TRANSLATE INTENTS TO LOCALISED PHASE DRIFTS & FORCE INTEGRATION ---
        for act in self.actors:
            if act['type'] == "player":
                act['vx'] = eeg_vx
                act['vy'] = eeg_vy
                act['tq'] = eeg_tq
                act['freq_val'] = float(alch_freq)
                act['spatial_val'] = float(alch_spatial)
                act_c0_spec = self._evaluate_cognitive_esports_loop_for_actor(act, eeg_c0_spectrum, is_real_data, dt)
                
                # Symmetrically project player movement-induced forces onto fluid grid (Unified)
                self.u, self.v = apply_unified_actor_forces(
                    self.device, self.res, self.WIDTH, self.HEIGHT, act['pin_pos'], self.pin_x, self.pin_y, act['angle'],
                    act['edge_intact'], self.u, self.v, self.wall_density, act['vx'], act['vy'], act['tq'], dt, p_blend,
                    p_node_radius, self.x_indices, self.y_indices, self.player_density,
                    45.0, 120.0
                )
            else:
                # Bot AI logic translates its tactics to virtual controller axes
                self._execute_bot_ai(act)
                act_c0_spec = self._evaluate_cognitive_esports_loop_for_actor(act, None, False, dt)
                
                # Symmetrically project bot forces onto fluid grid (Unified)
                self.u, self.v = apply_unified_actor_forces(
                    self.device, self.res, self.WIDTH, self.HEIGHT, act['pin_pos'], self.pin_x, self.pin_y, act['angle'],
                    act['edge_intact'], self.u, self.v, self.wall_density, act['vx'], act['vy'], act['tq'], dt, b_blend,
                    b_node_radius, self.x_indices, self.y_indices, self.bot_density,
                    45.0, 120.0
                )
                
            # Only apply movement-induced phase drift if not in assist mode to prevent self-destruction
            if not act.get('assist_mode_active', False):
                tq_drift = act['tq'] * combat_config.TORQUE_PHASE_DRIFT_SPEED * dt
                act['node_phases'] += tq_drift * torch.arange(16, device=self.device) / 16.0
                
                if abs(act['vx']) > 0.05 or abs(act['vy']) > 0.05:
                    angle_intent = math.atan2(-act['vy'], act['vx'])
                    node_angles = torch.atan2(self.pin_y, self.pin_x)
                    projection = torch.cos(node_angles - angle_intent)
                    act['node_phases'] += projection * combat_config.MOVEMENT_PHASE_DRIFT_SPEED * dt

        # --- 3. DYNAMIC DOMAIN EXPLOSIONS & PASSIVE AURAS ---
        for act in self.actors:
            # Dynamic target projections based on continuous ratio matching [1.2.8]
            target_f = float(act['vector'][0] * 1.0 + act['vector'][2] * (-1.0))
            target_s = float(act['vector'][0] * 1.0 + act['vector'][2] * (-1.0))
            
            if not act['is_slag']:
                # Continuous distance matching in spatial-frequency BCI coordinates
                freq_match = abs(act['freq_val'] - target_f) < 0.65
                spatial_match = abs(act['spatial_val'] - target_s) < 0.65
                
                # Symmetrical unstable phase transition for Chaotic Warp bot (random outbursts)
                if act['style'] == "Chaotic Warp" and random.random() < 0.015:
                    act['domain_charge'] = 1.0
                    
                if freq_match and spatial_match:
                    act['domain_charge'] += dt * 1.8
                    if act['domain_charge'] >= 1.0:
                        self._inject_domain_explosion(act['pos'], act['vector'], 2.0, act['is_yang'])
                        act['explosions_triggered'] += 1
                        act['domain_charge'] = 0.0
                else:
                    act['domain_charge'] = max(0.0, act['domain_charge'] - dt * 0.8)

            # Symmetrically deposit local elements from all 16 individual nodes on GPU
            # Makes mixed slimes paint gorgeous, multi-colored, rotating fractal wave trails! [1.2.8]
            pin_gx = ((act['pin_pos'][:, 0] / self.WIDTH) * self.res).long()
            pin_gy = ((act['pin_pos'][:, 1] / self.HEIGHT) * self.res).long()
            
            valid_nodes = (pin_gx > 0) & (pin_gx < self.res - 1) & (pin_gy > 0) & (pin_gy < self.res - 1) & act['edge_intact']
            if valid_nodes.any():
                gx_n = pin_gx[valid_nodes]
                gy_n = pin_gy[valid_nodes]
                vec_n = act['node_vectors'][valid_nodes] # [N_active, 3]
                
                # Apply local coordinate mask grids centered on each node
                dx = self.x_indices.unsqueeze(0) - gx_n.reshape(-1, 1, 1)
                dy = self.y_indices.unsqueeze(0) - gy_n.reshape(-1, 1, 1)
                mask = torch.exp(-(dx**2 + dy**2) / (5.0**2)) # [N_active, res, res]
                
                # Symmetrically deposit on the 3 channels
                for c in range(3):
                    weight = vec_n[:, c].reshape(-1, 1, 1) * mask * 2.2 * dt
                    self.density_complex[0, c*2] = torch.clamp(self.density_complex[0, c*2] + torch.sum(weight, dim=0), -2.5, 2.5)

        # --- 4. FLUID SOLVER ADVECTION & INTER-PHASE MODULATION ---
        self.u = torch.nan_to_num(self.u, nan=0.0) * 0.90 
        self.v = torch.nan_to_num(self.v, nan=0.0) * 0.90
        # Highly reduced damping factor (from 0.95 to 0.994) to allow long-lived, high-contrast, folding fractal structures! [1.2.2]
        self.density_complex = torch.nan_to_num(self.density_complex, nan=0.0) * 0.994

        R_re, R_im = self.density_complex[0, 0], self.density_complex[0, 1]
        G_re, G_im = self.density_complex[0, 2], self.density_complex[0, 3]
        B_re, B_im = self.density_complex[0, 4], self.density_complex[0, 5]
        
        amp_R = torch.hypot(R_re, R_im) + 1e-8
        amp_G = torch.hypot(G_re, G_im) + 1e-8
        amp_B = torch.hypot(B_re, B_im) + 1e-8
        
        phase_R = torch.atan2(R_im, R_re)
        phase_B = torch.atan2(B_im, B_re)
        
        phase_diff = phase_R - phase_B
        lock_in_force = amp_G * torch.sin(phase_diff) * 28.0 * dt
        
        # --- GPU IMMISCIBILITY & CAHN-HILLIARD BOUNDARY SHARPENING ---
        # Symmetrically forces all three element fields (Red, Green, Blue) to remain unmixed
        # and form spectacular crisp fractal boundaries under chaotic advection.
        amp_R_4d = torch.hypot(self.density_complex[:, 0:1], self.density_complex[:, 1:2]) + 1e-8
        amp_G_4d = torch.hypot(self.density_complex[:, 2:3], self.density_complex[:, 3:4]) + 1e-8
        amp_B_4d = torch.hypot(self.density_complex[:, 4:5], self.density_complex[:, 5:6]) + 1e-8
        
        R_pad = F.pad(amp_R_4d, (1, 1, 1, 1), mode='circular')
        G_pad = F.pad(amp_G_4d, (1, 1, 1, 1), mode='circular')
        B_pad = F.pad(amp_B_4d, (1, 1, 1, 1), mode='circular')
        
        laplacian_R = R_pad[:, :, 1:-1, 2:] + R_pad[:, :, 1:-1, :-2] + R_pad[:, :, 2:, 1:-1] + R_pad[:, :, :-2, 1:-1] - 4.0 * amp_R_4d
        laplacian_G = G_pad[:, :, 1:-1, 2:] + G_pad[:, :, 1:-1, :-2] + G_pad[:, :, 2:, 1:-1] + G_pad[:, :, :-2, 1:-1] - 4.0 * amp_G_4d
        laplacian_B = B_pad[:, :, 1:-1, 2:] + B_pad[:, :, 1:-1, :-2] + B_pad[:, :, 2:, 1:-1] + B_pad[:, :, :-2, 1:-1] - 4.0 * amp_B_4d
        
        # Textbook binary Cahn-Hilliard Phase separation potentials with 0.12 total density noise gating
        # Prevents background erasure while creating stunning sharp folding fractals at the clashing interfaces [1.2.8]
        mask_active = (amp_R_4d + amp_G_4d + amp_B_4d > 0.12).float()
        
        sharpening_R = (amp_R_4d * (amp_R_4d - 0.3) * (1.0 - amp_R_4d) + 0.10 * laplacian_R) * mask_active
        sharpening_G = (amp_G_4d * (amp_G_4d - 0.3) * (1.0 - amp_G_4d) + 0.10 * laplacian_G) * mask_active
        sharpening_B = (amp_B_4d * (amp_B_4d - 0.3) * (1.0 - amp_B_4d) + 0.10 * laplacian_B) * mask_active
        
        self.density_complex[:, 0:2] = torch.clamp(self.density_complex[:, 0:2] + sharpening_R * 4.0 * dt, -2.5, 2.5)
        self.density_complex[:, 2:4] = torch.clamp(self.density_complex[:, 2:4] + sharpening_G * 4.0 * dt, -2.5, 2.5)
        self.density_complex[:, 4:6] = torch.clamp(self.density_complex[:, 4:6] + sharpening_B * 4.0 * dt, -2.5, 2.5)

        pac_coupling = self.actors[0]['stabilization_factor'] if is_real_data else 0.5
        amp_R_new = amp_R * (1.0 + pac_coupling * amp_B * torch.cos(phase_B) * dt * 2.0)
        
        phase_R_new = phase_R - lock_in_force
        phase_B_new = phase_B + lock_in_force
        
        self.density_complex[0, 0] = amp_R_new * torch.cos(phase_R_new)
        self.density_complex[0, 1] = amp_R_new * torch.sin(phase_R_new)
        self.density_complex[0, 4] = amp_B * torch.cos(phase_B_new)
        self.density_complex[0, 5] = amp_B * torch.sin(phase_B_new)

        self._apply_domain_rule_imposition(dt)

        # --- 5. PHYSICAL STRESS & TENSION DETECTOR FOR ALL ACTORS ---
        for act in self.actors:
            disruption = self._evaluate_kuramoto_neural_decoherence(act, dt)
            act['pin_pos'][act['edge_intact']] += disruption[act['edge_intact']] * dt

        # --- 6. WIN / LOSS COHESION STATE MONITOR ---
        player_alive = self.actors[0]['integrity'] >= 0.20
        bot_alive = self.actors[1]['integrity'] >= 0.20
        
        if not player_alive and not bot_alive:
            self.winner = "Draw"
        elif not player_alive:
            self.winner = "Rogue Cultivator"
        elif not bot_alive:
            self.winner = "Player"

        # --- 7. SYMMETRIC POSITION UPDATES & REDIRECT ---
        for act in self.actors:
            blend_val = p_blend if act['type'] == "player" else b_blend
            scale_val = p_scale if act['type'] == "player" else b_scale
            
            # Symmetrically update slime boundaries using identical physical laws
            act['pin_pos'], act['edge_intact'], new_com = update_unified_slime_kinematics(
                self.device, self.res, self.WIDTH, self.HEIGHT, act['pin_pos'], self.pin_x, self.pin_y, act['angle'],
                act['edge_intact'], self.u, self.v, self.wall_density, act['tq'], dt, scale_val, blend_val, {}, 25.0
            )
            act['pos'].copy_(new_com)

        # Symmetrically inject phase noise from ruptured nodes completely on the GPU
        for act in self.actors:
            broken_mask = ~act['edge_intact']
            if broken_mask.any():
                broken_nodes = act['pin_pos'][broken_mask]
                gx_b = ((broken_nodes[:, 0] / self.WIDTH) * self.res).long()
                gy_b = ((broken_nodes[:, 1] / self.HEIGHT) * self.res).long()
                
                valid_mask = (gx_b > 0) & (gx_b < self.res - 1) & (gy_b > 0) & (gy_b < self.res - 1)
                if valid_mask.any():
                    gx_valid = gx_b[valid_mask]
                    gy_valid = gy_b[valid_mask]
                    
                    # Compute pseudo-random high-entropy noise arrays on GPU
                    noise = (torch.rand((6, gy_valid.shape[0]), device=self.device) * 2.0 - 1.0) * 0.05 * dt
                    for c in range(6):
                        self.density_complex[0, c, gy_valid, gx_valid] += noise[c]

        # Safeguard Clamping limits
        self.density_complex = torch.clamp(self.density_complex, -2.5, 2.5)
        self.u = torch.clamp(self.u, -65.0, 65.0)
        self.v = torch.clamp(self.v, -65.0, 65.0)

        # --- 8. ADVECTION & DIVERGENCE PROJECTION OF WORLD GRID ---
        self.density_complex = self.solver.advect(self.density_complex, self.u, self.v, dt)
        
        # Calculate instantaneous real propagating wave fronts with interference fringes
        time_factor = self.combat_time * 12.0
        wave_R = self.density_complex[:, 0:1] * math.cos(time_factor) - self.density_complex[:, 1:2] * math.sin(time_factor)
        wave_G = self.density_complex[:, 2:3] * math.cos(time_factor * 0.5) - self.density_complex[:, 3:4] * math.sin(time_factor * 0.5)
        wave_B = self.density_complex[:, 4:5] * math.cos(time_factor * 0.25) - self.density_complex[:, 5:6] * math.sin(time_factor * 0.25)
        
        # Render high-contrast self-similar interference stripes (physical fractals on GPU) [1.1.1, 1.2.3]
        R = torch.abs(wave_R)
        G = torch.abs(wave_G)
        B = torch.abs(wave_B)
        
        self.density = torch.cat([R, G, B], dim=1) 
        self.density = torch.clamp(self.density, 0.0, 1.0)
        
        self.u = self.solver.advect(self.u, self.u, self.v, dt)
        self.v = self.solver.vvect(self.v, self.u, self.v, dt) if hasattr(self.solver, 'vvect') else self.solver.advect(self.v, self.u, self.v, dt)
        self.u, self.v = self.solver.project(self.u, self.v, self.wall_density, None)
        
        self._update_render_density(self.actors[0]['pin_pos'], self.player_density, self.actors[0]['edge_intact'])
        self._update_render_density(self.actors[1]['pin_pos'], self.bot_density, self.actors[1]['edge_intact'])

        # --- 9. SYNCHRONISE BACK TO LEGACY PROPERTIES FOR COMPATIBILITY ---
        self._sync_legacy_properties()

    def _update_render_density(self, pin_pos, density_tensor, edge_intact_mask):
        active_nodes = pin_pos[edge_intact_mask]
        if active_nodes.numel() == 0:
            density_tensor.zero_()
            return
        gx = torch.remainder(active_nodes[:, 0] / self.WIDTH * self.res, self.res)
        gy = torch.remainder(active_nodes[:, 1] / self.HEIGHT * self.res, self.res)
        
        dx = torch.remainder(self.x_indices.unsqueeze(0) - gx.reshape(-1, 1, 1) + self.res/2, self.res) - self.res/2
        dy = torch.remainder(self.y_indices.unsqueeze(0) - gy.reshape(-1, 1, 1) + self.res/2, self.res) - self.res/2
        influence = torch.exp(-(dx**2 + dy**2) / 10.0)
        density_tensor.copy_(torch.max(influence, dim=0, keepdim=True)[0].unsqueeze(0))
