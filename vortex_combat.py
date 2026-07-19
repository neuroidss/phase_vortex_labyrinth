# vortex_combat.py
import torch
import torch.nn.functional as F
import math
import numpy as np
import random
import os
from vortex_fluid import FluidSolver
from implicit_config import COORDS_16_X, COORDS_16_Y
from vortex_softbody import update_neighbor_springs, apply_cohesion_constraint
from vortex_unified_physics import calculate_covariance_angle, update_unified_slime_kinematics
import combat_config
from spectral_material_engine import SpectralMaterialEngine

# =====================================================================
# MANDATE: Never use single-point approximations for character positions or actions.
# Always resolve actions and fields through the complete 16-node distributed
# spatial coordinate mapping (holographic micro-array principle).
#
# STRICT LOCALITY MANDATE: No action-at-a-distance (no teleports, no instant
# non-local health/integrity restoration, no direct coordinate pulling). All forces
# and restorative effects must propagate strictly through fluid advection and 
# direct node-to-grid field sampling.
#
# PROHIBITION OF FRACTIONAL PHYSICS: The physical fields (velocity fields u, v
# and the 100-channel spectral density grid) are strictly unified, objective, and
# team-blind. Any attempt to duplicate or split the physical field into team-specific
# fractions is strictly prohibited.
# =====================================================================

class PhaseVortexCombat:
    """
    Auto-Battler Arena Module.
    Simulates multi-actor squad combat using active, viscoelastic metamaterials.
    Features ONE unified team-blind Navier-Stokes fluid field on GPU.
    Damage is strictly mechanical, arising from fluid shear rates and physical clashes.
    """
    def __init__(self, device, width, height, res, team0_data, team1_data, difficulty=1):
        self.device = device
        self.WIDTH, self.HEIGHT = width, height
        self.res = res
        self.solver = FluidSolver(res, device)
        
        self.material_engine = SpectralMaterialEngine(res, device)
        
        self.y_indices, self.x_indices = torch.meshgrid(
            torch.arange(res, device=device, dtype=torch.float32),
            torch.arange(res, device=device, dtype=torch.float32), indexing='ij'
        )
        
        self.pin_x = torch.tensor(COORDS_16_X, dtype=torch.float32, device=device)
        self.pin_y = torch.tensor(COORDS_16_Y, dtype=torch.float32, device=device)
        
        self.u = torch.zeros((1, 1, res, res), device=device)
        self.v = torch.zeros((1, 1, res, res), device=device)
        
        # Symmetrically unified team-blind spectral field on GPU
        self.density_spectral = torch.zeros((1, 100, res, res), device=device)
        
        self.team0_density = torch.zeros((1, 1, res, res), device=device)
        self.team1_density = torch.zeros((1, 1, res, res), device=device)
        
        y_grid, x_grid = torch.meshgrid(torch.linspace(-1, 1, res, device=device), torch.linspace(-1, 1, res, device=device), indexing='ij')
        dist_from_center = torch.sqrt(x_grid**2 + y_grid**2)
        self.wall_density = (dist_from_center > 0.85).float().view(1, 1, res, res)
        
        self.combat_time = 0.0
        self.winner = None
        self.difficulty = difficulty

        self.actors = []
        self.floating_texts = []
        self.projectiles = []
        
        self.predicted_wp = 50.0
        self.implied_vol = 0.1
        self.drift_mu = 0.0
        
        # TACTICAL ENVIRONMENT EVENT LOG
        self.combat_log = ["CHAMPIONSHIP STARTED."]
        
        self.log_filepath = "battle_prediction_log.csv"
        self.log_file = open(self.log_filepath, "w")
        self.log_file.write("Time_Sec,T0_Integ,T1_Integ,Spot_S,Drift_Mu,Vol_Sigma,WP_Pct,Theta_Decay\n")
        self.log_file.flush()
        
        for i, unit in enumerate(team0_data):
            v = torch.tensor(unit['vector'], dtype=torch.float32, device=device)
            v /= (torch.norm(v) + 1e-8)
            spawn_x = width * 0.2 + (i % 2) * 40.0
            spawn_y = height * 0.3 + i * (height * 0.4 / max(1, len(team0_data)-1))
            self._add_actor(actor_id=f"t0_{i}", team=0, data=unit, vector=v, spawn_pos=torch.tensor([spawn_x, spawn_y], dtype=torch.float32, device=device))

        for i, unit in enumerate(team1_data):
            v = torch.tensor(unit['vector'], dtype=torch.float32, device=device)
            v /= (torch.norm(v) + 1e-8)
            spawn_x = width * 0.8 - (i % 2) * 40.0
            spawn_y = height * 0.3 + i * (height * 0.4 / max(1, len(team1_data)-1))
            self._add_actor(actor_id=f"t1_{i}", team=1, data=unit, vector=v, spawn_pos=torch.tensor([spawn_x, spawn_y], dtype=torch.float32, device=device))

    def __del__(self):
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

    def log_event(self, msg):
        """ Logs unique wave events to the terminal HUD. """
        if not self.combat_log or self.combat_log[-1] != msg:
            self.combat_log.append(msg)
            if len(self.combat_log) > 10:
                self.combat_log.pop(0)

    def _add_actor(self, actor_id, team, data, vector, spawn_pos):
        start_angle = 0.0 if team == 0 else math.pi
        cos_a, sin_a = math.cos(start_angle), math.sin(start_angle)
        rot_x = self.pin_x * cos_a + self.pin_y * sin_a
        rot_y = -self.pin_x * sin_a + self.pin_y * cos_a

        pin_pos = torch.zeros((16, 2), dtype=torch.float32, device=self.device)
        pin_pos[:, 0] = spawn_pos[0] + rot_x * 1.5
        pin_pos[:, 1] = spawn_pos[1] + rot_y * 1.5
        
        quality = data.get('quality', 90.0)
        base_K = (quality / 100.0) * 40.0 + 15.0
        
        style = data.get('style', 'Fighter')
        
        node_angles = torch.atan2(self.pin_y, self.pin_x) 
        cos_angles = torch.cos(node_angles)
        
        is_front = cos_angles > 0.3
        is_back  = cos_angles < -0.3
        is_side  = torch.abs(cos_angles) <= 0.3

        # Symmetrical 100-channel continuous power spectrum with a realistic 1/f background noise floor
        node_spectra = torch.zeros(16, 100, dtype=torch.float32, device=self.device)
        freqs = torch.linspace(1.0, 100.0, 100, device=self.device)
        
        # 1/f power-law background noise (non-zero baseline across all 100 bins)
        baseline = 1.0 / (freqs + 2.0)
        
        # Primary oscillatory band peaks
        theta_peak = torch.exp(-((freqs - 6.0)**2)/16.0)  # Theta (4-8Hz, Cohesion/Shields)
        beta_peak  = torch.exp(-((freqs - 24.0)**2)/16.0) # Beta (18-36Hz, Propulsion/Thrust)
        gamma_peak = torch.exp(-((freqs - 80.0)**2)/16.0) # Gamma (60-100Hz, Solitons/Offense)
        
        # Symmetrically initialize baseline noise on all 16 nodes
        node_spectra[:] = baseline.unsqueeze(0) * 1.5

        if style == 'Fighter':
            # Fighter (High Beta, agile movement with decent Theta armor)
            node_spectra[is_front] += beta_peak * 3.5 + theta_peak * 1.5 + gamma_peak * 0.5
            node_spectra[is_back]  += beta_peak * 4.0 + theta_peak * 1.2 + gamma_peak * 0.3
            node_spectra[is_side]  += beta_peak * 3.0 + theta_peak * 1.8 + gamma_peak * 0.4
        elif style == 'Mage':
            # Mage (High Gamma, ranged soliton streams with protective Theta/Beta baseline)
            node_spectra[is_front] += gamma_peak * 4.5 + theta_peak * 1.5 + beta_peak * 0.8
            node_spectra[is_back]  += gamma_peak * 3.5 + theta_peak * 1.8 + beta_peak * 0.6
            node_spectra[is_side]  += gamma_peak * 4.0 + theta_peak * 1.6 + beta_peak * 0.7
        elif style == 'Tank':
            # Tank (High Theta, massive structural defense and stable advection)
            node_spectra[is_front] += theta_peak * 4.5 + beta_peak * 1.5 + gamma_peak * 0.3
            node_spectra[is_back]  += theta_peak * 4.0 + beta_peak * 1.2 + gamma_peak * 0.2
            node_spectra[is_side]  += theta_peak * 4.2 + beta_peak * 1.0 + gamma_peak * 0.2
        else: # Support / Healer
            # Healer (Balanced Triad with enhanced Theta coherence stabilization)
            node_spectra[is_front] += theta_peak * 3.0 + beta_peak * 1.5 + gamma_peak * 1.5
            node_spectra[is_back]  += theta_peak * 2.8 + beta_peak * 1.2 + gamma_peak * 1.2
            node_spectra[is_side]  += theta_peak * 3.2 + beta_peak * 1.4 + gamma_peak * 1.4
            
        self.actors.append({
            'id': actor_id, 'team': team, 'is_dead': False,
            'custom_name': data.get('custom_name', f"Unit {actor_id}"),
            'style': style,
            'quality': quality, 'vector': vector,
            'node_spectra': node_spectra,
            'node_phases': torch.zeros(16, dtype=torch.float32, device=self.device),
            'K': base_K, 'K_active': base_K, 'integrity': 1.0,
            'pos': spawn_pos.clone(), 'pin_pos': pin_pos,
            'edge_intact': torch.ones(16, dtype=torch.bool, device=self.device),
            'angle': start_angle, 'vx': 0.0, 'vy': 0.0, 'tq': 0.0,
            'shear_stress': 0.0, 'explosions_triggered': 0,
            'is_player': data.get('is_player', False),
            'target_pos': None,
            'target_name': "None",
            'ai_state_desc': "IDLE",
            'attack_cooldown': 0.0,
            'ult_charge': 0.0,
            'blitz_timer': 0.0,
            'sinkhole_timer': 0.0
        })

    def _execute_bot_ai(self, actor):
        if actor['is_dead']: return
        
        enemies = [a for a in self.actors if a['team'] != actor['team'] and not a['is_dead']]
        if not enemies:
            actor['vx'], actor['vy'], actor['tq'] = 0.0, 0.0, 0.0
            return

        # Purely physical target-seeking AI: find the closest enemy and move towards them
        closest_enemy = min(enemies, key=lambda e: torch.norm(e['pos'] - actor['pos']).item())
        dir_vec = closest_enemy['pos'] - actor['pos']
        dist = torch.norm(dir_vec).item() + 1e-5
        
        actor['target_pos'] = closest_enemy['pos']
        actor['target_name'] = closest_enemy['custom_name']
        
        difficulty_mult = min(1.2, 0.6 + 0.15 * self.difficulty)
        
        # Determine movement speed based on role
        if actor['style'] == "Tank":
            speed_mult = 0.55
            actor['ai_state_desc'] = "GUARD"
        elif actor['style'] == "Mage":
            # Ranged: stay at a distance of 180 pixels, don't run into walls
            if dist > 180.0:
                speed_mult = 0.5
            elif dist < 120.0:
                speed_mult = -0.3
            else:
                speed_mult = 0.0
            actor['ai_state_desc'] = "CAST"
        elif actor['style'] == "Healer":
            # Support: stay close to allies first, or move to closest enemy if alone
            allies = [a for a in self.actors if a['team'] == actor['team'] and not a['is_dead'] and a is not actor]
            if allies:
                closest_ally = min(allies, key=lambda a: torch.norm(a['pos'] - actor['pos']).item())
                dir_ally = closest_ally['pos'] - actor['pos']
                dist_ally = torch.norm(dir_ally).item() + 1e-5
                dir_vec = dir_ally
                dist = dist_ally
                speed_mult = 0.65 if dist_ally > 50.0 else 0.0
                actor['target_pos'] = closest_ally['pos']
                actor['target_name'] = closest_ally['custom_name']
                actor['ai_state_desc'] = "SUPPORT"
            else:
                speed_mult = 0.4
                actor['ai_state_desc'] = "FIGHT"
        else: # Fighter
            speed_mult = 0.85
            # Double move speed during active blitz ultimate
            if actor.get('blitz_timer', 0.0) > 0.0:
                speed_mult = 2.0
            actor['ai_state_desc'] = "BLITZ" if actor.get('blitz_timer', 0.0) > 0.0 else "STRIKE"
            
        actor['vx'] = (dir_vec[0] / dist).item() * speed_mult * difficulty_mult
        actor['vy'] = (dir_vec[1] / dist).item() * speed_mult * difficulty_mult
        actor['tq'] = math.sin(self.combat_time * 2.0) * 0.2

    def _apply_softbody_repulsion(self, dt):
        for i, act_a in enumerate(self.actors):
            if act_a['is_dead']: continue
            for j, act_b in enumerate(self.actors):
                if j <= i or act_b['is_dead']: continue
                diff = act_a['pos'] - act_b['pos']
                dist = torch.norm(diff).item() + 1e-5
                if dist < combat_config.BODY_COLLISION_RADIUS:
                    overlap = combat_config.BODY_COLLISION_RADIUS - dist
                    
                    # Safe physical displacement to prevent teleports
                    push_magnitude = overlap * combat_config.BODY_REPULSION_STIFFNESS * dt
                    push_magnitude_clamped = min(1.5, push_magnitude)
                    
                    push_force = (diff / dist) * push_magnitude_clamped
                    act_a['pin_pos'] += push_force
                    act_b['pin_pos'] -= push_force

                    # PHYSICAL FLUID REBOUND IMPULSE:
                    # Only hostile collisions (enemies) trigger the violent, damaging fluid rebound blast!
                    if act_a['team'] != act_b['team']:
                        # Rebound velocities are fully distributed around the actual nodes of both slimes
                        rebound_dir = diff / dist
                        impulse_strength = 25.0 * dt * 20.0
                        
                        pin_gx_a = ((act_a['pin_pos'][:, 0] / self.WIDTH) * self.res).reshape(16, 1, 1)
                        pin_gy_a = ((act_a['pin_pos'][:, 1] / self.HEIGHT) * self.res).reshape(16, 1, 1)
                        dx_a = self.x_indices.unsqueeze(0) - pin_gx_a
                        dy_a = self.y_indices.unsqueeze(0) - pin_gy_a
                        node_mask_a = torch.exp(-(dx_a**2 + dy_a**2) / (3.0**2))
                        sum_mask_a = torch.sum(node_mask_a * act_a['edge_intact'].float().view(16, 1, 1), dim=0, keepdim=True)
                        
                        pin_gx_b = ((act_b['pin_pos'][:, 0] / self.WIDTH) * self.res).reshape(16, 1, 1)
                        pin_gy_b = ((act_b['pin_pos'][:, 1] / self.HEIGHT) * self.res).reshape(16, 1, 1)
                        dx_b = self.x_indices.unsqueeze(0) - pin_gx_b
                        dy_b = self.y_indices.unsqueeze(0) - pin_gy_b
                        node_mask_b = torch.exp(-(dx_b**2 + dy_b**2) / (3.0**2))
                        sum_mask_b = torch.sum(node_mask_b * act_b['edge_intact'].float().view(16, 1, 1), dim=0, keepdim=True)
                        
                        combined_mask = torch.clamp(sum_mask_a + sum_mask_b, 0.0, 1.0)
                        
                        self.u += combined_mask * rebound_dir[0].item() * impulse_strength * 0.1
                        self.v += combined_mask * rebound_dir[1].item() * impulse_strength * 0.1
                        
                        # Ingest a burst of high-frequency Gamma indicating a physical crash
                        self.density_spectral[:, 80:85] += combined_mask * 8.0 * dt

    def step(self, dt, time_sec, eeg_c0_spectrum, eeg_vx, eeg_vy, eeg_tq, is_real_data, compression, scale_factor, eeg_freqs=None, alch_freq=0.0, alch_spatial=0.0):
        self.combat_time += dt
        if self.winner is not None: 
            if self.log_file: 
                self.log_file.close()
                self.log_file = None
            return 

        alive_texts = []
        for ft in self.floating_texts:
            ft['life'] -= dt
            ft['pos'][1] -= 25.0 * dt 
            if ft['life'] > 0:
                alive_texts.append(ft)
        self.floating_texts = alive_texts

        p_blend = max(-1.0, min(1.0, compression))
        p_scale = scale_factor
        
        # --- UPDATE ULTIMATES & COOLDOWNS ---
        for act in self.actors:
            if act['is_dead']: continue
            act['attack_cooldown'] = max(0.0, act.get('attack_cooldown', 0.0) - dt)
            act['blitz_timer'] = max(0.0, act.get('blitz_timer', 0.0) - dt)
            act['sinkhole_timer'] = max(0.0, act.get('sinkhole_timer', 0.0) - dt)
            
            # Ult charge rate (passive + scaled by active shear rates)
            charge_rate = 0.03 + act['shear_stress'] * 0.005
            act['ult_charge'] = min(1.0, act.get('ult_charge', 0.0) + charge_rate * dt)
            
            # --- TRIGGER CLASS SPECIFIC ULTIMATES ---
            if act['ult_charge'] >= 1.0:
                act['ult_charge'] = 0.0
                self.log_event(f"--- {act['custom_name'].upper()} CASTS ULTIMATE! ---")
                
                if act['style'] == "Mage":
                    # Mage Ultimate: Launch a massive supernova soliton wave packet
                    closest_enemy = min([e for e in self.actors if e['team'] != act['team'] and not e['is_dead']], key=lambda e: torch.norm(e['pos'] - act['pos']).item(), default=None)
                    if closest_enemy:
                        dir_attack = closest_enemy['pos'] - act['pos']
                        dist_attack = torch.norm(dir_attack).item() + 1e-5
                        dir_attack_norm = dir_attack / dist_attack
                        self.projectiles.append({
                            'pos': act['pos'].clone(),
                            'vel': dir_attack_norm * 180.0,
                            'team': act['team'],
                            'radius': 30.0,
                            'style': 'supernova',
                            'life': 4.0
                        })
                elif act['style'] == "Fighter":
                    # Fighter Ultimate: Activate Hyper-Drive Blitz thrusting
                    act['blitz_timer'] = 3.0
                elif act['style'] == "Tank":
                    # Tank Ultimate: Activate gravitational confinement sinkhole
                    act['sinkhole_timer'] = 3.0
                elif act['style'] == "Healer":
                    # Healer Ultimate: Resonance Sanctuary physically emits a massive calming shockwave across the grid
                    pin_gx = ((act['pin_pos'][:, 0] / self.WIDTH) * self.res).long().clamp(0, self.res - 1)
                    pin_gy = ((act['pin_pos'][:, 1] / self.HEIGHT) * self.res).long().clamp(0, self.res - 1)
                    valid = act['edge_intact']
                    
                    if valid.any():
                        gx_n, gy_n = pin_gx[valid], pin_gy[valid]
                        dx = self.x_indices.unsqueeze(0) - gx_n.reshape(-1, 1, 1)
                        dy = self.y_indices.unsqueeze(0) - gy_n.reshape(-1, 1, 1)
                        
                        # Wide, slow dispersing mask around each node
                        mask = torch.exp(-(dx**2 + dy**2) / (6.0**2)) 
                        sum_mask = torch.sum(mask, dim=0, keepdim=True)
                        
                        # Ingest extreme, stable Theta support density into the unified grid
                        self.density_spectral[:, 4:12] += sum_mask * 12.0 * dt
                        
                        # Gently disperse the plume outwards in a non-destructive way
                        dy_m = self.y_indices.unsqueeze(0) - gy_n.reshape(-1, 1, 1)
                        dx_m = self.x_indices.unsqueeze(0) - gx_n.reshape(-1, 1, 1)
                        dist_m = torch.sqrt(dx_m**2 + dy_m**2) + 1e-5
                        dir_x_m = dx_m / dist_m
                        dir_y_m = dy_m / dist_m
                        
                        # Slow, outward flow to distribute the mist
                        dispersion_x = torch.sum(dir_x_m * mask, dim=0, keepdim=True) * 15.0 * dt
                        dispersion_y = torch.sum(dir_y_m * mask, dim=0, keepdim=True) * 15.0 * dt
                        self.u += dispersion_x
                        self.v += dispersion_y
                        
                        self.log_event("RESONANCE PLUME INJECTED!")

        # --- UPDATE ACTIVE PROJECTILE ENTITIES ---
        remaining_projectiles = []
        for proj in self.projectiles:
            proj['pos'] += proj['vel'] * dt
            proj['life'] -= dt
            
            # Map projectile coordinates to the grid
            gx = int((proj['pos'][0] / self.WIDTH) * self.res)
            gy = int((proj['pos'][1] / self.HEIGHT) * self.res)
            
            if 2 <= gx < self.res-2 and 2 <= gy < self.res-2:
                # Continuous advection of soliton energy along the grid path
                if proj['style'] == 'supernova':
                    self.density_spectral[0, 80:90, gy-3:gy+4, gx-3:gx+4] += 15.0 * dt
                    self.u[0, 0, gy-3:gy+4, gx-3:gx+4] += proj['vel'][0].item() * 0.1
                    self.v[0, 0, gy-3:gy+4, gx-3:gx+4] += proj['vel'][1].item() * 0.1
                else:
                    self.density_spectral[0, 80:85, gy-1:gy+2, gx-1:gx+2] += 6.0 * dt
                    self.u[0, 0, gy-1:gy+2, gx-1:gx+2] += proj['vel'][0].item() * 0.15
                    self.v[0, 0, gy-1:gy+2, gx-1:gx+2] += proj['vel'][1].item() * 0.15
            
            # Detect collisions with opposing team actors
            impacted = False
            for enemy in self.actors:
                if enemy['team'] != proj['team'] and not enemy['is_dead']:
                    dist_enemy = torch.norm(enemy['pos'] - proj['pos']).item()
                    if dist_enemy < combat_config.BODY_COLLISION_RADIUS * 1.1:
                        impacted = True
                        
                        # DIRECT HEAVY DAMAGE FROM MAGICAL PROJECTILES (Magic hits with high impact)
                        proj_damage = 0.0
                        if proj['style'] == 'supernova':
                            proj_damage = 0.40  # Massive ultimate area-of-effect nuke
                        elif proj['style'] == 'normal_soliton':
                            proj_damage = 0.16  # High-damage Mage standard projectile
                        else:  # defensive_soliton
                            proj_damage = 0.08  # Smaller Healer utility projectile
                            
                        # Apply direct damage immediately to the enemy
                        enemy['integrity'] = max(0.0, enemy['integrity'] - proj_damage)
                        if enemy['integrity'] <= 0.0:
                            enemy['is_dead'] = True
                            self.log_event(f"{enemy['custom_name']} WAS ANNIHILATED!")
                        
                        # Ingest massive multi-directional shockwave explosion directly into the Navier-Stokes velocity grid
                        if 2 <= gx < self.res-2 and 2 <= gy < self.res-2:
                            dy, dx = torch.meshgrid(torch.arange(5, device=self.device) - 2, torch.arange(5, device=self.device) - 2, indexing='ij')
                            dist_blast = torch.sqrt(dx**2 + dy**2).float() + 1e-5
                            dir_x = (dx / dist_blast)
                            dir_y = (dy / dist_blast)
                            
                            blast_strength = 120.0 if proj['style'] == 'supernova' else 40.0
                            self.u[0, 0, gy-2:gy+3, gx-2:gx+3] += dir_x * blast_strength
                            self.v[0, 0, gy-2:gy+3, gx-2:gx+3] += dir_y * blast_strength
                            self.density_spectral[0, 80:90, gy-2:gy+3, gx-2:gx+3] += blast_strength * 0.5
                        
                        self.log_event(f"Soliton detonated on {enemy['custom_name'][:4]}")
                        break
                        
            if not impacted and proj['life'] > 0.0:
                remaining_projectiles.append(proj)
        self.projectiles = remaining_projectiles

        # --- UPDATE BEHAVIORS & PHYSICS STEPS ---
        for act in self.actors:
            if not act['is_dead']:
                act['K_active'] += (act['K'] - act.get('K_active', act['K'])) * dt * 1.5
                if act.get('is_player', False) and is_real_data:
                    act['vx'], act['vy'], act['tq'] = eeg_vx, eeg_vy, eeg_tq
                    act['ai_state_desc'] = "MANUAL"
                else:
                    self._execute_bot_ai(act)

        # --- HOLOGRAPHIC SPECIFIC CLASS ATTACKS ---
        for act in self.actors:
            if act['is_dead']: continue
            pin_gx = ((act['pin_pos'][:, 0] / self.WIDTH) * self.res).long().clamp(0, self.res - 1)
            pin_gy = ((act['pin_pos'][:, 1] / self.HEIGHT) * self.res).long().clamp(0, self.res - 1)
            valid = act['edge_intact']
            
            if valid.any():
                gx_n, gy_n = pin_gx[valid], pin_gy[valid]
                dx = self.x_indices.unsqueeze(0) - gx_n.reshape(-1, 1, 1)
                dy = self.y_indices.unsqueeze(0) - gy_n.reshape(-1, 1, 1)
                
                mask = torch.exp(-(dx**2 + dy**2) / (2.5**2)) 
                active_spectra = act['node_spectra'][valid] 
                
                spec_volume = torch.einsum('vc, vhw -> chw', active_spectra, mask)
                self.density_spectral[0] += spec_volume * 2.5 * dt

            # Node coordinate differences for spatial projection
            pin_gx_node = ((act['pin_pos'][:, 0] / self.WIDTH) * self.res).reshape(16, 1, 1)
            pin_gy_node = ((act['pin_pos'][:, 1] / self.HEIGHT) * self.res).reshape(16, 1, 1)
            dx_node = self.x_indices.unsqueeze(0) - pin_gx_node
            dy_node = self.y_indices.unsqueeze(0) - pin_gy_node
            
            # 16-channel spatial node footprint
            node_mask = torch.exp(-(dx_node**2 + dy_node**2) / (3.0**2))
            is_active = act['edge_intact'].float().view(16, 1, 1)
            active_mask = node_mask * is_active
            sum_mask = torch.sum(active_mask, dim=0, keepdim=True)

            closest_enemy = min([e for e in self.actors if e['team'] != act['team'] and not e['is_dead']], key=lambda e: torch.norm(e['pos'] - act['pos']).item(), default=None)
            
            if act['style'] == "Mage":
                # Mages fire soliton projectiles towards target on cooldown
                if closest_enemy and act['attack_cooldown'] <= 0.0:
                    dir_attack = closest_enemy['pos'] - act['pos']
                    dist_attack = torch.norm(dir_attack).item() + 1e-5
                    vel_proj = (dir_attack / dist_attack) * 250.0
                    self.projectiles.append({
                        'pos': act['pos'].clone(),
                        'vel': vel_proj,
                        'team': act['team'],
                        'radius': 12.0,
                        'style': 'normal_soliton',
                        'life': 3.0
                    })
                    act['attack_cooldown'] = 1.5

                # Symmetrically project baseline flow velocity
                if closest_enemy:
                    dir_attack = closest_enemy['pos'] - act['pos']
                    dist_attack = torch.norm(dir_attack).item() + 1e-5
                    dir_attack_norm = dir_attack / dist_attack
                    jet_speed = 8.0 * (act['quality'] / 100.0)
                    self.u += sum_mask * dir_attack_norm[0].item() * jet_speed * dt * 0.4
                    self.v += sum_mask * dir_attack_norm[1].item() * jet_speed * dt * 0.4
                    self.density_spectral[:, 80:85] += sum_mask * 1.5 * dt

            elif act['style'] == "Fighter":
                # Fighters perform a physical Beta-thrust dash uniformly from all active nodes
                if closest_enemy:
                    dir_attack = closest_enemy['pos'] - act['pos']
                    dist_attack = torch.norm(dir_attack).item() + 1e-5
                    dir_attack_norm = dir_attack / dist_attack
                    
                    thrust_speed = 10.0 * (act['quality'] / 100.0)
                    if act.get('blitz_timer', 0.0) > 0.0:
                        thrust_speed *= 3.0 # Ultimate hyper speed
                    
                    self.u += sum_mask * dir_attack_norm[0].item() * thrust_speed * dt * 0.4
                    self.v += sum_mask * dir_attack_norm[1].item() * thrust_speed * dt * 0.4
                    self.density_spectral[:, 24:28] += sum_mask * 1.2 * dt

            elif act['style'] == "Tank":
                # Tanks emit a dense, highly viscous Theta field and a distributed physical kinetic bash
                self.density_spectral[:, 4:9] += sum_mask * 2.5 * dt
                if closest_enemy:
                    dir_attack = closest_enemy['pos'] - act['pos']
                    dist_attack = torch.norm(dir_attack).item() + 1e-5
                    dir_attack_norm = dir_attack / dist_attack
                    push_speed = 8.0 * (act['quality'] / 100.0)
                    self.u += sum_mask * dir_attack_norm[0].item() * push_speed * dt * 0.25
                    self.v += sum_mask * dir_attack_norm[1].item() * push_speed * dt * 0.25

                # Tank Ultimate: Gravitational confinement sinkhole
                if act.get('sinkhole_timer', 0.0) > 0.0:
                    tx = (act['pos'][0] / self.WIDTH) * self.res
                    ty = (act['pos'][1] / self.HEIGHT) * self.res
                    dx = tx - self.x_indices
                    dy = ty - self.y_indices
                    dist_grid = torch.sqrt(dx**2 + dy**2) + 1e-5
                    
                    pull_mask = torch.exp(-dist_grid / 15.0)
                    dir_x = dx / dist_grid
                    dir_y = dy / dist_grid
                    
                    torque_x = -dir_y
                    torque_y = dir_x
                    
                    self.u[0, 0] += (dir_x * 45.0 + torque_x * 25.0) * pull_mask * dt
                    self.v[0, 0] += (dir_y * 45.0 + torque_y * 25.0) * pull_mask * dt

            elif act['style'] == "Healer":
                # Healers physically pacify fluid velocity uniformly across all active nodes
                self.u *= (1.0 - sum_mask * 0.2 * dt * 10.0)
                self.v *= (1.0 - sum_mask * 0.2 * dt * 10.0)
                self.density_spectral[:, 6:12] += sum_mask * 1.0 * dt
                
                # Healers can also fire tiny, low-velocity defensive solitons on cooldown
                if closest_enemy and act['attack_cooldown'] <= 0.0:
                    dir_attack = closest_enemy['pos'] - act['pos']
                    dist_attack = torch.norm(dir_attack).item() + 1e-5
                    vel_proj = (dir_attack / dist_attack) * 120.0
                    self.projectiles.append({
                        'pos': act['pos'].clone(),
                        'vel': vel_proj,
                        'team': act['team'],
                        'radius': 8.0,
                        'style': 'defensive_soliton',
                        'life': 2.0
                    })
                    act['attack_cooldown'] = 2.0

        # Environment density limit
        self.density_spectral = torch.clamp(self.density_spectral, 0.0, 5.0)

        self._apply_softbody_repulsion(dt)

        self.u = torch.nan_to_num(self.u, nan=0.0) * 0.90 
        self.v = torch.nan_to_num(self.v, nan=0.0) * 0.90
        
        # Increased damping (0.90) so the environment clears instantly of stagnant "haze"!
        self.density_spectral = torch.nan_to_num(self.density_spectral, nan=0.0) * 0.90

        global_integrity = sum(a['integrity'] for a in self.actors if not a['is_dead']) / max(1, len(self.actors))
        
        viscosity, surface_tension, buoyancy = self.material_engine.compute_rheology_fields(
            self.density_spectral, integrity_map=global_integrity
        )

        self.density_spectral = self.apply_negentropic_fractal_boundaries(self.density_spectral, self.u, self.v, dt)

        # Semi-implicit integration of drag/viscosity for unconditional numerical stability
        self.u = (self.u + buoyancy[:, 0:1] * dt) / (1.0 + viscosity * dt)
        self.v = (self.v + buoyancy[:, 1:2] * dt) / (1.0 + viscosity * dt)
        
        self.density_spectral = self.solver.advect(self.density_spectral, self.u, self.v, dt)
        self.u = self.solver.advect(self.u, self.u, self.v, dt)
        self.v = self.solver.vvect(self.v, self.u, self.v, dt) if hasattr(self.solver, 'vvect') else self.solver.advect(self.v, self.u, self.v, dt)
        self.u, self.v = self.solver.project(self.u, self.v, self.wall_density, None)

        theta_band = torch.sum(self.density_spectral[:, 4:9, :, :], dim=1, keepdim=True)
        beta_band = torch.sum(self.density_spectral[:, 18:37, :, :], dim=1, keepdim=True)
        gamma_band = torch.sum(self.density_spectral[:, 60:101, :, :], dim=1, keepdim=True)
        self.density = torch.clamp(torch.cat([beta_band, theta_band, gamma_band], dim=1), 0.0, 1.0)
        
        old_integrity = {act['id']: act['integrity'] for act in self.actors}

        # Calculate team Combat Power (CP) for OPTIONAL DAMAGE SCALING
        t0_power = sum(a['quality'] for a in self.actors if a['team'] == 0)
        t1_power = sum(a['quality'] for a in self.actors if a['team'] == 1)
        
        ratio_t0 = t0_power / (t1_power + 1e-5) 
        ratio_t1 = t1_power / (t0_power + 1e-5) 

        # --- FIX: UNIFIED ENVIRONMENT SHEAR RATE CALCULATION ---
        u_pad = F.pad(self.u, (1, 1, 1, 1), mode='replicate')
        v_pad = F.pad(self.v, (1, 1, 1, 1), mode='replicate')
        du_dy = 0.5 * (u_pad[:, :, 2:, 1:-1] - u_pad[:, :, :-2, 1:-1])
        dv_dx = 0.5 * (v_pad[:, :, 1:-1, 2:] - v_pad[:, :, 1:-1, :-2])
        du_dx = 0.5 * (u_pad[:, :, 1:-1, 2:] - u_pad[:, :, 1:-1, :-2])
        
        # Unified non-linear shear stress field
        shear_field = torch.sqrt((du_dy + dv_dx)**2 + 4.0 * du_dx**2)

        for act in self.actors:
            if act['is_dead']: continue
            blend_val = compression if act.get('is_player', False) else 0.0
            scale_val = p_scale if act.get('is_player', False) else 1.25
            
            act['pin_pos'], act['edge_intact'], new_com = update_unified_slime_kinematics(
                self.device, self.res, self.WIDTH, self.HEIGHT, act['pin_pos'], self.pin_x, self.pin_y, act['angle'],
                act['edge_intact'], self.u, self.v, self.wall_density, act['tq'], dt, scale_val, blend_val, {}, 25.0,
                is_captured_mask=None, bci_mode='3_axis', eeg_c0_spectrum=eeg_c0_spectrum, eeg_freqs=eeg_freqs, vx=act['vx'], vy=act['vy']
            )
            act['pos'].copy_(new_com)
            act['angle'], _ = calculate_covariance_angle(act['pin_pos'], self.pin_x, self.pin_y, act['angle'], act['tq'], 0.0, self.device)

            # --- ELEMENTAL IMMUNITY: COSINE SIMILARITY (Spectral Embedding) ---
            px_norm = torch.clamp((act['pin_pos'][:, 0] / self.WIDTH) * 2.0 - 1.0, -1.0, 1.0)
            py_norm = torch.clamp((act['pin_pos'][:, 1] / self.HEIGHT) * 2.0 - 1.0, -1.0, 1.0)
            grid_uv = torch.stack([px_norm, py_norm], dim=1).view(1, 1, 16, 2)
            
            # FIX 2: Sample shear rate at the slime node locations
            local_shear = F.grid_sample(shear_field, grid_uv, align_corners=True).squeeze() # [16]
            node_theta_emission = torch.sum(act['node_spectra'][:, 4:9], dim=1) # [16]
            
            # Sample local Theta-band density from the physical grid at the node locations
            theta_density_grid = torch.sum(self.density_spectral[:, 4:9, :, :], dim=1, keepdim=True)
            local_grid_theta = F.grid_sample(theta_density_grid, grid_uv, align_corners=True).squeeze()
            
            # Total Theta defense is node-level emission + sampled grid-level density
            effective_theta = node_theta_emission + local_grid_theta * 1.5
            
            # Theta viscosity slice dampens the velocity gradient (scaled down from 100.0 to 6.0)
            effective_shear = torch.clamp(local_shear * 6.0 - effective_theta * 0.4, min=0.0)
            act['shear_stress'] = torch.mean(effective_shear).item()
            
            # High local grid Theta density suppresses phase noise and stabilizes the phase network
            phase_noise_suppression = 1.0 / (1.0 + torch.mean(local_grid_theta).item() * 2.0)
            
            # Reduced phase chaos level to prevent nodes from popping spontaneously
            scramble_rate = (effective_shear * 8.0) + ((~act['edge_intact']).float() * 0.5)
            phase_noise = (torch.rand(16, device=self.device) * 2.0 - 1.0) * scramble_rate * phase_noise_suppression
            
            phases = act['node_phases']
            idx_next = torch.remainder(torch.arange(16, device=self.device) + 1, 16)
            idx_prev = torch.remainder(torch.arange(16, device=self.device) - 1, 16)
            coupling = torch.sin(phases[idx_next] - phases) + torch.sin(phases[idx_prev] - phases)
            
            # FIX 3: Completely removed destructive drift which caused mass suicides
            
            theta_power = torch.sum(act['node_spectra'][:, 4:9]).item() / 16.0
            K = act.get('K_active', 30.0) + (theta_power * 15.0)
            
            phases += (K * 0.5 * coupling + phase_noise) * dt
            act['node_phases'] = torch.remainder(phases + math.pi, 2 * math.pi) - math.pi
            
            cos_sum = torch.cos(act['node_phases']).mean()
            sin_sum = torch.sin(act['node_phases']).mean()
            inst_coherence = math.hypot(cos_sum.item(), sin_sum.item())
            num_broken = (~act['edge_intact']).sum().item()
            
            # --- SMOOTH PHYSICAL DAMAGE & AUTOPOIETIC REGENERATION ---
            base_dmg = 0.0
            
            # Robust physical boundary safety check to prevent wall-collision self-destruction
            # Evaluates distance to outer circular boundary for all 16 nodes individually
            pin_normalized = (act['pin_pos'] / torch.tensor([self.WIDTH, self.HEIGHT], device=self.device)) * 2.0 - 1.0
            node_dists = torch.norm(pin_normalized, dim=1)
            is_near_wall = (node_dists > 0.76).any().item()
            
            if is_near_wall:
                # Maintain spring networks and block phase scattering on wall touch
                act['edge_intact'][:] = True
                act['node_phases'][:] = 0.0
                inst_coherence = 1.0
                act['shear_stress'] = 0.0

                # Dampen the local fluid velocities at the wall to prevent high-velocity squeezing
                gx = int((act['pos'][0] / self.WIDTH) * self.res)
                gy = int((act['pos'][1] / self.HEIGHT) * self.res)
                if 2 <= gx < self.res-2 and 2 <= gy < self.res-2:
                    self.u[0, 0, gy-2:gy+3, gx-2:gx+3] *= 0.5
                    self.v[0, 0, gy-2:gy+3, gx-2:gx+3] *= 0.5

            # Symmetrically apply friendly healer support to nearby allies
            is_near_healer = False
            for ally in self.actors:
                if ally['team'] == act['team'] and ally['style'] == "Healer" and not ally['is_dead'] and ally is not act:
                    if torch.norm(act['pos'] - ally['pos']).item() < 120.0:
                        is_near_healer = True
                        break

            if is_near_healer:
                # Restore phase alignment and repair structural integrity
                act['edge_intact'][:] = True
                act['node_phases'][:] = 0.0
                inst_coherence = 1.0
                act['shear_stress'] = 0.0

            # --- AUTOPOIETIC REGENERATION (UNIVERSAL SELF-HEALING) ---
            # Restructured to prevent endless stalemates.
            # Non-healer characters can only repair structure if supported by Healer's mist.
            if inst_coherence > 0.82 and act['shear_stress'] < 0.15:
                # Baseline self-repair is exclusive to the Healer class (Support)
                base_regen = 0.008 if act['style'] == "Healer" else 0.000
                
                # Active supportive healing is directly proportional to the sampled friendly Theta density on the grid
                grid_heal_rate = torch.mean(local_grid_theta).item() * 0.015
                
                heal_rate = (base_regen + grid_heal_rate) * (act['quality'] / 100.0)
                act['integrity'] = min(1.0, act['integrity'] + heal_rate * dt)
                
                # Slowly restore broken springs if highly stable
                if inst_coherence > 0.90:
                    act['edge_intact'][:] = True
            
            # 1. Coherence loss damage (phases desynchronized by enemy noise)
            # Coherence damage threshold lowered to 0.15.
            # Healthy brain desynchronization (0.2-0.35) no longer kills the slimes!
            if inst_coherence < 0.15:
                base_dmg += (0.15 - inst_coherence) * 0.5  
                
            # Enemy fluid damage threshold raised to 0.50
            if act['shear_stress'] > 0.5:
                base_dmg += (act['shear_stress'] - 0.5) * 0.5
                self.log_event(f"{act['custom_name']} takes spectral damage!")
                
            # Body collision during ramming, partially absorbed by Theta armor (acts as a percentage damper)
            for enemy in self.actors:
                if enemy['team'] != act['team'] and not enemy['is_dead']:
                    dist = torch.norm(act['pos'] - enemy['pos']).item()
                    if dist < combat_config.BODY_COLLISION_RADIUS * 1.1:
                        clash_pressure = (combat_config.BODY_COLLISION_RADIUS * 1.1 - dist) * 0.8
                        armor_absorption = min(0.60, theta_power * 0.12) # Up to 60% damage reduction
                        absorbed_damage = clash_pressure * (1.0 - armor_absorption) * 0.25
                        base_dmg += absorbed_damage
                        self.log_event(f"{act['custom_name']} CLASHED with {enemy['custom_name'][:4]}!")

            # Damage scaling based on team Combat Power (CP)
            power_factor = ratio_t0 if act['team'] == 0 else ratio_t1
            act['integrity'] -= (base_dmg / (power_factor + 1e-5)) * dt
                
            if act['integrity'] <= 0.0:
                act['integrity'] = 0.0
                act['is_dead'] = True
                self.log_event(f"{act['custom_name']} WAS ANNIHILATED!")

        team_alive = {0: any(a['team'] == 0 and not a['is_dead'] for a in self.actors), 
                      1: any(a['team'] == 1 and not a['is_dead'] for a in self.actors)}

        # ===================== OPTIONAL REAL-TIME CALCULATION =====================
        t0_active = [a for a in self.actors if a['team'] == 0 and not a['is_dead']]
        t1_active = [a for a in self.actors if a['team'] == 1 and not a['is_dead']]
        
        t0_integ = sum(a['integrity'] for a in t0_active) / max(1, len([a for a in self.actors if a['team'] == 0]))
        t1_integ = sum(a['integrity'] for a in t1_active) / max(1, len([a for a in self.actors if a['team'] == 1]))
        
        Spot_S = t0_integ - t1_integ # Spot Price [-1.0 ... 1.0]
        
        t0_gamma = sum(torch.sum(a['node_spectra'][:, 60:101]).item() for a in t0_active)
        t0_theta = sum(torch.sum(a['node_spectra'][:, 4:9]).item() for a in t0_active)
        t1_gamma = sum(torch.sum(a['node_spectra'][:, 60:101]).item() for a in t1_active)
        t1_theta = sum(torch.sum(a['node_spectra'][:, 4:9]).item() for a in t1_active)
        
        # Synergistic drift (Mu)
        raw_mu = (t0_gamma - t1_theta * 0.5) - (t1_gamma - t0_theta * 0.5)
        self.drift_mu = max(-2.0, min(2.0, raw_mu * 0.05))
        
        # Volatility (Sigma)
        all_active = [a for a in self.actors if not a['is_dead']]
        avg_dissonance = sum(a['shear_stress'] for a in all_active) / max(1, len(all_active))
        self.implied_vol = max(0.1, avg_dissonance * 5.0)
        
        tau = max(0.1, 30.0 - self.combat_time)
        
        d = (Spot_S + self.drift_mu * tau) / (self.implied_vol * math.sqrt(tau))
        
        # Clip exponent strictly to prevent NaN/Overflow on predict WP sigmoid evaluation
        exponent = -d * 1.5
        exponent_clipped = max(-700.0, min(700.0, exponent))
        self.predicted_wp = 1.0 / (1.0 + math.exp(exponent_clipped)) 
        
        if self.log_file:
            self.log_file.write(f"{self.combat_time:.3f},{t0_integ:.4f},{t1_integ:.4f},{Spot_S:.4f},{self.drift_mu:.4f},{self.implied_vol:.4f},{self.predicted_wp*100.0:.2f},{tau:.2f}\n")
            self.log_file.flush()

        for act in self.actors:
            # Floating healing popup numbers
            if act['integrity'] > old_integrity[act['id']]:
                heal_gained = act['integrity'] - old_integrity[act['id']]
                if heal_gained > 0.003:
                    val = int(heal_gained * 1000)
                    color = (100, 255, 100) # Pure vibrant green for healing numbers
                    jx = (random.random() * 2.0 - 1.0) * 15.0
                    jy = (random.random() * 2.0 - 1.0) * 10.0
                    self.floating_texts.append({
                        'pos': [act['pos'][0].item() + jx, act['pos'][1].item() - 25.0 + jy],
                        'text': f"+{val}", 'life': 1.0, 'color': color
                    })
            # Floating damage popup numbers
            elif old_integrity[act['id']] > act['integrity']:
                dmg_taken = old_integrity[act['id']] - act['integrity']
                if dmg_taken > 0.003: 
                    val = int(dmg_taken * 1000) 
                    color = (255, 100, 100) if act['team'] == 0 else (0, 255, 255) 
                    jx = (random.random() * 2.0 - 1.0) * 15.0
                    jy = (random.random() * 2.0 - 1.0) * 10.0
                    self.floating_texts.append({
                        'pos': [act['pos'][0].item() + jx, act['pos'][1].item() - 25.0 + jy],
                        'text': f"-{val}", 'life': 1.0, 'color': color
                    })
                
        if not team_alive[0] and not team_alive[1]: self.winner = "Draw"
        elif not team_alive[0]: self.winner = "Team 1"
        elif not team_alive[1]: self.winner = "Team 0"
            
        self.team0_density.zero_()
        self.team1_density.zero_()
        for act in self.actors:
            target_density = self.team0_density if act['team'] == 0 else self.team1_density
            self._update_render_density(act['pin_pos'], target_density, act['edge_intact'])

    def _update_render_density(self, pin_pos, density_tensor, edge_intact_mask):
        active_nodes = pin_pos[edge_intact_mask]
        if active_nodes.numel() == 0: return
        gx = ((active_nodes[:, 0] / self.WIDTH) * self.res).clamp(0, self.res - 1).long()
        gy = ((active_nodes[:, 1] / self.HEIGHT) * self.res).clamp(0, self.res - 1).long()
        dx = self.x_indices.unsqueeze(0) - gx.reshape(-1, 1, 1)
        dy = self.y_indices.unsqueeze(0) - gy.reshape(-1, 1, 1)
        influence = torch.exp(-(dx**2 + dy**2) / 10.0)
        density_tensor.copy_(torch.max(torch.max(influence, dim=0, keepdim=True)[0].unsqueeze(0), density_tensor))

    def apply_negentropic_fractal_boundaries(self, density_spectral, u, v, dt):
        """ Creates Gamma solitons specifically at the collision points between Beta and Theta flows """
        theta_band = torch.sum(density_spectral[:, 4:9, :, :], dim=1, keepdim=True)
        beta_band = torch.sum(density_spectral[:, 18:37, :, :], dim=1, keepdim=True)
        
        theta_pad = F.pad(theta_band, (1, 1, 1, 1), mode='replicate')
        beta_pad = F.pad(beta_band, (1, 1, 1, 1), mode='replicate')
        
        grad_theta_x = 0.5 * (theta_pad[:, :, 1:-1, 2:] - theta_pad[:, :, 1:-1, :-2])
        grad_theta_y = 0.5 * (theta_pad[:, :, 2:, 1:-1] - theta_pad[:, :, :-2, 1:-1])
        grad_beta_x = 0.5 * (beta_pad[:, :, 1:-1, 2:] - beta_pad[:, :, 1:-1, :-2])
        grad_beta_y = 0.5 * (beta_pad[:, :, 2:, 1:-1] - beta_pad[:, :, :-2, 1:-1])
        
        dot_product = grad_theta_x * grad_beta_x + grad_theta_y * grad_beta_y
        interface_intensity = torch.relu(-dot_product) * 2.0 
        
        fluid_speed = torch.sqrt(u**2 + v**2) + 1e-8
        kinetic_compression = interface_intensity * fluid_speed
        
        conversion_rate = 4.5 * dt
        gamma_spawn = kinetic_compression * conversion_rate
        
        density_spectral[:, 80:81, :, :] += gamma_spawn
        density_spectral[:, 18:37, :, :] = torch.clamp(
            density_spectral[:, 18:37, :, :] - (gamma_spawn / 19.0), 0.0, 10.0
        )
        return density_spectral
