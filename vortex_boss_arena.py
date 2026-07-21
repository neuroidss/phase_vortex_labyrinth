# EXOCORTEX_SQUAD_ARENA.py
import pygame
import torch
import torch.nn.functional as F
import math
import numpy as np
import time
import random
import sys

# Ensure global compatibility: all internal logic, comments, and variables are strictly in English.
# TACTICAL SPLIT: 6 Classes mapped onto 6 distinct non-linear spectral frequencies.
# WEAPON SOLITONS: Independent, physics-grounded self-propulsion and waveguides for Bow, Shuriken & Magic.

WIDTH, HEIGHT = 1200, 800  
ARENA_WIDTH = 800
COMPUTE_RES = 128
TOURNAMENT_SEED = 202607
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StandaloneFluidSolver:
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

    def compute_laplacian(self, field):
        field_pad = F.pad(field, (1, 1, 1, 1), mode='replicate')
        return (field_pad[:, :, 1:-1, 2:] + field_pad[:, :, 1:-1, :-2] +
                field_pad[:, :, 2:, 1:-1] + field_pad[:, :, :-2, 1:-1] - 4.0 * field)

    def project(self, u, v, wall_density):
        block_mask = (wall_density > 0.4).float()
        u = u * (1.0 - block_mask)
        v = v * (1.0 - block_mask)
        
        u_pad = F.pad(u, (1, 1, 1, 1), mode='replicate')
        v_pad = F.pad(v, (1, 1, 1, 1), mode='replicate')
        div = 0.5 * (u_pad[:, :, 1:-1, 2:] - u_pad[:, :, 1:-1, :-2] + 
                     v_pad[:, :, 2:, 1:-1] - v_pad[:, :, :-2, 1:-1])
        
        p = torch.zeros_like(u)
        for _ in range(10):
            p_pad = F.pad(p, (1, 1, 1, 1), mode='replicate')
            p = 0.25 * (p_pad[:, :, 1:-1, 2:] + p_pad[:, :, 1:-1, :-2] + 
                        p_pad[:, :, 2:, 1:-1] + p_pad[:, :, :-2, 1:-1] - div)
                    
        p_pad = F.pad(p, (1, 1, 1, 1), mode='replicate')
        u -= 0.5 * (p_pad[:, :, 1:-1, 2:] - p_pad[:, :, 1:-1, :-2])
        v -= 0.5 * (p_pad[:, :, 2:, 1:-1] - p_pad[:, :, :-2, 1:-1])
        
        return u * (1.0 - block_mask), v * (1.0 - block_mask)

class SoftbodyActor:
    def __init__(self, spawn_pos, team, color, name, style, weapon, creature, level, quality):
        self.pos = torch.tensor(spawn_pos, dtype=torch.float32, device=device)
        self.team = team
        self.color = color
        self.name = name
        self.style = style  
        self.weapon = weapon  
        self.creature = creature  
        self.level = level
        self.quality = quality
        
        self.integrity = 1.0
        self.ult_charge = 0.0
        self.is_dead = False
        self.current_action = "[IDLE]" 
        
        self.breather_phase = random.uniform(0.0, 2.0 * math.pi)
        self.breather_speed = 3.5 if style == "Fighter" else 1.8
        
        self.pin_x = torch.tensor([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], device=device) * 1.5
        self.pin_y = torch.tensor([-2.72, -7.43, -4.77, -10.15, -10.14, -4.77, -7.42, -2.73, 2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], device=device) * 1.5
        
        self.pin_pos = torch.zeros((16, 2), dtype=torch.float32, device=device)
        self.pin_pos[:, 0] = self.pos[0] + self.pin_x
        self.pin_pos[:, 1] = self.pos[1] + self.pin_y
        self.edge_intact = torch.ones(16, dtype=torch.bool, device=device)
        self.node_phases = torch.zeros(16, dtype=torch.float32, device=device)
        self.angle = 0.0
        
        self.idx_I = torch.arange(16, device=device)
        self.idx_J = torch.remainder(self.idx_I + 1, 16)
        
        self.cooldown = random.uniform(0.1, 1.5)
        self.channel_duration = 0.0
        self.channel_target = None
        self.power = self.calculate_power()

    def calculate_power(self):
        base = 1000 + self.level * 150
        qual_mult = 0.5 + (self.quality / 100.0)
        weapon_mults = {"Sword": 1.1, "Bow": 1.2, "Magic": 1.25, "Heal": 0.9, "Shield": 0.95, "Shuriken": 1.15}
        creature_mults = {"Dendro": 1.0, "Necro": 1.05, "Demon": 1.15, "Furry": 1.0, "Human": 1.0, "Angel": 1.1}
        return int(base * qual_mult * weapon_mults.get(self.weapon, 1.0) * creature_mults.get(self.creature, 1.0))

    def calculate_covariance(self):
        com = self.pin_pos.mean(dim=0)
        dx_local = self.pin_pos[:, 0] - com[0]
        dy_local = self.pin_pos[:, 1] - com[1]
        cross_cov = torch.sum(self.pin_y * dx_local - self.pin_x * dy_local)
        dot_cov = torch.sum(self.pin_x * dx_local + self.pin_y * dy_local) + 1e-5
        raw_angle = torch.atan2(cross_cov, dot_cov).item()
        
        angle_diff = (raw_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
        self.angle += angle_diff * 0.40
        return com

    def update_springs(self, ideal_pos):
        d_curr = torch.norm(self.pin_pos[self.idx_I] - self.pin_pos[self.idx_J], dim=1)
        d_rest = torch.norm(ideal_pos[self.idx_I] - ideal_pos[self.idx_J], dim=1)
        strain = d_curr / (d_rest + 1e-5)
        
        self.edge_intact = self.edge_intact & ~(strain > 2.8 + self.integrity * 2.5)
        self.edge_intact = self.edge_intact | (strain < 1.05)
        
        dir_mutual = (self.pin_pos[self.idx_J] - self.pin_pos[self.idx_I]) / (d_curr.unsqueeze(1) + 1e-5)
        f_mag = 200.0 * (self.integrity + 0.15) * (d_curr - d_rest)
        
        f_mutual_I = dir_mutual * f_mag.unsqueeze(1) * self.edge_intact.float().unsqueeze(1)
        f_mutual_prev = torch.roll(f_mutual_I, shifts=1, dims=0)
        return torch.clamp(f_mutual_I - f_mutual_prev, -300.0, 350.0)

def apply_cohesion_constraint(pin_pos, ideal_pos, scale, cohesion_level):
    max_allowed = scale * (20.0 - cohesion_level * 15.0) if cohesion_level < 0.0 else scale * (5.0 + (1.0 - cohesion_level) * 15.0)
    diff = pin_pos - ideal_pos
    dist = torch.norm(diff, dim=1, keepdim=True) + 1e-5
    over_limit = dist > max_allowed
    if over_limit.any():
        return torch.where(over_limit, ideal_pos + (diff / dist) * max_allowed, pin_pos)
    return pin_pos

class BossFightArena:
    def __init__(self, master_seed=TOURNAMENT_SEED, current_stage=1, ally_level=20, alch_resources=0.0):
        self.master_seed = master_seed
        self.current_stage = current_stage
        self.ally_level = ally_level
        self.alch_resources = alch_resources
        
        self.solver = StandaloneFluidSolver(COMPUTE_RES, device)
        self.u = torch.zeros((1, 1, COMPUTE_RES, COMPUTE_RES), device=device)
        self.v = torch.zeros((1, 1, COMPUTE_RES, COMPUTE_RES), device=device)
        self.density_spectral = torch.zeros((1, 100, COMPUTE_RES, COMPUTE_RES), device=device)
        
        y, x = self.solver.grid_y, self.solver.grid_x
        self.static_walls = ((torch.abs(x) > 0.94) | (torch.abs(y) > 0.94)).float().view(1, 1, COMPUTE_RES, COMPUTE_RES)
        
        self.freqs_hz = torch.linspace(1.0, 100.0, 100, device=device).view(1, 100, 1, 1)
        # OPTIMIZATION: Wide red window centered at 65.0 to render Low, Mid and High Gamma beautifully
        self.w_red   = torch.exp(-((self.freqs_hz - 65.0) ** 2) / 600.0)
        self.w_green = torch.exp(-((self.freqs_hz - 24.0) ** 2) / 144.0)
        # OPTIMIZATION: Centered at 8.0 to render both 6Hz and 10Hz as beautiful Blue/Cyan
        self.w_blue  = torch.exp(-((self.freqs_hz - 8.0) ** 2) / 100.0)
        self.wall_color = torch.tensor([50, 15, 60], dtype=torch.uint8, device=device).view(3, 1)

        self.actors = []
        self.damage_numbers = []
        self.combat_time = 0.0
        self.winner = None
        self.synergies = {}
        self.transition_timer = 2.2 

        self.generate_campaign_matchup()
        self.update_synergies()

    def log_event(self, msg):
        if not hasattr(self, 'combat_log'): self.combat_log = []
        if not self.combat_log or self.combat_log[-1] != msg:
            self.combat_log.append(msg)
            if len(self.combat_log) > 8: self.combat_log.pop(0)

    def generate_campaign_matchup(self):
        allies_configs = [
            {"style": "Tank", "weapon": "Shield", "creature": "Angel", "pos": [150, 200], "name": "Gabriel Aegis"},
            {"style": "Tank", "weapon": "Shield", "creature": "Dendro", "pos": [150, 450], "name": "Yggdrasil Oak"},
            {"style": "Fighter", "weapon": "Sword", "creature": "Furry", "pos": [220, 150], "name": "Fenrir Strike"},
            {"style": "Fighter", "weapon": "Sword", "creature": "Human", "pos": [220, 320], "name": "King Arthur"},
            {"style": "Fighter", "weapon": "Bow", "creature": "Human", "pos": [220, 500], "name": "Robin Arrow"},
            {"style": "Fighter", "weapon": "Shuriken", "creature": "Necro", "pos": [200, 580], "name": "Shadow Kunai"},
            {"style": "Mage", "weapon": "Magic", "creature": "Demon", "pos": [100, 300], "name": "Lilith Abyss"},
            {"style": "Healer", "weapon": "Heal", "creature": "Angel", "pos": [80, 400], "name": "Raphael Cure"}
        ]
        
        for cfg in allies_configs:
            color = (0, 180 + random.randint(0, 75), 180 + random.randint(0, 75))
            act = SoftbodyActor(cfg["pos"], 0, color, cfg["name"], cfg["style"], cfg["weapon"], cfg["creature"], self.ally_level, 95)
            self.actors.append(act)

        local_rand = random.Random(self.master_seed + self.current_stage * 1337)
        weapons = ["Sword", "Bow", "Magic", "Heal", "Shield", "Shuriken"]
        creatures = ["Dendro", "Necro", "Demon", "Furry", "Human", "Angel"]
        enemy_styles = ["Fighter", "Fighter", "Mage", "Healer", "Tank", "Fighter", "Mage", "Tank"]
        
        enemy_lv = int(5 * (1.08) ** self.current_stage) + self.current_stage * 2
        is_boss_stage = (self.current_stage % 5 == 0)
        enemy_names = ["Hundun Guard", "Lich Thrall", "Astaroth Spawn", "Behemoth Pup", "Plague Drone", "Hanzo Disciple", "Azazel Spark", "Baphomet Disc"]
        fractal_mult = 1.0 + (0.25 if self.current_stage%5==0 else 0) + (0.5 if self.current_stage%10==0 else 0)
        
        for i in range(8):
            e_pos = [600 + local_rand.randint(-40, 120), 100 + i * 85]
            if i == 0 and is_boss_stage:
                color = (180 + local_rand.randint(0, 75), 30, 0)
                act = SoftbodyActor(e_pos, 1, color, "BOSS", "Tank", "Shield", local_rand.choice(creatures), int(enemy_lv * fractal_mult) + 10, 100)
                act.power *= 3
                act.pin_x *= 2.5
                act.pin_y *= 2.5
            else:
                color = (150 + local_rand.randint(0, 50), 80, 40)
                act = SoftbodyActor(e_pos, 1, color, enemy_names[i], enemy_styles[i], local_rand.choice(weapons), local_rand.choice(creatures), enemy_lv, 85)
            self.actors.append(act)

    def update_synergies(self):
        syn = {}
        for act in self.actors:
            if not act.is_dead:
                syn[f"team{act.team}_{act.weapon}"] = syn.get(f"team{act.team}_{act.weapon}", 0) + 1
                syn[f"team{act.team}_{act.creature}"] = syn.get(f"team{act.team}_{act.creature}", 0) + 1
        self.synergies = syn

    # --- FIRE-AND-FORGET WEAPONS (Aerodynamic 3x3/2x2 Solitons) ---
    def fire_bow(self, source, target_pos):
        s_pos = source.pos.cpu().numpy()
        dir_x, dir_y = target_pos[0] - s_pos[0], target_pos[1] - s_pos[1]
        dist = math.hypot(dir_x, dir_y) + 1e-5
        dir_x, dir_y = dir_x / dist, dir_y / dist
        
        # Spawns at 65px (completely outside the softbody radius)
        inj_x, inj_y = s_pos[0] + dir_x * 65.0, s_pos[1] + dir_y * 65.0
        gx, gy = int((inj_x / ARENA_WIDTH) * COMPUTE_RES), int((inj_y / HEIGHT) * COMPUTE_RES)
        
        success = False
        if 3 <= gx < COMPUTE_RES - 3 and 3 <= gy < COMPUTE_RES - 3:
            success = True
            # Spectral Bow team routing: Team 0 = 60Hz, Team 1 = 65Hz
            freq = 60.0 if source.team == 0 else 65.0
            # Solid 3x3 bullet kernel that survives flight decay
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    weight = 1.0 - (dx*dx + dy*dy)/3.0
                    self.density_spectral[0, int(freq), gy+dy, gx+dx] += weight * 45.0
                    self.u[0, 0, gy+dy, gx+dx] += dir_x * 1500.0 * weight
                    self.v[0, 0, gy+dy, gx+dx] += dir_y * 1500.0 * weight
        source.current_action = "[SHOOTING]" if success else "[BLOCKED]"
        return success

    def fire_shuriken(self, source, target_pos):
        s_pos = source.pos.cpu().numpy()
        dir_x, dir_y = target_pos[0] - s_pos[0], target_pos[1] - s_pos[1]
        dist = math.hypot(dir_x, dir_y) + 1e-5
        dx, dy = dir_x / dist, dir_y / dist
        
        success = False
        angles = [-0.3, 0.0, 0.3]
        # Spectral Shuriken team routing: Team 0 = 40Hz, Team 1 = 45Hz
        freq = 40.0 if source.team == 0 else 45.0
        for a in angles:
            rx, ry = dx * math.cos(a) - dy * math.sin(a), dx * math.sin(a) + dy * math.cos(a)
            inj_x, inj_y = s_pos[0] + rx * 55.0, s_pos[1] + ry * 55.0
            gx, gy = int((inj_x / ARENA_WIDTH) * COMPUTE_RES), int((inj_y / HEIGHT) * COMPUTE_RES)
            if 2 <= gx < COMPUTE_RES - 2 and 2 <= gy < COMPUTE_RES - 2:
                success = True
                # Inject three 2x2 dense shurikens
                for dy in range(-1, 1):
                    for dx in range(-1, 1):
                        self.density_spectral[0, int(freq), gy+dy, gx+dx] += 25.0
                        self.u[0, 0, gy+dy, gx+dx] += rx * 1100.0
                        self.v[0, 0, gy+dy, gx+dx] += ry * 1100.0
        return success

    # --- CHANNELING WEAPONS (Continuous energy streams) ---
    def channel_magic(self, source, target_pos, dt):
        s_pos = source.pos.cpu().numpy()
        dir_x, dir_y = target_pos[0] - s_pos[0], target_pos[1] - s_pos[1]
        dist = math.hypot(dir_x, dir_y) + 1e-5
        dir_x, dir_y = dir_x / dist, dir_y / dist
        
        inj_x, inj_y = s_pos[0] + dir_x * 35.0, s_pos[1] + dir_y * 35.0
        pulse = 1.0 + 0.6 * math.sin(self.combat_time * 15.0 + source.pos[0].item())
        
        gx, gy = int((inj_x / ARENA_WIDTH) * COMPUTE_RES), int((inj_y / HEIGHT) * COMPUTE_RES)
        if 2 <= gx < COMPUTE_RES - 2 and 2 <= gy < COMPUTE_RES - 2:
            # Spectral Magic team routing: Team 0 = 80Hz, Team 1 = 85Hz
            freq = 80.0 if source.team == 0 else 85.0
            self.density_spectral[0, int(freq), gy-1:gy+2, gx-1:gx+2] += 30.0 * dt * pulse
            self.u[0, 0, gy-1:gy+2, gx-1:gx+2] += dir_x * 800.0 * dt * pulse
            self.v[0, 0, gy-1:gy+2, gx-1:gx+2] += dir_y * 800.0 * dt * pulse

    def channel_sword(self, source, target_pos, dt):
        s_pos = source.pos.cpu().numpy()
        dir_x, dir_y = target_pos[0] - s_pos[0], target_pos[1] - s_pos[1]
        dist = math.hypot(dir_x, dir_y) + 1e-5
        dir_x, dir_y = dir_x / dist, dir_y / dist
        
        for offset in [-12, 0, 12]:
            px, py = dir_x * math.cos(1.57) - dir_y * math.sin(1.57), dir_x * math.sin(1.57) + dir_y * math.cos(1.57)
            inj_x, inj_y = s_pos[0] + dir_x * 25.0 + px * offset, s_pos[1] + dir_y * 25.0 + py * offset
            gx, gy = int((inj_x / ARENA_WIDTH) * COMPUTE_RES), int((inj_y / HEIGHT) * COMPUTE_RES)
            
            if 2 <= gx < COMPUTE_RES - 2 and 2 <= gy < COMPUTE_RES - 2:
                # Spectral Sword team routing: Team 0 = 80Hz, Team 1 = 85Hz
                freq = 80.0 if source.team == 0 else 85.0
                # Green (Beta) for kinetic disruption & visual blade
                self.density_spectral[0, 24, gy, gx] += 50.0 * dt
                # Red (Gamma) for actual damage
                self.density_spectral[0, int(freq), gy, gx] += 35.0 * dt
                # Heavy forward push
                self.u[0, 0, gy, gx] += dir_x * 1200.0 * dt
                self.v[0, 0, gy, gx] += dir_y * 1200.0 * dt

    def channel_shield(self, source, dt):
        """ SHIELD IS MOVED TO ALPHA SPECTRUM (10Hz). No self-healing! """
        s_pos = source.pos.cpu().numpy()
        gx, gy = int((s_pos[0] / ARENA_WIDTH) * COMPUTE_RES), int((s_pos[1] / HEIGHT) * COMPUTE_RES)
        if 3 <= gx < COMPUTE_RES - 3 and 3 <= gy < COMPUTE_RES - 3:
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if dx*dx + dy*dy <= 9:
                        weight = 1.0 - (dx*dx + dy*dy)/10.0
                        # Injecting into Alpha frequency 10.0
                        self.density_spectral[0, 10, gy+dy, gx+dx] += weight * 60.0 * dt # Heavy Blue

    def channel_heal(self, source, dt):
        """ HEAL MIST IS MOVED TO THETA SPECTRUM (6Hz). """
        s_pos = source.pos.cpu().numpy()
        gx, gy = int((s_pos[0] / ARENA_WIDTH) * COMPUTE_RES), int((s_pos[1] / HEIGHT) * COMPUTE_RES)
        if 4 <= gx < COMPUTE_RES - 4 and 4 <= gy < COMPUTE_RES - 4:
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    if dx*dx + dy*dy <= 16:
                        weight = 1.0 - (dx*dx + dy*dy)/17.0
                        dir_x, dir_y = float(dx), float(dy)
                        dist = math.hypot(dir_x, dir_y) + 1e-5
                        # Injecting into Theta frequency 6.0
                        self.density_spectral[0, 6, gy+dy, gx+dx] += weight * 20.0 * dt
                        # Radial outward push
                        self.u[0, 0, gy+dy, gx+dx] += (dir_x/dist) * 60.0 * dt
                        self.v[0, 0, gy+dy, gx+dx] += (dir_y/dist) * 60.0 * dt

    def cast_ultimate(self, actor):
        enemies = [a for a in self.actors if a.team != actor.team and not a.is_dead]
        if not enemies: return
        target = min(enemies, key=lambda e: torch.norm(e.pos - actor.pos).item())
        self.log_event(f"{actor.name[:12].upper()} CASTS ULT!")
        t_pos = target.pos.cpu().numpy()
        
        if actor.weapon == "Bow":
            for _ in range(5): self.fire_bow(actor, t_pos + np.random.randn(2)*20)
        elif actor.weapon == "Shuriken":
            for _ in range(3): self.fire_shuriken(actor, t_pos + np.random.randn(2)*15)
        elif actor.weapon == "Sword":
            self.strike_sword(actor, t_pos)
            self.strike_sword(actor, t_pos)
        elif actor.weapon == "Magic":
            self.channel_magic(actor, t_pos, 1.0)
        elif actor.weapon == "Shield":
            self.channel_shield(actor, 2.0) 
        else: # Healer
            self.channel_heal(actor, 2.0) 

    def step(self, dt):
        self.alch_resources += 5.0 * (1.05) ** self.current_stage * dt
        if self.alch_resources >= 100.0 * (1.06) ** self.ally_level:
            self.alch_resources -= 100.0 * (1.06) ** self.ally_level
            self.ally_level += 1
            for act in self.actors:
                if act.team == 0: act.level = self.ally_level; act.power = act.calculate_power()

        if self.winner is not None:
            self.transition_timer -= dt
            return

        self.combat_time += dt
        self.update_synergies()
        self.resolve_solitonic_collisions(dt)

        # ====================================================================
        # NON-LINEAR SCHRÖDINGER PHYSICS (DIVERSE RHEOLOGY & BALANCED COHESION)
        # ====================================================================
        lap_spec = self.solver.compute_laplacian(self.density_spectral)
        
        # 1. INDEPENDENT ACTIVE COMPACTNESS: Shurikens (40Hz), Bow (60Hz) and Magic (80Hz) 
        # now each have custom self-focusing equations! No infinite runaway freezing!
        shuriken_spec = self.density_spectral[:, 37:50]
        bow_spec      = self.density_spectral[:, 50:70]
        magic_spec    = self.density_spectral[:, 70:90]
        
        # Shurikens (40Hz) self-focus moderately (allows some scatter)
        self.density_spectral[:, 37:50] += 6.0 * shuriken_spec * (shuriken_spec - 0.3) * (1.5 - shuriken_spec) * dt
        # Bow (60Hz) self-focuses strongly (holds laser-tight lines in flight)
        self.density_spectral[:, 50:70] += 12.0 * bow_spec * (bow_spec - 0.2) * (2.2 - bow_spec) * dt
        # Magic (80Hz) self-focuses solidly (holds mycelium streams)
        self.density_spectral[:, 70:90] += 8.0 * magic_spec * (magic_spec - 0.35) * (1.8 - magic_spec) * dt
        
        # Base diffusion for all
        self.density_spectral = torch.clamp(self.density_spectral + lap_spec * (0.04 / torch.sqrt(self.freqs_hz)) * dt, 0.0, 3.5)

        # Frequency drift
        psi_prev, psi_next = torch.roll(self.density_spectral, 1, 1), torch.roll(self.density_spectral, -1, 1)
        self.density_spectral = torch.clamp(self.density_spectral + 0.5 * self.density_spectral * (psi_prev - psi_next) * dt, 0.0, 3.5)

        # 3. GLOBAL ENTROPY EVAPORATION (Frame-rate independent)
        # Gamma active attack classes decay according to their respective tactical roles
        self.density_spectral[:, 70:90] *= torch.clamp(torch.tensor(1.0 - 1.2 * dt, device=device), 0.0, 1.0) # Magic (80Hz)
        self.density_spectral[:, 50:70] *= torch.clamp(torch.tensor(1.0 - 0.4 * dt, device=device), 0.0, 1.0) # Bow (60Hz) - long flight sniper!
        self.density_spectral[:, 37:50] *= torch.clamp(torch.tensor(1.0 - 1.8 * dt, device=device), 0.0, 1.0) # Shuriken (40Hz) - moderate scatter!
        self.density_spectral[:, 18:37] *= torch.clamp(torch.tensor(1.0 - 4.5 * dt, device=device), 0.0, 1.0) # Beta (Green) - dies very fast
        self.density_spectral[:, 4:9] *= torch.clamp(torch.tensor(1.0 - 2.5 * dt, device=device), 0.0, 1.0)   # Theta (Blue/Heal)
        self.density_spectral[:, 9:13] *= torch.clamp(torch.tensor(1.0 - 2.5 * dt, device=device), 0.0, 1.0)  # Alpha (Blue/Shield)

        # Fracture Reaction
        theta_sum = torch.sum(self.density_spectral[:, 4:9], dim=1, keepdim=True)
        beta_sum = torch.sum(self.density_spectral[:, 18:37], dim=1, keepdim=True)
        
        grad_t_x = 0.5 * (F.pad(theta_sum, (1,1,1,1), mode='replicate')[:, :, 1:-1, 2:] - F.pad(theta_sum, (1,1,1,1), mode='replicate')[:, :, 1:-1, :-2])
        grad_t_y = 0.5 * (F.pad(theta_sum, (1,1,1,1), mode='replicate')[:, :, 2:, 1:-1] - F.pad(theta_sum, (1,1,1,1), mode='replicate')[:, :, :-2, 1:-1])
        grad_b_x = 0.5 * (F.pad(beta_sum, (1,1,1,1), mode='replicate')[:, :, 1:-1, 2:] - F.pad(beta_sum, (1,1,1,1), mode='replicate')[:, :, 1:-1, :-2])
        grad_b_y = 0.5 * (F.pad(beta_sum, (1,1,1,1), mode='replicate')[:, :, 2:, 1:-1] - F.pad(beta_sum, (1,1,1,1), mode='replicate')[:, :, :-2, 1:-1])
        
        clash_dot = grad_t_x * grad_b_x + grad_t_y * grad_b_y
        interface = torch.clamp(torch.relu(-clash_dot) * 2.0, 0.0, 2.0)
        
        kinetic_clash = torch.clamp(torch.relu(-interface * (self.u * grad_t_x + self.v * grad_t_y)), 0.0, 50.0)
        self.density_spectral[:, 70:85, :, :] += kinetic_clash * 1.5 * dt
        self.u += grad_t_x * kinetic_clash * 5.0 * dt
        self.v += grad_t_y * kinetic_clash * 5.0 * dt

        # 5. FLUID FRICTION & WEAK SUPERFLUIDITY
        shuriken_band = torch.sum(self.density_spectral[:, 37:50], dim=1, keepdim=True) 
        bow_band      = torch.sum(self.density_spectral[:, 50:70], dim=1, keepdim=True) 
        magic_band    = torch.sum(self.density_spectral[:, 70:90], dim=1, keepdim=True) 
        
        active_projectiles = shuriken_band + bow_band + magic_band
        gamma_presence = torch.clamp(active_projectiles / 1.5, 0.0, 1.0)
        
        # Superfluidity: Friction is high in empty space (4.5), but drops near zero inside solitons (0.2)
        friction = 4.5 - gamma_presence * 4.3
        self.u *= torch.clamp(1.0 - friction * dt, 0.0, 1.0)
        self.v *= torch.clamp(1.0 - friction * dt, 0.0, 1.0)
        
        self.u = torch.clamp(self.u, -250.0, 250.0)
        self.v = torch.clamp(self.v, -250.0, 250.0)

        # Transverse Squeezing waveguides computed individually for each attack band
        for band, squeeze_power in [(shuriken_band, 50.0), (bow_band, 150.0), (magic_band, 80.0)]:
            band_pad = F.pad(band, (1, 1, 1, 1), mode='replicate')
            bgx = 0.5 * (band_pad[:, :, 1:-1, 2:] - band_pad[:, :, 1:-1, :-2])
            bgy = 0.5 * (band_pad[:, :, 2:, 1:-1] - band_pad[:, :, :-2, 1:-1])
            
            speed = torch.sqrt(self.u**2 + self.v**2) + 1e-5
            dot_g_v = bgx * (self.u/speed) + bgy * (self.v/speed)
            
            self.u += (bgx - dot_g_v * (self.u/speed)) * squeeze_power * dt
            self.v += (bgy - dot_g_v * (self.v/speed)) * squeeze_power * dt
            
        # Active Matter Self-Propulsion engine driving each class
        speed = torch.sqrt(self.u**2 + self.v**2) + 1e-5
        self.u += (self.u/speed) * (shuriken_band * 120.0 + bow_band * 450.0 + magic_band * 180.0) * dt
        self.v += (self.v/speed) * (shuriken_band * 120.0 + bow_band * 450.0 + magic_band * 180.0) * dt
        
        # Beta (Green) creates local turbulent vortices to simulate sword slashes
        beta_band = torch.sum(self.density_spectral[:, 18:37], dim=1, keepdim=True)
        gx_b = 0.5 * (F.pad(beta_band, (1,1,1,1), mode='replicate')[:, :, 1:-1, 2:] - F.pad(beta_band, (1,1,1,1), mode='replicate')[:, :, 1:-1, :-2])
        gy_b = 0.5 * (F.pad(beta_band, (1,1,1,1), mode='replicate')[:, :, 2:, 1:-1] - F.pad(beta_band, (1,1,1,1), mode='replicate')[:, :, :-2, 1:-1])
        self.u += -gy_b * 200.0 * dt 
        self.v += gx_b * 200.0 * dt

        # --- TARGET ABSORPTION (ENTROPY SINK) ---
        t0_pwr = sum(a.power for a in self.actors if a.team == 0 and not a.is_dead)
        t1_pwr = sum(a.power for a in self.actors if a.team == 1 and not a.is_dead)
        dmg_modifier = math.sqrt(t0_pwr / (t1_pwr + 1e-5))

        for act in self.actors:
            if act.is_dead: continue
            
            allies = [a for a in self.actors if a.team == act.team and not a.is_dead and a is not act]
            
            for i in range(16):
                nx, ny = act.pin_pos[i, 0].item(), act.pin_pos[i, 1].item()
                gx, gy = int((nx / ARENA_WIDTH) * COMPUTE_RES), int((ny / HEIGHT) * COMPUTE_RES)
                
                if 0 <= gx < COMPUTE_RES and 0 <= gy < COMPUTE_RES:
                    theta_val = self.density_spectral[0, 4:9, gy, gx].sum().item()   # Heal (6Hz)
                    alpha_val = self.density_spectral[0, 9:13, gy, gx].sum().item()  # Shield (10Hz)
                    
                    # --- SPECTRAL TEAM ROUTING: DAMAGE CHECKS (75Hz vs 85Hz sub-channels) ---
                    if act.team == 0:
                        # Team 0 takes damage only from Team 1's low, mid, high Gamma attacks (45, 65, 85)
                        gamma_val = self.density_spectral[0, [45, 65, 85], gy, gx].sum().item()
                    else:
                        # Team 1 takes damage only from Team 0's low, mid, high Gamma attacks (40, 60, 80)
                        gamma_val = self.density_spectral[0, [40, 60, 80], gy, gx].sum().item()
                    
                    # --- SHIELD BARRIER ABSORPTION PHYSICS ---
                    if alpha_val > 0.30:
                        # Shield actively absorbs Gamma (damage) and dampens fluid velocity (kinetic barrier)
                        self.density_spectral[0, 60:100, gy, gx] *= 0.15 
                        self.u[0, 0, gy, gx] *= 0.25 
                        self.v[0, 0, gy, gx] *= 0.25 
                        gamma_val *= 0.15
                    
                    if gamma_val > 0.60: 
                        pwr_scale = max(0.15, min(4.5, dmg_modifier if act.team == 1 else (1.0 / (dmg_modifier + 1e-5))))
                        dmg_rate = gamma_val * 0.08 * pwr_scale 
                        
                        act.node_phases[i] += 4.5 * dt
                        act.edge_intact[i] = False  
                        act.integrity = max(0.0, act.integrity - dmg_rate * dt)
                        act.ult_charge = min(1.0, act.ult_charge + dmg_rate * dt * 0.5)
                        
                        if random.random() < 0.15:
                            self.damage_numbers.append({"pos": [nx, ny], "text": f"-{int(dmg_rate * 100)}", "life": 1.0, "color": (255, 100, 100) if act.team == 0 else (0, 255, 255)})

                        # Target absorbs wave energy
                        self.density_spectral[0, 60:100, gy, gx] *= 0.1
                        self.u[0, 0, gy, gx] *= 0.1
                        self.v[0, 0, gy, gx] *= 0.1
                        
                    # --- SHIELD BALANCING & ROLE ABSORPTION (Theta ONLY heals, Shield does not!) ---
                    if theta_val > 0.30 and act.integrity < 1.0:
                        is_tank = (act.style == "Tank")
                        if not is_tank:
                            heal_rate = theta_val * 0.045 
                            act.integrity = min(1.0, act.integrity + heal_rate * dt)
                            act.edge_intact[i] = True  
                        else:
                            friendly_healers = [e for e in allies if e.weapon == "Heal"]
                            has_friendly_healer_nearby = any(torch.norm(e.pos - act.pos).item() < 140.0 for e in friendly_healers)
                            if has_friendly_healer_nearby:
                                heal_rate = theta_val * 0.025 
                                act.integrity = min(1.0, act.integrity + heal_rate * dt)
                                act.edge_intact[i] = True  
                        
                        self.density_spectral[0, 4:9, gy, gx] *= torch.clamp(torch.tensor(1.0 - 2.5*dt, device=device), 0.0, 1.0)
            
            if act.integrity <= 0.005: act.is_dead = True

        # ADVECTION & PROJECTION
        self.density_spectral = self.solver.advect(self.density_spectral, self.u, self.v, dt)
        self.u = self.solver.advect(self.u, self.u, self.v, dt)
        self.v = self.solver.advect(self.v, self.u, self.v, dt)
        self.u, self.v = self.solver.project(self.u, self.v, self.static_walls)

        # AUTONOMOUS ACTORS STEP
        for act in self.actors:
            if act.is_dead: 
                act.current_action = "[DEAD]"
                continue
            
            enemies = [e for e in self.actors if e.team != act.team and not e.is_dead]
            allies = [a for a in self.actors if a.team == act.team and not a.is_dead and a is not act]
            
            steer_dir = np.array([0.0, 0.0])
            
            if enemies:
                closest_enemy = min(enemies, key=lambda e: torch.norm(e.pos - act.pos).item())
                dist_enemy = torch.norm(closest_enemy.pos - act.pos).item()
                dir_enemy = (closest_enemy.pos - act.pos).cpu().numpy() / (dist_enemy + 1e-5)
                
                # --- COMBAT AI BY WEAPON ---
                if act.weapon == "Magic":
                    if dist_enemy > 250.0: steer_dir = dir_enemy; act.current_action = "[SEEKING]"
                    elif dist_enemy < 180.0: steer_dir = -dir_enemy; act.current_action = "[KITING]"
                    else: act.current_action = "[AIMING]"
                    
                    if act.channel_duration > 0:
                        act.channel_duration -= dt
                        if act.channel_target and not act.channel_target.is_dead:
                            self.channel_magic(act, act.channel_target.pos.cpu().numpy(), dt)
                            steer_dir = np.array([0.0, 0.0])
                        else: act.channel_duration = 0.0 
                    else:
                        act.cooldown -= dt
                        if act.cooldown <= 0:
                            act.channel_target = closest_enemy
                            act.channel_duration = random.uniform(2.0, 3.5)
                            act.cooldown = random.uniform(1.5, 3.0)
                            
                elif act.weapon == "Heal" or act.weapon == "Shield":
                    steer_dir = -dir_enemy
                    if act.channel_duration > 0:
                        act.channel_duration -= dt
                        steer_dir = np.array([0.0, 0.0]) 
                        if act.weapon == "Heal": self.channel_heal(act, dt) 
                        else: self.channel_shield(act, dt) 
                    else:
                        act.cooldown -= dt
                        act.current_action = "[RELOADING]"
                        if act.cooldown <= 0:
                            act.channel_duration = random.uniform(1.0, 2.5)
                            act.cooldown = random.uniform(2.0, 4.0)

                elif act.weapon == "Sword":
                    steer_dir = dir_enemy; act.current_action = "[RUSHING]"
                    if act.channel_duration > 0:
                        act.channel_duration -= dt
                        if act.channel_target and not act.channel_target.is_dead:
                            self.channel_sword(act, act.channel_target.pos.cpu().numpy(), dt)
                            steer_dir = dir_enemy * 0.5 
                        else: act.channel_duration = 0.0
                    else:
                        act.cooldown -= dt
                        if act.cooldown <= 0 and dist_enemy < 70.0:
                            act.channel_target = closest_enemy
                            act.channel_duration = random.uniform(0.5, 1.0)
                            act.cooldown = random.uniform(0.5, 1.0)
                            
                elif act.weapon == "Bow":
                    # FIXED: Orbiting / Strafing movement logic for dynamic kiters (No index errors!)
                    tangent_vec = np.array([-dir_enemy[1], dir_enemy[0]])
                    if dist_enemy > 320.0:
                        steer_dir = dir_enemy * 0.7 + tangent_vec * 0.3
                        act.current_action = "[SEEKING]"
                    elif dist_enemy < 220.0:
                        steer_dir = -dir_enemy * 0.7 + tangent_vec * 0.3
                        act.current_action = "[KITING]"
                    else:
                        # Circle around optimal range
                        steer_dir = tangent_vec * 1.0
                        act.current_action = "[STRAFING]"
                    
                    act.cooldown -= dt
                    if act.cooldown <= 0:
                        success = self.fire_bow(act, closest_enemy.pos.cpu().numpy())
                        act.cooldown = random.uniform(1.2, 1.8) if success else 0.1
                        
                elif act.weapon == "Shuriken":
                    # FIXED: Swift Ninja movement: quick circle-strafes and recoil dodges (No index errors!)
                    tangent_vec = np.array([-dir_enemy[1], dir_enemy[0]])
                    if dist_enemy > 180.0:
                        steer_dir = dir_enemy * 0.8 + tangent_vec * 0.2
                        act.current_action = "[SEEKING]"
                    elif dist_enemy < 110.0:
                        steer_dir = -dir_enemy * 0.9 + tangent_vec * 0.1
                        act.current_action = "[KITING]"
                    else:
                        steer_dir = tangent_vec * 1.0
                        act.current_action = "[STRAFING]"
                    
                    act.cooldown -= dt
                    if act.cooldown <= 0:
                        success = self.fire_shuriken(act, closest_enemy.pos.cpu().numpy())
                        if success:
                            # Elastic dodge jump backward on release!
                            act.pin_pos[:, 0] -= dir_enemy[0] * 35.0
                            act.pin_pos[:, 1] -= dir_enemy[1] * 35.0
                            act.current_action = "[BACKDODGE]"
                            act.cooldown = random.uniform(1.4, 2.0)
                        else:
                            act.cooldown = 0.1
            
            move_speed = 100.0 if act.style == "Fighter" else (60.0 if act.style == "Tank" else 45.0)
            if act.creature == "Furry": move_speed *= 1.4 
            if act.creature == "Demon": move_speed *= 0.8 
            
            act.pin_pos[:, 0] += steer_dir[0] * move_speed * dt
            act.pin_pos[:, 1] += steer_dir[1] * move_speed * dt
            
            act.breather_phase += act.breather_speed * dt
            act.pos.copy_(act.calculate_covariance())
            
            breath_scale = 1.0 + 0.15 * math.sin(act.breather_phase)
            ideal_x = act.pin_x * math.cos(act.angle) * breath_scale + act.pin_y * math.sin(act.angle) * breath_scale
            ideal_y = -act.pin_x * math.sin(act.angle) * breath_scale + act.pin_y * math.cos(act.angle) * breath_scale
            ideal_pos = act.pos.unsqueeze(0) + torch.stack([ideal_x, ideal_y], dim=1)
            
            f_spring = act.update_springs(ideal_pos)
            f_restore = (ideal_pos - act.pin_pos) * (18.0 + act.integrity * 25.0)
            
            node_uv = torch.clamp((act.pin_pos / torch.tensor([ARENA_WIDTH, HEIGHT], device=device)) * 2.0 - 1.0, -1.0, 1.0).view(1, 1, 16, 2)
            fluid_vel = F.grid_sample(torch.stack([self.u, self.v], dim=1).view(1,2,COMPUTE_RES,COMPUTE_RES), node_uv, align_corners=True).squeeze().t() * 75.0
            
            mask = act.edge_intact
            act.pin_pos[mask] += (fluid_vel[mask] + f_restore[mask] + f_spring[mask]) * dt
            
            act.pin_pos[:, 0] = torch.clamp(act.pin_pos[:, 0], 40.0, ARENA_WIDTH - 40.0)
            act.pin_pos[:, 1] = torch.clamp(act.pin_pos[:, 1], 40.0, HEIGHT - 40.0)
            act.pin_pos = apply_cohesion_constraint(act.pin_pos, ideal_pos, 1.25, 0.5)

        # WINNER LOGIC
        if not any(a.team == 0 and not a.is_dead for a in self.actors) and not any(a.team == 1 and not a.is_dead for a in self.actors): 
            self.winner = "Draw"
        elif not any(a.team == 0 and not a.is_dead for a in self.actors): 
            self.winner = "Team 1 (Horde)"
        elif not any(a.team == 1 and not a.is_dead for a in self.actors): 
            self.winner = "Team 0 (Alliance)"

        self.damage_numbers = [ft for ft in self.damage_numbers if (ft.update({'life': ft['life'] - dt, 'pos': [ft['pos'][0], ft['pos'][1] - 35.0*dt]}) or ft['life'] > 0)]

    def resolve_solitonic_collisions(self, dt):
        for i, act_a in enumerate(self.actors):
            if act_a.is_dead: continue
            for j, act_b in enumerate(self.actors):
                if j <= i or act_b.is_dead: continue
                diff = act_a.pos - act_b.pos
                dist = torch.norm(diff).item() + 1e-5
                min_sep = 65.0 if (act_a.name in ["MINI-BOSS", "BOSS", "SUPERBOSS"] or act_b.name in ["MINI-BOSS", "BOSS", "SUPERBOSS"]) else 42.0
                if dist < min_sep:
                    overlap = min_sep - dist
                    act_a.node_phases += 0.8 * overlap * dt
                    act_b.node_phases -= 0.8 * overlap * dt
                    push = (diff / dist) * overlap * 0.52
                    act_a.pin_pos += push
                    act_b.pin_pos -= push

def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("EXOCORTEX SQUAD ARENA: COMBAT ARSENAL UPDATE")
    clock = pygame.time.Clock()
    
    font_large = pygame.font.SysFont("Consolas", 16, bold=True)
    font_med = pygame.font.SysFont("Consolas", 13, bold=True)
    font_small = pygame.font.SysFont("Consolas", 11)
    
    current_stage, ally_level, alch_resources = 1, 20, 0.0
    arena = BossFightArena(master_seed=TOURNAMENT_SEED, current_stage=current_stage, ally_level=ally_level, alch_resources=alch_resources)
    
    running = True
    last_time = time.time()
    
    while running:
        dt = min(0.032, time.time() - last_time)
        last_time = time.time()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r: arena = BossFightArena(master_seed=TOURNAMENT_SEED, current_stage=current_stage, ally_level=ally_level, alch_resources=alch_resources)

        arena.step(dt)
        ally_level, alch_resources = arena.ally_level, arena.alch_resources

        if arena.winner is not None and arena.transition_timer <= 0.0:
            if "Alliance" in arena.winner: current_stage += 1; alch_resources += 500.0 * (1.08) ** current_stage
            arena = BossFightArena(master_seed=TOURNAMENT_SEED, current_stage=current_stage, ally_level=ally_level, alch_resources=alch_resources)

        # --- 1. OPTIMIZED RENDER FLUID ---
        R = torch.clamp(torch.sum(arena.density_spectral * arena.w_red, dim=1) * 255.0, 0.0, 255.0)
        G = torch.clamp(torch.sum(arena.density_spectral * arena.w_green, dim=1) * 255.0, 0.0, 255.0)
        B = torch.clamp(torch.sum(arena.density_spectral * arena.w_blue, dim=1) * 255.0, 0.0, 255.0)
        
        rgb_tensor = torch.stack([R[0], G[0], B[0]], dim=0).to(torch.uint8)
        rgb_tensor[:, arena.static_walls[0, 0] > 0.5] = arena.wall_color
        
        surf_large = pygame.transform.scale(pygame.surfarray.make_surface(np.transpose(rgb_tensor.permute(1, 2, 0).cpu().numpy(), (1, 0, 2))), (ARENA_WIDTH, HEIGHT))
        screen.blit(surf_large, (0, 0))

        # --- 2. RENDER NEON STRESS-LINES ---
        u_np, v_np = arena.u[0, 0].cpu().numpy(), arena.v[0, 0].cpu().numpy()
        for i in range(4, COMPUTE_RES - 4, 8):
            for j in range(4, COMPUTE_RES - 4, 8):
                if (speed := math.hypot(u_np[i, j], v_np[i, j])) > 10.0:
                    pygame.draw.line(screen, (0, int(130 + min(125, speed * 2.0)), 255), 
                                     (int((j / COMPUTE_RES) * ARENA_WIDTH), int((i / COMPUTE_RES) * HEIGHT)), 
                                     (int((j / COMPUTE_RES) * ARENA_WIDTH + (u_np[i, j] / speed) * min(15.0, speed * 0.15)), int((i / COMPUTE_RES) * HEIGHT + (v_np[i, j] / speed) * min(15.0, speed * 0.15))), 1)

        # --- 3. DRAW SOFTBODIES & FLOATING HP BARS ---
        for act in arena.actors:
            if act.is_dead: continue
            nodes = act.pin_pos.cpu().numpy()
            
            for i in range(16):
                if act.edge_intact[i]:
                    pygame.draw.line(screen, act.color, (int(nodes[i, 0]), int(nodes[i, 1])), (int(nodes[(i + 1) % 16, 0]), int(nodes[(i + 1) % 16, 1])), 3)
                pygame.draw.circle(screen, act.color if act.edge_intact[i] else (255, 50, 50), (int(nodes[i, 0]), int(nodes[i, 1])), 3)
                
            bx, by = int(act.pos[0].item() - 20), int(act.pos[1].item() - 35)
            pygame.draw.rect(screen, (40, 40, 50), (bx, by, 40, 4))
            pygame.draw.rect(screen, (0, 255, 150) if act.team == 0 else (255, 100, 100), (bx, by, int(40 * act.integrity), 4))
            
            name_surf = font_small.render(f"{act.name[:8]}", True, (255, 255, 255))
            screen.blit(name_surf, (bx + 20 - name_surf.get_width()//2, by - 12))

        # --- 4. DRAW DAMAGE FLASH TEXTS ---
        for ft in arena.damage_numbers:
            ts = font_large.render(ft['text'], True, ft['color'])
            ts.set_alpha(int(max(0, min(1.0, ft['life'])) * 255))
            screen.blit(ts, (int(ft['pos'][0] - ts.get_width()//2), int(ft['pos'][1])))

        # --- 5. RENDER SIDEBAR HUD ---
        sidebar_x = ARENA_WIDTH
        pygame.draw.rect(screen, (15, 12, 22), pygame.Rect(sidebar_x, 0, WIDTH - ARENA_WIDTH, HEIGHT))
        pygame.draw.line(screen, (0, 255, 200), (sidebar_x, 0), (sidebar_x, HEIGHT), 2)

        t0_hp = sum(a.integrity for a in arena.actors if a.team == 0 and not a.is_dead) / 8.0
        t1_hp = sum(a.integrity for a in arena.actors if a.team == 1 and not a.is_dead) / 8.0

        screen.blit(font_large.render(f"ALLIANCE POWER: {sum(a.power for a in arena.actors if a.team==0)}", True, (0, 255, 200)), (sidebar_x + 15, 12))
        screen.blit(font_large.render(f"HORDE POWER: {sum(a.power for a in arena.actors if a.team==1)}", True, (255, 100, 100)), (sidebar_x + 15, 32))
        screen.blit(font_small.render(f"Vault: {int(alch_resources)} pts | Level: {ally_level}", True, (0, 255, 180)), (sidebar_x + 15, 48))

        pygame.draw.rect(screen, (0, 255, 200), (sidebar_x + 15, 108, int(170 * t0_hp), 8))
        pygame.draw.rect(screen, (255, 100, 100), (sidebar_x + 215, 108, int(170 * t1_hp), 8))

        card_y = 124
        for i in range(8):
            for t_idx, act in enumerate([a for a in arena.actors if a.team == 0][:8] + [a for a in arena.actors if a.team == 1][:8]):
                if t_idx % 8 != i: continue
                cx = sidebar_x + 15 if act.team == 0 else sidebar_x + 210
                pygame.draw.rect(screen, (10, 40, 40) if act.team==0 and not act.is_dead else ((50, 25, 20) if not act.is_dead else (30,30,35)), pygame.Rect(cx, card_y, 175, 52))
                
                status_color = (0, 255, 200) if act.team == 0 else (255, 100, 100)
                if act.channel_duration > 0: status_color = (255, 200, 255) 
                
                # --- HUD AI ACTION FEEDBACK ---
                screen.blit(font_med.render(f"{act.name[:13]}", True, (255, 255, 255) if not act.is_dead else (120, 120, 120)), (cx + 7, card_y + 4))
                screen.blit(font_small.render(f"Lv.{act.level} {act.weapon}", True, status_color if not act.is_dead else (100, 100, 100)), (cx + 7, card_y + 20))
                
                action_color = (255, 255, 100) if "BLOCKED" in act.current_action else ((100, 255, 100) if "SHOOTING" in act.current_action or "STRIKE" in act.current_action or "CHANNELING" in act.current_action or "HEALING" in act.current_action or "GUARDING" in act.current_action or "STRAFING" in act.current_action or "BACKDODGE" in act.current_action else (150, 150, 150))
                screen.blit(font_small.render(act.current_action, True, action_color if not act.is_dead else (100, 100, 100)), (cx + 7, card_y + 35))
                
                if not act.is_dead:
                    pygame.draw.rect(screen, (255, 220, 0), (cx + 100, card_y + 42, int(65 * act.ult_charge), 4))
            card_y += 56

        if arena.winner is not None:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))
            res_text = pygame.font.SysFont("Consolas", 38, bold=True).render(f"STAGE {current_stage} {'COMPLETED' if 'Alliance' in arena.winner else 'FAILED'}", True, (0, 255, 180) if 'Alliance' in arena.winner else (255, 80, 80))
            screen.blit(res_text, (WIDTH // 2 - res_text.get_width()//2, HEIGHT // 2 - 40))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
