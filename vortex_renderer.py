# vortex_renderer.py
import pygame
import torch
import torch.nn.functional as F
import numpy as np
import math
from implicit_config import ALCHEMY_ENTITIES_CONFIG, SEMANTIC_PILLS_DB

class VortexRenderer:
    def __init__(self, width, height, zoom_factor=1.35):
        self.WIDTH = width
        self.HEIGHT = height
        self.ZOOM = zoom_factor
        self.font = None

    def render_field(self, arena):
        theta = -arena.player_angle
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        
        px_norm = (arena.player_pos[0] / self.WIDTH) * 2.0 - 1.0
        py_norm = (arena.player_pos[1] / self.HEIGHT) * 2.0 - 1.0
        
        M = torch.tensor([[
            [cos_t * self.ZOOM, -sin_t * self.ZOOM, px_norm],
            [sin_t * self.ZOOM,  cos_t * self.ZOOM, py_norm]
        ]], dtype=torch.float32, device=arena.device)
        
        grid = F.affine_grid(M, size=(1, 3, arena.res, arena.res), align_corners=True)
        
        hide_tiled = arena.cfg.get('hide_tiled_labyrinths', True)
        if not hide_tiled:
            grid = torch.remainder(grid + 1.0, 2.0) - 1.0
        
        cam_density = F.grid_sample(arena.density, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        cam_walls = F.grid_sample(arena.wall_density, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        cam_player = F.grid_sample(arena.player_density, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        vis = torch.clamp(cam_density[0].permute(1, 2, 0), 0.0, 1.0)
        vis = torch.where(vis == 0.0, torch.tensor([12.0/255.0]*3, device=arena.device), vis)
        
        wall_val = cam_walls[0, 0].unsqueeze(-1)
        wall_color = torch.tensor([0.9, 0.15, 0.6], device=arena.device).view(1, 1, 3)
        vis = vis * (1.0 - wall_val * 0.94) + wall_color * wall_val * 0.85
        
        player_val = cam_player[0, 0].unsqueeze(-1)
        player_val_contrasted = torch.clamp((player_val - 0.04) / 0.75, 0.0, 1.0)
        jelly_color = torch.tensor([0.0, 0.45, 0.65], device=arena.device).view(1, 1, 3)
        membrane_color = torch.tensor([0.2, 1.0, 0.95], device=arena.device).view(1, 1, 3)
        membrane_mask = torch.clamp(1.0 - torch.abs(player_val_contrasted - 0.18) / 0.08, 0.0, 1.0) ** 3.0
        
        vis = vis * (1.0 - player_val_contrasted * 0.6) + jelly_color * player_val_contrasted * 0.6
        vis = vis * (1.0 - membrane_mask * 0.8) + membrane_color * membrane_mask * 0.95
        
        cam_orig = F.grid_sample(arena.orig_obstacles, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        orig_val = cam_orig[0, 0].unsqueeze(-1)
        vis = torch.where(orig_val > 0.5, vis * 0.45 + torch.tensor([0.22, 0.05, 0.5], device=arena.device) * 0.55, vis)
        
        vis_transposed = vis.permute(2, 0, 1).unsqueeze(0) 
        vis_resized = F.interpolate(vis_transposed, size=(self.HEIGHT, self.WIDTH), mode='bilinear', align_corners=True).squeeze(0)
        vis_resized = vis_resized.permute(1, 2, 0) 
        
        rgb = (vis_resized * 255).to(torch.uint8).cpu().numpy()
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        return surf

    def draw_tension_lines(self, surface, arena):
        u_cpu, v_cpu = arena.u[0, 0].cpu().numpy(), arena.v[0, 0].cpu().numpy()
        theta, cos_t, sin_t = -arena.player_angle, math.cos(-arena.player_angle), math.sin(-arena.player_angle)
        step = arena.res // 24
        
        for i in range(1, 23):
            for j in range(1, 23):
                gy, gx = i * step, j * step
                wx, wy = (gx / float(arena.res)) * self.WIDTH, (gy / float(arena.res)) * self.HEIGHT
                
                dx = (wx - arena.player_pos[0].item() + self.WIDTH/2) % self.WIDTH - self.WIDTH/2
                dy = (wy - arena.player_pos[1].item() + self.HEIGHT/2) % self.HEIGHT - self.HEIGHT/2
                
                sx = self.WIDTH / 2.0 + (dx * cos_t + dy * sin_t) / self.ZOOM
                sy = self.HEIGHT / 2.0 + (-dx * sin_t + dy * cos_t) / self.ZOOM
                
                if 0 <= sx <= self.WIDTH and 0 <= sy <= self.HEIGHT:
                    vx_w, vy_w = float(u_cpu[gy, gx]), float(v_cpu[gy, gx])
                    vx_cam = (vx_w * cos_t + vy_w * sin_t) / self.ZOOM
                    vy_cam = (-vx_w * sin_t + vy_w * cos_t) / self.ZOOM
                    
                    speed = math.hypot(vx_cam, vy_cam)
                    if speed > 0.5:
                        max_len = 15.0
                        draw_vx = (vx_cam / speed) * max_len if speed > max_len else vx_cam
                        draw_vy = (vy_cam / speed) * max_len if speed > max_len else vy_cam
                        
                        ex, ey = sx + draw_vx * 1.5, sy + draw_vy * 1.5
                        if math.isfinite(ex) and math.isfinite(ey):
                            col_f = min(1.0, speed / 40.0)
                            pygame.draw.line(surface, (0, int(150 + col_f * 105), int(255 - col_f * 100)), (int(sx), int(sy)), (int(ex), int(ey)), 1)

    def draw_coherence_bridges(self, surface, arena):
        c0_matrix = arena.eeg_c0_matrix.cpu().numpy()
        theta, cos_t, sin_t = -arena.player_angle, math.cos(-arena.player_angle), math.sin(-arena.player_angle)
        
        screen_coords = []
        for i in range(16):
            px, py = arena.pin_pos[i, 0].item(), arena.pin_pos[i, 1].item()
            dx = (px - arena.player_pos[0].item() + self.WIDTH/2) % self.WIDTH - self.WIDTH/2
            dy = (py - arena.player_pos[1].item() + self.HEIGHT/2) % self.HEIGHT - self.HEIGHT/2
            
            sx = self.WIDTH / 2.0 + (dx * cos_t + dy * sin_t) / self.ZOOM
            sy = self.HEIGHT / 2.0 + (-dx * sin_t + dy * cos_t) / self.ZOOM
            screen_coords.append((int(sx), int(sy)))
            
        max_val = c0_matrix.max() if c0_matrix.max() > 1e-5 else 1.0
        for i in range(16):
            for j in range(i + 1, 16):
                val = float(c0_matrix[i, j])
                if val > 0.05:
                    ratio = val / max_val
                    col = (int(50 + 205 * ratio), int(255 - 155 * ratio), 255)
                    thickness = max(1, int(ratio * 4.5))
                    pygame.draw.line(surface, col, screen_coords[i], screen_coords[j], thickness)

    def draw_electrode_sensors(self, surface, arena):
        if hasattr(arena, 'eeg_c0_matrix') and arena.eeg_c0_matrix.sum().item() > 0.05:
            self.draw_coherence_bridges(surface, arena)

        theta, cos_t, sin_t = -arena.player_angle, math.cos(-arena.player_angle), math.sin(-arena.player_angle)
        
        for i in range(16):
            is_cap = arena.pin_captured[i]
            col = (255, 0, 255) if is_cap else (0, 255, 255)
            
            px, py = arena.pin_pos[i, 0].item(), arena.pin_pos[i, 1].item()
            dx = (px - arena.player_pos[0].item() + self.WIDTH/2) % self.WIDTH - self.WIDTH/2
            dy = (py - arena.player_pos[1].item() + self.HEIGHT/2) % self.HEIGHT - self.HEIGHT/2
            
            sx = self.WIDTH / 2.0 + (dx * cos_t + dy * sin_t) / self.ZOOM
            sy = self.HEIGHT / 2.0 + (-dx * sin_t + dy * cos_t) / self.ZOOM
            
            pygame.draw.circle(surface, (*col, 100), (int(sx), int(sy)), 8, 1)
            pygame.draw.circle(surface, col, (int(sx), int(sy)), 2)

    def draw_rune_zones(self, surface, arena):
        if not hasattr(arena, 'rune_zones') or not arena.rune_zones:
            return
            
        theta = -arena.player_angle
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        
        if self.font is None:
            self.font = pygame.font.SysFont("Consolas", 14, bold=True)
            
        for zone in arena.rune_zones:
            zx, zy = zone['pos'][0].item(), zone['pos'][1].item()
            
            dx = (zx - arena.player_pos[0].item() + self.WIDTH/2) % self.WIDTH - self.WIDTH/2
            dy = (zy - arena.player_pos[1].item() + self.HEIGHT/2) % self.HEIGHT - self.HEIGHT/2
            
            sx = self.WIDTH / 2.0 + (dx * cos_t + dy * sin_t) / self.ZOOM
            sy = self.HEIGHT / 2.0 + (-dx * sin_t + dy * cos_t) / self.ZOOM
            
            if 0 <= sx <= self.WIDTH and 0 <= sy <= self.HEIGHT:
                rad = int(zone['radius'] / self.ZOOM)
                
                if zone['completed']:
                    clf = zone['classification']
                    color = (0, 255, 120)
                    pygame.draw.circle(surface, color, (int(sx), int(sy)), rad, 3)
                    text_str = f"RUNE: {clf.upper()}"
                    text_main = self.font.render(text_str, True, color)
                    surface.blit(text_main, (int(sx) - text_main.get_width()//2, int(sy) - rad - 20))
                else:
                    pulse = int(100 + 40 * math.sin(pygame.time.get_ticks() * 0.005))
                    color = (pulse, pulse, 50)
                    pygame.draw.circle(surface, color, (int(sx), int(sy)), rad, 1)

    def draw_debug_window(self, surface, arena):
        if not arena.cfg.get('show_debug_window', True):
            return
            
        panel_w, panel_h = 300, 310
        panel_x, panel_y = self.WIDTH - panel_w - 15, 15
        
        s = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        s.fill((10, 15, 25, 215))
        surface.blit(s, (panel_x, panel_y))
        pygame.draw.rect(surface, (0, 255, 200), (panel_x, panel_y, panel_w, panel_h), 1)
        
        if self.font is None:
            self.font = pygame.font.SysFont("Consolas", 13, bold=True)
            
        title = self.font.render("EXOCORTEX SPECTROSCOPY", True, (0, 255, 200))
        surface.blit(title, (panel_x + 10, panel_y + 8))
        
        q = arena.pill_quality
        if q >= 95.0: grade = "Divine Core"
        elif q >= 80.0: grade = "Saint Core"
        elif q >= 60.0: grade = "Spiritual Core"
        elif q >= 40.0: grade = "Mortal Core"
        else: grade = "Impure Slag"

        y_offset = panel_y + 30
        
        metrics = [
            f"Attractor Freq: {arena.target_freq_desc}",
            f"Attractor Spat: {arena.target_spat_desc}",
            "-" * 35,
            f"Freq Match  : {arena.score_resonance * 100:.1f}%",
            f"Envelope    : {arena.score_containment * 100:.1f}%",
            f"Temp Stabil : {arena.score_temp * 100:.1f}% ({arena.cauldron_temp:.0f} K)",
            f"Vortex Spin : {arena.score_vortex * 100:.1f}%",
            "-" * 35,
            f"Progress    : {arena.smelting_progress * 100:.1f}%",
            f"Emergent ID : {arena.emergent_pill_name}",
            f"ID Match    : {arena.emergent_pill_similarity * 100:.1f}%",
            f"Pill Quality: {arena.pill_quality:.1f}% [{grade}]",
        ]
        
        for met in metrics:
            if met.startswith("-"):
                text = self.font.render(met, True, (0, 255, 200))
            elif "Match" in met or "Envelope" in met or "Stabil" in met or "Spin" in met:
                text = self.font.render(met, True, (200, 255, 220))
            elif "Quality" in met:
                col = (0, 255, 100) if q >= 80.0 else ((255, 200, 0) if q >= 40.0 else (255, 100, 100))
                text = self.font.render(met, True, col)
            elif "Emergent" in met:
                text = self.font.render(met, True, (255, 150, 255))
            else:
                text = self.font.render(met, True, (255, 255, 255))
            surface.blit(text, (panel_x + 12, y_offset))
            y_offset += 16
            
        # Радар фаз
        cx, cy = panel_x + panel_w // 2, y_offset + 32
        radar_rad = 28
        pygame.draw.circle(surface, (40, 60, 80), (cx, cy), radar_rad, 1)
        
        com = arena.player_pos
        cos_p, sin_p = math.cos(-arena.player_angle), math.sin(-arena.player_angle)
        
        for i in range(16):
            ix = arena.pin_x[i].item() / 12.0 * radar_rad
            iy = arena.pin_y[i].item() / 12.0 * radar_rad
            pygame.draw.circle(surface, (0, 150, 100), (int(cx + ix), int(cy + iy)), 2, 1)
            
            rx = arena.pin_pos[i, 0].item() - com[0].item()
            ry = arena.pin_pos[i, 1].item() - com[1].item()
            loc_x = (rx * cos_p - ry * sin_p) / (arena.cell_w * 0.4) * radar_rad
            loc_y = (rx * sin_p + ry * cos_p) / (arena.cell_w * 0.4) * radar_rad
            pygame.draw.circle(surface, (255, 220, 50), (int(cx + loc_x), int(cy + loc_y)), 2)

    def draw_alchemy_ui(self, surface, arena):
        if not hasattr(arena, 'alchemy_entities'): return
        
        theta = -arena.player_angle
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        def project(wx, wy):
            dx = (wx - arena.player_pos[0].item() + self.WIDTH/2) % self.WIDTH - self.WIDTH/2
            dy = (wy - arena.player_pos[1].item() + self.HEIGHT/2) % self.HEIGHT - self.HEIGHT/2
            sx = self.WIDTH / 2.0 + (dx * cos_t + dy * sin_t) / self.ZOOM
            sy = self.HEIGHT / 2.0 + (-dx * sin_t + dy * cos_t) / self.ZOOM
            return int(sx), int(sy)

        # Отрисовка Котла
        cx, cy = project(arena.cauldron_pos[0].item(), arena.cauldron_pos[1].item())
        rad_cauldron = int(arena.cell_w * 1.5 / self.ZOOM)
        pygame.draw.circle(surface, (150, 150, 150), (cx, cy), rad_cauldron, 2)

        # Отрисовка Сущностей на основе их векторов из конфигурации
        for ent in arena.alchemy_entities:
            ex, ey = project(ent['pos'][0].item(), ent['pos'][1].item())
            
            if ent['type'] == 'pill': 
                vec = ent['vector']
                col = (int(vec[0]*255), int(vec[1]*255), int(vec[2]*255))
                icon = "EXOCORTEX PILL"
            else:
                ent_cfg = next((c for c in ALCHEMY_ENTITIES_CONFIG if c['type'] == ent['type']), None)
                if ent_cfg:
                    col = ent_cfg['color']
                    icon = f"{ent_cfg['name']} ({ent_cfg['freq']:.0f}Hz)"
                else:
                    col, icon = (255, 255, 255), f"COGNIT: {ent['type'].upper()} ({ent['freq']:.0f}Hz)"

            pulse = int(40 + 40 * math.sin(pygame.time.get_ticks() * 0.005))
            pygame.draw.circle(surface, col, (ex, ey), 12)
            pygame.draw.circle(surface, (255, 255, 255), (ex, ey), 12 + pulse//10, 2)
            
            if self.font is None: self.font = pygame.font.SysFont("Consolas", 14, bold=True)
            text = self.font.render(icon, True, (255, 255, 255))
            surface.blit(text, (ex - text.get_width()//2, ey - 22))

        # UI Прогресса Плавки
        if arena.smelting_progress > 0.0 and not arena.pill_created:
            bar_w, bar_h = 300, 24
            bx, by = self.WIDTH // 2 - bar_w // 2, self.HEIGHT - 80
            pygame.draw.rect(surface, (40, 40, 40), (bx, by, bar_w, bar_h))
            pygame.draw.rect(surface, (255, 180, 0), (bx, by, int(bar_w * arena.smelting_progress), bar_h))
            pygame.draw.rect(surface, (255, 255, 255), (bx, by, bar_w, bar_h), 2)
            
            pump_active = arena.score_temp > 0.5 and (arena.score_resonance * arena.score_containment) > 0.5
            status_text = "BIFURCATION PUMP ACTIVE" if pump_active else "ALIGNMENT REQUIRED..."
            color = (0, 255, 255) if pump_active else (255, 255, 255)
            
            text = self.font.render(status_text, True, color)
            surface.blit(text, (self.WIDTH // 2 - text.get_width()//2, by - 25))
            
        elif arena.pill_created:
            text = self.font.render(f"PILL FORGED: {arena.emergent_pill_name}", True, (255, 200, 0))
            surface.blit(text, (self.WIDTH // 2 - text.get_width()//2, self.HEIGHT - 60))

    def draw_ui(self, surface, arena):
        self.draw_rune_zones(surface, arena)
        self.draw_debug_window(surface, arena)
        self.draw_alchemy_ui(surface, arena)

        cell_w = (self.WIDTH * 0.8) / arena.maze.dim
        gx = (self.WIDTH * 0.1) + (arena.goal_cell[0] + 0.5) * cell_w
        gy = (self.HEIGHT * 0.1) + (arena.goal_cell[1] + 0.5) * cell_w
        
        dx = (gx - arena.player_pos[0].item() + self.WIDTH/2) % self.WIDTH - self.WIDTH/2
        dy = (gy - arena.player_pos[1].item() + self.HEIGHT/2) % self.HEIGHT - self.HEIGHT/2
        
        theta = -arena.player_angle
        sx = self.WIDTH / 2.0 + (dx * math.cos(theta) + dy * math.sin(theta)) / self.ZOOM
        sy = self.HEIGHT / 2.0 + (-dx * math.sin(theta) + dy * math.cos(theta)) / self.ZOOM
        
        if 0 <= sx <= self.WIDTH and 0 <= sy <= self.HEIGHT:
            pygame.draw.circle(surface, (0, 255, 100), (int(sx), int(sy)), int(cell_w * 0.25 / self.ZOOM), 1)

        cx, cy = self.WIDTH // 2, self.HEIGHT // 2
        pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 14, 1)
        pygame.draw.circle(surface, (0, 255, 255), (cx, cy), 4)
