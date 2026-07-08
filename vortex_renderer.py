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
        self.title_font = None

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
        
        hide_tiled = getattr(arena, 'cfg', {}).get('hide_tiled_labyrinths', True)
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
        
        # Player Slime rendering with real-time color morphing based on Frequency input
        player_val = cam_player[0, 0].unsqueeze(-1)
        player_val_contrasted = torch.clamp((player_val - 0.04) / 0.75, 0.0, 1.0)
        
        p_color = [0.0, 0.45, 0.65] 
        if hasattr(arena, 'player_pill_name') and arena.player_pill_name in SEMANTIC_PILLS_DB:
            c = SEMANTIC_PILLS_DB[arena.player_pill_name]['color']
            p_color = [c[0]/255.0, c[1]/255.0, c[2]/255.0]
            
        # Frequency morph: shift towards red (Gamma) or purple (Theta)
        if hasattr(arena, 'player_freq_val'):
            freq_shift = float(arena.player_freq_val)
            if freq_shift > 0.0:
                p_color[0] = min(1.0, p_color[0] + freq_shift * 0.7)
                p_color[1] = max(0.0, p_color[1] - freq_shift * 0.4)
            else:
                p_color[2] = min(1.0, p_color[2] + abs(freq_shift) * 0.8)
                p_color[0] = min(1.0, p_color[0] + abs(freq_shift) * 0.3)
                
        jelly_color = torch.tensor(p_color, device=arena.device).view(1, 1, 3)
        membrane_color = torch.tensor([1.0, 1.0, 1.0], device=arena.device).view(1, 1, 3)
        membrane_mask = torch.clamp(1.0 - torch.abs(player_val_contrasted - 0.18) / 0.08, 0.0, 1.0) ** 3.0
        
        vis = vis * (1.0 - player_val_contrasted * 0.6) + jelly_color * player_val_contrasted * 0.6
        vis = vis * (1.0 - membrane_mask * 0.8) + membrane_color * membrane_mask * 0.95
        
        # Bot Slime rendering
        if hasattr(arena, 'bot_density'):
            cam_bot = F.grid_sample(arena.bot_density, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            bot_val = cam_bot[0, 0].unsqueeze(-1)
            bot_val_contrasted = torch.clamp((bot_val - 0.04) / 0.75, 0.0, 1.0)
            
            b_color = (0.6, 0.0, 0.0) 
            if hasattr(arena, 'bot_pill_name') and arena.bot_pill_name in SEMANTIC_PILLS_DB:
                c = SEMANTIC_PILLS_DB[arena.bot_pill_name]['color']
                b_color = [c[0]/255.0, c[1]/255.0, c[2]/255.0]
                
            bot_jelly = torch.tensor(b_color, device=arena.device).view(1, 1, 3)
            bot_membrane = torch.tensor([1.0, 0.2, 0.2], device=arena.device).view(1, 1, 3)
            b_membrane_mask = torch.clamp(1.0 - torch.abs(bot_val_contrasted - 0.18) / 0.08, 0.0, 1.0) ** 3.0
            
            vis = vis * (1.0 - bot_val_contrasted * 0.8) + bot_jelly * bot_val_contrasted * 0.8
            vis = vis * (1.0 - b_membrane_mask * 0.9) + bot_membrane * b_membrane_mask * 0.9

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

    def draw_electrode_sensors(self, surface, arena):
        theta, cos_t, sin_t = -arena.player_angle, math.cos(-arena.player_angle), math.sin(-arena.player_angle)
        
        # Draw player nodes (cyan)
        for i in range(16):
            if hasattr(arena, 'player_edge_intact') and not arena.player_edge_intact[i]: continue 
            
            px, py = (arena.pin_pos[i, 0].item(), arena.pin_pos[i, 1].item()) if hasattr(arena, 'pin_pos') else (arena.player_pin_pos[i, 0].item(), arena.player_pin_pos[i, 1].item())
            dx = (px - arena.player_pos[0].item() + self.WIDTH/2) % self.WIDTH - self.WIDTH/2
            dy = (py - arena.player_pos[1].item() + self.HEIGHT/2) % self.HEIGHT - self.HEIGHT/2
            sx = self.WIDTH / 2.0 + (dx * cos_t + dy * sin_t) / self.ZOOM
            sy = self.HEIGHT / 2.0 + (-dx * sin_t + dy * cos_t) / self.ZOOM
            
            pygame.draw.circle(surface, (0, 255, 255, 100), (int(sx), int(sy)), 8, 1)
            pygame.draw.circle(surface, (0, 255, 255), (int(sx), int(sy)), 2)
            
        # Draw bot nodes (deep orange/red) if in combat arena
        if hasattr(arena, 'bot_pin_pos'):
            for i in range(16):
                if not arena.bot_edge_intact[i]: continue
                px, py = arena.bot_pin_pos[i, 0].item(), arena.bot_pin_pos[i, 1].item()
                dx = (px - arena.player_pos[0].item() + self.WIDTH/2) % self.WIDTH - self.WIDTH/2
                dy = (py - arena.player_pos[1].item() + self.HEIGHT/2) % self.HEIGHT - self.HEIGHT/2
                sx = self.WIDTH / 2.0 + (dx * cos_t + dy * sin_t) / self.ZOOM
                sy = self.HEIGHT / 2.0 + (-dx * sin_t + dy * cos_t) / self.ZOOM
                
                pygame.draw.circle(surface, (255, 100, 0, 100), (int(sx), int(sy)), 8, 1)
                pygame.draw.circle(surface, (255, 100, 0), (int(sx), int(sy)), 2)

    def draw_combat_debug(self, surface, arena):
        """ Draws a real-time spectroscopy panel with coupled oscillator diagnostic visuals """
        panel_w, panel_h = 320, 310
        panel_x, panel_y = self.WIDTH - panel_w - 15, 15
        
        s = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        s.fill((15, 10, 20, 215)) 
        surface.blit(s, (panel_x, panel_y))
        pygame.draw.rect(surface, (255, 100, 100), (panel_x, panel_y, panel_w, panel_h), 1)
        
        if self.font is None:
            self.font = pygame.font.SysFont("Consolas", 13, bold=True)
            
        title = self.font.render("CONNECTOME SPECTROSCOPY (DIAG)", True, (255, 100, 100))
        surface.blit(title, (panel_x + 10, panel_y + 8))
        
        y_offset = panel_y + 30
        
        # Raw Diagnostic Extraction
        raw_ax = getattr(arena, 'raw_axes', [])
        raw_btn = getattr(arena, 'raw_buttons', [])
        
        if raw_ax:
            ax_str = ", ".join([f"{val:.2f}" for val in raw_ax[:6]])
            ax_line = f"Raw Axes: [{ax_str}]"
        else:
            ax_line = "Raw Axes: No Gamepad Driver"
            
        if raw_btn:
            pressed_btns = [i for i, btn in enumerate(raw_btn) if btn]
            btn_line = f"Raw Btns: {pressed_btns}"
        else:
            btn_line = "Raw Btns: None"

        # Calculate continuous float health based on Kuramoto phase order parameter H
        p_integ = getattr(arena, 'player_integrity', 1.0)
        b_integ = getattr(arena, 'bot_integrity', 1.0)

        metrics = [
            f"Your Core   : {arena.player_pill_name}",
            f"Rogue Core  : {arena.bot_pill_name}",
            f"Coupling (K): {arena.player_K:.1f} vs {arena.bot_K:.1f}",
            "-" * 37,
            ax_line,
            btn_line,
            f"Freq (Z/C/LB/RB): {arena.player_freq_val:.2f}",
            f"Spat (X/V/L2/R2): {arena.player_spatial_val:.2f}",
            "-" * 37,
            f"Your Dissonance: {arena.player_shear_stress:.2f}",
            f"Bot Dissonance : {arena.bot_shear_stress:.2f}",
            f"Your Density   : {getattr(arena, 'player_density_val', 0.0):.3f}",
            f"Bot Density    : {getattr(arena, 'bot_density_val', 0.0):.3f}",
            f"Your Jitter F  : {getattr(arena, 'player_jitter_avg', 0.0):.1f} N",
            f"Bot Jitter F   : {getattr(arena, 'bot_jitter_avg', 0.0):.1f} N",
            f"Absorbed E     : {arena.energy_absorbed:.2f} / 12.0",
            f"Clash Border   : {arena.clash_intensity * 100:.1f}% (Feigenbaum)",
            "-" * 37,
            f"Phase Integrity (P): {p_integ*100:.1f}%",
            f"Phase Integrity (B): {b_integ*100:.1f}%",
        ]
        
        for met in metrics:
            if met.startswith("-"):
                text = self.font.render(met, True, (255, 100, 100))
            elif "Axes" in met or "Btns" in met:
                text = self.font.render(met, True, (0, 255, 100) if raw_ax else (150, 150, 150))
            elif "Dissonance" in met or "Clash" in met:
                text = self.font.render(met, True, (255, 200, 150))
            elif "Absorbed" in met:
                text = self.font.render(met, True, (100, 255, 100) if arena.energy_absorbed > 0 else (255, 255, 255))
            elif "Freq" in met or "Spat" in met:
                text = self.font.render(met, True, (0, 255, 255))
            elif "Jitter" in met:
                text = self.font.render(met, True, (255, 100, 100) if "Bot" in met else (100, 255, 255))
            elif "Integrity" in met:
                val = p_integ if "(P)" in met else b_integ
                col = (0, 255, 100) if val > 0.7 else ((255, 200, 0) if val > 0.4 else (255, 50, 50))
                text = self.font.render(met, True, col)
            else:
                text = self.font.render(met, True, (255, 255, 255))
            surface.blit(text, (panel_x + 12, y_offset))
            y_offset += 12

        # Draw the polari-oscillator Kuramoto Circle (Connectome Phase Radar)
        cx, cy = panel_x + panel_w - 45, panel_y + 110
        pygame.draw.circle(surface, (50, 40, 60), (cx, cy), 30, 1)
        
        p_phases = getattr(arena, 'player_node_phases', None)
        if p_phases is not None:
            p_phases_cpu = p_phases.cpu().numpy()
            for i in range(16):
                px = cx + int(27 * math.cos(p_phases_cpu[i]))
                py = cy + int(27 * math.sin(p_phases_cpu[i]))
                pygame.draw.circle(surface, (0, 255, 200), (px, py), 2)
                
        b_phases = getattr(arena, 'bot_node_phases', None)
        if b_phases is not None:
            b_phases_cpu = b_phases.cpu().numpy()
            for i in range(16):
                bx = cx + int(18 * math.cos(b_phases_cpu[i]))
                by = cy + int(18 * math.sin(b_phases_cpu[i]))
                pygame.draw.circle(surface, (255, 50, 100), (bx, by), 2)

    def draw_combat_ui(self, surface, arena):
        """ Render UI specific to the Arena Domain Clash """
        if self.font is None:
            self.font = pygame.font.SysFont("Consolas", 18, bold=True)
            self.title_font = pygame.font.SysFont("Consolas", 42, bold=True)
            
        self.draw_combat_debug(surface, arena)
            
        # Top HUD
        diff_text = self.font.render(f"TRIBULATION LEVEL: {arena.difficulty}", True, (255, 100, 100))
        surface.blit(diff_text, (self.WIDTH // 2 - diff_text.get_width()//2, 20))
        
        p_integ = getattr(arena, 'player_integrity', 1.0)
        b_integ = getattr(arena, 'bot_integrity', 1.0)

        # Draw continuous Phase-Locked health bars (cohesion derived from Kuramoto order parameter)
        p_text = self.font.render(f"YOUR PHASIC INTEGRITY: {p_integ*100:.1f}%", True, (0, 255, 200))
        surface.blit(p_text, (50, self.HEIGHT - 50))
        
        b_text = self.font.render(f"ROGUE INTEGRITY: {b_integ*100:.1f}%", True, (255, 50, 50))
        surface.blit(b_text, (self.WIDTH - b_text.get_width() - 50, self.HEIGHT - 50))
        
        # Pill Matchups and visual Domain shockwave indicators
        p_ch_val = int(arena.player_domain_charge * 100.0)
        p_pill_text = self.font.render(f"DOMAIN: {arena.player_pill_name} [Pulse: {p_ch_val}%]", True, (200, 255, 220))
        surface.blit(p_pill_text, (50, self.HEIGHT - 80))
        
        b_ch_val = int(arena.bot_domain_charge * 100.0)
        b_pill_text = self.font.render(f"DOMAIN: {arena.bot_pill_name} [Pulse: {b_ch_val}%]", True, (255, 200, 200))
        surface.blit(b_pill_text, (self.WIDTH - b_pill_text.get_width() - 50, self.HEIGHT - 80))

        # Visual indicator of the domain charging aura around the player
        if hasattr(arena, 'player_domain_charge') and arena.player_domain_charge > 0.05:
            cx, cy = self.WIDTH // 2, self.HEIGHT // 2
            aura_radius = int(25.0 * arena.player_domain_charge / self.ZOOM)
            pygame.draw.circle(surface, (255, 0, 255, 80), (cx, cy), aura_radius + 15, 2)

        # Winner Splash Screen
        if arena.winner:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            surface.blit(overlay, (0, 0))
            
            result = "DOMAIN SUPREME - VICTORY!" if arena.winner == "Player" else "DOMAIN SHATTERED - DEFEAT"
            color = (0, 255, 100) if arena.winner == "Player" else (255, 50, 50)
            
            res_text = self.title_font.render(result, True, color)
            surface.blit(res_text, (self.WIDTH // 2 - res_text.get_width()//2, self.HEIGHT // 2 - 50))
            
            info = self.font.render("Transitioning to Next Cycle...", True, (200, 200, 200))
            surface.blit(info, (self.WIDTH // 2 - info.get_width()//2, self.HEIGHT // 2 + 20))

    def draw_debug_window(self, surface, arena):
        # Prevent Labyrinth debug from drawing during combat
        if hasattr(arena, 'bot_density'): return
        
        if not getattr(arena, 'cfg', {}).get('show_debug_window', True):
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
        if hasattr(arena, 'bot_density'): return 
        if not hasattr(arena, 'alchemy_entities'): return
        
        theta = -arena.player_angle
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        def project(wx, wy):
            dx = (wx - arena.player_pos[0].item() + self.WIDTH/2) % self.WIDTH - self.WIDTH/2
            dy = (wy - arena.player_pos[1].item() + self.HEIGHT/2) % self.HEIGHT - self.HEIGHT/2
            sx = self.WIDTH / 2.0 + (dx * cos_t + dy * sin_t) / self.ZOOM
            sy = self.HEIGHT / 2.0 + (-dx * sin_t + dy * cos_t) / self.ZOOM
            return int(sx), int(sy)

        cx, cy = project(arena.cauldron_pos[0].item(), arena.cauldron_pos[1].item())
        rad_cauldron = int(arena.cell_w * 1.5 / self.ZOOM)
        pygame.draw.circle(surface, (150, 150, 150), (cx, cy), rad_cauldron, 2)

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
        if hasattr(arena, 'bot_density'):
            self.draw_combat_ui(surface, arena)
            return
            
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
