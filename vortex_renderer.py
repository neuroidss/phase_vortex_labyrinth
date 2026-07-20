# vortex_renderer.py
import pygame
import torch
import torch.nn.functional as F
import numpy as np
import math
from implicit_config import ALCHEMY_ENTITIES_CONFIG, SEMANTIC_PILLS_DB

class VortexRenderer:
    """
    Affine Rendering Engine with Integrated Connectome Feedback.
    Handles the spatial coordinates transformation and maps non-linear 
    wave dynamics (NLSE) onto high-fidelity visual assets.
    """
    def __init__(self, width, height, zoom_factor=1.35):
        self.WIDTH = width
        self.HEIGHT = height
        self.ZOOM = zoom_factor
        self.debug_font = None
        self.alchemy_font = None
        self.hud_font = None
        self.title_font = None

    def render_field(self, arena):
        if hasattr(arena, 'maze'):
            theta = -arena.player_angle
            px_norm = (arena.player_pos[0] / self.WIDTH) * 2.0 - 1.0
            py_norm = (arena.player_pos[1] / self.HEIGHT) * 2.0 - 1.0
        else:
            theta = 0.0
            px_norm, py_norm = 0.0, 0.0

        cos_t, sin_t = math.cos(theta), math.sin(theta)
        
        M = torch.tensor([[
            [cos_t * self.ZOOM, -sin_t * self.ZOOM, px_norm],
            [sin_t * self.ZOOM,  cos_t * self.ZOOM, py_norm]
        ]], dtype=torch.float32, device=arena.device)
        
        grid = F.affine_grid(M, size=(1, 3, arena.res, arena.res), align_corners=True)
        
        cam_density = F.grid_sample(arena.density, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        vis = torch.clamp(cam_density[0].permute(1, 2, 0), 0.0, 1.0)
        vis = torch.where(vis == 0.0, torch.tensor([12.0/255.0, 15.0/255.0, 20.0/255.0], device=arena.device), vis)
        
        cam_walls = F.grid_sample(arena.wall_density, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        wall_val = cam_walls[0, 0].unsqueeze(-1)
        wall_color = torch.tensor([0.9, 0.15, 0.6], device=arena.device).view(1, 1, 3)
        vis = vis * (1.0 - wall_val * 0.94) + wall_color * wall_val * 0.85
        
        if hasattr(arena, 'team0_density') or hasattr(arena, 'player_density'):
            dens0 = arena.team0_density if hasattr(arena, 'team0_density') else arena.player_density
            dens1 = arena.team1_density if hasattr(arena, 'team1_density') else getattr(arena, 'bot_density', None)
            
            cam_t0 = F.grid_sample(dens0, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            t0_val = torch.clamp((cam_t0[0, 0].unsqueeze(-1) - 0.04) / 0.75, 0.0, 1.0)
            c0 = torch.tensor([0.0, 0.8, 0.8], device=arena.device).view(1, 1, 3)
            vis = vis * (1.0 - t0_val * 0.6) + c0 * t0_val * 0.6

            if dens1 is not None:
                cam_t1 = F.grid_sample(dens1, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
                t1_val = torch.clamp((cam_t1[0, 0].unsqueeze(-1) - 0.04) / 0.75, 0.0, 1.0)
                c1 = torch.tensor([0.8, 0.2, 0.2], device=arena.device).view(1, 1, 3)
                vis = vis * (1.0 - t1_val * 0.6) + c1 * t1_val * 0.6
            
        vis_transposed = vis.permute(2, 0, 1).unsqueeze(0) 
        vis_resized = F.interpolate(vis_transposed, size=(self.HEIGHT, self.WIDTH), mode='bilinear', align_corners=True).squeeze(0)
        vis_resized = vis_resized.permute(1, 2, 0) 
        
        rgb = (vis_resized * 255).to(torch.uint8).cpu().numpy()
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        return surf

    def draw_tension_lines(self, surface, arena):
        u_cpu, v_cpu = arena.u[0, 0].cpu().numpy(), arena.v[0, 0].cpu().numpy()
        if hasattr(arena, 'maze'):
            theta = -arena.player_angle
            cx, cy = arena.player_pos[0].item(), arena.player_pos[1].item()
        else:
            theta = 0.0
            cx, cy = self.WIDTH / 2.0, self.HEIGHT / 2.0

        cos_t, sin_t = math.cos(theta), math.sin(theta)
        step = arena.res // 24
        
        for i in range(1, 23):
            for j in range(1, 23):
                gy, gx = i * step, j * step
                wx, wy = (gx / float(arena.res)) * self.WIDTH, (gy / float(arena.res)) * self.HEIGHT
                
                dx, dy = wx - cx, wy - cy
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

    def draw_soliton_telemetry(self, surface, arena):
        """ Рендерит неоновые видоискатели и данные над летящими солитонами Шрёдингера """
        if not hasattr(arena, 'tracked_solitons') or not arena.tracked_solitons: return
        if self.debug_font is None: self.debug_font = pygame.font.SysFont("Consolas", 11, bold=True)
        
        if hasattr(arena, 'maze'):
            theta = -arena.player_angle
            cx, cy = arena.player_pos[0].item(), arena.player_pos[1].item()
        else:
            theta = 0.0
            cx, cy = self.WIDTH / 2.0, self.HEIGHT / 2.0

        cos_t, sin_t = math.cos(theta), math.sin(theta)
        
        for sol in arena.tracked_solitons:
            px, py = sol['pos'][0], sol['pos'][1]
            sx = self.WIDTH / 2.0 + ((px - cx) * cos_t + (py - cy) * sin_t) / self.ZOOM
            sy = self.HEIGHT / 2.0 + (-(px - cx) * sin_t + (py - cy) * cos_t) / self.ZOOM
            
            if 0 <= sx <= self.WIDTH and 0 <= sy <= self.HEIGHT:
                # Рисуем неоновый фиолетовый прицел захвата солитона
                rad = int(22 / self.ZOOM)
                color = (230, 80, 255) # Неоновый фиолетовый
                pygame.draw.circle(surface, color, (int(sx), int(sy)), rad, 1)
                pygame.draw.circle(surface, color, (int(sx), int(sy)), 3)
                
                # Тонкие перекрестия видоискателя
                pygame.draw.line(surface, color, (int(sx - rad - 6), int(sy)), (int(sx - rad + 3), int(sy)), 1)
                pygame.draw.line(surface, color, (int(sx + rad - 3), int(sy)), (int(sx + rad + 6), int(sy)), 1)
                pygame.draw.line(surface, color, (int(sx), int(sy - rad - 6)), (int(sx), int(sy - rad + 3)), 1)
                pygame.draw.line(surface, color, (int(sx), int(sy + rad - 3)), (int(sx), int(sy + rad + 6)), 1)
                
                # Спектроскопическая инфо-строка
                stat_str = f"SOLITON [Amp:{sol['amp']:.1f} | Vel:{sol['speed']:.0f} | Coh:{sol['stability']:.0f}%]"
                text_surf = self.debug_font.render(stat_str, True, (255, 180, 255))
                surface.blit(text_surf, (int(sx - text_surf.get_width()//2), int(sy + rad + 6)))

    def draw_electrode_sensors(self, surface, arena):
        if hasattr(arena, 'maze'):
            theta = -arena.player_angle
            cx, cy = arena.player_pos[0].item(), arena.player_pos[1].item()
        else:
            theta = 0.0
            cx, cy = self.WIDTH / 2.0, self.HEIGHT / 2.0

        cos_t, sin_t = math.cos(theta), math.sin(theta)

        if hasattr(arena, 'actors'):
            actor_list = arena.actors
        else:
            actor_list = [{'is_dead': False, 'team': 0, 'edge_intact': getattr(arena, 'player_edge_intact', [True]*16), 'pin_pos': arena.pin_pos, 'node_phases': torch.zeros(16), 'integrity': 1.0, 'pos': arena.player_pos}]

        if self.debug_font is None: self.debug_font = pygame.font.SysFont("Consolas", 11, bold=True)

        for act in actor_list:
            if act.get('is_dead', False): continue
            
            color = (0, 255, 255) if act['team'] == 0 else (255, 100, 50)
            alpha_col = (color[0], color[1], color[2], 100)
            
            px, py = act['pos'][0].item(), act['pos'][1].item()
            sx_c = self.WIDTH / 2.0 + ((px - cx) * cos_t + (py - cy) * sin_t) / self.ZOOM
            sy_c = self.HEIGHT / 2.0 + (-(px - cx) * sin_t + (py - cy) * cos_t) / self.ZOOM
            
            bar_w = 40
            bar_h = 4
            bx = int(sx_c - bar_w/2)
            by = int(sy_c - 25)
            
            if 0 <= sx_c <= self.WIDTH and 0 <= sy_c <= self.HEIGHT:
                pygame.draw.rect(surface, (40, 40, 50), (bx, by, bar_w, bar_h))
                health_w = int(bar_w * act['integrity'])
                pygame.draw.rect(surface, (0, 255, 150) if act['team'] == 0 else (255, 100, 100), (bx, by, health_w, bar_h))
                
                cos_sum = torch.cos(act['node_phases']).mean()
                sin_sum = torch.sin(act['node_phases']).mean()
                inst_coh = math.hypot(cos_sum.item(), sin_sum.item())
                
                ai_state = act.get('ai_state_desc', 'FIGHTING')
                name_str = f"{act['custom_name'][:4]} [{ai_state}] ({inst_coh*100:.0f}%)"
                text_surf = self.debug_font.render(name_str, True, (255, 255, 255))
                surface.blit(text_surf, (int(sx_c - text_surf.get_width()/2), by - 14))

                if 'target_pos' in act and act['target_pos'] is not None:
                    tx, ty = act['target_pos'][0].item(), act['target_pos'][1].item()
                    tsx = self.WIDTH / 2.0 + ((tx - cx) * cos_t + (ty - cy) * sin_t) / self.ZOOM
                    tsy = self.HEIGHT / 2.0 + (-(tx - cx) * sin_t + (ty - cy) * cos_t) / self.ZOOM
                    
                    if 0 <= tsx <= self.WIDTH and 0 <= tsy <= self.HEIGHT:
                        pygame.draw.line(surface, color, (int(sx_c), int(sy_c)), (int(tsx), int(tsy)), 1)
                        pygame.draw.circle(surface, color, (int(tsx), int(tsy)), 4, 1)

            for i in range(16):
                next_i = (i + 1) % 16
                node1_intact = act['edge_intact'][i]
                node2_intact = act['edge_intact'][next_i]
                
                px1, py1 = act['pin_pos'][i, 0].item(), act['pin_pos'][i, 1].item()
                px2, py2 = act['pin_pos'][next_i, 0].item(), act['pin_pos'][next_i, 1].item()
                
                sx1 = self.WIDTH / 2.0 + ((px1 - cx) * cos_t + (py1 - cy) * sin_t) / self.ZOOM
                sy1 = self.HEIGHT / 2.0 + (-(px1 - cx) * sin_t + (py1 - cy) * cos_t) / self.ZOOM
                sx2 = self.WIDTH / 2.0 + ((px2 - cx) * cos_t + (py2 - cy) * sin_t) / self.ZOOM
                sy2 = self.HEIGHT / 2.0 + (-(px2 - cx) * sin_t + (py2 - cy) * cos_t) / self.ZOOM
                
                if node1_intact and node2_intact:
                    pygame.draw.line(surface, alpha_col, (int(sx1), int(sy1)), (int(sx2), int(sy2)), 2)
                else:
                    pygame.draw.line(surface, (255, 0, 0, 150), (int(sx1), int(sy1)), (int(sx2), int(sy2)), 1)

            for i in range(16):
                if not act['edge_intact'][i]: continue 
                
                px, py = act['pin_pos'][i, 0].item(), act['pin_pos'][i, 1].item()
                sx = self.WIDTH / 2.0 + ((px - cx) * cos_t + (py - cy) * sin_t) / self.ZOOM
                sy = self.HEIGHT / 2.0 + (-(px - cx) * sin_t + (py - cy) * cos_t) / self.ZOOM
                
                if 0 <= sx <= self.WIDTH and 0 <= sy <= self.HEIGHT:
                    pygame.draw.circle(surface, alpha_col, (int(sx), int(sy)), 6, 1)
                    pygame.draw.circle(surface, color, (int(sx), int(sy)), 2)
                    
        # Вызываем спектроскопический сканер солитонов (всегда рисуется при включенных сенсорах)
        self.draw_soliton_telemetry(surface, arena)

    def draw_floating_combat_text(self, surface, arena):
        """ Отрисовка всплывающих цифр урона над юнитами """
        if not hasattr(arena, 'floating_texts') or not arena.floating_texts: return
        if self.hud_font is None: self.hud_font = pygame.font.SysFont("Consolas", 18, bold=True)
            
        if hasattr(arena, 'maze'):
            theta = -arena.player_angle
            cx, cy = arena.player_pos[0].item(), arena.player_pos[1].item()
        else:
            theta = 0.0
            cx, cy = self.WIDTH / 2.0, self.HEIGHT / 2.0

        cos_t, sin_t = math.cos(theta), math.sin(theta)

        for ft in arena.floating_texts:
            px, py = ft['pos'][0], ft['pos'][1]
            sx = self.WIDTH / 2.0 + ((px - cx) * cos_t + (py - cy) * sin_t) / self.ZOOM
            sy = self.HEIGHT / 2.0 + (-(px - cx) * sin_t + (py - cy) * cos_t) / self.ZOOM
            
            if 0 <= sx <= self.WIDTH and 0 <= sy <= self.HEIGHT:
                alpha = int(max(0, min(1.0, ft['life'])) * 255)
                text_surf = self.hud_font.render(ft['text'], True, ft['color'])
                text_surf.set_alpha(alpha)
                surface.blit(text_surf, (int(sx) - text_surf.get_width()//2, int(sy)))

    def draw_combat_debug(self, surface, arena):
        if not hasattr(arena, 'actors'): return
        
        panel_w, panel_h = 320, 310
        panel_x, panel_y = self.WIDTH - panel_w - 15, 15
        
        s = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        s.fill((15, 10, 20, 215)) 
        surface.blit(s, (panel_x, panel_y))
        pygame.draw.rect(surface, (255, 100, 100), (panel_x, panel_y, panel_w, panel_h), 1)
        
        if self.debug_font is None:
            self.debug_font = pygame.font.SysFont("Consolas", 13, bold=True)
            
        title = self.debug_font.render("NLSE SPECTROSCOPY (DIAG)", True, (255, 100, 100))
        surface.blit(title, (panel_x + 10, panel_y + 8))
        
        y_offset = panel_y + 30
        
        p0 = next((a for a in arena.actors if a['team'] == 0 and not a['is_dead']), None)
        p1 = next((a for a in arena.actors if a['team'] == 1 and not a['is_dead']), None)

        if not p0 and not p1: return

        p0_name = p0['custom_name'] if p0 else "NONE"
        p1_name = p1['custom_name'] if p1 else "NONE"
        
        p0_K = p0['K_active'] if p0 else 0.0
        p1_K = p1['K_active'] if p1 else 0.0
        
        p0_diss = p0['shear_stress'] if p0 else 0.0
        p1_diss = p1['shear_stress'] if p1 else 0.0

        p0_int = p0['integrity'] if p0 else 0.0
        p1_int = p1['integrity'] if p1 else 0.0

        coupling_str = f"Coupling (K): {p0_K:.1f} vs {p1_K:.1f}"

        metrics = [
            f"Team 0 Lead : {p0_name}",
            f"Team 1 Lead : {p1_name}",
            coupling_str,
            "-" * 37,
            f"T0 Dissonance : {p0_diss:.2f}",
            f"T1 Dissonance : {p1_diss:.2f}",
            f"T0 Explosions : {p0.get('explosions_triggered', 0) if p0 else 0}",
            f"T1 Explosions : {p1.get('explosions_triggered', 0) if p1 else 0}",
            f"NLSE Clash Field: {getattr(arena, 'clash_intensity', 0.0) * 100:.1f}%",
            "-" * 37,
            f"Phase Integrity T0: {p0_int*100:.1f}%",
            f"Phase Integrity T1: {p1_int*100:.1f}%",
        ]
        
        for met in metrics:
            if met.startswith("-"): text = self.debug_font.render(met, True, (255, 100, 100))
            elif "Dissonance" in met or "Clash" in met: text = self.debug_font.render(met, True, (255, 200, 150))
            elif "Integrity T0" in met:
                col = (0, 255, 100) if p0_int > 0.7 else ((255, 200, 0) if p0_int > 0.4 else (255, 50, 50))
                text = self.debug_font.render(met, True, col)
            elif "Integrity T1" in met:
                col = (0, 255, 100) if p1_int > 0.7 else ((255, 200, 0) if p1_int > 0.4 else (255, 50, 50))
                text = self.debug_font.render(met, True, col)
            else: text = self.debug_font.render(met, True, (255, 255, 255))
            surface.blit(text, (panel_x + 12, y_offset))
            y_offset += 14

        cx, cy = panel_x + panel_w - 45, panel_y + 110
        pygame.draw.circle(surface, (50, 40, 60), (cx, cy), 30, 1)
        
        if p0:
            phases_cpu = p0['node_phases'].cpu().numpy()
            for i in range(16):
                px = cx + int(27 * math.cos(phases_cpu[i]))
                py = cy + int(27 * math.sin(phases_cpu[i]))
                pygame.draw.circle(surface, (0, 255, 200), (px, py), 2)
                
        if p1:
            phases_cpu = p1['node_phases'].cpu().numpy()
            for i in range(16):
                bx = cx + int(18 * math.cos(phases_cpu[i]))
                by = cy + int(18 * math.sin(phases_cpu[i]))
                pygame.draw.circle(surface, (255, 50, 100), (bx, by), 2)

    def draw_combat_ui(self, surface, arena):
        if self.hud_font is None: self.hud_font = pygame.font.SysFont("Consolas", 18, bold=True)
        if self.title_font is None: self.title_font = pygame.font.SysFont("Consolas", 42, bold=True)
        
        self.draw_combat_debug(surface, arena)
        self.draw_floating_combat_text(surface, arena)
        
        diff_text = self.hud_font.render(f"TRIBULATION LEVEL: {getattr(arena, 'difficulty', 1)}", True, (255, 100, 100))
        surface.blit(diff_text, (self.WIDTH // 2 - diff_text.get_width()//2, 20))
        
        t0_living = [a for a in arena.actors if a['team'] == 0 and not a['is_dead']]
        t1_living = [a for a in arena.actors if a['team'] == 1 and not a['is_dead']]
        t0_integ = sum(a['integrity'] for a in t0_living) / max(1, len(t0_living)) if t0_living else 0.0
        t1_integ = sum(a['integrity'] for a in t1_living) / max(1, len(t1_living)) if t1_living else 0.0

        p_text = self.hud_font.render(f"TEAM 0 INTEGRITY: {t0_integ*100:.1f}%", True, (0, 255, 200))
        surface.blit(p_text, (50, self.HEIGHT - 50))
        
        b_text = self.hud_font.render(f"TEAM 1 INTEGRITY: {t1_integ*100:.1f}%", True, (255, 50, 50))
        surface.blit(b_text, (self.WIDTH - b_text.get_width() - 50, self.HEIGHT - 50))

        if getattr(arena, 'winner', None):
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            surface.blit(overlay, (0, 0))
            
            result = f"VICTORY: {arena.winner}!"
            color = (0, 255, 100) if "0" in arena.winner else (255, 50, 50)
            
            res_text = self.title_font.render(result, True, color)
            surface.blit(res_text, (self.WIDTH // 2 - res_text.get_width()//2, self.HEIGHT // 2 - 50))

    def draw_debug_window(self, surface, arena):
        if not hasattr(arena, 'cauldron_temp'): return
            
        panel_w, panel_h = 300, 310
        panel_x, panel_y = self.WIDTH - panel_w - 15, 15
        s = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        s.fill((10, 15, 25, 215))
        surface.blit(s, (panel_x, panel_y))
        pygame.draw.rect(surface, (0, 255, 200), (panel_x, panel_y, panel_w, panel_h), 1)
        if self.debug_font is None: self.debug_font = pygame.font.SysFont("Consolas", 13, bold=True)
        title = self.debug_font.render("EXOCORTEX SPECTROSCOPY", True, (0, 255, 200))
        surface.blit(title, (panel_x + 10, panel_y + 8))
        q = getattr(arena, 'pill_quality', 0.0)
        grade = "Divine Core" if q >= 95.0 else ("Saint Core" if q >= 80.0 else ("Spiritual Core" if q >= 60.0 else ("Mortal Core" if q >= 40.0 else "Impure Slag")))
        y_offset = panel_y + 30
        metrics = [
            f"Attractor Freq: {getattr(arena, 'target_freq_desc', '0Hz')}",
            f"Attractor Spat: {getattr(arena, 'target_spat_desc', '0')}",
            "-" * 35,
            f"Freq Match  : {getattr(arena, 'score_resonance', 0.0) * 100:.1f}%",
            f"Envelope    : {getattr(arena, 'score_containment', 0.0) * 100:.1f}%",
            f"Temp Stabil : {getattr(arena, 'score_temp', 0.0) * 100:.1f}%",
            f"Vortex Spin : {getattr(arena, 'score_vortex', 0.0) * 100:.1f}%",
            "-" * 35,
            f"Progress    : {getattr(arena, 'smelting_progress', 0.0) * 100:.1f}%",
            f"Emergent ID : {getattr(arena, 'emergent_pill_name', 'None')}",
            f"ID Match    : {getattr(arena, 'emergent_pill_similarity', 0.0) * 100:.1f}%",
            f"Pill Quality: {q:.1f}% [{grade}]",
        ]
        for met in metrics:
            if met.startswith("-"): text = self.debug_font.render(met, True, (0, 255, 200))
            elif "Match" in met or "Envelope" in met or "Stabil" in met or "Spin" in met: text = self.debug_font.render(met, True, (200, 255, 220))
            elif "Quality" in met: text = self.debug_font.render(met, True, (0, 255, 100) if q >= 80.0 else ((255, 200, 0) if q >= 40.0 else (255, 100, 100)))
            elif "Emergent" in met: text = self.debug_font.render(met, True, (255, 150, 255))
            else: text = self.debug_font.render(met, True, (255, 255, 255))
            surface.blit(text, (panel_x + 12, y_offset))
            y_offset += 16
        cx, cy = panel_x + panel_w // 2, y_offset + 32
        radar_rad = 28
        pygame.draw.circle(surface, (40, 60, 80), (cx, cy), radar_rad, 1)
        com = arena.player_pos
        cos_p, sin_p = math.cos(-arena.player_angle), math.sin(-arena.player_angle)
        if hasattr(arena, 'pin_x'):
            for i in range(16):
                ix, iy = arena.pin_x[i].item() / 12.0 * radar_rad, arena.pin_y[i].item() / 12.0 * radar_rad
                pygame.draw.circle(surface, (0, 150, 100), (int(cx + ix), int(cy + iy)), 2, 1)
                rx, ry = arena.pin_pos[i, 0].item() - com[0].item(), arena.pin_pos[i, 1].item() - com[1].item()
                loc_x = (rx * cos_p - ry * sin_p) / (getattr(arena, 'cell_w', 20.0) * 0.4) * radar_rad
                loc_y = (rx * sin_p + ry * cos_p) / (getattr(arena, 'cell_w', 20.0) * 0.4) * radar_rad
                pygame.draw.circle(surface, (255, 220, 50), (int(cx + loc_x), int(cy + loc_y)), 2)

    def draw_alchemy_ui(self, surface, arena):
        if not hasattr(arena, 'alchemy_entities'): return
        theta = -arena.player_angle
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        def project(wx, wy):
            dx, dy = wx - arena.player_pos[0].item(), wy - arena.player_pos[1].item()
            sx = self.WIDTH / 2.0 + (dx * cos_t + dy * sin_t) / self.ZOOM
            sy = self.HEIGHT / 2.0 + (-dx * sin_t + dy * cos_t) / self.ZOOM
            return int(sx), int(sy)
        cx, cy = project(arena.cauldron_pos[0].item(), arena.cauldron_pos[1].item())
        rad_cauldron = int(getattr(arena, 'cell_w', 20.0) * 1.5 / self.ZOOM)
        pygame.draw.circle(surface, (150, 150, 150), (cx, cy), rad_cauldron, 2)
        for ent in arena.alchemy_entities:
            ex, ey = project(ent['pos'][0].item(), ent['pos'][1].item())
            if 0 <= ex <= self.WIDTH and 0 <= ey <= self.HEIGHT:
                if ent['type'] == 'pill': 
                    vec = ent['vector']
                    col, icon = (int(vec[0]*255), int(vec[1]*255), int(vec[2]*255)), "EXOCORTEX PILL"
                else:
                    ent_cfg = next((c for c in ALCHEMY_ENTITIES_CONFIG if c['type'] == ent['type']), None)
                    if ent_cfg: col, icon = ent_cfg['color'], f"{ent_cfg['name']} ({ent_cfg['freq']:.0f}Hz)"
                    else: col, icon = (255, 255, 255), f"COGNIT: {ent['type'].upper()} ({ent['freq']:.0f}Hz)"
                pulse = int(40 + 40 * math.sin(pygame.time.get_ticks() * 0.005))
                pygame.draw.circle(surface, col, (ex, ey), 12)
                pygame.draw.circle(surface, (255, 255, 255), (ex, ey), 12 + pulse//10, 2)
                if self.alchemy_font is None: self.alchemy_font = pygame.font.SysFont("Consolas", 14, bold=True)
                text_surf = self.alchemy_font.render(icon, True, (255, 255, 255))
                surface.blit(text_surf, (ex - text_surf.get_width()//2, ey - 32))
        if getattr(arena, 'pill_created', False):
            gx, gy = project(arena.portal_pos[0].item(), arena.portal_pos[1].item())
            if 0 <= gx <= self.WIDTH and 0 <= gy <= self.HEIGHT:
                portal_pulse = int(10 * math.sin(pygame.time.get_ticks() * 0.01))
                pygame.draw.circle(surface, (255, 180, 0), (gx, gy), int(getattr(arena, 'cell_w', 20.0) * 0.6 / self.ZOOM) + portal_pulse, 2)

    def draw_ui(self, surface, arena):
        if hasattr(arena, 'team1_density') or hasattr(arena, 'actors'):
            self.draw_combat_ui(surface, arena)
            return
        self.draw_debug_window(surface, arena)
        self.draw_alchemy_ui(surface, arena)
        if hasattr(arena, 'goal_cell') and hasattr(arena, 'cell_w') and hasattr(arena.maze, 'dim'):
            cell_w = arena.cell_w
            gx, gy = (self.WIDTH * 0.1) + (arena.goal_cell[0] + 0.5) * cell_w, (self.HEIGHT * 0.1) + (arena.goal_cell[1] + 0.5) * cell_w
            dx, dy = gx - arena.player_pos[0].item(), gy - arena.player_pos[1].item()
            theta = -arena.player_angle
            sx = self.WIDTH / 2.0 + (dx * math.cos(theta) + dy * math.sin(theta)) / self.ZOOM
            sy = self.HEIGHT / 2.0 + (-dx * math.sin(theta) + dy * math.cos(theta)) / self.ZOOM
            if 0 <= sx <= self.WIDTH and 0 <= sy <= self.HEIGHT:
                pygame.draw.circle(surface, (0, 255, 100), (int(sx), int(sy)), int(cell_w * 0.25 / self.ZOOM), 1)
        cx, cy = self.WIDTH // 2, self.HEIGHT // 2
        pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 14, 1)
        pygame.draw.circle(surface, (0, 255, 255), (cx, cy), 4)
