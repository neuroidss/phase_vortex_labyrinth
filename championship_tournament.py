# championship_tournament.py
import pygame
import torch
import sys
import time
from vortex_combat import PhaseVortexCombat
from vortex_renderer import VortexRenderer
import combat_config

MATCHUPS = [
    {
        "name": "Match 1: Frontline Breach vs Soliton Artillery",
        "team0": [
            combat_config.BOT_ARCHETYPES["Vanguard"][0],
            combat_config.BOT_ARCHETYPES["Assault"][0],
            combat_config.BOT_ARCHETYPES["Assault"][0]
        ],
        "team1": [
            combat_config.BOT_ARCHETYPES["Vanguard"][0],
            combat_config.BOT_ARCHETYPES["Artillery"][0],
            combat_config.BOT_ARCHETYPES["Support"][0]
        ],
        "prediction": "Team 1 Artillery will rain Gamma solitons from behind their Vanguard shield, breaking Team 0's Assault."
    }
]

WIDTH, HEIGHT = 800, 800
COMPUTE_RES = 128
ZOOM_OUT_FACTOR = 1.35

def main():
    try:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Exocortex Championship: Auto-Battler NLSE")
        clock = pygame.time.Clock()
        
        font_title = pygame.font.SysFont("Consolas", 18, bold=True)
        font_desc = pygame.font.SysFont("Consolas", 12, bold=False)
        font_hud = pygame.font.SysFont("Consolas", 14, bold=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        renderer = VortexRenderer(WIDTH, HEIGHT, ZOOM_OUT_FACTOR)
        
        current_match_idx = 0
        match_stats = {i: {"bot0_wins": 0, "bot1_wins": 0} for i in range(len(MATCHUPS))}
        
        def init_match(idx):
            m_cfg = MATCHUPS[idx]
            arena = PhaseVortexCombat(
                device, WIDTH, HEIGHT, COMPUTE_RES, 
                team0_data=m_cfg["team0"], team1_data=m_cfg["team1"], difficulty=4
            )
            arena.raw_axes = [0.0] * 6
            arena.raw_buttons = [0] * 8
            return arena, m_cfg
            
        arena, current_match = init_match(current_match_idx)
        running = True
        round_timer = 0.0
        transition_timer = 0.0
        
        while running:
            dt = min(0.032, clock.tick() / 1000.0)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    arena, current_match = init_match(current_match_idx)
                    round_timer = 0.0
                    transition_timer = 0.0

            if transition_timer <= 0.0:
                round_timer += dt
                arena.step(dt, round_timer, None, 0.0, 0.0, 0.0, False, 0.0, 1.25)
                if arena.winner is not None:
                    if arena.winner == "Team 0": match_stats[current_match_idx]["bot0_wins"] += 1
                    elif arena.winner == "Team 1": match_stats[current_match_idx]["bot1_wins"] += 1
                    transition_timer = 3.0
            else:
                transition_timer -= dt
                if transition_timer <= 0.0:
                    current_match_idx = (current_match_idx + 1) % len(MATCHUPS)
                    arena, current_match = init_match(current_match_idx)
                    round_timer = 0.0

            t0_living = [a for a in arena.actors if a['team'] == 0 and not a['is_dead']]
            t1_living = [a for a in arena.actors if a['team'] == 1 and not a['is_dead']]
            t0_integ = sum(a['integrity'] for a in t0_living) / len(current_match["team0"]) if t0_living else 0.0
            t1_integ = sum(a['integrity'] for a in t1_living) / len(current_match["team1"]) if t1_living else 0.0

            prob_0 = (t0_integ**2) / (t0_integ**2 + t1_integ**2 + 1e-5)
            prob_1 = (t1_integ**2) / (t0_integ**2 + t1_integ**2 + 1e-5)
                    
            screen.blit(renderer.render_field(arena), (0, 0))
            renderer.draw_electrode_sensors(screen, arena)
            # ВАЖНО: Отрисовка урона теперь работает и в турнире!
            renderer.draw_floating_combat_text(screen, arena)
            
            panel = pygame.Surface((WIDTH, 175), pygame.SRCALPHA)
            panel.fill((10, 12, 18, 230))
            screen.blit(panel, (0, HEIGHT - 175))
            pygame.draw.line(screen, (0, 255, 200), (0, HEIGHT - 180), (WIDTH, HEIGHT - 180), 2)
            
            bar_x, bar_y, bar_w, bar_h = 20, HEIGHT - 165, WIDTH - 40, 14
            pygame.draw.rect(screen, (30, 30, 45), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(screen, (0, 255, 150), (bar_x, bar_y, int(bar_w * prob_0), bar_h))
            pygame.draw.rect(screen, (255, 100, 100), (bar_x + int(bar_w * prob_0), bar_y, bar_w - int(bar_w * prob_0), bar_h))
            
            m_title = font_title.render(current_match["name"].upper(), True, (0, 255, 200))
            screen.blit(m_title, (20, HEIGHT - 140))
            
            p_surf = font_hud.render(f"TEAM 0 (Cyan): {t0_integ*100:.1f}% Integrity", True, (0, 255, 150))
            b_surf = font_hud.render(f"TEAM 1 (Red): {t1_integ*100:.1f}% Integrity", True, (255, 100, 100))
            screen.blit(p_surf, (20, HEIGHT - 92))
            screen.blit(b_surf, (WIDTH // 2 + 20, HEIGHT - 92))
            
            prompt_str = f"TIME: {round_timer:05.2f}s | Winner: {arena.winner if arena.winner else 'FIGHT'}"
            prompt_surf = font_desc.render(prompt_str, True, (150, 150, 150))
            screen.blit(prompt_surf, (20, HEIGHT - 35))
            
            renderer.draw_combat_debug(screen, arena)
            pygame.display.flip()
            
        pygame.quit()
        sys.exit()
    except Exception as e:
        import traceback
        traceback.print_exc()
        pygame.quit()

if __name__ == "__main__":
    main()
