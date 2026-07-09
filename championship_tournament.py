# championship_tournament.py
import pygame
import torch
import sys
import time
import math
import numpy as np
import random
import combat_config
from vortex_combat import PhaseVortexCombat
from vortex_renderer import VortexRenderer

# --- RECT-GRID TOURNAMENT MATCHUPS CONFIGURATION ---
# Symmetrical Rock-Paper-Scissors cycle + Living Hybrid Demonstration
MATCHUPS = [
    {
        "name": "Match 1: Aggressive Yang vs Defensive Yin",
        "bot0": {"type": "bot", "pill_name": "Pure Yang Core", "custom_name": "Elder Pyro", "style": "Aggressive Yang", "quality": 95.0, "vector": [1.0, 0.0, 0.0], "desc": "Fiery Gamma charges."},
        "bot1": {"type": "bot", "pill_name": "Deep Yin Core", "custom_name": "Frost Weaver", "style": "Defensive Yin", "quality": 95.0, "vector": [0.0, 0.0, 1.0], "desc": "Defensive Water parries."},
        "prediction": "Frost Weaver (Yin/Water) is predicted to douse and break Elder Pyro (Yang/Fire) via parries and defensive dissipation."
    },
    {
        "name": "Match 2: Speed Yang vs Elusive Yin",
        "bot0": {"type": "bot", "pill_name": "Pure Yang Core", "custom_name": "Solar Flare", "style": "Speed Yang", "quality": 90.0, "vector": [1.0, 0.0, 0.0], "desc": "High velocity orbits."},
        "bot1": {"type": "bot", "pill_name": "Deep Yin Core", "custom_name": "Lunar Shadow", "style": "Elusive Yin", "quality": 90.0, "vector": [0.0, 0.0, 1.0], "desc": "Evasive maneuvers."},
        "prediction": "Lunar Shadow is predicted to evade Solar Flare's charges and win via slow domain-exhaustion."
    },
    {
        "name": "Match 3: Freezing Flame vs SMR Catalyst",
        "bot0": {"type": "bot", "pill_name": "Turbulent Anomaly", "custom_name": "Glacial Phoenix", "style": "Steady Triad", "quality": 95.0, "vector": [0.707, 0.0, 0.707], "desc": "Ice and Fire blended core (Mixed Nodes)."},
        "bot1": {"type": "bot", "pill_name": "Foundation Pill", "custom_name": "Zen Disciple", "style": "Balanced Triad", "quality": 95.0, "vector": [0.0, 1.0, 0.0], "desc": "Resonant SMR focus."},
        "prediction": "Glacial Phoenix (Freezing Flame) is predicted to break Zen Disciple (Grass) as its Yang nodes incinerate the green catalyst."
    },
    {
        "name": "Match 4: SMR Catalyst vs Defensive Yin",
        "bot0": {"type": "bot", "pill_name": "Foundation Pill", "custom_name": "Zen Disciple", "style": "Steady Triad", "quality": 95.0, "vector": [0.0, 1.0, 0.0], "desc": "Resonant SMR focus."},
        "bot1": {"type": "bot", "pill_name": "Deep Yin Core", "custom_name": "Frost Weaver", "style": "Defensive Yin", "quality": 95.0, "vector": [0.0, 0.0, 1.0], "desc": "Defensive Water parries."},
        "prediction": "Zen Disciple (Catalyst/Grass) is predicted to absorb and lock Frost Weaver (Yin/Water) via high phase-coupling."
    }
]

WIDTH, HEIGHT = 800, 800
COMPUTE_RES = 128
ZOOM_OUT_FACTOR = 1.35

def main():
    try:
        pygame.init()
        pygame.font.init()
        
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Exocortex Championship: Automated Parameters Testbed")
        clock = pygame.time.Clock()
        
        font_title = pygame.font.SysFont("Consolas", 18, bold=True)
        font_desc = pygame.font.SysFont("Consolas", 12, bold=False)
        font_hud = pygame.font.SysFont("Consolas", 14, bold=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        renderer = VortexRenderer(WIDTH, HEIGHT, ZOOM_OUT_FACTOR)
        
        current_match_idx = 0
        match_stats = {i: {"bot0_wins": 0, "bot1_wins": 0} for i in range(len(MATCHUPS))}
        
        # Function to boot up current matchup
        def init_match(idx):
            m_cfg = MATCHUPS[idx]
            arena = PhaseVortexCombat(
                device, WIDTH, HEIGHT, COMPUTE_RES, None, difficulty=4,
                actor0_data=m_cfg["bot0"], actor1_data=m_cfg["bot1"]
            )
            # Add raw input placeholders to avoid rendering crashes
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
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Manual skip to next matchup
                        current_match_idx = (current_match_idx + 1) % len(MATCHUPS)
                        arena, current_match = init_match(current_match_idx)
                        round_timer = 0.0
                        transition_timer = 0.0
                    if event.key == pygame.K_r:
                        # Restart current matchup
                        arena, current_match = init_match(current_match_idx)
                        round_timer = 0.0
                        transition_timer = 0.0

            # Execute a clean physics step in autonomous Bot vs Bot mode
            if transition_timer <= 0.0:
                round_timer += dt
                arena.step(dt, round_timer, None, 0.0, 0.0, 0.0, False, 0.0, 1.25)
                
                # Check for victory conditions
                if arena.winner is not None:
                    p_integ = arena.actors[0]['integrity']
                    b_integ = arena.actors[1]['integrity']
                    
                    # Record statistics
                    if arena.winner == "Player": # Actor 0
                        match_stats[current_match_idx]["bot0_wins"] += 1
                    elif arena.winner == "Rogue Cultivator": # Actor 1
                        match_stats[current_match_idx]["bot1_wins"] += 1
                    
                    # Record and write telemetry log for detailed physical calibration
                    with open("tournament_history.log", "a", encoding="utf-8") as f:
                        log_str = (
                            f"--- MATCH COMPLETED at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
                            f"Matchup   : {current_match['name']}\n"
                            f"Prediction: {current_match['prediction']}\n"
                            f"Winner     : {arena.winner} won!\n"
                            f"Duration   : {round_timer:.2f} seconds\n"
                            f"End Phasic Integrity: {current_match['bot0']['custom_name']} = {p_integ*100:.1f}% | "
                            f"{current_match['bot1']['custom_name']} = {b_integ*100:.1f}%\n"
                            f"K Coupling: {arena.actors[0]['K_active']:.1f} vs {arena.actors[1]['K_active']:.1f}\n"
                            f"Dissonance: {arena.actors[0]['shear_stress']:.2f} vs {arena.actors[1]['shear_stress']:.2f}\n"
                            f"Explosions: {arena.actors[0]['custom_name']} = {arena.actors[0]['explosions_triggered']} | "
                            f"{arena.actors[1]['custom_name']} = {arena.actors[1]['explosions_triggered']}\n"
                            f"--------------------------------------------------\n\n"
                        )
                        f.write(log_str)
                        print(log_str)

                    transition_timer = 3.0
            else:
                transition_timer -= dt
                if transition_timer <= 0.0:
                    # Move to next matchup automatically on finish
                    current_match_idx = (current_match_idx + 1) % len(MATCHUPS)
                    arena, current_match = init_match(current_match_idx)
                    round_timer = 0.0

            p_integ = arena.actors[0]['integrity']
            b_integ = arena.actors[1]['integrity']

            # Calculate win probabilities based on Phasic Integrity
            p0_sq = p_integ ** 2
            p1_sq = b_integ ** 2
            total_p = p0_sq + p1_sq + 1e-5
            prob_0 = p0_sq / total_p
            prob_1 = p1_sq / total_p
                    
            # Render visual viewport
            screen.blit(renderer.render_field(arena), (0, 0))
            renderer.draw_electrode_sensors(screen, arena)
            
            # --- CHAMPIONSHIP HUD PANEL ---
            panel = pygame.Surface((WIDTH, 175), pygame.SRCALPHA)
            panel.fill((10, 12, 18, 230))
            screen.blit(panel, (0, HEIGHT - 175))
            pygame.draw.line(screen, (0, 255, 200), (0, HEIGHT - 180), (WIDTH, HEIGHT - 180), 2)
            
            # Draw live dual-colored horizontal prediction bar (Green vs Red)
            bar_x = 20
            bar_y = HEIGHT - 165
            bar_w = WIDTH - 40
            bar_h = 14
            pygame.draw.rect(screen, (30, 30, 45), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(screen, (0, 255, 150), (bar_x, bar_y, int(bar_w * prob_0), bar_h))
            pygame.draw.rect(screen, (255, 100, 100), (bar_x + int(bar_w * prob_0), bar_y, bar_w - int(bar_w * prob_0), bar_h))
            pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, bar_w, bar_h), 1)

            pred_text = font_desc.render(f"LIVE WIN PREDICTION: {current_match['bot0']['custom_name']} ({prob_0*100:.1f}%) vs {current_match['bot1']['custom_name']} ({prob_1*100:.1f}%)", True, (255, 255, 255))
            screen.blit(pred_text, (bar_x, bar_y - 18))

            # Draw Match Title and Archetypes style
            m_title = font_title.render(current_match["name"].upper(), True, (0, 255, 200))
            screen.blit(m_title, (20, HEIGHT - 140))
            
            # Draw Theoretical predicted outcome vs empirical stats
            m_pred = font_desc.render(f"THEORETICAL PREDICTION: {current_match['prediction']}", True, (255, 180, 50))
            screen.blit(m_pred, (20, HEIGHT - 118))
            
            p_str = f"BOT 0 [{current_match['bot0']['custom_name']}]: {p_integ*100:.1f}% Phasic Integrity"
            b_str = f"BOT 1 [{current_match['bot1']['custom_name']}]: {b_integ*100:.1f}% Phasic Integrity"
            
            p_surf = font_hud.render(p_str, True, (0, 255, 150))
            b_surf = font_hud.render(b_str, True, (255, 100, 100))
            screen.blit(p_surf, (20, HEIGHT - 92))
            screen.blit(b_surf, (WIDTH // 2 + 20, HEIGHT - 92))
            
            # Display statistics of victories
            bot0_wins = match_stats[current_match_idx]["bot0_wins"]
            bot1_wins = match_stats[current_match_idx]["bot1_wins"]
            stats_str = f"MATCH STATS (Empirical): BOT 0 wins: {bot0_wins} | BOT 1 wins: {bot1_wins}"
            stats_surf = font_hud.render(stats_str, True, (255, 255, 255))
            screen.blit(stats_surf, (20, HEIGHT - 65))
            
            # Display testbed prompt
            prompt_str = f"TIME: {round_timer:05.2f}s | Press [SPACE] to skip match | Press [R] to restart"
            prompt_surf = font_desc.render(prompt_str, True, (150, 150, 150))
            screen.blit(prompt_surf, (20, HEIGHT - 35))
            
            # Symmetrically draw the connectome spectroscopy panel
            renderer.draw_combat_debug(screen, arena)
            
            pygame.display.flip()
            
        pygame.quit()
        sys.exit()
    except Exception as e:
        print("[CRITICAL EXCEPTION IN TOURNAMENT LOOP]:")
        import traceback
        traceback.print_exc()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()
