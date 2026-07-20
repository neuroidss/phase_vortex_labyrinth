# gacha_hub.py
import pygame
import sys
import json
import os
import random
import torch
import math
from vortex_physics import PhaseVortexArena
from vortex_combat import PhaseVortexCombat
from vortex_renderer import VortexRenderer
from input_manager import UnifiedInputManager
import combat_config

SAVE_FILE = "gacha_save.json"
COMPUTE_RES = 128
TOURNAMENT_SEED = 202607
ZOOM_OUT_FACTOR = 1.35  

def load_profile():
    defaults = {
        "currency": 5000,
        "inventory": [
            {"id": "u_0", "arch": "Vanguard", "idx": 0, "level": 1, "exp_invested": 0},
            {"id": "u_1", "arch": "Assault", "idx": 0, "level": 1, "exp_invested": 0}
        ],
        "squad_ids": ["u_0", "u_1", None],
        "next_uid": 2,
        "campaign_level": 1
    }
    
    p = defaults.copy()
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                p = loaded
        except Exception as e:
            print(f"[WARNING] Error reading profile: {e}. Restoring defaults.")
            p = defaults

    for k, v in defaults.items():
        if k not in p:
            p[k] = v
    
    if not isinstance(p["inventory"], list):
        p["inventory"] = defaults["inventory"]
    
    valid_ids = set()
    clean_inventory = []
    for item in p["inventory"]:
        if isinstance(item, dict) and "id" in item and "arch" in item:
            item["level"] = max(1, item.get("level", 1))
            item["exp_invested"] = max(0, item.get("exp_invested", 0))
            item["idx"] = item.get("idx", 0)
            valid_ids.add(item["id"])
            clean_inventory.append(item)
    p["inventory"] = clean_inventory

    if not isinstance(p["squad_ids"], list):
        p["squad_ids"] = [None, None, None]
    while len(p["squad_ids"]) < 3:
        p["squad_ids"].append(None)
    p["squad_ids"] = p["squad_ids"][:3]
    
    p["squad_ids"] = [sid if (sid in valid_ids) else None for sid in p["squad_ids"]]
    
    if not isinstance(p["currency"], (int, float)):
        p["currency"] = 5000
    else:
        p["currency"] = int(p["currency"])

    if not isinstance(p["campaign_level"], int):
        p["campaign_level"] = 1
        
    return p

def save_profile(p):
    temp_file = SAVE_FILE + ".tmp"
    try:
        with open(temp_file, "w") as f:
            json.dump(p, f)
        os.replace(temp_file, SAVE_FILE)
    except Exception as e:
        print(f"[ERROR] Failed to save profile atomically: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)

def get_unit_data(unit_def):
    arch_class = combat_config.BOT_ARCHETYPES[unit_def["arch"]]
    base = arch_class[unit_def["idx"]]
    quality = 20.0 + (unit_def["level"] * 5.0)
    return {
        "custom_name": base["name"],
        "style": base["style"],
        "desc": base["desc"],
        "vector": base["vector"],
        "quality": quality,
        "is_player": False
    }

def get_enemy_squad_for_level(lvl):
    is_boss = (lvl % 10 == 0)
    is_mini = (lvl % 5 == 0) and not is_boss
    
    enemies = []
    
    if is_boss:
        enemies.append({
            "custom_name": f"HUNDUN BOSS (Lv.{lvl})", "style": "Tank", "desc": "Extreme shielding.",
            "vector": [0.1, 0.8, 0.1], "quality": 50.0 + lvl * 4.0
        })
        enemies.append({
            "custom_name": f"Soliton Demolisher (Lv.{lvl})", "style": "Mage", "desc": "Heavy damage.",
            "vector": [0.0, 0.2, 0.8], "quality": 40.0 + lvl * 4.0
        })
        enemies.append({
            "custom_name": f"Astral Restorer (Lv.{lvl})", "style": "Healer", "desc": "Phase restorer.",
            "vector": [0.1, 0.6, 0.3], "quality": 40.0 + lvl * 4.0
        })
    elif is_mini:
        enemies.append({
            "custom_name": f"Goliath Mini-Boss (Lv.{lvl})", "style": "Tank", "desc": "Hardened frame.",
            "vector": [0.2, 0.7, 0.1], "quality": 35.0 + lvl * 3.0
        })
        enemies.append({
            "custom_name": f"Assault Raider (Lv.{lvl})", "style": "Fighter", "desc": "Aggressive Beta push.",
            "vector": [0.7, 0.2, 0.1], "quality": 25.0 + lvl * 3.0
        })
    else:
        num_enemies = min(3, 1 + (lvl // 3))
        for i in range(num_enemies):
            arch_name = random.choice(list(combat_config.BOT_ARCHETYPES.keys()))
            base = combat_config.BOT_ARCHETYPES[arch_name][0]
            enemies.append({
                "custom_name": f"Corrupted {base['name']} (Lv.{lvl})",
                "style": base["style"], "desc": base["desc"], "vector": base["vector"],
                "quality": 20.0 + lvl * 2.5
            })
            
    return enemies

def calculate_prebattle_wp(team0_data, team1_data):
    if not team0_data or not team1_data:
        return 50.0
        
    t0_power = sum(u['quality'] for u in team0_data)
    t1_power = sum(u['quality'] for u in team1_data)
    
    t0_beta  = sum(float(u['vector'][0]) for u in team0_data)
    t0_theta = sum(float(u['vector'][1]) for u in team0_data)
    t0_gamma = sum(float(u['vector'][2]) for u in team0_data)
    
    t1_beta  = sum(float(u['vector'][0]) for u in team1_data)
    t1_theta = sum(float(u['vector'][1]) for u in team1_data)
    t1_gamma = sum(float(u['vector'][2]) for u in team1_data)
    
    arena_control = t0_beta / (t0_beta + t1_beta + 1e-5)
    
    t0_penetration = max(0.0, t0_gamma - (t1_theta * 0.7))
    t1_penetration = max(0.0, t1_gamma - (t0_theta * 0.7))
    
    t0_eff_dmg = (t0_penetration + arena_control * 0.3) * t0_power
    t1_eff_dmg = (t1_penetration + (1.0 - arena_control) * 0.3) * t1_power
    
    ttk_t1 = t1_power / (t0_eff_dmg + 1e-5)
    ttk_t0 = t0_power / (t1_eff_dmg + 1e-5)
    
    advantage = ttk_t0 - ttk_t1 
    
    exponent = -advantage * 0.8
    exponent_clipped = max(-700.0, min(700.0, exponent))
    
    wp = 1.0 / (1.0 + math.exp(exponent_clipped))
    return wp * 100.0

def main():
    pygame.init()
    pygame.joystick.init() 
    
    WIDTH, HEIGHT = 1120, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Quantum Alchemy: Gacha Hub")
    clock = pygame.time.Clock()
    
    joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
    for j in joysticks: j.init()
    
    font_large = pygame.font.SysFont("Consolas", 24, bold=True)
    font_med = pygame.font.SysFont("Consolas", 16, bold=True)
    font_small = pygame.font.SysFont("Consolas", 13)
    
    profile = load_profile()
    state = "HUB"
    inv_scroll_y = 0  # Смещение инвентаря
    
    arena = None
    renderer = None
    input_manager = UnifiedInputManager(800, 800)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    combat_timer = 0.0
    transition_timer = 0.0
    
    emerg_pill_name = "Foundation Pill"
    emerg_quality = 100.0
    
    # Принудительно освобождаем мышь для начального меню HUB
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)

    def draw_button(surface, text, rect, color=(50,50,80), text_col=(255,255,255)):
        mx, my = pygame.mouse.get_pos()
        hover = rect.collidepoint(mx, my)
        bg = (min(255, color[0]+30), min(255, color[1]+30), min(255, color[2]+30)) if hover else color
        pygame.draw.rect(surface, bg, rect)
        pygame.draw.rect(surface, (255,255,255), rect, 1)
        tsurf = font_med.render(text, True, text_col)
        surface.blit(tsurf, (rect.x + rect.w//2 - tsurf.get_width()//2, rect.y + rect.h//2 - tsurf.get_height()//2))
        return hover

    def pull_gacha():
        if profile["currency"] >= 100:
            profile["currency"] -= 100
            archs = list(combat_config.BOT_ARCHETYPES.keys())
            chosen_arch = random.choice(archs)
            arch_len = len(combat_config.BOT_ARCHETYPES[chosen_arch])
            chosen_idx = random.randint(0, arch_len - 1)
            
            uid = f"u_{profile['next_uid']}"
            profile['next_uid'] += 1
            profile["inventory"].append({
                "id": uid, "arch": chosen_arch, "idx": chosen_idx, "level": 1, "exp_invested": 0
            })
            save_profile(profile)

    def mint_custom_pill(pill_name, quality):
        arch_map = {
            "Foundation Pill": "Support",
            "Pure Yang Core": "Assault",
            "Deep Yin Core": "Vanguard",
            "Turbulent Anomaly": "Artillery"
        }
        arch = arch_map.get(pill_name, "Support")
        
        uid = f"u_{profile['next_uid']}"
        profile['next_uid'] += 1
        
        calculated_level = max(1, min(10, int((quality - 20.0) / 8.0)))
        
        profile["inventory"].append({
            "id": uid, "arch": arch, "idx": 0, "level": calculated_level, "exp_invested": 0
        })
        save_profile(profile)
        
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        mx, my = pygame.mouse.get_pos()
        clicked = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: clicked = True
            
            # Обработка скролла в HUB
            if event.type == pygame.MOUSEWHEEL and state == "HUB":
                inv_scroll_y += event.y * 40
                max_scroll = min(0, HEIGHT - 150 - len(profile["inventory"]) * 64)
                if max_scroll > 0: max_scroll = 0
                inv_scroll_y = max(max_scroll, min(0, inv_scroll_y))

        if state == "HUB":
            # Убеждаемся, что курсор виден в HUB
            if not pygame.mouse.get_visible():
                pygame.mouse.set_visible(True)
                pygame.event.set_grab(False)
                
            screen.fill((10, 12, 18))
            
            header = font_large.render(f"ALCHEMY CAULDRON HUB  |  Quantum Prisms: {profile['currency']}", True, (0, 255, 200))
            screen.blit(header, (20, 20))
            
            inv_title = font_med.render("Unlocked Cultivators (Equip, Level Up, 100% Refund):", True, (255, 255, 255))
            screen.blit(inv_title, (20, 70))
            
            y_off = 100 + inv_scroll_y
            
            # Клиппинг, чтобы инвентарь не вылезал на верхние надписи
            screen.set_clip(pygame.Rect(0, 95, 770, HEIGHT - 95))
            
            for u in profile["inventory"]:
                udata = get_unit_data(u)
                in_squad = u["id"] in profile["squad_ids"]
                bg_col = (20, 50, 50) if in_squad else (22, 22, 30)
                
                pygame.draw.rect(screen, bg_col, (20, y_off, 740, 56))
                pygame.draw.rect(screen, (80, 80, 100), (20, y_off, 740, 56), 1)
                
                t1 = font_med.render(f"Lv.{u['level']} {udata['custom_name']} ({udata['style']})", True, (0, 255, 200) if in_squad else (255, 255, 255))
                screen.blit(t1, (30, y_off + 8))
                t2 = font_small.render(udata['desc'], True, (150, 150, 150))
                screen.blit(t2, (30, y_off + 32))
                
                # Кнопка SELL (Продажа и полное удаление, если не в отряде)
                sell_rect = pygame.Rect(355, y_off + 10, 75, 36)
                if draw_button(screen, "SELL", sell_rect, (140, 40, 40) if not in_squad else (60, 40, 40)):
                    if clicked and not in_squad:
                        profile["currency"] += u["exp_invested"] + 50
                        profile["inventory"].remove(u)
                        save_profile(profile)
                        break # Выходим из цикла, т.к. список изменился
                
                eq_rect = pygame.Rect(440, y_off + 10, 85, 36)
                if draw_button(screen, "UNEQUIP" if in_squad else "EQUIP", eq_rect, (80,20,20) if in_squad else (20,20,80)):
                    if clicked:
                        if in_squad:
                            profile["squad_ids"] = [i if i != u["id"] else None for i in profile["squad_ids"]]
                        else:
                            for slot in range(3):
                                if profile["squad_ids"][slot] is None:
                                    profile["squad_ids"][slot] = u["id"]
                                    break
                        save_profile(profile)

                up_rect = pygame.Rect(535, y_off + 10, 110, 36)
                if draw_button(screen, "UP (100)", up_rect, (20, 80, 20)):
                    if clicked and profile["currency"] >= 100:
                        profile["currency"] -= 100
                        u["level"] += 1
                        u["exp_invested"] += 100
                        save_profile(profile)
                        
                ref_rect = pygame.Rect(655, y_off + 10, 95, 36)
                if draw_button(screen, "REFUND", ref_rect, (80, 80, 20)):
                    if clicked and u["exp_invested"] > 0:
                        profile["currency"] += u["exp_invested"]
                        u["level"] = 1
                        u["exp_invested"] = 0
                        save_profile(profile)

                y_off += 64

            screen.set_clip(None) # Убираем клиппинг для боковой панели

            sidebar_x = 780
            pygame.draw.line(screen, (80, 80, 100), (sidebar_x, 0), (sidebar_x, HEIGHT), 2)
            
            gacha_rect = pygame.Rect(sidebar_x + 20, 20, 280, 50)
            if draw_button(screen, "SUMMON CULTIVATOR (100)", gacha_rect, color=(80, 20, 80)):
                if clicked: pull_gacha()
                
            smelt_rect = pygame.Rect(sidebar_x + 20, 80, 280, 50)
            if draw_button(screen, "SMELT JINDAN CORE (250)", smelt_rect, color=(160, 100, 20)):
                if clicked and profile["currency"] >= 250:
                    profile["currency"] -= 250
                    save_profile(profile)
                    
                    arena = PhaseVortexArena(device, 800, 800, COMPUTE_RES, seed=TOURNAMENT_SEED + random.randint(0, 50000))
                    renderer = VortexRenderer(800, 800, ZOOM_OUT_FACTOR)
                    combat_timer = 0.0
                    transition_timer = 0.0
                    state = "SMELTING"
                    pygame.mouse.set_visible(False)
                    pygame.event.set_grab(True)
                
            stage_lvl = profile["campaign_level"]
            is_boss_stage = (stage_lvl % 10 == 0)
            is_mini_stage = (stage_lvl % 5 == 0) and not is_boss_stage
            
            stage_title_color = (255, 100, 100) if is_boss_stage else ((255, 200, 50) if is_mini_stage else (0, 255, 100))
            stage_tag = "[BOSS]" if is_boss_stage else ("[MINI-BOSS]" if is_mini_stage else "[NORMAL]")
            
            stage_text = font_large.render(f"STAGE {stage_lvl} {stage_tag}", True, stage_title_color)
            screen.blit(stage_text, (sidebar_x + 20, 145))
            
            team0_raw = []
            for uid in profile["squad_ids"]:
                if uid:
                    unit_def = next((u for u in profile["inventory"] if u["id"] == uid), None)
                    if unit_def: team0_raw.append(get_unit_data(unit_def))
            team1_raw = get_enemy_squad_for_level(stage_lvl)
            
            my_power = sum(u['quality'] for u in team0_raw)
            enemy_power = sum(u['quality'] for u in team1_raw)
            
            power_text = font_med.render(f"MY POWER: {my_power:.0f}  |  ENEMY: {enemy_power:.0f}", True, (255, 255, 255))
            screen.blit(power_text, (sidebar_x + 20, 185))
            
            pre_wp = calculate_prebattle_wp(team0_raw, team1_raw)
            if pre_wp >= 75.0:
                wp_color, wp_desc = (0, 255, 100), f"WIN CHANCE: {pre_wp:.1f}% (TACTICAL ADVANTAGE)"
            elif pre_wp >= 40.0:
                wp_color, wp_desc = (255, 200, 0), f"WIN CHANCE: {pre_wp:.1f}% (FAIR FIGHT)"
            else:
                wp_color, wp_desc = (255, 50, 50), f"WIN CHANCE: {pre_wp:.1f}% (SEVERE DISADVANTAGE)"
                
            wp_surf = font_small.render(wp_desc, True, wp_color)
            screen.blit(wp_surf, (sidebar_x + 20, 210))
            
            combat_rect = pygame.Rect(sidebar_x + 20, 230, 280, 44)
            if draw_button(screen, "LAUNCH SQUAD", combat_rect, color=(20, 80, 20)):
                if clicked:
                    if not team0_raw:
                        print("Assemble squad first!")
                    else:
                        arena = PhaseVortexCombat(device, 800, 800, 128, team0_data=team0_raw, team1_data=team1_raw, difficulty=stage_lvl)
                        renderer = VortexRenderer(800, 800, ZOOM_OUT_FACTOR)
                        combat_timer = 0.0
                        transition_timer = 0.0
                        state = "COMBAT"
                        pygame.mouse.set_visible(False)
                        pygame.event.set_grab(True)

            sq_title = font_med.render("Deployed Squad (Max 3):", True, (255, 255, 255))
            screen.blit(sq_title, (sidebar_x + 20, 290))
            sy = 290
            for slot, uid in enumerate(profile["squad_ids"]):
                pygame.draw.rect(screen, (20, 20, 30), (sidebar_x + 20, sy, 280, 44))
                pygame.draw.rect(screen, (100, 100, 100), (sidebar_x + 20, sy, 280, 44), 1)
                if uid:
                    unit_def = next((u for u in profile["inventory"] if u["id"] == uid), None)
                    name = get_unit_data(unit_def)["custom_name"] if unit_def else "ERROR"
                    ts = font_small.render(f"Slot {slot+1}: {name}", True, (0, 255, 200))
                else:
                    ts = font_small.render(f"Slot {slot+1}: EMPTY", True, (100, 100, 100))
                screen.blit(ts, (sidebar_x + 35, sy + 14))
                sy += 54

            guide_y = HEIGHT - 180
            pygame.draw.rect(screen, (15, 15, 22), (sidebar_x + 20, guide_y, 280, 160))
            pygame.draw.rect(screen, (80, 80, 100), (sidebar_x + 20, guide_y, 280, 160), 1)
            screen.blit(font_med.render("ALCHEMICAL ROLES:", True, (255, 180, 50)), (sidebar_x + 30, guide_y + 10))
            screen.blit(font_small.render("Tanks: Theta shields & block", True, (200, 200, 200)), (sidebar_x + 30, guide_y + 35))
            screen.blit(font_small.render("Fighters: Fast kinetic drift", True, (200, 200, 200)), (sidebar_x + 30, guide_y + 55))
            screen.blit(font_small.render("Mages: Launch soliton packets", True, (200, 200, 200)), (sidebar_x + 30, guide_y + 75))
            screen.blit(font_small.render("Healers: Phase-lock repair", True, (200, 200, 200)), (sidebar_x + 30, guide_y + 95))
            screen.blit(font_small.render("Press R in combat to reset", True, (150, 150, 150)), (sidebar_x + 30, guide_y + 125))

            pygame.display.flip()

        elif state == "COMBAT":
            if transition_timer <= 0.0:
                combat_timer += dt
                is_real_data, eeg_vx, eeg_vy, eeg_tq, compression, alch_freq, alch_spatial = input_manager.process_inputs(joysticks, dt)
                
                arena.step(dt, combat_timer, None, eeg_vx, eeg_vy, eeg_tq, is_real_data, compression, 1.25)
                if arena.winner is not None:
                    transition_timer = 3.0
                    if arena.winner == "Team 0":
                        stage_lvl = profile["campaign_level"]
                        reward = 1000 if (stage_lvl % 10 == 0) else (300 if (stage_lvl % 5 == 0) else 150)
                        profile["currency"] += reward
                        profile["campaign_level"] += 1
                        save_profile(profile)
            else:
                transition_timer -= dt
                if transition_timer <= 0.0:
                    state = "HUB"
                    pygame.mouse.set_visible(True)
                    pygame.event.set_grab(False)
                    
            screen.fill((0,0,0))
            surf = renderer.render_field(arena)
            screen.blit(surf, (0, 0))
            
            renderer.draw_electrode_sensors(screen, arena)
            renderer.draw_floating_combat_text(screen, arena)
            renderer.draw_combat_ui(screen, arena)
            
            sidebar_x = 800
            pygame.draw.rect(screen, (15, 12, 22), (sidebar_x, 0, 320, HEIGHT))
            pygame.draw.line(screen, (0, 255, 200), (sidebar_x, 0), (sidebar_x, HEIGHT), 2)
            
            screen.blit(font_large.render("PORTFOLIO DECAY", True, (0, 255, 200)), (sidebar_x + 20, 20))
            screen.blit(font_small.render("-" * 35, True, (0, 120, 100)), (sidebar_x + 20, 50))
            
            wp_line = font_med.render(f"WIN PROB: {arena.predicted_wp * 100.0:.1f}%", True, (0, 255, 150) if arena.predicted_wp > 0.5 else (255, 100, 100))
            screen.blit(wp_line, (sidebar_x + 20, 70))
            
            vol_line = font_small.render(f"Implied Vol (Sigma): {arena.implied_vol:.3f}", True, (255, 200, 0))
            screen.blit(vol_line, (sidebar_x + 20, 100))
            
            drift_line = font_small.render(f"Expected Drift (Mu): {arena.drift_mu:.3f}", True, (150, 150, 255))
            screen.blit(drift_line, (sidebar_x + 20, 120))
            
            tau_val = max(0.0, 30.0 - arena.combat_time)
            tau_line = font_small.render(f"Time to Expire (Tau): {tau_val:.2f}s", True, (150, 150, 150))
            screen.blit(tau_line, (sidebar_x + 20, 140))
            
            graph_y = 200
            pygame.draw.rect(screen, (30, 30, 45), (sidebar_x + 20, graph_y, 280, 150))
            pygame.draw.rect(screen, (0, 255, 200), (sidebar_x + 20, graph_y, 280, 150), 1)
            
            wp_height = int(arena.predicted_wp * 148.0)
            pygame.draw.rect(screen, (0, 150, 100) if arena.predicted_wp > 0.5 else (150, 50, 50), (sidebar_x + 22, graph_y + 149 - wp_height, 276, wp_height))
            
            feed_y = 370
            pygame.draw.rect(screen, (20, 20, 30), (sidebar_x + 20, feed_y, 280, 180))
            pygame.draw.rect(screen, (0, 255, 200), (sidebar_x + 20, feed_y, 280, 180), 1)
            
            screen.blit(font_small.render("TACTICAL COMBAT FEED", True, (0, 255, 200)), (sidebar_x + 30, feed_y + 8))
            
            log_y = feed_y + 30
            for entry in getattr(arena, 'combat_log', [])[-8:]: 
                ts = font_small.render(entry, True, (255, 255, 255))
                screen.blit(ts, (sidebar_x + 30, log_y))
                log_y += 16
            
            screen.blit(font_small.render("Telemetry logged to battle_prediction_log.csv", True, (150, 150, 150)), (sidebar_x + 20, HEIGHT - 30))
            
            pygame.display.flip()

        elif state == "SMELTING":
            if transition_timer <= 0.0:
                combat_timer += dt
                is_real_data, eeg_vx, eeg_vy, eeg_tq, compression, alch_freq, alch_spatial = input_manager.process_inputs(joysticks, dt)
                
                arena.step(dt, combat_timer, None, eeg_vx, eeg_vy, eeg_tq, is_real_data, compression, 1.25, None, alch_freq, alch_spatial)
                
                emerg_pill_name = arena.emergent_pill_name
                emerg_quality = arena.pill_quality
                
                if arena.pill_created and arena.pin_captured.all():
                    transition_timer = 3.0
                    mint_custom_pill(emerg_pill_name, emerg_quality)
            else:
                transition_timer -= dt
                if transition_timer <= 0.0:
                    state = "HUB"
                    pygame.mouse.set_visible(True)
                    pygame.event.set_grab(False)

            screen.fill((0,0,0))
            surf = renderer.render_field(arena)
            screen.blit(surf, (0, 0))
            
            renderer.draw_electrode_sensors(screen, arena)
            renderer.draw_ui(screen, arena)
            
            sidebar_x = 780
            skip_rect = pygame.Rect(sidebar_x + 20, HEIGHT - 100, 280, 50)
            if draw_button(screen, "SKIP SMELTING (AUTO MINT)", skip_rect, color=(160, 20, 20)):
                if clicked:
                    mint_custom_pill(random.choice(["Foundation Pill", "Pure Yang Core", "Deep Yin Core", "Turbulent Anomaly"]), random.uniform(50.0, 95.0))
                    state = "HUB"
                    pygame.mouse.set_visible(True)
                    pygame.event.set_grab(False)
                    
            pygame.display.flip()

if __name__ == "__main__":
    main()
