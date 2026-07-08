# phase_vortex_labyrinth.py
import pygame
import torch
import sys
import traceback
import time
import numpy as np
import random
import math
from implicit_config import ALCHEMY_ENTITIES_CONFIG

try:
    from neuro_driver import RealNeuroDriver
    from symbiotic_engine import SymbioticEngineGPU
    HAS_NEURO = True
except ImportError:
    HAS_NEURO = False

from vortex_physics import PhaseVortexArena
from vortex_renderer import VortexRenderer
from input_manager import UnifiedInputManager

WIDTH, HEIGHT = 800, 800
COMPUTE_RES = 128
ZOOM_OUT_FACTOR = 1.35
TOURNAMENT_SEED = 202607


class AudioSonificationManager:
    """
    Класс пространственной звуковой сонафикации локальной физики вокруг слайма.
    Динамически адаптируется под реестр сущностей ALCHEMY_ENTITIES_CONFIG.
    """
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.phase_r = 0.0
        self.phase_g = 0.0
        self.phase_b = 0.0
        self.phase_wall = 0.0
        
        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(self.sample_rate, -16, 2, 512)
            pygame.mixer.init()
            
        self.channel = pygame.mixer.Channel(0)
        self.channel.set_volume(0.35)

    def update(self, arena):
        # 1. Получаем координаты слайма на расчетной сетке
        px, py = arena.player_pos[0].item(), arena.player_pos[1].item()
        px_grid = int((px / arena.WIDTH) * arena.res)
        py_grid = int((py / arena.HEIGHT) * arena.res)
        
        gx = max(0, min(arena.res - 1, px_grid))
        gy = max(0, min(arena.res - 1, py_grid))
        
        # 2. Высокопроизводительный сбор физических величин из комплексных полей
        local_data = torch.stack([
            arena.density_complex[0, 0, gy, gx],
            arena.density_complex[0, 1, gy, gx],
            arena.density_complex[0, 2, gy, gx],
            arena.density_complex[0, 3, gy, gx],
            arena.density_complex[0, 4, gy, gx],
            arena.density_complex[0, 5, gy, gx],
            arena.u[0, 0, gy, gx],
            arena.v[0, 0, gy, gx],
            arena.wall_density[0, 0, gy, gx]
        ]).cpu().numpy()
        
        r_re, r_im, g_re, g_im, b_re, b_im, u_val, v_val, wall_val = local_data
        
        # 3. Вычисление амплитуды и фазы по теореме Пифагора и тригонометрии
        R_val = math.hypot(r_re, r_im)
        G_val = math.hypot(g_re, g_im)
        B_val = math.hypot(b_re, b_im)
        
        R_phase = math.atan2(r_im, r_re + 1e-8)
        G_phase = math.atan2(g_im, g_re + 1e-8)
        B_phase = math.atan2(b_im, b_re + 1e-8)
        
        speed = math.hypot(u_val, v_val)
        
        # 4. Динамическая привязка частот-генераторов на основе ALCHEMY_ENTITIES_CONFIG
        ent_r = next((c for c in ALCHEMY_ENTITIES_CONFIG if c.get('rgb') == 0), None)
        ent_g = next((c for c in ALCHEMY_ENTITIES_CONFIG if c.get('rgb') == 2), None)
        ent_b = next((c for c in ALCHEMY_ENTITIES_CONFIG if c.get('rgb') == 4), None)

        base_r_freq = ent_r['freq'] if ent_r else 80.0
        base_g_freq = ent_g['freq'] if ent_g else 14.0
        base_b_freq = ent_b['freq'] if ent_b else 6.0
        
        # Преобразуем низкие биологические частоты ЭЭГ в хорошо слышимый аудиодиапазон
        freq_r = 300.0 + base_r_freq * 1.5 + 60.0 * math.sin(R_phase)
        freq_g = 180.0 + base_g_freq * 3.0 + 30.0 * math.sin(G_phase)
        freq_b = 90.0 + base_b_freq * 5.0 + 15.0 * math.sin(B_phase)
        freq_wall = 1000.0                         # Сонар препятствий (Гц)
        
        # Определение громкостей звуковых каналов на основе плотности полей
        amp_r = min(0.5, R_val * 0.5)
        amp_g = min(0.5, G_val * 0.5)
        amp_b = min(0.5, B_val * 0.5)
        amp_noise = min(0.12, speed * 0.015)
        amp_wall = min(0.25, wall_val * 0.6)
        
        # Синтез аудиоданных
        num_samples = 512
        samples = np.zeros(num_samples, dtype=np.float32)
        dt = 1.0 / self.sample_rate
        
        for i in range(num_samples):
            self.phase_r += 2 * math.pi * freq_r * dt
            self.phase_g += 2 * math.pi * freq_g * dt
            self.phase_b += 2 * math.pi * freq_b * dt
            self.phase_wall += 2 * math.pi * freq_wall * dt
            
            self.phase_r %= (2 * math.pi)
            self.phase_g %= (2 * math.pi)
            self.phase_b %= (2 * math.pi)
            self.phase_wall %= (2 * math.pi)
            
            sig_r = amp_r * math.sin(self.phase_r)
            sig_g = amp_g * math.sin(self.phase_g)
            sig_b = amp_b * math.sin(self.phase_b)
            sig_wall = amp_wall * math.sin(self.phase_wall)
            noise_val = (random.random() * 2.0 - 1.0) * amp_noise
            
            samples[i] = sig_r + sig_g + sig_b + noise_val + sig_wall
            
        samples = np.tanh(samples)
        audio_int16 = (samples * 32767).astype(np.int16)
        
        # Стерео-дублирование
        audio_stereo = np.column_stack((audio_int16, audio_int16))
        
        sound = pygame.sndarray.make_sound(audio_stereo)
        
        if not self.channel.get_busy():
            self.channel.play(sound)
        elif self.channel.get_queue() is None:
            self.channel.queue(sound)


def main():
    try:
        pygame.init()
        pygame.font.init()
        pygame.joystick.init()
        
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Quantum Alchemy Reactor")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Consolas", 28, bold=True)
        
        input_manager = UnifiedInputManager(WIDTH, HEIGHT)
        
        joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
        for j in joysticks: j.init()

        if HAS_NEURO:
            driver = RealNeuroDriver()
            driver.start_lsl_scanning_thread()
            driver.start_ble_scanning_thread()
            neuro_engine = SymbioticEngineGPU(device_name='cuda')
            device = neuro_engine.device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        arena = PhaseVortexArena(device, WIDTH, HEIGHT, COMPUTE_RES, seed=TOURNAMENT_SEED)
        renderer = VortexRenderer(WIDTH, HEIGHT, ZOOM_OUT_FACTOR)
        audio_manager = AudioSonificationManager(sample_rate=44100)
        
        show_lines = False
        show_sensors = True

        run_start_time = time.time()
        last_finish_time = 0.0
        frame_times = []
        
        running = True
        while running:
            dt = min(0.032, clock.tick() / 1000.0)
            time_sec = pygame.time.get_ticks() / 1000.0
            
            curr_time = time.time()
            frame_times.append(curr_time)
            frame_times = [t for t in frame_times if curr_time - t < 1.0]
            current_fps = len(frame_times)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        arena.cfg['vorticity_sensitivity'] = max(0.0, arena.cfg['vorticity_sensitivity'] - 0.05)
                    if event.key == pygame.K_2:
                        arena.cfg['vorticity_sensitivity'] = min(2.0, arena.cfg['vorticity_sensitivity'] + 0.05)
                    if event.key == pygame.K_l:
                        show_lines = not show_lines
                    if event.key == pygame.K_k:
                        show_sensors = not show_sensors
                    if event.key == pygame.K_ESCAPE:
                        input_manager.toggle_mouse_lock()

            is_real_data, eeg_vx, eeg_vy, eeg_tq, ui_compression, alch_freq, alch_spatial = input_manager.process_inputs(joysticks, dt)
            eeg_c0_spectrum = None
            eeg_freqs = None

            if HAS_NEURO:
                active_slots = [i for i in range(5) if driver.workers[i].is_connected or any(v == i for v in driver.lsl_inlets.values())]
                if active_slots:
                    is_real_data = True
                    for slot_idx in active_slots:
                        q = driver.queues[slot_idx]
                        q_len = len(q)
                        if q_len > 0:
                            samples = [q.popleft() for _ in range(q_len)]
                            K = len(samples)
                            if K >= 500:
                                samples = samples[-500:]
                                neuro_engine.pinned_cpu_buffer[slot_idx*16:(slot_idx+1)*16, :] = torch.tensor(samples, dtype=torch.float32).T
                            else:
                                neuro_engine.pinned_cpu_buffer[slot_idx*16:(slot_idx+1)*16, :-K] = neuro_engine.pinned_cpu_buffer[slot_idx*16:(slot_idx+1)*16, K:].clone()
                                neuro_engine.pinned_cpu_buffer[slot_idx*16:(slot_idx+1)*16, -K:] = torch.tensor(samples, dtype=torch.float32).T
                    
                    comp_val = ui_compression if arena.cfg.get('constrict_frequency_on_compression', True) else 0.0
                    c0_spec, freqs, bci_vx, bci_vy, bci_tq = neuro_engine.get_predictive_ciplv(len(active_slots) * 16, comp_val)
                    eeg_c0_spectrum = c0_spec[:16, :16, :]
                    eeg_freqs = freqs
                    
                    raw_bci_vx = bci_vx.item() if hasattr(bci_vx, 'item') else float(bci_vx)
                    raw_bci_vy = bci_vy.item() if hasattr(bci_vy, 'item') else float(bci_vy)
                    raw_bci_tq = bci_tq.item() if hasattr(bci_tq, 'item') else float(bci_tq)
                    
                    eeg_vx = max(-1.0, min(1.0, raw_bci_vx / 350.0))
                    eeg_vy = max(-1.0, min(1.0, raw_bci_vy / 350.0))
                    eeg_tq = max(-1.0, min(1.0, raw_bci_tq / 60.0))

            if ui_compression > 0.0:
                scale = 1.25 - ui_compression * 1.10
            else:
                scale = 1.25 - ui_compression * 5.25
            
            prev_captured = arena.pin_captured.sum().item()
            
            arena.step(dt, time_sec, eeg_c0_spectrum, eeg_vx, eeg_vy, eeg_tq, is_real_data, ui_compression, scale, eeg_freqs, alch_freq, alch_spatial)
            
            audio_manager.update(arena)
            
            new_captured = arena.pin_captured.sum().item()
            
            if new_captured == 0 and prev_captured == 16:
                last_finish_time = time.time() - run_start_time
                run_start_time = time.time()
                
            current_run_time = time.time() - run_start_time
            
            screen.blit(renderer.render_field(arena), (0, 0))
            
            if show_lines:
                renderer.draw_tension_lines(screen, arena)
            if show_sensors:
                renderer.draw_electrode_sensors(screen, arena)
                renderer.draw_ui(screen, arena) 
                
            time_str = f"TIME: {current_run_time:06.3f}s"
            shadow = font.render(time_str, True, (0, 0, 0))
            text = font.render(time_str, True, (0, 255, 200))
            screen.blit(shadow, (22, 22))
            screen.blit(text, (20, 20))
            
            fps_str = f"FPS: {current_fps} Hz"
            fps_color = (0, 255, 100) if current_fps >= 60 else (255, 100, 100)
            f_shadow = font.render(fps_str, True, (0, 0, 0))
            f_text = font.render(fps_str, True, fps_color)
            screen.blit(f_shadow, (22, 52))
            screen.blit(f_text, (20, 50))
            
            if last_finish_time > 0:
                prev_str = f"PREV: {last_finish_time:06.3f}s"
                p_shadow = font.render(prev_str, True, (0, 0, 0))
                p_text = font.render(prev_str, True, (150, 150, 150))
                screen.blit(p_shadow, (22, 82))
                screen.blit(p_text, (20, 80))
                
            pygame.display.flip()

        if HAS_NEURO: driver.scanner_running = False
        pygame.quit()
        sys.exit()
    except Exception as e:
        print("[CRITICAL EXCEPTION IN MAIN LOOP]:")
        traceback.print_exc()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()
