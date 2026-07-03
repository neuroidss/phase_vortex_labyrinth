# phase_vortex_labyrinth.py
import pygame
import torch
import sys
import traceback
import time

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

def main():
    try:
        pygame.init()
        pygame.font.init()
        pygame.joystick.init()
        
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Fluid Vortex Labyrinth (Full-Spectrum Time-Attack)")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Consolas", 28, bold=True)
        
        # Инициализируем обособленный модуль управления
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
        
        show_lines = False
        show_sensors = False

        run_start_time = time.time()
        last_finish_time = 0.0
        
        running = True
        while running:
            dt = min(0.032, clock.tick(60) / 1000.0)
            time_sec = pygame.time.get_ticks() / 1000.0
            
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
                        # Переключаем захват мыши через менеджер ввода
                        input_manager.toggle_mouse_lock()

            # --- ОПРОС УНИВЕРСАЛЬНОГО ВВОДА ---
            is_real_data, eeg_vx, eeg_vy, eeg_tq, ui_compression = input_manager.process_inputs(joysticks, dt)
            eeg_c0_spectrum = None
            eeg_freqs = None

            # --- BCI (Прямая проекция всех частот) ---
            if HAS_NEURO:
                active_slots = [i for i in range(5) if driver.workers[i].is_connected or any(v == i for v in driver.lsl_inlets.values())]
                if active_slots:
                    is_real_data = True
                    for slot_idx in active_slots:
                        q = driver.queues[slot_idx]
                        while len(q) > 0:
                            neuro_engine.pinned_cpu_buffer[slot_idx*16:(slot_idx+1)*16, :-1] = neuro_engine.pinned_cpu_buffer[slot_idx*16:(slot_idx+1)*16, 1:].clone()
                            neuro_engine.pinned_cpu_buffer[slot_idx*16:(slot_idx+1)*16, -1] = torch.tensor(q.popleft())
                    
                    # Передаем текущую компрессию для динамического сужения STFT-фильтра к 18-36 Гц
                    c0_spec, freqs, bci_vx, bci_vy, bci_tq = neuro_engine.get_predictive_ciplv(len(active_slots) * 16, ui_compression)
                    eeg_c0_spectrum = c0_spec[:16, :16, :]
                    eeg_freqs = freqs
                    
                    # Переносим BCI-интенции напрямую в управляющие векторы вместо их обнуления
                    raw_bci_vx = bci_vx.item() if hasattr(bci_vx, 'item') else float(bci_vx)
                    raw_bci_vy = bci_vy.item() if hasattr(bci_vy, 'item') else float(bci_vy)
                    raw_bci_tq = bci_tq.item() if hasattr(bci_tq, 'item') else float(bci_tq)
                    
                    # Калибруем масштабы под стандартный диапазон стиков [-1.0 ... 1.0]
                    eeg_vx = max(-1.0, min(1.0, raw_bci_vx / 350.0))
                    eeg_vy = max(-1.0, min(1.0, raw_bci_vy / 350.0))
                    eeg_tq = max(-1.0, min(1.0, raw_bci_tq / 60.0))

            # Динамически адаптируем масштаб отрисовки в зависимости от сжатия
            if ui_compression > 0.0:
                scale = 1.25 - ui_compression * 1.10
            else:
                scale = 1.25 - ui_compression * 5.25
            
            # Считываем количество захваченных нод до шага физики
            prev_captured = arena.pin_captured.sum().item()
            
            arena.step(dt, time_sec, eeg_c0_spectrum, eeg_vx, eeg_vy, eeg_tq, is_real_data, ui_compression, scale, eeg_freqs)
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
            
            if last_finish_time > 0:
                prev_str = f"PREV: {last_finish_time:06.3f}s"
                p_shadow = font.render(prev_str, True, (0, 0, 0))
                p_text = font.render(prev_str, True, (150, 150, 150))
                screen.blit(p_shadow, (22, 52))
                screen.blit(p_text, (20, 50))
                
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
