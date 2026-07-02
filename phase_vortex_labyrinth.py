# phase_vortex_labyrinth.py
import pygame
import torch
import sys
import traceback

try:
    from neuro_driver import RealNeuroDriver
    from symbiotic_engine import SymbioticEngineGPU
    HAS_NEURO = True
except ImportError:
    HAS_NEURO = False

from vortex_physics import PhaseVortexArena
from vortex_renderer import VortexRenderer

WIDTH, HEIGHT = 800, 800
COMPUTE_RES = 128
ZOOM_OUT_FACTOR = 1.35

def main():
    try:
        pygame.init()
        pygame.joystick.init()
        
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Fluid Vortex Labyrinth (Immersive Mode)")
        clock = pygame.time.Clock()
        
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

        arena = PhaseVortexArena(device, WIDTH, HEIGHT, COMPUTE_RES)
        renderer = VortexRenderer(WIDTH, HEIGHT, ZOOM_OUT_FACTOR)
        
        ui_compression = 0.0
        
        # --- ПАРАМЕТРЫ ИММЕРСИВНОСТИ (Отключены по умолчанию) ---
        show_lines = False
        show_sensors = False

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

            keys = pygame.key.get_pressed()
            ui_compression = min(1.0, ui_compression + dt * 2.0) if keys[pygame.K_SPACE] else max(0.0, ui_compression - dt * 2.0)

            is_real_data, eeg_vx, eeg_vy, eeg_tq, eeg_c0 = False, 0.0, 0.0, 0.0, None
            if any(keys[k] for k in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN, pygame.K_a, pygame.K_d, pygame.K_w, pygame.K_s, pygame.K_q, pygame.K_e]):
                is_real_data = True
                if keys[pygame.K_LEFT] or keys[pygame.K_a]: eeg_vx -= 1.0
                if keys[pygame.K_RIGHT] or keys[pygame.K_d]: eeg_vx += 1.0
                if keys[pygame.K_UP] or keys[pygame.K_w]: eeg_vy -= 1.0
                if keys[pygame.K_DOWN] or keys[pygame.K_s]: eeg_vy += 1.0
                if keys[pygame.K_q]: eeg_tq -= 1.0
                if keys[pygame.K_e]: eeg_tq += 1.0

            if HAS_NEURO:
                active_slots = [i for i in range(5) if driver.workers[i].is_connected or any(v == i for v in driver.lsl_inlets.values())]
                if active_slots:
                    is_real_data = True
                    for slot_idx in active_slots:
                        q = driver.queues[slot_idx]
                        while len(q) > 0:
                            neuro_engine.pinned_cpu_buffer[slot_idx*16:(slot_idx+1)*16, :-1] = neuro_engine.pinned_cpu_buffer[slot_idx*16:(slot_idx+1)*16, 1:].clone()
                            neuro_engine.pinned_cpu_buffer[slot_idx*16:(slot_idx+1)*16, -1] = torch.tensor(q.popleft())
                    c0_gpu, _, _, vx, vy, tq, _, _ = neuro_engine.get_predictive_ciplv(len(active_slots) * 16)
                    eeg_c0 = c0_gpu[:16, :16]
                    eeg_vx, eeg_vy, eeg_tq = vx.item() * 0.012, -vy.item() * 0.012, tq.item() * 0.012

            if len(joysticks) > 0 and joysticks[0].get_numaxes() >= 5:
                ui_compression = (joysticks[0].get_axis(4) + 1.0) / 2.0
                if any(abs(joysticks[0].get_axis(a)) > 0.05 for a in range(joysticks[0].get_numaxes())):
                    is_real_data = True

            scale = 1.5 + (1.0 - ui_compression) * 5.0
            arena.step(dt, time_sec, eeg_c0, eeg_vx, eeg_vy, eeg_tq, is_real_data, ui_compression, scale)
            
            screen.blit(renderer.render_field(arena), (0, 0))
            
            if show_lines:
                renderer.draw_tension_lines(screen, arena)
            if show_sensors:
                renderer.draw_electrode_sensors(screen, arena)
                renderer.draw_ui(screen, arena) 
                
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
