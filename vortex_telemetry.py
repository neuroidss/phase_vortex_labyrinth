# vortex_telemetry.py
import numpy as np
import torch

def classify_control_style(telemetry):
    if len(telemetry['deformation']) < 5:
        return "Gamepad"
        
    defs = np.array(telemetry['deformation'])
    angles = np.array(telemetry['angles'])
    accels = np.array(telemetry['acceleration'])
    vel_mags = np.array(telemetry['vel_mags'])
    
    pi_4 = np.pi / 4.0
    angles = np.mod(angles + np.pi, 2 * np.pi) - np.pi
    grid_devs = np.minimum(np.abs(angles % pi_4), pi_4 - np.abs(angles % pi_4))
    mean_grid_dev = np.mean(grid_devs)
    accel_jitter = np.std(accels)
    mean_def = np.mean(defs)
    
    if accel_jitter < 3.5:
        return "AI (Autopilot)"
    if mean_grid_dev < 0.06:
        return "Keyboard"
    if mean_def > 14.0:
        return "Neuroslime (Direct EEG)"
    if accel_jitter > 28.0:
        return "Neurogamepad (EEG-Stick)"
        
    return "Gamepad"


def update_rune_zones(rune_zones, player_pos, pin_pos, pin_captured, pin_vel, smooth_vorticity, player_angle, dt, device):
    for zone in rune_zones:
        if zone['completed']:
            continue
            
        dx_z = player_pos[0] - zone['pos'][0]
        dy_z = player_pos[1] - zone['pos'][1]
        dist_z = torch.sqrt(dx_z**2 + dy_z**2).item()
        
        if dist_z < zone['radius'] * 1.2:
            local_energy = abs(smooth_vorticity) * 2.5 + 0.15
            zone['charge'] = min(1.0, zone['charge'] + local_energy * dt * 0.4)
            
            active_vels = pin_vel[~pin_captured]
            v_com = active_vels.mean(dim=0) if active_vels.numel() > 0 else torch.zeros(2, device=device)
            v_mag = torch.norm(v_com).item()
            
            accel = torch.norm(v_com - zone['last_vel']).item() / (dt + 1e-5)
            zone['last_vel'].copy_(v_com)
            
            node_dists = torch.norm(pin_pos - player_pos, dim=1)
            deform = torch.std(node_dists).item()
            
            zone['telemetry']['deformation'].append(float(deform))
            zone['telemetry']['vorticity'].append(float(smooth_vorticity))
            zone['telemetry']['acceleration'].append(float(accel))
            zone['telemetry']['angles'].append(float(player_angle))
            zone['telemetry']['vel_mags'].append(float(v_mag))
            
            if zone['charge'] >= 1.0:
                zone['completed'] = True
                zone['classification'] = classify_control_style(zone['telemetry'])
        else:
            zone['charge'] = max(0.0, zone['charge'] - dt * 0.1)
