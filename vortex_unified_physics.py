# vortex_unified_physics.py
import math
import torch
import torch.nn.functional as F
from vortex_softbody import update_neighbor_springs, apply_cohesion_constraint

def calculate_covariance_angle(pin_pos, pin_x, pin_y, current_angle, tq, dt, device):
    """
    Calculates actor orientation angle via node coordinates covariance.
    """
    com = pin_pos.mean(dim=0)
    actual_local_x = pin_pos[:, 0] - com[0]
    actual_local_y = pin_pos[:, 1] - com[1]
    
    cross_cov = torch.sum(pin_y * actual_local_x - pin_x * actual_local_y)
    dot_cov = torch.sum(pin_x * actual_local_x + pin_y * actual_local_y) + 1e-5
    
    raw_angle = torch.atan2(cross_cov, dot_cov).item()
    angle_diff = (raw_angle - current_angle + math.pi) % (2 * math.pi) - math.pi
    
    new_angle = current_angle + angle_diff * 0.40
    new_angle -= tq * 3.5 * dt
    return new_angle, com

def extract_semantic_core_vector(eeg_c0_spectrum, eeg_freqs, device):
    """
    Translates raw 60-bin EEG spectrum into the 3-axis Alchemical Vector (Working Memory 2.0 Mapping).
    Returns absolute energy integrals: motor_e (Yang/Red), gamma_e (Catalyst/Green), theta_e (Yin/Blue), 
    and the normalized 3D tensor representing the user's instantaneous Elemental Core.
    """
    if eeg_c0_spectrum is None or eeg_freqs is None:
        return 0.0, 0.0, 0.0, torch.tensor([0.577, 0.577, 0.577], dtype=torch.float32, device=device)
        
    # 1. 18-36 Hz (Motor Imagery) mapped to Yang (Red / Translation)
    w_motor = torch.exp(-((eeg_freqs - 27.0) ** 2) / 100.0).view(1, 1, -1)
    # 2. 4-8 Hz (Theta) mapped to Yin (Blue / Radial Cohesion)
    w_theta = torch.exp(-((eeg_freqs - 6.0) ** 2) / 16.0).view(1, 1, -1)
    # 3. 60+ Hz (High Gamma) mapped to Catalyst (Green / Vorticity) - Clean Gaussian peak centered at 75 Hz
    w_gamma = torch.exp(-((eeg_freqs - 75.0) ** 2) / 400.0).view(1, 1, -1)
    
    motor_e = torch.sum(torch.abs(eeg_c0_spectrum) * w_motor).item()
    theta_e = torch.sum(torch.abs(eeg_c0_spectrum) * w_theta).item()
    gamma_e = torch.sum(torch.abs(eeg_c0_spectrum) * w_gamma).item()
    
    raw_vec = torch.tensor([motor_e, gamma_e, theta_e], dtype=torch.float32, device=device)
    norm_vec = raw_vec / (torch.norm(raw_vec) + 1e-8)
    
    return motor_e, gamma_e, theta_e, norm_vec

def apply_unified_actor_forces(device, res, WIDTH, HEIGHT, pin_pos, pin_x, pin_y, current_angle,
                               edge_intact, u, v, wall_density, vx, vy, tq, dt, blend,
                               node_radius, x_indices, y_indices, player_density, cohesion_force, cohesion_gravity,
                               bci_mode='3_axis', eeg_c0_spectrum=None, eeg_freqs=None):
    """
    Symmetrically projects analog movement forces onto the 2D fluid grid.
    Fully supports 120-jet direct high-dimensional BCI mapping without reduction.
    """
    com = pin_pos.mean(dim=0)
    actual_local_x = pin_pos[:, 0] - com[0]
    actual_local_y = pin_pos[:, 1] - com[1]
    
    dist_local = torch.sqrt(actual_local_x**2 + actual_local_y**2) + 1e-5
    tangent_x = -actual_local_y / dist_local
    tangent_y = actual_local_x / dist_local

    # 3-Axis Neurogamepad Base Logic (Fallback)
    forward_speed, strafe_speed = -vy, vx
    world_vx = -math.sin(current_angle) * forward_speed + math.cos(current_angle) * strafe_speed
    world_vy = -math.cos(current_angle) * forward_speed - math.sin(current_angle) * strafe_speed
    
    force_mult_gp = 400.0 * (1.0 + max(0.0, blend) * 2.0)
    node_bci_force_x = torch.full((16,), world_vx, device=device) * force_mult_gp
    node_bci_force_y = torch.full((16,), world_vy, device=device) * force_mult_gp
    node_bci_force_x += tangent_x * tq * 200.0
    node_bci_force_y += tangent_y * tq * 200.0
    
    # Base spatial mapping grids
    pin_gx = torch.remainder((pin_pos[:, 0] / WIDTH) * res, res)
    pin_gy = torch.remainder((pin_pos[:, 1] / HEIGHT) * res, res)
    
    dx_shape = torch.remainder(x_indices.unsqueeze(0) - pin_gx.reshape(16, 1, 1) + res/2, res) - res/2
    dy_shape = torch.remainder(y_indices.unsqueeze(0) - pin_gy.reshape(16, 1, 1) + res/2, res) - res/2
    
    is_active_1d = edge_intact.float()

    # === 120 JETS NATIVE SPECTROSCOPY (0 DIMENSIONALITY REDUCTION) ===
    if bci_mode == '120_jets' and eeg_c0_spectrum is not None and eeg_freqs is not None:
        pos_x = pin_pos[:, 0].unsqueeze(1) - pin_pos[:, 0].unsqueeze(0)
        pos_y = pin_pos[:, 1].unsqueeze(1) - pin_pos[:, 1].unsqueeze(0)
        dist_matrix = torch.sqrt(pos_x**2 + pos_y**2) + 1e-5
        
        dir_x = pos_x / dist_matrix
        dir_y = pos_y / dist_matrix
        tangent_x_matrix = -dir_y
        tangent_y_matrix = dir_x
        
        # --- SEMANTIC PHYSICS DISPERSION (FRACTAL OVERLAPS) ---
        # 1. 18-36 Hz (Motor Imagery / High Beta) -> Linear Translation (Yang)
        w_trans = torch.exp(-((eeg_freqs - 27.0)**2) / 100.0).unsqueeze(0).unsqueeze(0)
        
        # 2. 4-8 Hz (Theta Container / Past) -> Radial Constriction/Cohesion (Yin)
        w_radial = torch.exp(-((eeg_freqs - 6.0)**2) / 16.0).unsqueeze(0).unsqueeze(0)
        
        # 3. 60+ Hz (High Gamma / Future / Entropy) -> Shear Vorticity (Catalyst)
        w_shear = torch.exp(-((eeg_freqs - 75.0)**2) / 400.0).unsqueeze(0).unsqueeze(0)
        
        # Integrate across the entire 60-bin frequency spectrum simultaneously 
        trans_flux_x = torch.sum(eeg_c0_spectrum * w_trans * dir_x.unsqueeze(2), dim=2) 
        trans_flux_y = torch.sum(eeg_c0_spectrum * w_trans * dir_y.unsqueeze(2), dim=2)
        
        rad_flux_x = torch.sum(eeg_c0_spectrum * w_radial * dir_x.unsqueeze(2), dim=2)
        rad_flux_y = torch.sum(eeg_c0_spectrum * w_radial * dir_y.unsqueeze(2), dim=2)
        
        shear_flux_x = torch.sum(eeg_c0_spectrum * w_shear * tangent_x_matrix.unsqueeze(2), dim=2)
        shear_flux_y = torch.sum(eeg_c0_spectrum * w_shear * tangent_y_matrix.unsqueeze(2), dim=2)
        
        # Independent boundary jet projection directly onto the fluid grid
        jet_force_x = torch.clamp(torch.sum(trans_flux_x * 85000.0 + rad_flux_x * 45000.0 + shear_flux_x * 125000.0, dim=1), -85000.0, 85000.0)
        jet_force_y = torch.clamp(torch.sum(trans_flux_y * 85000.0 + rad_flux_y * 45000.0 + shear_flux_y * 125000.0, dim=1), -85000.0, 85000.0)
        
        node_bci_force_x += jet_force_x
        node_bci_force_y += jet_force_y

    node_influence = torch.exp(-(dx_shape**2 + dy_shape**2) / node_radius) * is_active_1d.reshape(16, 1, 1)
    node_influence_norm = node_influence / (torch.sum(node_influence, dim=(1, 2), keepdim=True) + 1e-8)
    
    bci_force_grid_x = torch.sum(node_influence_norm * node_bci_force_x.reshape(16, 1, 1), dim=0)
    bci_force_grid_y = torch.sum(node_influence_norm * node_bci_force_y.reshape(16, 1, 1), dim=0)

    # Cohesion Gradient
    rho_pad = F.pad(player_density, (1, 1, 1, 1), mode='replicate')
    grad_x_rho = 0.5 * (rho_pad[:, :, 1:-1, 2:] - rho_pad[:, :, 1:-1, :-2])
    grad_y_rho = 0.5 * (rho_pad[:, :, 2:, 1:-1] - rho_pad[:, :, :-2, 1:-1])
    cohesion_coeff = cohesion_force + max(0.0, blend) * 180.0
    f_cohesion_x = grad_x_rho[0, 0] * cohesion_coeff
    f_cohesion_y = grad_y_rho[0, 0] * cohesion_coeff

    com_grid_x = (com[0] / WIDTH) * 2.0 - 1.0
    com_grid_y = (com[1] / HEIGHT) * 2.0 - 1.0
    
    grid_x = torch.linspace(-1.0, 1.0, res, device=device)
    grid_y = torch.linspace(-1.0, 1.0, res, device=device)
    y_grid, x_grid = torch.meshgrid(grid_y, grid_x, indexing='ij')
    
    dx_com = com_grid_x - x_grid
    dy_com = com_grid_y - y_grid
    gravity_coeff = cohesion_gravity * (1.0 + max(0.0, blend) * 1.5)
    f_gravity_x = dx_com * player_density[0, 0] * gravity_coeff
    f_gravity_y = dy_com * player_density[0, 0] * gravity_coeff

    u[0, 0] += (bci_force_grid_x * 1.5 + f_cohesion_x + f_gravity_x) * dt
    v[0, 0] += (bci_force_grid_y * 1.5 + f_cohesion_y + f_gravity_y) * dt
    
    return u, v

def update_unified_slime_kinematics(device, res, WIDTH, HEIGHT, pin_pos, pin_x, pin_y, current_angle,
                                    edge_intact, u, v, wall_density, tq, dt, scale, blend, config,
                                    cell_w, is_captured_mask=None,
                                    bci_mode='3_axis', eeg_c0_spectrum=None, eeg_freqs=None, vx=0.0, vy=0.0):
    """
    Symmetrically updates physical coordinates under restoring forces and fluid vectors.
    """
    com = pin_pos.mean(dim=0)
    cos_p = math.cos(current_angle)
    sin_p = math.sin(current_angle)
    ideal_x = pin_x * cos_p + pin_y * sin_p
    ideal_y = -pin_x * sin_p + pin_y * cos_p
    ideal_x_scaled = ideal_x * scale
    ideal_y_scaled = ideal_y * scale
    ideal_pos = com.unsqueeze(0) + torch.stack([ideal_x_scaled, ideal_y_scaled], dim=1)

    pin_uv_raw = (pin_pos / torch.tensor([WIDTH, HEIGHT], dtype=torch.float32, device=device)) * 2.0 - 1.0
    pin_uv = torch.clamp(pin_uv_raw, -1.0, 1.0)
    
    sampled_u = F.grid_sample(u, pin_uv.view(1, 1, 16, 2), mode='bilinear', align_corners=True).squeeze()
    sampled_v = F.grid_sample(v, pin_uv.view(1, 1, 16, 2), mode='bilinear', align_corners=True).squeeze()
    fluid_vel = torch.stack([sampled_u, sampled_v], dim=1) * 80.0

    inner_smooth = F.avg_pool2d(wall_density, kernel_size=5, stride=1, padding=2)
    w_inner_pad = F.pad(inner_smooth, (1, 1, 1, 1), mode='replicate')
    grad_x_in = 0.5 * (w_inner_pad[:, :, 1:-1, 2:] - w_inner_pad[:, :, 1:-1, :-2])
    grad_y_in = 0.5 * (w_inner_pad[:, :, 2:, 1:-1] - w_inner_pad[:, :, :-2, 1:-1])
    w_gx = F.grid_sample(grad_x_in, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
    w_gy = F.grid_sample(grad_y_in, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
    
    grad_norm = torch.sqrt(w_gx**2 + w_gy**2) + 1e-5
    dir_out_x, dir_out_y = -w_gx / grad_norm, -w_gy / grad_norm

    w_val_smooth_in = F.grid_sample(inner_smooth, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
    f_wall = torch.stack([dir_out_x * w_val_smooth_in * 3500.0, dir_out_y * w_val_smooth_in * 3500.0], dim=1)

    if blend >= 0.0: 
        shape_pull = 15.0 + blend * 35.0
    else: 
        shape_pull = 15.0 + blend * 10.0 
        
    f_restore = (ideal_pos - pin_pos) * shape_pull
    f_spring, edge_intact = update_neighbor_springs(pin_pos, ideal_pos, edge_intact, blend, device)

    slip_factor = 0.85 + blend * 0.15
    slip_factor = max(0.65, min(1.0, slip_factor))

    actual_local_x = pin_pos[:, 0] - com[0]
    actual_local_y = pin_pos[:, 1] - com[1]
    dist_local = torch.sqrt(actual_local_x**2 + actual_local_y**2) + 1e-5
    tangent_x = -actual_local_y / dist_local
    tangent_y = actual_local_x / dist_local

    pin_vel = fluid_vel * slip_factor + f_wall + f_spring * 0.15 + f_restore

    forward_speed, strafe_speed = -vy, vx
    world_vx = -math.sin(current_angle) * forward_speed + cos_p * strafe_speed
    world_vy = -cos_p * forward_speed - math.sin(current_angle) * strafe_speed
    
    pin_vel[:, 0] += world_vx * 250.0
    pin_vel[:, 1] += world_vy * 250.0
    pin_vel[:, 0] += tangent_x * tq * 150.0
    pin_vel[:, 1] += tangent_y * tq * 150.0

    # === 120 JETS NATIVE KINEMATICS (WORKING MEMORY 2.0 MAPPING) ===
    if bci_mode == '120_jets' and eeg_c0_spectrum is not None and eeg_freqs is not None:
        pos_x = pin_pos[:, 0].unsqueeze(1) - pin_pos[:, 0].unsqueeze(0)
        pos_y = pin_pos[:, 1].unsqueeze(1) - pin_pos[:, 1].unsqueeze(0)
        dist_matrix = torch.sqrt(pos_x**2 + pos_y**2) + 1e-5
        dir_x = pos_x / dist_matrix
        dir_y = pos_y / dist_matrix
        tangent_x_matrix = -dir_y
        tangent_y_matrix = dir_x
        
        # Extract direct kinetic nodes from identical spectroscopy bands
        w_trans = torch.exp(-((eeg_freqs - 27.0)**2) / 100.0).unsqueeze(0).unsqueeze(0)
        w_radial = torch.exp(-((eeg_freqs - 6.0)**2) / 16.0).unsqueeze(0).unsqueeze(0)
        w_shear = torch.exp(-((eeg_freqs - 75.0)**2) / 400.0).unsqueeze(0).unsqueeze(0)
        
        trans_flux_x = torch.sum(eeg_c0_spectrum * w_trans * dir_x.unsqueeze(2), dim=2)
        trans_flux_y = torch.sum(eeg_c0_spectrum * w_trans * dir_y.unsqueeze(2), dim=2)
        rad_flux_x = torch.sum(eeg_c0_spectrum * w_radial * dir_x.unsqueeze(2), dim=2)
        rad_flux_y = torch.sum(eeg_c0_spectrum * w_radial * dir_y.unsqueeze(2), dim=2)
        shear_flux_x = torch.sum(eeg_c0_spectrum * w_shear * tangent_x_matrix.unsqueeze(2), dim=2)
        shear_flux_y = torch.sum(eeg_c0_spectrum * w_shear * tangent_y_matrix.unsqueeze(2), dim=2)
        
        # Extremely responsive pure translation, avoiding artificial smoothing
        jet_vx = torch.clamp(torch.sum(trans_flux_x * 3500.0 + rad_flux_x * 1500.0 + shear_flux_x * 6500.0, dim=1), -2500.0, 2500.0)
        jet_vy = torch.clamp(torch.sum(trans_flux_y * 3500.0 + rad_flux_y * 1500.0 + shear_flux_y * 6500.0, dim=1), -2500.0, 2500.0)
        
        pin_vel[:, 0] += jet_vx
        pin_vel[:, 1] += jet_vy

    limit_inner = config.get('inner_wall_penetration_limit', 0.04)
    w_val_sharp = F.grid_sample(wall_density, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
    blocking_inner = torch.clamp(w_val_sharp / limit_inner, 0.0, 1.0)
    
    dot_inner = pin_vel[:, 0] * dir_out_x + pin_vel[:, 1] * dir_out_y
    moving_into_inner = dot_inner < 0
    
    pin_vel[:, 0] -= torch.where(moving_into_inner, dot_inner * dir_out_x * blocking_inner, torch.zeros_like(pin_vel[:, 0]))
    pin_vel[:, 1] -= torch.where(moving_into_inner, dot_inner * dir_out_y * blocking_inner, torch.zeros_like(pin_vel[:, 1]))
    
    pin_vel = torch.clamp(pin_vel, -350.0, 350.0)
    
    if is_captured_mask is None:
        is_captured_mask = torch.zeros(16, dtype=torch.bool, device=device)
        
    pin_pos[~is_captured_mask] += pin_vel[~is_captured_mask] * dt

    pin_pos[:, 0] = torch.clamp(pin_pos[:, 0], 30.0, WIDTH - 30.0)
    pin_pos[:, 1] = torch.clamp(pin_pos[:, 1], 30.0, HEIGHT - 30.0)
    
    new_com = pin_pos.mean(dim=0)
    ideal_pos_new = new_com.unsqueeze(0) + torch.stack([ideal_x_scaled, ideal_y_scaled], dim=1)
    pin_pos = apply_cohesion_constraint(pin_pos, ideal_pos_new, is_captured_mask, scale, blend)
    
    pin_uv_post = torch.clamp((pin_pos / torch.tensor([WIDTH, HEIGHT], dtype=torch.float32, device=device)) * 2.0 - 1.0, -1.0, 1.0)
    w_val_sharp_post = F.grid_sample(wall_density, pin_uv_post.view(1, 1, 16, 2), align_corners=True).squeeze()
    
    push_mult = 12.0 + blend * 8.0
    pushed_wall = w_val_sharp_post > limit_inner
    if pushed_wall.any():
        pin_pos[pushed_wall, 0] += dir_out_x[pushed_wall] * (w_val_sharp_post[pushed_wall] - limit_inner) * push_mult
        pin_pos[pushed_wall, 1] += dir_out_y[pushed_wall] * (w_val_sharp_post[pushed_wall] - limit_inner) * push_mult

    return pin_pos, edge_intact, new_com
