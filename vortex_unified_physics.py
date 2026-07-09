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

def apply_unified_actor_forces(device, res, WIDTH, HEIGHT, pin_pos, pin_x, pin_y, current_angle,
                               edge_intact, u, v, wall_density, vx, vy, tq, dt, blend,
                               node_radius, x_indices, y_indices, player_density, cohesion_force, cohesion_gravity):
    """
    Symmetrically projects analog movement forces onto the 2D fluid grid.
    """
    com = pin_pos.mean(dim=0)
    actual_local_x = pin_pos[:, 0] - com[0]
    actual_local_y = pin_pos[:, 1] - com[1]
    
    dist_local = torch.sqrt(actual_local_x**2 + actual_local_y**2) + 1e-5
    tangent_x = -actual_local_y / dist_local
    tangent_y = actual_local_x / dist_local

    forward_speed, strafe_speed = -vy, vx
    world_vx = -math.sin(current_angle) * forward_speed + math.cos(current_angle) * strafe_speed
    world_vy = -math.cos(current_angle) * forward_speed - math.sin(current_angle) * strafe_speed
    
    force_mult_gp = 400.0 * (1.0 + max(0.0, blend) * 2.0)
    node_bci_force_x = torch.full((16,), world_vx, device=device) * force_mult_gp
    node_bci_force_y = torch.full((16,), world_vy, device=device) * force_mult_gp
    node_bci_force_x += tangent_x * tq * 200.0
    node_bci_force_y += tangent_y * tq * 200.0
    
    pin_gx = torch.remainder((pin_pos[:, 0] / WIDTH) * res, res)
    pin_gy = torch.remainder((pin_pos[:, 1] / HEIGHT) * res, res)
    
    dx_shape = torch.remainder(x_indices.unsqueeze(0) - pin_gx.reshape(16, 1, 1) + res/2, res) - res/2
    dy_shape = torch.remainder(y_indices.unsqueeze(0) - pin_gy.reshape(16, 1, 1) + res/2, res) - res/2
    
    is_active_1d = edge_intact.float()
    node_influence = torch.exp(-(dx_shape**2 + dy_shape**2) / node_radius) * is_active_1d.reshape(16, 1, 1)
    node_influence_norm = node_influence / (torch.sum(node_influence, dim=(1, 2), keepdim=True) + 1e-8)
    
    bci_force_grid_x = torch.sum(node_influence_norm * node_bci_force_x.reshape(16, 1, 1), dim=0)
    bci_force_grid_y = torch.sum(node_influence_norm * node_bci_force_y.reshape(16, 1, 1), dim=0)

    # Cohesion Gradient
    rho_pad = F.pad(player_density, (1, 1, 1, 1), mode='circular')
    grad_x_rho = 0.5 * (rho_pad[:, :, 1:-1, 2:] - rho_pad[:, :, 1:-1, :-2])
    grad_y_rho = 0.5 * (rho_pad[:, :, 2:, 1:-1] - rho_pad[:, :, :-2, 1:-1])
    cohesion_coeff = cohesion_force + max(0.0, blend) * 180.0
    f_cohesion_x = grad_x_rho[0, 0] * cohesion_coeff
    f_cohesion_y = grad_y_rho[0, 0] * cohesion_coeff

    # Centripetal gravity
    com_grid_x = (com[0] / WIDTH) * 2.0 - 1.0
    com_grid_y = (com[1] / HEIGHT) * 2.0 - 1.0
    
    grid_x = torch.linspace(-1.0, 1.0, res, device=device)
    grid_y = torch.linspace(-1.0, 1.0, res, device=device)
    y_grid, x_grid = torch.meshgrid(grid_y, grid_x, indexing='ij')
    
    dx_com = torch.remainder((com_grid_x - x_grid) + 1.0, 2.0) - 1.0
    dy_com = torch.remainder((com_grid_y - y_grid) + 1.0, 2.0) - 1.0
    gravity_coeff = cohesion_gravity * (1.0 + max(0.0, blend) * 1.5)
    f_gravity_x = dx_com * player_density[0, 0] * gravity_coeff
    f_gravity_y = dy_com * player_density[0, 0] * gravity_coeff

    u[0, 0] += (bci_force_grid_x * 1.5 + f_cohesion_x + f_gravity_x) * dt
    v[0, 0] += (bci_force_grid_y * 1.5 + f_cohesion_y + f_gravity_y) * dt
    
    return u, v

def update_unified_slime_kinematics(device, res, WIDTH, HEIGHT, pin_pos, pin_x, pin_y, current_angle,
                                    edge_intact, u, v, wall_density, tq, dt, scale, blend, config,
                                    cell_w, is_captured_mask=None):
    """
    Symmetrically updates physical coordinates under restoring forces and fluid vectors.
    """
    com = pin_pos.mean(dim=0)
    cos_p, sin_p = math.cos(current_angle), math.sin(current_angle)
    ideal_x = pin_x * cos_p + pin_y * sin_p
    ideal_y = -pin_x * sin_p + pin_y * cos_p
    ideal_x_scaled = ideal_x * scale
    ideal_y_scaled = ideal_y * scale
    ideal_pos = com.unsqueeze(0) + torch.stack([ideal_x_scaled, ideal_y_scaled], dim=1)

    pin_uv_raw = (pin_pos / torch.tensor([WIDTH, HEIGHT], dtype=torch.float32, device=device)) * 2.0 - 1.0
    pin_uv = torch.remainder(pin_uv_raw + 1.0, 2.0) - 1.0
    
    sampled_u = F.grid_sample(u, pin_uv.view(1, 1, 16, 2), mode='bilinear', align_corners=True).squeeze()
    sampled_v = F.grid_sample(v, pin_uv.view(1, 1, 16, 2), mode='bilinear', align_corners=True).squeeze()
    fluid_vel = torch.stack([sampled_u, sampled_v], dim=1) * 80.0

    # Boundary wall forces
    inner_smooth = F.avg_pool2d(wall_density, kernel_size=5, stride=1, padding=2)
    w_inner_pad = F.pad(inner_smooth, (1, 1, 1, 1), mode='circular')
    grad_x_in = 0.5 * (w_inner_pad[:, :, 1:-1, 2:] - w_inner_pad[:, :, 1:-1, :-2])
    grad_y_in = 0.5 * (w_inner_pad[:, :, 2:, 1:-1] - w_inner_pad[:, :, :-2, 1:-1])
    w_gx = F.grid_sample(grad_x_in, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
    w_gy = F.grid_sample(grad_y_in, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
    
    grad_norm = torch.sqrt(w_gx**2 + w_gy**2) + 1e-5
    dir_out_x, dir_out_y = -w_gx / grad_norm, -w_gy / grad_norm

    w_val_smooth_in = F.grid_sample(inner_smooth, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
    f_wall = torch.stack([dir_out_x * w_val_smooth_in * 3500.0, dir_out_y * w_val_smooth_in * 3500.0], dim=1)

    # Restoring structural elasticity
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
    pin_vel[:, 0] += tangent_x * tq * 100.0
    pin_vel[:, 1] += tangent_y * tq * 100.0

    # Defensive bounds
    limit_inner = config.get('inner_wall_penetration_limit', 0.04)
    w_val_sharp = F.grid_sample(wall_density, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
    blocking_inner = torch.clamp(w_val_sharp / limit_inner, 0.0, 1.0)
    
    dot_inner = pin_vel[:, 0] * dir_out_x + pin_vel[:, 1] * dir_out_y
    moving_into_inner = dot_inner < 0
    
    pin_vel[:, 0] -= torch.where(moving_into_inner, dot_inner * dir_out_x * blocking_inner, torch.zeros_like(pin_vel[:, 0]))
    pin_vel[:, 1] -= torch.where(moving_into_inner, dot_inner * dir_out_y * blocking_inner, torch.zeros_like(pin_vel[:, 1]))
    
    pin_vel = torch.clamp(pin_vel, -180.0, 180.0)
    
    if is_captured_mask is None:
        is_captured_mask = torch.zeros(16, dtype=torch.bool, device=device)
        
    pin_pos[~is_captured_mask] += pin_vel[~is_captured_mask] * dt
    
    new_com = pin_pos.mean(dim=0)
    ideal_pos_new = new_com.unsqueeze(0) + torch.stack([ideal_x_scaled, ideal_y_scaled], dim=1)
    pin_pos = apply_cohesion_constraint(pin_pos, ideal_pos_new, is_captured_mask, scale, blend)
    
    # Secondary wall repulsion constraint
    pin_uv_post = torch.remainder((pin_pos / torch.tensor([WIDTH, HEIGHT], dtype=torch.float32, device=device)) * 2.0, 2.0) - 1.0
    w_val_sharp_post = F.grid_sample(wall_density, pin_uv_post.view(1, 1, 16, 2), align_corners=True).squeeze()
    
    push_mult = 12.0 + blend * 8.0
    pushed_wall = w_val_sharp_post > limit_inner
    if pushed_wall.any():
        pin_pos[pushed_wall, 0] += dir_out_x[pushed_wall] * (w_val_sharp_post[pushed_wall] - limit_inner) * push_mult
        pin_pos[pushed_wall, 1] += dir_out_y[pushed_wall] * (w_val_sharp_post[pushed_wall] - limit_inner) * push_mult

    return pin_pos, edge_intact, new_com
