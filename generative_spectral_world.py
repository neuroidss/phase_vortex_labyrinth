# generative_spectral_world.py
import torch
import torch.nn.functional as F
import math

class MultiDevicePhaseWorldEngine:
    """
    Implements a zero-dimensionality-reduction generative physical space.
    Maps localized 16-channel ciPLV phase matrices from multiple BCI nodes
    directly onto non-equilibrium thermodynamic operations (Delta, Alpha, Gamma-Past, Gamma-Future).
    Supports dual-avatar cross-coherence binding.
    """
    def __init__(self, res, width, height, device):
        self.res = res
        self.WIDTH, self.HEIGHT = width, height
        self.device = device
        
        dy, dx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, res, device=device),
            torch.linspace(-1.0, 1.0, res, device=device), indexing='ij'
        )
        self.grid_x = dx
        self.grid_y = dy

        self.parent_pos = torch.tensor([width * 0.35, height * 0.5], dtype=torch.float32, device=device)
        self.child_pos  = torch.tensor([width * 0.45, height * 0.5], dtype=torch.float32, device=device)
        self.glymphatic_pulsation_phase = 0.0

    def compute_laplacian(self, field):
        field_pad = F.pad(field, (1, 1, 1, 1), mode='replicate')
        return (field_pad[:, :, 1:-1, 2:] + field_pad[:, :, 1:-1, :-2] +
                field_pad[:, :, 2:, 1:-1] + field_pad[:, :, :-2, 1:-1] - 4.0 * field)

    def apply_negentropic_fractal_boundaries(self, density_complex, u, v, dt):
        """
        The core of the SMR-to-Wave bridging for the 6-channel RGB representation.
        Blue (4,5) = Theta/Water (Shields), Green (2,3) = Beta/Qi (Movement), Red (0,1) = Gamma/Fire (Projectiles).
        """
        gamma_mag = torch.hypot(density_complex[:, 0:1, :, :], density_complex[:, 1:2, :, :]) # Red
        beta_mag = torch.hypot(density_complex[:, 2:3, :, :], density_complex[:, 3:4, :, :])  # Green
        theta_mag = torch.hypot(density_complex[:, 4:5, :, :], density_complex[:, 5:6, :, :]) # Blue
        
        theta_pad = F.pad(theta_mag, (1, 1, 1, 1), mode='replicate')
        beta_pad = F.pad(beta_mag, (1, 1, 1, 1), mode='replicate')
        
        grad_theta_x = 0.5 * (theta_pad[:, :, 1:-1, 2:] - theta_pad[:, :, 1:-1, :-2])
        grad_theta_y = 0.5 * (theta_pad[:, :, 2:, 1:-1] - theta_pad[:, :, :-2, 1:-1])
        grad_beta_x = 0.5 * (beta_pad[:, :, 1:-1, 2:] - beta_pad[:, :, 1:-1, :-2])
        grad_beta_y = 0.5 * (beta_pad[:, :, 2:, 1:-1] - beta_pad[:, :, :-2, 1:-1])
        
        # ФИЗИЧЕСКИЙ ФИКС БАГА САМОУНИЧТОЖЕНИЯ
        dot_product = grad_theta_x * grad_beta_x + grad_theta_y * grad_beta_y
        interface_intensity = torch.relu(-dot_product) * 2.0 
        
        # 1. NEGENTROPY (Boundary Sharpening)
        lap_theta = self.compute_laplacian(theta_mag)
        lap_beta = self.compute_laplacian(beta_mag)
        
        sharpening_force = 1.5 * interface_intensity
        theta_change = -sharpening_force * lap_theta * dt
        beta_change = -sharpening_force * lap_beta * dt
        
        density_complex[:, 4:6, :, :] += (theta_change / 2.0)
        density_complex[:, 2:4, :, :] += (beta_change / 2.0)
        
        # 2. CROSS-FREQUENCY CONVERSION (Beta -> Gamma Projection)
        fluid_speed = torch.sqrt(u**2 + v**2) + 1e-8
        kinetic_compression = interface_intensity * fluid_speed
        
        conversion_rate = 8.5 * dt
        gamma_spawn = kinetic_compression * conversion_rate
        
        density_complex[:, 0:2, :, :] += gamma_spawn
        
        density_complex[:, 2:4, :, :] = torch.clamp(
            density_complex[:, 2:4, :, :] - (gamma_spawn / 2.0), -2.5, 2.5
        )
        
        return density_complex

    def step_phase_kinematics(self, u, v, density_complex, rho_stone, 
                              eeg_c0_spec_dev0, eeg_freqs_dev0, 
                              eeg_c0_spec_dev1, eeg_freqs_dev1, dt):
        """
        Executes non-arbitrary, physics-grounded interactions based on local physical phase states
        and BCI cross-coherence inputs.
        """
        density_complex = self.apply_negentropic_fractal_boundaries(density_complex, u, v, dt)

        cross_coupling_strength = 0.0
        if eeg_c0_spec_dev0 is not None and eeg_c0_spec_dev1 is not None:
            norm_dev0 = eeg_c0_spec_dev0 / (torch.norm(eeg_c0_spec_dev0) + 1e-8)
            norm_dev1 = eeg_c0_spec_dev1 / (torch.norm(eeg_c0_spec_dev1) + 1e-8)
            cross_coupling_strength = torch.sum(norm_dev0 * norm_dev1).item()
            cross_coupling_strength = max(0.0, min(1.0, cross_coupling_strength))

        p_uv = torch.stack([
            (self.parent_pos[0] / self.WIDTH) * 2.0 - 1.0,
            (self.parent_pos[1] / self.HEIGHT) * 2.0 - 1.0
        ]).view(1, 1, 1, 2)
        
        local_stone_density = F.grid_sample(rho_stone, p_uv, align_corners=True).squeeze().item()
        local_water_density = F.grid_sample(
            torch.hypot(density_complex[:, 4:5], density_complex[:, 5:6]),
            p_uv, align_corners=True
        ).squeeze().item()
        
        w_beta = torch.exp(-((eeg_freqs_dev0 - 27.0) ** 2) / 81.0).view(1, 1, -1)
        beta_coherence = torch.sum(torch.abs(eeg_c0_spec_dev0) * w_beta, dim=2)
        
        coords_x = torch.tensor([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], device=self.device)
        coords_y = torch.tensor([-2.72, -7.43, -4.77, -10.15, -10.14, -4.77, -7.42, -2.73, 2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], device=self.device)
        dX = coords_x.unsqueeze(0) - coords_x.unsqueeze(1)
        dY = coords_y.unsqueeze(0) - coords_y.unsqueeze(1)
        
        intent_x = torch.sum(beta_coherence * dX).item() * 0.12
        intent_y = torch.sum(beta_coherence * dY).item() * 0.12
        intent_mag = math.hypot(intent_x, intent_y) + 1e-5
        
        permeability = 1.0
        is_drilling = False
        
        if local_stone_density > 0.5:
            drilling_power = intent_mag * (1.0 + cross_coupling_strength)
            if drilling_power > 15.0:
                permeability = 0.15 * (drilling_power / 15.0)
                is_drilling = True
            else:
                permeability = 0.0 
        elif local_water_density > 0.3:
            permeability = 0.65
        else:
            permeability = 1.0

        if permeability > 0.0:
            self.parent_pos[0] += (intent_x / intent_mag) * permeability * 150.0 * dt
            self.parent_pos[1] += (intent_y / intent_mag) * permeability * 150.0 * dt

        w_delta = torch.exp(-((eeg_freqs_dev0 - 2.0) ** 2) / 1.0).view(1, 1, -1)
        delta_power = torch.sum(torch.abs(eeg_c0_spec_dev0) * w_delta).item()
        
        if delta_power > 0.35:
            self.glymphatic_pulsation_phase += 4.0 * dt
            pulse = math.sin(self.glymphatic_pulsation_phase) * (delta_power * 15.0)
            u *= 0.92
            v *= 0.92
            density_complex[:, 0:2] *= 0.88 

        w_alpha = torch.exp(-((eeg_freqs_dev0 - 10.0) ** 2) / 4.0).view(1, 1, -1)
        alpha_power = torch.sum(torch.abs(eeg_c0_spec_dev0) * w_alpha).item()
        
        if alpha_power > 0.45:
            p_x, p_y = (self.parent_pos[0]/self.WIDTH)*self.res, (self.parent_pos[1]/self.HEIGHT)*self.res
            dist_parent = torch.sqrt((self.grid_x - p_x)**2 + (self.grid_y - p_y)**2) + 1e-5
            sweep_wave = torch.exp(-((dist_parent - (alpha_power * 12.0)) ** 2) / 4.0)
            u[0, 0] *= (1.0 - sweep_wave)
            v[0, 0] *= (1.0 - sweep_wave)
            density_complex *= (1.0 - sweep_wave.unsqueeze(0).unsqueeze(0) * 0.5)

        w_low_gamma = torch.exp(-((eeg_freqs_dev0 - 40.0) ** 2) / 100.0).view(1, 1, -1)
        low_gamma_power = torch.sum(torch.abs(eeg_c0_spec_dev0) * w_low_gamma).item()
        
        if low_gamma_power > 0.50:
            p_x, p_y = (self.parent_pos[0]/self.WIDTH)*self.res, (self.parent_pos[1]/self.HEIGHT)*self.res
            dist_parent = torch.sqrt((self.grid_x - p_x)**2 + (self.grid_y - p_y)**2) + 1e-5
            sink_force = torch.exp(-dist_parent / 15.0) * (low_gamma_power * 12.0)
            radial_x = (p_x - self.grid_x) / dist_parent
            radial_y = (p_y - self.grid_y) / dist_parent
            u[0, 0] += radial_x * sink_force * dt
            v[0, 0] += radial_y * sink_force * dt

        w_high_gamma = torch.exp(-((eeg_freqs_dev0 - 80.0) ** 2) / 400.0).view(1, 1, -1)
        high_gamma_power = torch.sum(torch.abs(eeg_c0_spec_dev0) * w_high_gamma).item()
        
        if high_gamma_power > 0.65 and intent_mag < 0.5:
            recoil_dir_x = -(intent_x / intent_mag)
            recoil_dir_y = -(intent_y / intent_mag)
            
            recoil_force = high_gamma_power * 800.0 * dt
            self.parent_pos[0] += recoil_dir_x * recoil_force
            self.parent_pos[1] += recoil_dir_y * recoil_force
            
            proj_x = self.parent_pos[0] + (intent_x / intent_mag) * 45.0
            proj_y = self.parent_pos[1] + (intent_y / intent_mag) * 45.0
            proj_gx = int((proj_x / self.WIDTH) * self.res)
            proj_gy = int((proj_y / self.HEIGHT) * self.res)
            
            if 0 <= proj_gx < self.res and 0 <= proj_gy < self.res:
                density_complex[0, 0, proj_gy, proj_gx] += 2.0 
                density_complex[0, 1, proj_gy, proj_gx] += 2.0 

        if cross_coupling_strength > 0.05:
            dx_c = self.child_pos[0] - self.parent_pos[0]
            dy_c = self.child_pos[1] - self.parent_pos[1]
            dist_c = math.hypot(dx_c, dy_c) + 1e-5
            
            ideal_orbit_dist = 65.0
            w_qi = torch.exp(-((eeg_freqs_dev1 - 14.0) ** 2) / 16.0).view(1, 1, -1) if eeg_freqs_dev1 is not None else torch.zeros_like(w_high_gamma)
            qi_power = torch.sum(torch.abs(eeg_c0_spec_dev1) * w_qi).item() if eeg_c0_spec_dev1 is not None else 0.1
            
            angle_speed = (0.5 + qi_power * 4.0) * dt
            current_angle = math.atan2(dy_c, dx_c) + angle_speed
            
            target_cx = self.parent_pos[0] + math.cos(current_angle) * ideal_orbit_dist
            target_cy = self.parent_pos[1] + math.sin(current_angle) * ideal_orbit_dist
            
            spring_k = cross_coupling_strength * 18.0
            self.child_pos[0] += (target_cx - self.child_pos[0]) * spring_k * dt
            self.child_pos[1] += (target_cy - self.child_pos[1]) * spring_k * dt
        else:
            c_uv = torch.stack([
                (self.child_pos[0] / self.WIDTH) * 2.0 - 1.0,
                (self.child_pos[1] / self.HEIGHT) * 2.0 - 1.0
            ]).view(1, 1, 1, 2)
            
            sampled_u = F.grid_sample(u, c_uv, align_corners=True).squeeze().item()
            sampled_v = F.grid_sample(v, c_uv, align_corners=True).squeeze().item()
            
            self.child_pos[0] += sampled_u * 80.0 * dt
            self.child_pos[1] += sampled_v * 80.0 * dt

        self.parent_pos = torch.clamp(self.parent_pos, 30.0, torch.tensor([self.WIDTH - 30.0, self.HEIGHT - 30.0], device=self.device))
        self.child_pos  = torch.clamp(self.child_pos, 30.0, torch.tensor([self.WIDTH - 30.0, self.HEIGHT - 30.0], device=self.device))
        
        return u, v, density_complex
