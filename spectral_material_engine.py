# spectral_material_engine.py
import torch
import torch.nn.functional as F

class SpectralMaterialEngine:
    """
    Translates high-density ciPLV EEG spectral components into 
    point-wise hydrodynamic rheology fields (Viscoelasticity, Surface Tension, Buoyancy).
    Ensures zero-latency coupling between cognitive state and physical matter states.
    """
    def __init__(self, res, device):
        self.res = res
        self.device = device
        
    def compute_rheology_fields(self, density_spectral, integrity_map=None):
        """
        Calculates localized active-matter variables from the continuous 100-channel spectral density.
        Transforms the Navier-Stokes grid into a Non-Newtonian Viscoelastic Metamaterial.
        
        Args:
            density_spectral: Tensor of shape [1, 100, res, res] representing the 1-100Hz spectrum.
            integrity_map: Scalar float representing Kuramoto Phase Order parameter [0.0 ... 1.0]
            
        Returns:
            viscosity: Tensor [1, 1, res, res]
            surface_tension: Tensor [1, 1, res, res]
            buoyancy_force: Tensor [1, 2, res, res]
        """
        # 1. Extract functional bands representing biological cognitive states
        # Theta (4-8Hz): The Past / Memory / Cohesion / Shields
        theta_band = torch.sum(density_spectral[:, 4:9, :, :], dim=1, keepdim=True)
        # Beta (18-36Hz): The Present / Kinetic Flux / Motor Drive
        beta_band = torch.sum(density_spectral[:, 18:37, :, :], dim=1, keepdim=True)
        # Gamma (60-100Hz): The Future / Information Packets / Cavitation
        gamma_band = torch.sum(density_spectral[:, 60:101, :, :], dim=1, keepdim=True)
        
        # 2. VISCOELASTICITY FIELD (Non-Newtonian Shear Thickening)
        # Base state is highly fluid (like water)
        base_viscosity = 0.05
        
        # Theta acts as a shear-thickening agent. Where Theta is high, the fluid 
        # instantly crystallizes into a dense gel (Energy shield).
        # Gamma acts as a shear-thinning (cavitation) agent, ripping the fluid apart.
        viscosity = base_viscosity + (15.0 * theta_band) - (0.8 * gamma_band)
        
        # If integrity map (Kuramoto order parameter) is provided, high integrity 
        # acts as a multiplicative crystalline lattice structurer.
        if integrity_map is not None:
            viscosity *= (1.0 + integrity_map * 10.0)
            
        viscosity = torch.clamp(viscosity, 0.01, 200.0)
        
        # 3. SURFACE TENSION (Negentropic Boundary Strength)
        # Theta binds the fluid together. High surface tension resists mixing.
        surface_tension = 60.0 * theta_band / (beta_band + 1e-5)
        surface_tension = torch.clamp(surface_tension, 0.0, 150.0)
        
        # 4. ACTIVE BUOYANCY / CAVITATION PRESSURE
        # High Gamma (future packets) creates explosive expansion pressure.
        # Beta provides directed kinetic advection, not buoyancy.
        buoyancy_y = -15.81 * gamma_band
        # Random micro-convective plumes induced by Gamma cavitation
        buoyancy_x = torch.sin(gamma_band * 25.0) * gamma_band * 5.0
        buoyancy_force = torch.cat([buoyancy_x, buoyancy_y], dim=1)
        
        return viscosity, surface_tension, buoyancy_force
