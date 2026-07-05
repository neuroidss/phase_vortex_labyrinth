# vortex_softbody.py
import torch

def update_neighbor_springs(pin_pos, ideal_pos, edge_intact, cohesion, device):
    """
    Вычисляет упругие силы взаимного притяжения соседних нод по кольцу.
    Реализует динамический разрыв связей при растяжении и слияние при сближении.
    """
    idx_I = torch.arange(16, device=device)
    idx_J = torch.remainder(idx_I + 1, 16)
    
    pos_I, pos_J = pin_pos[idx_I], pin_pos[idx_J]
    ideal_I, ideal_J = ideal_pos[idx_I], ideal_pos[idx_J]
    
    d_curr = torch.norm(pos_I - pos_J, dim=1)
    d_rest = torch.norm(ideal_I - ideal_J, dim=1)
    strain = d_curr / (d_rest + 1e-5)
    
    if cohesion >= 0.0:
        T_tear = 2.0 + cohesion * 98.0  
    else:
        # Пружины рвутся сложнее даже в разжатом виде
        T_tear = 2.0 + (cohesion + 1.0) * 1.5 
        
    has_torn = strain > T_tear
    edge_intact = edge_intact & (~has_torn)
    
    has_closed = strain < 1.1
    edge_intact = edge_intact | has_closed
    
    dir_mutual = (pos_J - pos_I) / (d_curr.unsqueeze(1) + 1e-5)
    
    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Базовая жесткость пружин усилена!
    # При blend = -1.0 жесткость составляет 200, удерживая структуру геля
    k_neighbor = 200.0 * (max(0.0, cohesion) + 1.0)
    
    f_mag = k_neighbor * (d_curr - d_rest)
    
    f_mutual_I = dir_mutual * f_mag.unsqueeze(1) * edge_intact.float().unsqueeze(1)
    f_mutual_prev = torch.roll(f_mutual_I, shifts=1, dims=0)
    
    f_net_neighbor = f_mutual_I - f_mutual_prev
    f_net_neighbor_clamped = torch.clamp(f_net_neighbor, -400.0, 400.0)
    
    return f_net_neighbor_clamped, edge_intact


def apply_cohesion_constraint(pin_pos, ideal_pos, pin_captured, scale, cohesion_level):
    """
    Position-Based Dynamics (PBD) ограничитель деформации.
    Служит абсолютным предохранителем, плавно смыкающимся от -1 до 1.
    """
    if cohesion_level >= 0.0:
        max_allowed_drift = scale * (5.0 + (1.0 - cohesion_level) * 15.0)
    else:
        # В разжатом виде максимальный дрейф снижен с 50 до 20, 
        # чтобы узлы не расползались по стенам и не теряли гидродинамическую тягу
        max_allowed_drift = scale * (20.0 - cohesion_level * 15.0)
        
    diff_to_ideal = pin_pos - ideal_pos
    dist_to_ideal = torch.norm(diff_to_ideal, dim=1, keepdim=True) + 1e-5
    
    over_limit = (dist_to_ideal > max_allowed_drift) & (~pin_captured).unsqueeze(1)
    if over_limit.any():
        correction = (diff_to_ideal / dist_to_ideal) * max_allowed_drift
        mask_1d = over_limit.squeeze(1)
        pin_pos[mask_1d] = ideal_pos[mask_1d] + correction[mask_1d]
        
    return pin_pos
