# combat_config.py
"""
ULTRA-HIGH-DENSITY NEURO-COMBAT BALANCING CONFIGURATION
Calibrated for Non-linear Schrödinger-Navier-Stokes (NLSE) Auto-Battler Squads.
"""

# --- DEBUG & TESTING LAUNCH MODES ---
DIRECT_ARENA_BOOT = True 

# --- PHYSICAL COLLISION & DAMAGE MULTIPLIERS ---
PROXIMITY_DAMAGE_SCALE = 1.8
JITTER_FORCE_MAX_CLAMP = 14.0
JITTER_FORCE_MULTIPLIER = 0.45
BODY_REPULSION_STIFFNESS = 450.0
BODY_COLLISION_RADIUS = 35.0

# --- SPRING ELASTICITY & RUPTURE LIMITS ---
ELONGATION_LIMIT_BASE = 38.0
ELONGATION_LIMIT_STRETCH = 0.35
INTEGRITY_SHRINK_INFLUENCE = 0.3

# --- KURAMOTO PHASE DECOHERENCE (PHASE NOISE) ---
SCRAMBLE_RATE_SCALE = 1.8
SCRAMBLE_BROKEN_NODE_SCALE = 0.25
BEAM_SCRAMBLE_SCALE = 1.2

# --- REGENERATIVE PHASIC RECHARGE (STABILIZATION) ---
K_BOT_ASSIST_BASE = 25.0
K_BOT_ASSIST_QUALITY_SCALE = 20.0
K_BOT_ASSIST_SIMILARITY_SCALE = 25.0

# --- DECENTRALIZED GRADIENT LOCOMOTION DRIFT RATES ---
MOVEMENT_PHASE_DRIFT_SPEED = 0.15
TORQUE_PHASE_DRIFT_SPEED = 0.20

# =====================================================================
# NON-LINEAR SCHRÖDINGER WAVE DYNAMICS (NLSE)
# =====================================================================
G_MATRIX = [
    [ 1.5,  2.5,  4.5],  
    [ 2.5,  0.5,  2.5],  
    [ 4.5,  2.5, -6.0]   
]

# Calibrated weights for smooth, progressive advection acceleration
QC_WEIGHTS = [1200.0, 600.0, 2800.0] 
NLSE_SPEED = 220.0

BOT_ARCHETYPES = {
    "Vanguard": [{"name": "Heavy Aegis", "style": "Tank", "desc": "High Theta. Blocks enemies.", "vector": [0.2, 0.8, 0.0]}],
    "Assault": [{"name": "Striker", "style": "Fighter", "desc": "High Beta. Rushes the enemy.", "vector": [0.8, 0.2, 0.0]}],
    "Artillery": [{"name": "Soliton Weaver", "style": "Mage", "desc": "High Gamma. Soliton projectiles.", "vector": [0.0, 0.2, 0.8]}],
    "Support": [{"name": "Harmonic Guide", "style": "Healer", "desc": "Balanced Triad.", "vector": [0.33, 0.33, 0.33]}]
}
