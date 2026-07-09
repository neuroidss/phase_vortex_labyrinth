# combat_config.py
"""
ULTRA-HIGH-DENSITY NEURO-COMBAT BALANCING CONFIGURATION
Calibrate these physical and Kuramoto parameters to match the predicted championship results.
"""

# --- DEBUG & TESTING LAUNCH MODES ---
# Set to True to bypass Labyrinth entirely in main file and boot directly into the Arena
DIRECT_ARENA_BOOT = False 

# --- PHYSICAL COLLISION & DAMAGE MULTIPLIERS ---
# Multiplier for direct physical contact damage when slimes overlap
PROXIMITY_DAMAGE_SCALE = 1.8

# Multiplier for dissonance damage when slime phase vector mismatches local fluid aura
DISRUPTION_FORCE_SCALE = 15.0

# Max clamp limit for physical node vibration to prevent spring snap chain reactions
JITTER_FORCE_MAX_CLAMP = 14.0

# General scaling factor for the physical node displacement under stress
JITTER_FORCE_MULTIPLIER = 0.35

# --- SPRING ELASTICITY & RUPTURE LIMITS ---
# Minimum rest-state pixel distance limit before boundary springs snap
ELONGATION_LIMIT_BASE = 38.0

# Incremental spring stretch limit based on core quality metric
ELONGATION_LIMIT_STRETCH = 0.35

# Impact of Phasic Integrity on boundary shrinking (0.0 = completely rigid, 1.0 = highly dependent)
INTEGRITY_SHRINK_INFLUENCE = 0.3

# --- KURAMOTO PHASE DECOHERENCE (PHASE NOISE) ---
# Scaling factor for phase scrambling due to aura dissonance
SCRAMBLE_RATE_SCALE = 1.8

# Cumulative phase scrambling acceleration per broken spring node
SCRAMBLE_BROKEN_NODE_SCALE = 0.25

# Focused beam phase-noise scrambling multiplier (Player only, on high Gamma sync)
BEAM_SCRAMBLE_SCALE = 1.2

# --- REGENERATIVE PHASIC RECHARGE (STABILIZATION) ---
# Base Kuramoto coupling assist provided to bot actors to prevent self-destruction
K_BOT_ASSIST_BASE = 25.0

# Bot K assist scaling factor based on its core quality metric
K_BOT_ASSIST_QUALITY_SCALE = 20.0

# Bot K assist scaling factor based on its alignment similarity to local complex fluid
K_BOT_ASSIST_SIMILARITY_SCALE = 25.0

# --- DECENTRALIZED GRADIENT LOCOMOTION DRIFT RATES (SAFE CALIBRATED BALANCING) ---
# Velocity-to-phase gradient conversion factor for directional swimming (Keeps slimes stable during movement)
MOVEMENT_PHASE_DRIFT_SPEED = 0.15

# Torque-to-phase gradient conversion factor for circular current generation (Keeps slimes stable during spin)
TORQUE_PHASE_DRIFT_SPEED = 0.20

# --- SEMANTIC ROCK-PAPER-SCISSORS (RPS) MULTIPLIERS ---
# Primary component indices inside the 3D alchemical vector: 
# Index 0 = Yang (Fire / Red)
# Index 1 = SMR Catalyst (Qi / Earth / Wood / Green)
# Index 2 = Yin (Water / Blue)
# Rules of opposition: Yang (0) beats Yin (2), Yin (2) beats Catalyst (1), Catalyst (1) beats Yang (0)
RPS_ADVANTAGE_MULTIPLIER = 1.45
RPS_DISADVANTAGE_MULTIPLIER = 0.70

# --- RECOGNIZED CULTIVATOR BOT ARCHETYPES REGISTRY ---
# Fully isolated from execution engine code
BOT_ARCHETYPES = {
    "Pure Yang Core": [
        {"name": "Elder Pyro", "style": "Aggressive Yang", "desc": "Launches relentless fiery physical shockwaves."},
        {"name": "Solar Flare", "style": "Speed Yang", "desc": "High velocity Yang cultivator that charges rapidly."}
    ],
    "Deep Yin Core": [
        {"name": "Frost Weaver", "style": "Defensive Yin", "desc": "Uses defensive shielding to absorb and parry your energy."},
        {"name": "Lunar Shadow", "style": "Elusive Yin", "desc": "Keeps its distance while draining your domain."}
    ],
    "Foundation Pill": [
        {"name": "Balanced Disciple", "style": "Balanced Triad", "desc": "Uses a balanced blend of Yang, Yin, and SMR catalyst."},
        {"name": "Zen Initiate", "style": "Steady Triad", "desc": "Slow, steady, and extremely resilient to phase noise."}
    ],
    "Turbulent Anomaly": [
        {"name": "Chaos Hundun", "style": "Chaotic Warp", "desc": "Emits unpredictable high-entropy phase noise outbursts."},
        {"name": "Vortex Spinner", "style": "Vortex Spinner", "desc": "Spins violently, creating massive local fluid currents."}
    ]
}
