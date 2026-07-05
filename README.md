# 🌌 Sovereign Phase Vortex Labyrinth: Hydrodynamic BCI Esports Engine

An experimental, non-classical Brain-Computer Interface (BCI) esports engine built on the direct coupling of a 16-channel EEG array (**FreeEEG16** / LSL streams) with a continuous 2D Navier-Stokes fluid-particle simulation.

The engine completely discards traditional discrete button-based inputs, transitioning control entirely into **direct spatial perturbations of a continuous physical medium**. The controlled avatar (the "neuro-slime") and the surrounding labyrinth coexist within a seamless, non-Euclidean toroidal continuum.

[gameplay](https://drive.google.com/file/d/1x8jnGOfOT4AdYe6K6T-0jPwscZurOfV7/view?usp=sharing)

---

## 🧬 Core Architecture & The 120-Jet Engine

Unlike traditional BCI applications that reduce complex brain waves to a few simple commands (e.g., "Left", "Right", "Push"), this engine projects the entire topology of the brain's phase-synchronization into the physical world.

### 1. The 120-Jet Continuum (Maximum BCI Fidelity)
By default, the engine operates in the `120_jets` mode. Instead of just projecting the 16 physical electrodes as individual thrusters, the engine calculates the Phase Locking Value (ciPLV) for **every unique pair of electrodes** ($\frac{16 \times 15}{2} = 120$ connections).
*   Each of these 120 dynamic connections becomes an independent hydrodynamic nozzle (a jet) in the fluid solver, positioned exactly between the corresponding physical nodes of the slime.
*   The raw mental intention is injected directly into the Navier-Stokes grid as 120 independent vortices.
*   The slime does not have artificial "kinematic" speed applied to it. It moves **strictly because the 120 jets push the surrounding fluid, and the fluid carries the soft-body nodes**. 

### 2. The Slime Adhesion Continuum (Compression Axis)
The game does not use discrete "modes" or button toggles. Instead, it utilizes a continuous compression axis `[-1.0 ... 0.0 ... +1.0]` (mapped to analog triggers L2/R2, or the mouse/keyboard delta). This axis controls the micro-physics of the soft-body and its interaction with the environment:

*   **Expanded State (-1.0):** The Position-Based Dynamics (PBD) springs relax. The slime spreads out, becoming a flexible gel. Its 120 jets cover a wider area. In this state, the slime becomes "sticky" — it can seamlessly wrap around corners and crawl along walls without bouncing off, allowing for high-friction, tactical wall-crawling.
*   **Neutral State (0.0):** Balanced fluid movement. The internal springs hold the geometric shape of the FreeEEG16 array, but allow for organic deformation when squeezing through the labyrinth.
*   **Monolithic Compressed State (+1.0):** The 16 nodes tightly pack into a razor-sharp, unbreakable formation. The 120 jets align almost perfectly, merging their hydrodynamic forces into a single, high-velocity thruster. The slime's outer shell becomes highly repellent, allowing it to ricochet off walls and instantly break free from sticky corners.

### 3. Hybrid Predictive Camera & Ego-Centric Steering
To solve the "Catch-22" of soft-body rotation (where the camera waits for the slime to rotate, but the slime is anchored to the camera's template), the engine uses a **Predictive Kinematic Camera**:
*   **`coherence_relative_to_physical = True`**: The BCI thrust is always applied relative to the physical rotation of the slime itself. If you rotate the slime slightly in a hallway, "forward" immediately shifts to match the new nose-angle (tank-like steering).
*   The camera reads the user's rotational intent (`eeg_tq`) and proactively rotates the virtual ideal template. The PBD springs then physically *drag* the fluid nodes into the turn with immense torque, providing zero-latency, perfectly crisp steering even during heavy fluid deformation.

### 4. Pure Fluid Boundary Gliding
Traditional fluid engines suffer from the "dead water" problem (velocity is zero exactly at the wall boundary, trapping the player). We solved this by implementing a dynamic repulsive magnetic layer (`inner_wall_repulsion_scale`). It keeps the active jet nozzles exactly one millimeter away from the dead zone, ensuring that the 120 jets always inject force into "living" fluid, allowing smooth gliding along walls.

---

## 🏆 Esports Integrity & Tournament Standards

The engine is engineered specifically to meet the strict competitive standards required for international, professional tournament play:

### 1. Absolute Determinism (0% RNG)
The simulation loop is 100% deterministic. There are no random numbers used during gameplay. The trajectory of the slime, fluid shear patterns, and wall interactions are direct mathematical results of the athlete's phase-locking consistency and the laws of fluid dynamics. 

### 2. Artifact Immunity via Vectorized ciPLV
Because raw EEG signals can contain muscle artifacts (clenched jaw, blinking) that could be abused to generate inputs, the engine enforces strict biological verification. 
*   The parallel GPU spectrometer computes the **Corrected Imaginary Phase Locking Value (ciPLV)** for all 120 pairs in real-time.
*   This metric is mathematically immune to zero-lag synchronization, completely discarding physical currents spreading across the scalp (volume conduction/EMG leakage). Only genuine, non-zero-lag cortical phase couplings generate thrust in the fluid.

---

## 🎮 Gamepad & Neurogamepad Universal Input

For players training without an EEG headset, or for system calibration, the `input_manager.py` module translates manual inputs into the same bipolar difference space:
*   **Neurogamepad Mode:** Calculates a unified vector from the neuro-data or physical gamepad (WASD / Left Stick) and applies it to the fluid as a monolithic block, simulating the thrust of the `+1.0` compressed state without requiring full 120-jet EEG control.
*   **Mouselock & Triggers:** Entering ESCAPE locks the mouse cursor, enabling virtual relative mode. The horizontal mouse delta maps to continuous turning. Left Click / Space / R2 compresses the slime (+1.0), and Right Click / LShift / L2 expands it (-1.0).

---

## 🛠 Controls & Diagnostics

*   **L:** Toggle fluid velocity and tension vectors (hydrodynamic diagnostic mode).
*   **K:** Toggle 16-channel electrode sensors, 120-jet coherence bridges, and HUD markers.
*   **Escape:** Toggle mouse grab/release (Mouselock).
*   **LClick / Space / R2:** Compress slime (Monolithic state, high speed, high wall bounce).
*   **RClick / LShift / L2:** Decompress slime (Gel state, sticky wall-crawling).

---

## 📚 References & Scientific Grounding

1.  **Bruña, R., Maestú, F., & Pereda, E. (2018).** *Phase locking value revisited: teaching new tricks to an old dog.* Journal of Neural Engineering, 15(5), 056011.  
    **DOI:** [10.1088/1741-2552/aacfe4](https://doi.org/10.1088/1741-2552/aacfe4)
2.  **Miller, E. K., Lundqvist, M., & Bastos, A. M. (2018).** *Working Memory 2.0.* Neuron, 100(2), 463-475.  
    **DOI:** [10.1016/j.neuron.2018.09.023](https://doi.org/10.1016/j.neuron.2018.09.023)
3.  **Hawkins, J., Leadholm, N., & Clay, V. (2025).** *Hierarchy or Heterarchy? A Theory of Long-Range Connections for the Sensorimotor Brain.* arXiv preprint arXiv:2507.05888.  
    **arXiv Link:** [arXiv:2507.05888](https://arxiv.org/abs/2507.05888)


*Engineered for high-fidelity cortical dynamics mapping and competitive BCI sports. AGPL v3 licensed.*
```
