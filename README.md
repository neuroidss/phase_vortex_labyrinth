# 🌌 Sovereign Phase Vortex Labyrinth: Hydrodynamic BCI Esports Engine

An experimental, non-classical Brain-Computer Interface (BCI) esports engine built on the direct coupling of a 16-channel EEG array (**FreeEEG16** / LSL streams) with a continuous 2D Navier-Stokes fluid-particle simulation.

The engine completely discards traditional discrete button-based inputs, transitioning control entirely into **direct spatial perturbations of a continuous physical medium**. The controlled avatar (slime) and the surrounding labyrinth coexist within a seamless, non-Euclidean toroidal continuum.

[gameplay](https://drive.google.com/file/d/1x8jnGOfOT4AdYe6K6T-0jPwscZurOfV7/view?usp=sharing)

---

## 🧬 Core Architecture & Physical Principles

The engine is built on a coupled **Eulerian-Lagrangian Solver**, where the fluid is simulated on a grid of `128x128` (Eulerian), and the 16 electrodes from the occipito-parietal array are projected into the world as 16 dynamic marker particles of the slime's soft body (Lagrangian).

### 1. Incompressible Navier-Stokes Grid (Toroidal Universe)
The environment is a continuous `128x128` grid solving the incompressible Navier-Stokes equations for fluid velocity $(u, v)$ and density. 
*   **Toroidal Topology:** The pressure and velocity boundary conditions in the Navier-Stokes solver are configured in a periodic, toroidal mode (`mode='circular'`). Fluid flow exiting the right boundary instantly and without loss of momentum enters the left boundary.
*   **Unconstrained Particle Coordinates:** The absolute coordinates of the slime (`pin_pos`) are calculated in an unconstrained, infinite space (allowing coordinates to grow endlessly, e.g., $x = 15000$). This prevents abrupt camera jumps during wrapping.
*   **Periodic Mapping & Tiling:** When sampling local fluid velocities, the coordinates of the slime are folded back into the grid boundaries using the modulo operation:
```math
\vec{u}_{\text{sampled}} = \text{sample\_fluid}(\vec{x}_{\text{slime}} \pmod{\vec{w}})
```
    This results in a seamless, infinitely tileable representation of the physical world.

### 2. Holographic BCI Projection (Frequency-to-Physics Mapping)
The engine does not impose artificial frequency band boundaries (like hardcoded Alpha or Beta filters), accommodating the unique neural baseline of each individual athlete:
*   **Parallel STFT Spectrometer:** Every 4 ms, the engine computes a Short-Time Fourier Transform (STFT) with a Hann window, extracting the entire spectrum from 3 Hz to 100 Hz (with a resolution of ~1.95 Hz, yielding approximately 50 frequency bins) and strictly notch-filtering out the 50 Hz power-line noise.
*   **Vectorized ciPLV Analyzer:** The full spectral coherence tensor $[16, 16, 50]$ is computed in parallel on the GPU using vectorized batch matrix multiplication of normalized phase vectors:
    $$PLV(f) = \frac{Z(f) \cdot Z^H(f)}{T}$$
    This algebraic method (adapted from *Bruña et al., 2018*) avoids expensive trigonometric calculations on the CPU, resolving the entire spectrum of phase couplings on an NVIDIA RTX 3060 in under $0.5$ ms.
*   **Spatial Radius-Frequency Duality ($R \propto 1/f$):**
    Each frequency bin $f$ is projected into the fluid solver with its own mathematically mapped spatial footprint ($R$):
    *   **Low Frequencies (Theta/Alpha, 4-12 Hz) - *The Past / Memory*:** Large spatial footprint. Injecting force as a broad, laminar, cohesive wave. This maintains the structural shape and global volume of the slime.
    *   **Low Gamma Frequencies (30-50 Hz) - *Active Working Memory*:** Moderate spatial footprint. Handles local shape maintenance and active navigation.
    *   **High Gamma Frequencies (50-100 Hz) - *The Future / Intention*:** Ultra-small spatial footprint. Force is injected as a razor-sharp, pixel-perfect jet. This concentrated energy creates extreme local shear stress (vorticity), which is physically required to erode walls and pierce barriers.

---

## 🏆 Esports Integrity & Tournament Standards

The engine is engineered specifically to meet the strict competitive standards required for international, professional tournament play (BCI Esports):

### 1. Absolute Determinism (0% RNG)
The simulation loop is 100% deterministic and runs entirely on the GPU. There are no random numbers used during gameplay. The trajectory of the slime, fluid shear patterns, and wall erosion are direct mathematical results of the athlete's phase-locking consistency and the laws of fluid dynamics. Mental errors (such as phase desynchronization) are immediately punished by a loss of speed and momentum, analogous to a missed mechanical click in a traditional FPS shooter.

### 2. Seeded Time-Attack Labyrinths
The maze generator (`vortex_maze.py`) supports strict pseudorandom seeding. During tournaments, athletes compete on identical track layouts, allowing them to memorize paths and optimize trajectories.
*   **High-Skill Corner Cutting (Tactical Erosion):** Labyrinth walls are highly rigid and self-healing (`recovery = 1.8`). Slime particles repel elastically from them like a solid rubber bumper (`f_wall = 1500.0`). Cutting a corner through a wall is a high-risk, high-skill maneuver requiring continuous, high-amplitude focus in the High Gamma band for several seconds. If focus wavers for even a millisecond, the wall heals instantly, trapping the slime.

### 3. Telemetry Logging & Anti-Cheat
Because raw EEG signals can contain muscle artifacts (clenched jaw, blinking) that could be abused to generate inputs, the engine enforces strict biological verification:
*   **EMG Artifact Immunity:** The `ciPLV` metric is mathematically immune to zero-lag synchronization, completely discarding physical currents spreading across the scalp (volume conduction/source leakage).
*   **Deterministic Replay Logger:** The engine supports real-time logging of the raw EEG data streams and phase matrices. This allows tournament referees to replay any match, verifying the spectral purity of the inputs and ensuring that victory was achieved purely through authentic cortical modulation.

### 4. Spectator Readability
The mental athleticism of the players is visualized directly on-screen for live audiences and commentators:
*   Fluid velocity and tension vectors displays the physical force fields.
*   The active 16-channel electrode grid (toggle with **K**) renders the real-time ciPLV graph directly on top of the slime, showing which brain areas are cohering to generate momentum.

---

## 🎮 Gamepad-ified Universal Input Layer

For players training without an EEG headset, the separate `input_manager.py` module translates manual inputs into the same bipolar "coherence-like" difference space:
*   **Keyboard (WASD / Space-Shift):** No independent, unipolar half-axes. Inputs are processed through pairwise button subtraction: `Right - Left` (axis X), `Down - Up` (axis Y), and `Space (Compress) - LShift (Decompress)`.
*   **Mouse-Look (Mouselock):** Entering ESCAPE locks the mouse cursor, enabling virtual relative mode. The horizontal movement delta `dx` is mapped as a continuous turning axis `eeg_tq`, mirroring a gamepad stick. Left click compresses the slime, and Right click expands it.
*   **Gamepad Auto-Calibration:** The module automatically calibrates the trigger axes (L2/R2) on start to handle OS-specific axis variations, mapping them to a clean, balanced difference axis `[-1.0, 1.0]` for slime compression/expansion.

---

## 🛠 Controls & Hotkeys

*   **L:** Toggle fluid velocity and tension vectors (spectral diagnostic mode).
*   **K:** Toggle 16-channel electrode sensors and HUD markers.
*   **Escape:** Toggle mouse grab/release (Mouselock).
*   **LClick / Space / R2:** Compress slime (increase density, squeeze through tight passages).
*   **RClick / LShift / L2:** Decompress slime (expand shape, anchor to walls).

---

## 📚 References & Scientific Grounding

1.  **Bruña, R., Maestú, F., & Pereda, E. (2018).** *Phase locking value revisited: teaching new tricks to an old dog.* Journal of Neural Engineering, 15(5), 056011.  
    **DOI:** [10.1088/1741-2552/aacfe4](https://doi.org/10.1088/1741-2552/aacfe4)
2.  **Miller, E. K., Lundqvist, M., & Bastos, A. M. (2018).** *Working Memory 2.0.* Neuron, 100(2), 463-475.  
    **DOI:** [10.1016/j.neuron.2018.09.023](https://doi.org/10.1016/j.neuron.2018.09.023)
3.  **Hawkins, J., Leadholm, N., & Clay, V. (2025).** *Hierarchy or Heterarchy? A Theory of Long-Range Connections for the Sensorimotor Brain.* arXiv preprint arXiv:2507.05888.  
    **arXiv Link:** [arXiv:2507.05888](https://arxiv.org/abs/2507.05888)

*Engineered for high-fidelity cortical dynamics mapping and competitive BCI sports. AGPL v3 licensed.*

