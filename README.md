# 🌌 QUANTUM ALCHEMY REACTOR: The Neuro-Esports Phase Engine

**Quantum Alchemy Reactor (QAR)** is an experimental Brain-Computer Interface (BCI) esports framework and tactical auto-battler. It directly couples real-time human brain dynamics (via a 16-channel EEG micro-array *FreeEEG16-alpha2*) with a continuous, complex-valued 2D non-linear wave simulation evaluated on the GPU.

This project operates on the **Mathematical Universe Hypothesis** and the philosophical projections of **Sir Roger Penrose's "Three Worlds" diagram**:
*   **The Mental World** (comprising user attention, BCI states, or high-fidelity Gamepad axis manipulations) projects onto the underlying mathematical structures of the simulation.
*   **The Platonic World** (governed by mathematical truths, including the Nonlinear Schrödinger fluid fields, the Kuramoto Coupled Phase equations, and Feigenbaum's non-linear limits) processes these projections to organize the system.
*   **The Physical World** (realized through the high-speed GPU render viewport and the 44100Hz spatialized procedural audio buffer) represents, displays, and sonifies these emergent, dissipative structures in real-time.

[gamepad vs bot](https://drive.google.com/file/d/1iruVLrGPEa8bQdZrL-5MD4MuQo3FO1xa/view)

[bot vs bot](https://drive.google.com/file/d/1zm8lGpOFRbi65zBRely4jKHQlue9Avv4/view)

---

## ⚡ HIGH-DENSITY MICROSCOPY: THE FreeEEG16-alpha2 HARDWARE

Unlike standard consumer EEG headbands or sparse clinical caps utilizing the International 10-20 standard (where electrodes are distributed decimeters apart across the entire skull), the **FreeEEG16-alpha2 is an ultra-high-density micro-array packing all 16 channels onto a miniature 26 mm diameter footprint.**

### 🔬 Spatial Resolution Shift
This extreme density equivalent to high-end scientific research grids (such as the *g.tec Pangolin*) shifts the acquisition paradigm from global macro-scale EEG to localized micro-state tracking. 
*   **Miniature Footprint**: The actual coordinate space (as defined in `implicit_config.py`) maps the 16 contacts within a circle of roughly 13 mm radius (26 mm diameter). 
*   **Sub-Centimeter Cortical Patches**: Instead of measuring cross-hemispheric communication, the array targets highly localized cortical regions, capturing micro-spatial phase gradients ($\nabla\Phi$) and local synchronization within a single cortical area.

### 🧮 The ciPLV Mathematical Mandate
Because the electrodes are positioned mere millimeters apart, they are subject to extreme volume conduction. The electrical potentials propagating through the scalp and skull are subject to massive spatial low-pass filtering, causing different channels to record identical, instantaneous potential changes from the same underlying source.
*   **Standard PLV Failure**: Using a standard Phase Locking Value (PLV) in this environment is mathematically invalid, as volume conduction creates false-positive zero-lag phase locking across the entire micro-array.
*   **The ciPLV Solution**: The **Corrected Imaginary Phase Locking Value (ciPLV)** is physically and mathematically mandatory for this layout [1]. By isolating only the imaginary part of the cross-spectrum and normalizing it against the real component, the engine completely discards zero-lag volume conduction, exposing the genuine, non-zero-lag synaptic phase-locking occurring between localized cortical columns.

---

## 🌊 SPECTRUM DISPERSION ENGINE (ZERO DIMENSIONALITY REDUCTION)

QAR processes the complete spectro-spatial manifold of the brain natively. Instead of averaging the 60 frequency bins of the incoming `ciPLV` matrix down to a few rigid action axes, the engine passes the raw, unadulterated `[16, 16, 60]` tensor directly into the physical equations of motion.

### 🧬 Wave Dispersion Physics & Emergent Mechanics
Instead of hardcoding triggers like `if (frequency == 80) spawn_projectile()`, the physical behaviors of different classes and elements emerge purely from the continuous wave-fluid coupling.

#### A. Linear Translation / Bulk Thrust (Beta / 18–36 Hz) $\rightarrow$ Movement
Phase-locking in the motor planning bands (Beta) generates longer wavelengths. In the Madelung hydrodynamic representation, these smooth, large-scale phase gradients ($\nabla\Phi$) naturally drive a continuous **Quantum Current** ($\vec{J}$) that accelerates the Navier-Stokes velocity fields:
$$\vec{F}_{\text{fluid}} = \sum_c \gamma_c \text{Im}\left(\psi_c^* \nabla \psi_c\right) = \sum_c \gamma_c \left(R_c \nabla I_c - I_c \nabla R_c\right)$$
This current drives laminar fluid flows, acting like a physical propeller that pushes the fluid and propels the slime ring forward across the canvas.

#### B. Radial Cohesion / Protective Pulse (Theta / 4–8 Hz) $\rightarrow$ Shields
Phase-locking in the deep Theta carrier band generates radial gravity forces. This pulls the 16 nodes inward toward the center of mass, packing the gel and forming a highly stable, dense core resilient to external shear.

#### C. Vorticity / Localized Shear (Gamma / 60–100 Hz) $\rightarrow$ Projectiles (Solitons)
At high frequencies (Gamma), the wavelengths are very small and highly localized. Under a strongly attractive self-interaction potential ($g_{c,c} < 0$) in the non-linear Schrödinger coupling:
$$V_{\text{nl}, c}(x, y) = \sum_{c'} g_{c, c'} |\psi_{c'}(x, y)|^2$$
the high-frequency wave-packets overcome linear dispersion and collapse into stable, self-focusing, non-dispersing wave packets — **Solitons** [3]. These solitons travel across the fluid as physical, high-energy projectiles, completely decoupled from the bulk flow.

#### D. Cross-Phase Modulation (XPM) & Spectral Morphing
When a 80 Hz projectile-soliton flies through a 20 Hz Beta current, they do not pass through each other linearly. Because the non-linear potentials are coupled ($g_{c, c'} > 0$), the presence of the 20 Hz density modulates the phase of the 80 Hz wave. This **Cross-Phase Modulation** procedurally generates sidebands in flight [3]:
$$f_{\text{new}} = f_1 \pm n \cdot f_2 \implies [60\text{ Hz}, 100\text{ Hz}, 40\text{ Hz}]$$
The projectile physically morphs its spectral envelope during its trajectory, hitting the target at its exact resonant frequencies.

---

## 🌀 TOPOLOGICAL CONTAINMENT & PHASIC RECOVERY (CLASSES & ROLES)

QAR translates the continuous physical wave-fluid model into a strategic **Managerial Gacha Auto-Battler** campaign where units are organized into four unique roles:

*   **Vanguard (Tank - High Theta):** Focuses on low-frequency Theta waves. It generates dense, highly viscous fluid fields that absorb kinetic shockwaves and physically slow down incoming attacks, blocking the enemy frontline.
*   **Assault (Fighter - High Beta):** Focuses on the Beta band. Generates rapid, laminar currents to rush the enemy frontline and deliver close-range kinetic damage.
*   **Artillery (Mage - High Gamma):** Focuses on high-frequency Gamma waves. Stands completely still to prevent fluid turbulence, launching stable solitons that fly across the field and detonate upon hitting the enemy's resonant frequency.
*   **Support (Healer - Phase-Lock Recovery):** Emits a highly cohesive Theta-band field. When overlapping with allies, its field physically **boosts their Kuramoto coupling constant $K$**, pulling their scattered phases back into alignment [3]. This restores their health pool (`integrity`) over time and re-welds broken spring connectors.
*   **Topological Dual-Spin Solitons:** Alchemical entities represent stable, non-dissipating vortices. To prevent them from immediately dissolving, the engine implements a counter-rotating structure: an inner Yang Core (inner spin) and an outer Yin Ring (outer spin) that integrate to zero net angular momentum, maintaining stability.

---

## 🎮 GACHA HUB & PERSISTENT PROGRESSION

*   **The Cauldron Hub (`gacha_hub.py`):** The main entry point where players manage their roster, pull new units, and allocate resources.
*   **Persistent Saving:** Local profile data is safely stored in `gacha_save.json`, tracking currency (*Quantum Prisms*), unlocked characters, and campaign level.
*   **Upgrade & Refund System:** Spend currency to upgrade a unit's level (increasing its `quality` and Kuramoto $K$ stability). Leveling is 100% refundable without penalty, allowing players to freely reallocate resources and adapt to difficult stages.
*   **Campaign Stages:** Battle through a linear chain of campaign levels. Every 5 levels features a **Mini-boss** (reinforced Vanguard), and every 10 levels features a **Boss** (Tank, Mage, and Healer). Beating a boss grants a massive reward of +1000 Prisms.
*   **Exocortex Cauldron (BCI Smelting):** BCI players can enter the Cauldron to smelt custom, elite **Jindan (Golden Core)** units. The unit's baseline stats, class role, and Kuramoto phase stability ($K$) are determined by the average `ciPLV` phase coherence achieved during the smelting.

---

## 🔬 PEER-REVIEWED SCIENTIFIC FOUNDATION

Every mechanic in the QAR engine is derived from established computational neuroscience, complex systems, and non-equilibrium thermodynamics:

1.  **Corrected Imaginary Phase Locking Value (ciPLV)**: Used in the BCI spatial mapping to isolate true neocortical phase synchronization and completely eliminate zero-lag volume conduction.
    > *Bruña, R., Maestú, F., & Pereda, E. (2018). Phase locking value revisited: teaching new tricks to an old dog. Journal of Neural Engineering.* **[DOI: 10.1088/1741-2552/aacfe4]**
2.  **Working Memory 2.0 (Phase Rhythms)**: The push-pull mechanics of superficial (Gamma) and deep (Alpha/Beta/Theta) cortical layers during volitional focus dictate the game's spectral and spatial control axes [2].
    > *Miller, E. K., Lundqvist, M., & Bastos, A. M. (2018). Working Memory 2.0. Neuron.* **[DOI: 10.1016/j.neuron.2018.09.023]**
3.  **The Kuramoto Model (Connectome Coherence)**: Slime structural health is modeled as a network of coupled phase oscillators.
    > *Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence. Springer.* **[DOI: 10.1007/978-3-642-69689-3]**
4.  **The Edge of Chaos & Period-Doubling Bifurcations**: The thermodynamic extraction of entropy in the Cauldron and the chaotic clash boundaries in the Arena are governed by Feigenbaum's scaling.
    > *Feigenbaum, M. J. (1978). Quantitative universality for a class of nonlinear transformations. Journal of Statistical Physics.* **[DOI: 10.1007/BF01020332]**

---

## 📜 LORE AS COGNITIVE PHYSICS: THE ALCHEMICAL DICTIONARY

The world of QAR has suffered a *Decoherence Event*, shattering semantic reality into high-entropy phase noise (*Hundun*). The player acts as a *Cultivator* (Phase-locked operator) using internal mental coherence to forge islands of low-entropy meaning (*Pills*) and defend them on the combat Arena.

| Xianxia Concept | Physics/Signal Equivalent | Role in the Reactor Engine |
| :--- | :--- | :--- |
| **Jing (Essence / Цзин)** | **Yin-Water (6Hz Theta)** | Low-frequency structural container. Slows kinetic chaos and stabilizes the boundary. |
| **Qi (Energy / Ци)** | **SMR Catalyst (14Hz)** | The sensorimotor bridge. Synchronizes and integrates opposite polarities into a crystal lattice. |
| **Shen (Spirit / Шэнь)** | **Yang-Fire (80Hz Gamma)** | High-frequency informational activation. Generates torque and thermal kinetic energy. |
| **Xin Mo (Inner Demon)** | **Phase Noise & Artifacts** | Signal desynchronization (scramble) degrading the Kuramoto Order Parameter. |
| **Jindan (Golden Core)** | **Hydrodynamic Soliton** | A self-preserving standing wave packet born at the Edge of Chaos ($\delta\approx 4.669$). |
| **Tribulation (Heavenly Wrath)**| **Entropy Backpressure** | The violent, turbulent recoil generated when purging phase noise from the core into the labyrinth. |
| **Domain (Aura)** | **Resonant Wave Emission** | Radial projection of your Pill's vector, imposing your physical rules onto the surrounding fluid. |

---

## 🛠 REAL-TIME DIAGNOSTIC TELEMETRY

QAR features a comprehensive, hardware-facing diagnostic HUD panel designed to verify connection parameters and data integrity during live test trials:
*   **EEG SPS Indicator:** Actively monitors the raw packet ingestion rate from all connected BLE/LSL workers in real-time, providing immediate feedback up to a maximum rate of 250 SPS.
*   **Exocortex Spectroscopy Panel:** Displays live mathematical readings, including node-level dissonance, coupling ($K$), and shear stress.
*   **Polar Phase Radar:** A real-time vector display mapping the 16 Kuramoto node oscillators as they rotate and disperse along the unit circle.

---
*Engineered for cognitive connectome mapping and competitive BCI Esports. AGPL v3 licensed.*

