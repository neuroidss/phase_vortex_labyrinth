# 🌌 QUANTUM ALCHEMY REACTOR: The Neuro-Esports Phase Engine

**Quantum Alchemy Reactor (QAR)** is an experimental Brain-Computer Interface (BCI) esports framework and tactical auto-battler. It directly couples real-time human brain dynamics (via a 16-channel EEG micro-array *FreeEEG16-alpha2*) with a continuous, complex-valued 2D non-linear wave simulation evaluated on the GPU.

This project operates on the **Mathematical Universe Hypothesis** and the philosophical projections of **Sir Roger Penrose's "Three Worlds" diagram**:
*   **The Mental World** (comprising user attention, BCI states, or high-fidelity Gamepad axis manipulations) projects onto the underlying mathematical structures of the simulation.
*   **The Platonic World** (governed by mathematical truths, including the Nonlinear Schrödinger fluid fields, the Kuramoto Coupled Phase equations, and Feigenbaum's non-linear limits) processes these projections to organize the system.
*   **The Physical World** (realized through the high-speed GPU render viewport and the 44100Hz spatialized procedural audio buffer) represents, displays, and sonifies these emergent, dissipative structures in real-time.

---

## ⚡ THE HOLOGRAPHIC 26mm INTERFACE (FreeEEG16-alpha2)

Unlike standard consumer EEG headbands utilizing the International 10-20 standard (where electrodes are distributed decimeters apart), the **FreeEEG16-alpha2 is an ultra-high-density micro-array packing 16 channels onto a 26 mm footprint.**

### 🔬 Holographic Spatial Resolution
By restricting the spatial domain to 26 mm, the engine operates on a holographic principle. The macroscopic dynamics of the entire neocortex leave microscopic, continuous phase gradients ($\nabla\Phi$) across this localized 26 mm patch. 
The **Slime** in the game acts as your exact physical 26 mm probe. By steering its continuous phase matrices, the brain projects global kinetic intent and cognitive framing directly into the Navier-Stokes field.

### 🧮 The ciPLV Mathematical Mandate
Because the electrodes are positioned mere millimeters apart, they are subject to extreme volume conduction. The **Corrected Imaginary Phase Locking Value (ciPLV)** is physically and mathematically mandatory [1]. By isolating only the imaginary part of the cross-spectrum and normalizing it against the real component, the engine completely discards zero-lag volume conduction, exposing the genuine, non-zero-lag synaptic phase-locking occurring between localized cortical columns.
$$\text{ciPLV}_{i,j}(f) = \frac{\text{Im}(\langle e^{-i(\phi_i(f,t) - \phi_j(f,t))} \rangle)}{\sqrt{1 - \text{Re}(\langle e^{-i(\phi_i(f,t) - \phi_j(f,t))} \rangle)^2}}$$

---

## 📐 THE 120-JET CROSS-COHERENCE MANIFOLD (NO SINGLE-POINT REDUCTION)

To preserve the biological integrity of the viscoelastic simulation, **it is strictly forbidden to approximate slimes or characters as single geometric points (centers of mass) or simple disjoint nodes.** 

*   **120-Pairwise Manifold**: A 16-channel electrode micro-array yields exactly $\frac{16 \times 15}{2} = 120$ unique pairwise cross-coherence relationships.
*   **No Single-Point Approximations**: Resolving active-matter dynamics through single coordinates is prohibited. The simulation maps the complete, unreduced 120-dimensional spectro-spatial manifold—where each electrode pair possesses its own individual frequency spectrum of coherence—directly onto the Navier-Stokes velocity and density fields as 120 discrete physical "jets" of momentum.
*   **Preventing Internal Shear Stress**: Uniform projection across the entire 120-jet manifold ensures realistic, numerically stable solid-body translation and organic boundary interactions, preventing artificial local velocity gradients (`shear_stress`) from causing numerical self-destruction.

---

## 📐 THE STRICT LOCALITY MANDATE (NO ACTION-AT-A-DISTANCE)

To conform with continuous field physics, **all remote interactions are strictly localized.** The system completely rejects non-local operations such as instant teleportation, direct health/integrity injections across the grid, or direct coordinate pulling forces between remote units. All forces, healing properties, and stabilization parameters must propagate strictly via local fluid flow convection and node-to-grid field sampling.

*   **Advected Healing Mist (Resonance Plume)**: Healers generate a highly concentrated local Theta/Gamma spectral plume directly around their active nodes, introducing a gentle outward convective flow to disperse it. This mist is completely advected by the Navier-Stokes fluid currents across the grid. Friendly units must physically align and push the fluid flow to direct this healing mist over damaged frontline units to trigger their autopoietic regeneration.
*   **Physical Gravitational Vortices**: Tanks generate a localized low-pressure gravitational sinkhole centered at their coordinates on the GPU. This physically pulls nearby units and advects fluid density inward using a swirling spiral vortex, avoiding any non-physical coordinate-based teleportation.

---

## 🌊 VISCOELASTIC METAMATERIALS & NEGENTROPIC FRACTALS

QAR processes the complete spectro-spatial manifold natively. Instead of hardcoded triggers (`if frequency == 80: shoot()`), the engine features a completely emergent, multi-phase continuous Navier-Stokes solver. 

### A. Non-Newtonian Viscoelasticity (Theta / 4–8 Hz)
Theta band coherence acts as a structural binding agent. Where Theta is highly concentrated, the fluid on the GPU instantly transitions into a highly viscous, non-Newtonian gel [2]. This provides dynamic, physical energy shielding—protecting the Slime against high-frequency shocks and tearing.

### B. Bulk Thrust (Beta / 18–36 Hz) 
Phase-locking in the motor planning bands generates longer wavelengths that align with the 26mm Slime scale [2]. These large-scale phase gradients ($\nabla\Phi$) drive continuous **Madelung Quantum Currents**, acting as physical propellers that push the fluid and propel the slime ring via laminar advection.

### C. Negentropic Fractal Waveguides & CFC (Gamma / 60–100 Hz)
In standard fluid dynamics, violent mixing leads to chaotic entropy (heat). However, the simulated cortical fluid is an *active medium*. 
When a fast **Beta current** slams into a dense **Theta shield**, the boundary sharply folds, reducing entropy. This **Negentropic Fractal Boundary** acts as an acoustic lens: the kinetic energy is mechanically compressed until it undergoes **Cross-Frequency Conversion (CFC)**, detonating into a high-frequency **Gamma Soliton** (projectile) that rips through the field to deal cavitation damage [2].

---

## 🌀 TOPOLOGICAL CONTAINMENT & PHASIC RECOVERY (CLASSES & ROLES)

QAR translates the continuous physical wave-fluid model into a strategic **Managerial Gacha Auto-Battler** where human players and AI bots share the exact same physical constraints.

*   **Vanguard (Tank - High Theta):** Focuses on low-frequency Theta waves. It generates dense, highly viscous fluid fields that absorb kinetic shockwaves and physically slow down incoming attacks, blocking the enemy frontline.
*   **Assault (Fighter - High Beta):** Focuses on the Beta band. Generates rapid, laminar currents to rush the enemy frontline and deliver close-range kinetic advection.
*   **Artillery (Mage - High Gamma):** Focuses on high-frequency Gamma waves. Launches stable solitons that fly across the field and detonate upon hitting the enemy, causing phase-decoherence in the target's Kuramoto springs.
*   **Support (Healer - Phase-Lock Recovery):** Emits a highly cohesive Theta-band field. When overlapping with allies, its field physically **boosts their Kuramoto coupling constant $K$**, pulling their scattered phases back into alignment.

---

## 🎮 THE CONTINUOUS SPECTRUM BUILD GUIDE (TACTICAL SQUAD ASSEMBLY)

To prevent self-destruction and support advanced playstyles, every slime's internal coordinate framework is initialized with a **fully continuous power spectral density** ($1/f$ pink noise background floor) across all 100 frequency bins. This guarantees there are no absolute "zero-defense" holes in their physical node structures, allowing non-Tank units to withstand local fluid velocity gradients.

### 🧬 Class Spectral Distributions (Baseline $1/f$ + Oscillatory Peaks)
*   **Tank (Vanguard):** 
    *   *Theta (4-8Hz)*: **Heavy Cohesion** (`4.5` multiplier) distributed uniformly over all 16 nodes to absorb massive physical shear rates and protect the frontline.
    *   *Beta (18-36Hz)*: Moderate (`1.5`) for structural advection thrust.
    *   *Gamma (60-100Hz)*: Low (`0.3`).
*   **Fighter (Assault):** 
    *   *Beta (18-36Hz)*: **Heavy Propulsion** (`3.5` to `4.0` multiplier) to generate high-speed laminar currents.
    *   *Theta (4-8Hz)*: Moderate (`1.2` to `1.8`) to protect internal springs during rapid dashes.
    *   *Gamma (60-100Hz)*: Low-moderate (`0.5`).
*   **Mage (Artillery):** 
    *   *Gamma (60-100Hz)*: **Extreme Solitons** (`4.0` to `4.5` multiplier) to fire high-density ranged projectiles.
    *   *Theta (4-8Hz)*: Moderate (`1.5` to `1.8`) to shield front nodes from the shear rates generated by its own projectile launch.
    *   *Beta (18-36Hz)*: Low-moderate (`0.8`).
*   **Healer (Support):** 
    *   *Theta (4-8Hz)*: **Enhanced Coherence** (`3.0` multiplier) to generate stabilization sanctuaries.
    *   *Beta / Gamma*: Balanced (`1.2` to `1.5`) for flexible support dynamics.

### ⚡ Ultimate (Ult) Physical Grid Mechanics
When a slime's `'ult_charge'` reaches `1.0` (charged passively and dynamically by receiving or dealing shear-rate damage), it activates its Ultimate:
*   **Mage — "SUPERNOVA SOLITON":** Casts a massive, high-density Gamma soliton wave packet that travels across the Navier-Stokes grid, leaving a glowing trail, and detonates in a giant multi-directional fluid explosion on impact.
*   **Fighter — "HYPER-DRIVE BLITZ":** Gains 200% movement speed for 3.0 seconds, continuously ejecting high-velocity Beta thrust behind its nodes.
*   **Tank — "THETA CONFINEMENT":** Creates a gravitational sinkhole at its position for 3.0 seconds, physically pulling and dragging all enemy nodes into its viscous center.
*   **Healer — "RESONANCE SANCTUARY":** Instantly repairs 50% integrity of all alive allies and fully restores all their broken Kuramoto springs across the entire playfield.

---

## 📈 THE SUN TZU OPTION-BASED PREDICTOR

To support deep tactical play, the engine integrates a real-time **Option-on-Future Win Probability Predictor** modeled after financial option pricing theory (Black-Scholes-like probability decay).

$$d = \frac{S + \mu \cdot \tau}{\sigma \sqrt{\tau} + 1\text{e-}5}$$
$$P = \Phi(d) \approx \frac{1}{1 + e^{-1.5 \cdot d}}$$

*   **Spot Price ($S$):** The difference between the normalized structural integrity of the two teams: $S = Integrity_{T0} - Integrity_{T1}$.
*   **Time to Expiration ($\tau$):** The remaining duration of the combat contract (decaying towards the 30.0s time limit).
*   **Expected Drift ($\mu$):** The advantage score calculated from the spectral matchup. If Team 0's offensive Gamma penetrates Team 1's defensive Theta, the drift $\mu$ is positive.
*   **Volatility ($\sigma$):** The spatial phase-noise and shear rate of the fluid on the arena. Higher turbulence increases volatility, pulling the win probability $P$ towards $50\%$.
*   **Theta Decay:** As time to expiration ($\tau \to 0$) decreases, the uncertainty converges, forcing the Win Probability ($P$) to resolve decisively towards $100\%$ or $0\%$. Real-time metrics are exported to `battle_prediction_log.csv` for downstream model calibration.

---

## 🔬 PEER-REVIEWED SCIENTIFIC FOUNDATION

Every mechanic in the QAR engine is derived from established computational neuroscience and complex systems:

1.  **Corrected Imaginary Phase Locking Value (ciPLV)**: Used in the BCI spatial mapping to isolate true neocortical phase synchronization and eliminate volume conduction artifacts.
    > *Bruña, R., Maestú, F., & Pereda, E. (2018). Phase locking value revisited: teaching new tricks to an old dog. Journal of Neural Engineering.* **[DOI: 10.1088/1741-2552/aacfe4]**
2.  **Working Memory 2.0 (Phase Rhythms)**: The push-pull mechanics of superficial (Gamma) and deep (Alpha/Beta/Theta) cortical layers during volitional focus dictate the game's spectral control axes.
    > *Miller, E. K., Lundqvist, M., & Bastos, A. M. (2018). Working Memory 2.0. Neuron.* **[DOI: 10.1016/j.neuron.2018.09.023]**
3.  **The Kuramoto Model (Connectome Coherence)**: Slime structural health is modeled as a network of coupled phase oscillators.
    > *Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence. Springer.* **[DOI: 10.1007/978-3-642-69689-3]**

---

## 🛠 REAL-TIME DIAGNOSTIC TELEMETRY

QAR features a comprehensive, hardware-facing diagnostic HUD panel designed to verify connection parameters and data integrity during live test trials:
*   **EEG SPS Indicator:** Actively monitors the raw packet ingestion rate from all connected BLE/LSL workers.
*   **Exocortex Spectroscopy Panel:** Displays live mathematical readings, including node-level dissonance, coupling ($K$), and shear stress.
*   **Polar Phase Radar:** A real-time vector display mapping the 16 Kuramoto node oscillators as they rotate and disperse along the unit circle.

---
*Engineered for cognitive connectome mapping and competitive BCI Esports. AGPL v3 licensed.*
