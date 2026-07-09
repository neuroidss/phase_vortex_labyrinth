# 🌌 QUANTUM ALCHEMY REACTOR: The Neuro-Esports Phase Engine

**Quantum Alchemy Reactor (QAR)** is an experimental Brain-Computer Interface (BCI) esports framework. It directly couples real-time human brain dynamics (via a 16-channel EEG micro-array *FreeEEG16-alpha2*) with a continuous, complex-valued 2D quantum hydrodynamic simulation (Navier-Stokes equations) evaluated on the GPU.

This project operates on the **Mathematical Universe Hypothesis** and the philosophical projections of **Sir Roger Penrose's "Three Worlds" diagram**:
*   **The Mental World** (comprising user attention, BCI states, or high-fidelity Gamepad axis manipulations) projects onto the underlying mathematical structures of the simulation.
*   **The Platonic World** (governed by mathematical truths, including the Navier-Stokes fluid fields, the Kuramoto Coupled Phase equations, and Feigenbaum's non-linear limits) processes these projections to organize the system.
*   **The Physical World** (realized through the high-speed GPU render viewport and the 44100Hz spatialized procedural audio buffer) represents, displays, and sonifies these emergent, dissipative structures in real-time.

[gamepad vs bot](https://drive.google.com/file/d/1iruVLrGPEa8bQdZrL-5MD4MuQo3FO1xa/view)

[bot vs bot](https://drive.google.com/file/d/1zm8lGpOFRbi65zBRely4jKHQlue9Avv4/view)

---

## ⚡ ULTRA-HIGH-DENSITY MICROSCOPY: THE FreeEEG16-alpha2 HARDWARE

Unlike standard consumer EEG headbands or sparse clinical caps utilizing the International 10-20 standard (where electrodes are distributed decimeters apart across the entire skull), the **FreeEEG16-alpha2 is an ultra-high-density micro-array packing all 16 channels onto a miniature 26 mm diameter footprint.**

### 🔬 Pangolin-Class Spatial Resolution
This extreme density is equivalent to high-end scientific research grids (such as the *g.tec Pangolin*), shifting the acquisition paradigm from global macro-scale EEG to localized micro-state tracking. 
*   **Miniature Footprint**: The actual coordinate space (as defined in `implicit_config.py`) maps the 16 contacts within a circle of roughly 13 mm radius (26 mm diameter). 
*   **Sub-Centimeter Cortical Patches**: Instead of measuring cross-hemispheric communication, the array targets highly localized cortical regions, capturing micro-spatial phase gradients ($\nabla\Phi$) and local synchronization within a single cortical area.

### 🧮 The ciPLV Mathematical Mandate
Because the electrodes are positioned mere millimeters apart, they are subject to extreme volume conduction. The electrical potentials propagating through the scalp and skull are subject to massive spatial low-pass filtering, causing different channels to record identical, instantaneous potential changes from the same underlying source.
*   **Standard PLV Failure**: Using a standard Phase Locking Value (PLV) in this environment is mathematically invalid, as volume conduction creates false-positive zero-lag phase locking across the entire micro-array.
*   **The ciPLV Solution**: The **Corrected Imaginary Phase Locking Value (ciPLV)** is physically and mathematically mandatory for this layout [Bruña et al., 2018]. By isolating only the imaginary part of the cross-spectrum and normalizing it against the real component, the engine completely discards zero-lag volume conduction, exposing the genuine, non-zero-lag synaptic phase-locking occurring between localized cortical columns.

---

## 🔬 PEER-REVIEWED SCIENTIFIC FOUNDATION & DOIs

Every mechanic in the QAR engine is derived from established computational neuroscience, complex systems, and non-equilibrium thermodynamics:

1.  **Corrected Imaginary Phase Locking Value (ciPLV)**: Used in the BCI spatial mapping to isolate true neocortical phase synchronization and completely eliminate zero-lag volume conduction.
    > *Bruña, R., Maestú, F., & Pereda, E. (2018). Phase locking value revisited: teaching new tricks to an old dog. Journal of Neural Engineering.* **[DOI: 10.1088/1741-2552/aacfe4]**
2.  **Working Memory 2.0 (Phase Rhythms)**: The push-pull mechanics of superficial (Gamma) and deep (Alpha/Beta/Theta) cortical layers during volitional focus dictate the game's spectral and spatial control axes.
    > *Miller, E. K., Lundqvist, M., & Bastos, A. M. (2018). Working Memory 2.0. Neuron.* **[DOI: 10.1016/j.neuron.2018.09.023]**
3.  **The Kuramoto Model (Connectome Coherence)**: Slime structural health is modeled as a network of coupled phase oscillators.
    > *Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence. Springer.* **[DOI: 10.1007/978-3-642-69689-3]**
4.  **The Edge of Chaos & Period-Doubling Bifurcations**: The thermodynamic extraction of entropy in the Cauldron and the chaotic clash boundaries in the Arena are governed by Feigenbaum's scaling.
    > *Feigenbaum, M. J. (1978). Quantitative universality for a class of nonlinear transformations. Journal of Statistical Physics.* **[DOI: 10.1007/BF01020332]**
5.  **Penrose Three-World Ontology**: The epistemological foundation of the Mental-Platonic-Physical projection interface.
    > *Penrose, R. (1994). Shadows of the Mind: A Search for the Missing Science of Consciousness. Oxford University Press.* **[ISBN: 978-0198539780]**

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

## 🔄 THE ENDLESS ESPORTS LOOP: DRAFT $\to$ SMELT $\to$ CLASH

QAR operates as a highly competitive *Neuro-Action Roguelite* consisting of three interconnected phases:

### Phase 1: The Labyrinth (Strategic Gacha & Draft)
The stochastic maze generates a random set of raw, high-entropy *Cognits* (phase resonators) [vortex_maze.py]. Players must navigate and collect them [vortex_physics.py]. Because the spawn pool is random, players must adapt on the fly.
*   **Bypassing the Maze (Debug Skip)**: Developers can enable `self.SKIP_TO_CAULDRON = True` in `vortex_physics.py` to spawn directly below the Cauldron with all three alchemical entities pre-collected, speeding up physical calibration of the smelting and arena modules.

### Phase 2: The Cauldron (Bifurcation Smelting)
Ingredients dissolve into the $C^3$ complex-valued fluid [vortex_physics.py].
*   **Cross-Frequency Coupling**: Real-time non-linear interactions are solved inside `density_complex` to facilitate actual order generation. Qi (SMR/Green) acts as a Phase-Locking force pulling Yin and Yang into synchronization, while Yin (Theta) phase-modulates Yang (Gamma) amplitude (Phase-Amplitude Coupling).
*   **The Bifurcation Sieve**: To forge a *Divine Core*, players must hold the system precisely at the Edge of Chaos. The cauldron then acts as a dissipative pump, recursively separating phase noise (scaling via Feigenbaum's $\delta=4.669$) and blasting the high-entropy turbulence outward into the maze. Full circular padding prevents horizontal/vertical coordinate-axis locks.

### Phase 3: The Arena (Decentralized Domain Clash)
The forged Pill's vector is propped up as your resonant Core [vortex_physics.py]. You are teleported to a circular arena to face an endless ladder of Rogue Cultivators in a fully symmetric multi-actor framework [vortex_combat.py].

---

## 🧊🔥 FRACTALLY SEGMENTED CORE SLIMES (FREEZING FLAME)

QAR features a deep physical representation of multi-element hybrid bodies, supporting the legendary dual-polarity mechanics found in xianxia literature (such as clashing "freezing flame" nodes).

Instead of treating the cultivator's body as a uniform, mono-colored mass, **the 16 individual nodes of the softbody membrane are segmentally and fractally mapped to distinct element vectors.**
*   **Visual Wave Tracing**: As a mixed-core cultivator (such as the Yin-Yang hybrid `Glacial Phoenix`) moves and rotates under physical torques, its Fire nodes (Red) and Water nodes (Blue) trace separate, beautiful, clashing waves simultaneously into the fluid.
*   **Local Node-by-Node Stress**: Physical damage, dissonance, and Kuramoto phase scrambling are calculated individually for each node. pod-based alignment allows a spinning slime to strategically parry incoming waves by turning its matching elemental nodes toward the opponent, adding immense mechanical depth to esports matches.

---

## 🌊 ADVANCED PHASE PHYSICS: 3-PHASE CAHN-HILLIARD & INTERFERENCE

The combat arena relies entirely on non-equilibrium thermodynamics and fluid wave equations calculated on the GPU to drive the visuals and game-play loops without artificial script bottlenecks.

### 🧬 Symmetrical 3-Phase Cahn-Hilliard Separation
To prevent clashing elements from immediately blending into a homogeneous flat mud, the engine implements a masked, non-linear double-well phase separation potential $f(\phi)=\phi(\phi - 0.3)(1.0 - \phi)$ across Red, Green, and Blue channels.
*   **Vacuum Preservation**: If an element's density is below `0.12`, the self-sharpening term is bypassed. This allows newly emitted trails and distant wavefronts to diffuse and flow naturally.
*   **Interface Sharpening**: When opposing domains overlap (density $> 0.12$), the sharpening term triggers, forcing Fire, Water, and Grass to repel each other and form spectacular, razor-sharp Kelvin-Helmholtz vortices on the screen.

### 🔮 Propagating Wave Interference & $\nabla W_{RGB}$ Clash Damage
Elements are represented as real propagating physical waves in complex space:
$$W(t) = X \cos(\omega t) - Y \sin(\omega t)$$

The visual rendering displays absolute wave oscillations, mapping gorgeous high-contrast, self-similar interference stripes and hydrodynamic wave fractals in real-time.

*   **No Python Distance Loops**: Symmetrical collision damage and Kuramoto phase scrambling are driven entirely by the **GPU spatial gradient of the clashing wave-fields ($\nabla W_{RGB}$)**. If two wave fields overlap and clash violently, the intersecting gradients shoot up, causing physical vibration (jitter) and structural decay of the softbody.

---

## 🎮 THE ESPORTS INPUT BALANCE: GAMEPAD VS. 120-JET BCI

*   **The Gamepad (Gamepad Neuro-Assist)**: Maps trigger buttons to soft-body physics. **Left Trigger (L2) compresses** the gel, while **Right Trigger (R2) expands** it. Analog movements use stable physical translation formulas.
*   **The BCI (Organic Dominance)**: Real-time Multi-Frequency decomposition extracts Theta, SMR, and Gamma coherence matrices across the dense Pz micro-array. The player gains high-dimensional continuous control over all 16 boundary thrusters simultaneously.

---

## 🧪 AUTONOMOUS CHAMPIONSHIP TOURNAMENT TESTBED

To calibrate and test the balanced interactions of the physics engine without manually playing through the smelting loop, QAR contains a dedicated visual simulation harness:

```bash
python championship_tournament.py
```

### 🎯 Intransitive Rock-Paper-Scissors (RPS) Balance Cycle
The testbed features **4 distinct pure-element and hybrid matches** executing a flawless, closed balance loop driven entirely by Cahn-Hilliard gradients:

1.  **Yin/Water (Blue, Theta)** beats **Yang/Fire (Red, Gamma)**: Water filters and douses fire's high-frequency phase noise.
2.  **Yang/Fire (Red, Gamma)** beats **Catalyst/Grass (Green, SMR)**: Intense high-frequency thermal shockwaves incinerate and disrupt green catalyst structures.
3.  **Catalyst/Grass (Green, SMR)** beats **Yin/Water (Blue, Theta)**: SMR's high phase-coupling $K$ and absorbent grid structures bind and drain water's defensive shield.

This provides highly scenic, deterministic outcomes lasting exactly **15 to 35 seconds** per match.

---

## 🔊 ZERO-LATENCY PROCEDURAL SONIFICATION

QAR generates 100% procedural stereo audio with ~11ms latency to enable auditory-only navigation:
*   **Continuous Carrier Waves**: Amplitudes of Theta, SMR, and Gamma dictate the volume of binaural carrier frequencies [phase_vortex_labyrinth.py].
*   **Phase Modulation**: Spatial phase gradients modulate the pitch in real-time — you *hear* the phase shifts [phase_vortex_labyrinth.py].
*   **Kinetic Shear Noise**: Local velocity modulates wind-shear pink noise [phase_vortex_labyrinth.py].
*   **Visceral Combat Sensation**: Structural spring tears trigger sharp, high-frequency static pop/clicks, while successful phase energy absorption synthesizes beautiful, pure harmonic chimes [phase_vortex_labyrinth.py].

---

## 🛠 REAL-TIME SPECTROSCOPY PANEL (BATTLE DEBUG)

During combat, the expanded red panel provides a comprehensive physical diagnostic overview of the non-linear interaction:

```
[ CONNECTOME SPECTROSCOPY (DIAG) ]
----------------------------------
Your Core   : Deep Yin Core
Rogue Core  : Turbulent Anomaly (Frost Weaver)
Rogue Style : Defensive Yin
Coupling (K): 46.7 vs 54.2
Control Mode: FreeEEG16 Native
- - - - - - - - - - - - - - - - - 
Raw Axes    : [0.00, 0.00, -1.00, 0.00, 0.00, 0.00]
Raw Btns    : [2, 5]
Freq Output : -1.00 (Theta)
Spat Output : -1.00 (Shield)
- - - - - - - - - - - - - - - - - 
Your Dissonance: 0.24
Bot Dissonance : 0.81
Your Density   : 7.123
Bot Density    : 1.154
Your Jitter F  : 14.5 N
Bot Jitter F   : 142.3 N
Absorbed E     : 4.12 / 12.0
Clash Border   : 84.3% (Feigenbaum)
- - - - - - - - - - - - - - - - - 
Phase Integrity (P): 94.2%  [|||||||||.]
Phase Integrity (B): 32.1%  [|||.......]
```
*(The circular phase radar on the right visually maps the 16 nodes rotating and dispersing on the polar unit circle in real-time).*

---
*Engineered for cognitive connectome mapping and competitive BCI Esports. AGPL v3 licensed.*

