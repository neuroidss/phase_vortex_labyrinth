# 🌌 QUANTUM ALCHEMY REACTOR: The Neuro-Esports Phase Engine

**Quantum Alchemy Reactor (QAR)** is an experimental Brain-Computer Interface (BCI) esports framework. It directly couples real-time human brain dynamics (via a 16-channel EEG micro-array *FreeEEG16-alpha2*) with a continuous, complex-valued 2D quantum hydrodynamic simulation (Navier-Stokes equations) evaluated on the GPU.

This project operates on the **Mathematical Universe Hypothesis** and the philosophical projections of **Sir Roger Penrose's "Three Worlds" diagram**:
*   **The Mental World** (User attention, BCI states, or high-fidelity Gamepad axis manipulations) projects onto...
*   **The Platonic World** (Mathematical truths: Navier-Stokes fluid fields, the Kuramoto Coupled Phase equations, and Feigenbaum's non-linear limits) to manifest as dissipative, self-organizing structures in...
*   **The Physical World** (The high-speed GPU render viewport and the 44100Hz spatialized procedural audio buffer).

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
| **Jindan (Golden Core)** | **Hydrodynamic Soliton** | A self-preserving standing wave packet born at the Edge of Chaos ($\delta \approx 4.669$). |
| **Tribulation (Heavenly Wrath)**| **Entropy Backpressure** | The violent, turbulent recoil generated when purging phase noise from the core into the labyrinth. |
| **Domain (Aura)** | **Resonant Wave Emission** | Radial projection of your Pill's vector, imposing your physical rules onto the surrounding fluid. |

---

## 🔄 THE ENDLESS ESPORTS LOOP: DRAFT $\to$ SMELT $\to$ CLASH

QAR operates as a highly competitive *Neuro-Action Roguelite* consisting of three interconnected phases:

### Phase 1: The Labyrinth (Strategic Gacha & Draft)
The stochastic maze generates a random set of raw, high-entropy *Cognits* (phase resonators) [vortex_maze.py]. Players must navigate and collect them [vortex_physics.py]. Because the spawn pool is random, players must adapt on the fly: if they fail to gather a balanced triad, they must prepare to smelt a highly volatile unbalanced core, requiring immense BCI mental compensation [vortex_physics.py].

### Phase 2: The Cauldron (Bifurcation Smelting)
Ingredients dissolve into the $C^3$ complex-valued fluid [vortex_physics.py].
*   **No Hardcoded Recipes**: The resulting pill's identity is evaluated using **cosine similarity** between the cauldron's integrated phase state and the semantic database (`SEMANTIC_PILLS_DB`) [vortex_physics.py].
*   **The Bifurcation Sieve**: To forge a *Divine Core*, players must hold the system precisely at the Edge of Chaos [vortex_physics.py]. The cauldron then acts as a dissipative pump, recursively separating phase noise (scaling via Feigenbaum's $\delta = 4.669$) and blasting the high-entropy turbulence outward into the maze [vortex_physics.py].

### Phase 3: The Arena (Domain Clash Endless Ladder)
Upon passing the portal, the forged Pill's vector is propped up as your permanent resonant Core [vortex_physics.py]. You are teleported to a circular arena to face an endless ladder of Rogue Cultivators (Bots) whose difficulty scales infinitely [vortex_combat.py].

*   **Continuous Health (Kuramoto Order Parameter $H$)**: Binary HP is removed [vortex_combat.py]. Health is represented by **Phase Integrity** [vortex_combat.py, vortex_renderer.py]. External hostile waves scramble the phases of your 16 softbody nodes, driving $H \to 0$ [vortex_combat.py].
*   **Visceral Structural Damage**: Your spring stiffness is proportional to your Phase Integrity ($k \propto H$) [vortex_combat.py]. As integrity drops, your slime turns soft and is physically stretched by fluid shear [vortex_combat.py]. When a spring exceeds its tensile limit, it **snaps permanently**, and the severed node bleeds phase noise into your own domain [vortex_combat.py].
*   **Domain Rule Imposition**: The fluid is a battleground [vortex_combat.py]. In regions where your similarity exceeds the bot's ($S_p > S_b$), **you control the physical constants of the fluid** [vortex_combat.py]. If you are Yang, you accelerate whirlpools; if Yin, you freeze the fluid to paralyze the bot [vortex_combat.py].
*   **Active Domain Pulsing**: By aligning your physical triggers with your core frequency, you charge and detonate Domain Shockwaves, physically blasting the opponent's nodes apart [vortex_combat.py].

---

## 🎮 THE ESPORTS INPUT BALANCE: GAMEPAD VS. 120-JET BCI

*   **The Gamepad (Deterministic Precision)**: Uses trigger/bumper half-axes to morph the slime's shape (contracting into a stiff Core or expanding into a loose, receptive Shield) and shift frequencies manually [input_manager.py]. Gamepad inputs are highly responsive but constrained by discrete human motor limits (~150ms) [input_manager.py].
*   **The BCI (Organic Dominance)**: Brain state transitions occur in 20-40ms. When using the headset, the **120 individual cross-coherence pairs** among the 16 nodes are mapped to dynamic fluid micro-jets [vortex_combat.py]. This creates an organic, fractal "buffer" shield around the EEG player, allowing them to absorb shear stress that would immediately shatter a rigid gamepad player [vortex_combat.py].

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
Rogue Core  : Turbulent Anomaly
Coupling (K): 16.7 vs 14.2
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

