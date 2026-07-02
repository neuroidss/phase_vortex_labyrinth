# 🌌 Sovereign Phase Vortex Labyrinth: Hydrodynamic BCI Paradigm

An experimental, non-classical Brain-Computer Interface (BCI) sandbox designed for the 16-channel, ultra-high-density **FreeEEG16** dry electrode array. 

The system completely abolishes the traditional dualism of "rigid hitbox navigation", moving instead to a **Unified Hydrodynamic Medium**. Both the controlled avatar (slime) and the surrounding labyrinth are resolved within the exact same continuous physical grid, governed by real-time incompressible fluid dynamics.

---

## 🧬 Core Architecture & Physical Principles

The engine is built on a coupled **Eulerian-Lagrangian Solver**, where the fluid is simulated on a grid (Eulerian), and the brain's coherence points act as dynamic marker particles (Lagrangian) that are strictly bound to the flow.

### 1. Incompressible Navier-Stokes Grid (Stable Fluids)
The environment is a continuous `128x128` grid solving the incompressible Navier-Stokes equations for fluid velocity $(u, v)$ and density. 
*   **The Void:** The space outside the maze is modeled as a free hydrodynamic void (zero wall density), allowing fluid to flow out of bounds if the boundaries of the labyrinth are breached.
*   **Incompressible Obstacles:** The walls of the labyrinth are regions of high `wall_density`. The mathematical pressure projection step (`project`) ensures that any velocity normal to a solid wall is cancelled, forcing the fluid to flow strictly parallel to the obstacles.

### 2. Immersed Boundary Tracer Particles (The 16 Coherences)
The 16 dry electrode channels from the parietal array (Pz) are represented as **16 independent, dynamic tracer particles** (`pin_pos`).
*   **Passive Advection:** Unlike classical game avatars, these particles have no direct, coordinate-modifying motor engines. They move *strictly* by advecting along the local projected fluid velocity field: 
    $$\vec{v}_{pin} = \vec{v}_{fluid} + \vec{v}_{wall\_repulsion}$$
*   **Zero-Tunneling Wall Repulsion:** To prevent sub-pixel numerical drift from leaking into walls over time, a stable repulsion vector is calculated from the spatial gradient of the `wall_density` field, gently pushing particles away from walls. Combined with a strict speed limit ($150\text{ px/s}$), particles are physically incapable of phasing through solid walls.

### 3. Unified Two-Way Fluid Coupling (Shape Cohesion)
The elastic "surface tension" of the slime is resolved entirely through the fluid medium, rather than hardcoded coordinate overrides:
*   **Elastic Force Injection:** The shape-restoring spring forces pulling the particles back into their ideal 16-channel formation are calculated for each node.
*   **Fluid Propulsion:** Instead of moving the particle coordinates, these spring forces are injected directly into the fluid velocity grid as localized momentum forces (`u` and `v`). 
*   **Emergent Deformation:** The fluid flows towards the ideal shape, dragging the passive tracer particles with it. If a wall blocks the path, the fluid (and thus the particles) naturally squishes against it, sliding beautifully along the corridor.

### 4. Deformable Terrain (Erosion)
Walls are not magically privileged static blocks; they are high-viscosity fields subject to physical **shear stress erosion**.
*   When the fluid is pushed against walls with immense force (via focused EEG or manual inputs), the localized velocity buildup causes the wall to *erode* and melt on-screen.
*   As the wall density dissolves, the fluid (carrying the slime particles) can physically flow through the newly opened breach, creating dynamic, destructible pathways.

---

## 🔮 The Event Horizon: Localized Magical Portal

At the exit of the labyrinth lies a highly localized, passive vortex sink (a magical portal designed by the high technology of the Sages).

```
                      [Active Slime approaches Portal]
                                     │
                      [First node enters Event Horizon]
                      (dist < 0.25 of Labyrinth Cell)
                                     │
                                     ▼
                      [Node is captured and locked]
                     (Snaps to center, turns Purple)
                                     │
                                     ▼
                    [Global Center of Mass (CoM) Shifts]
                 (Calculated over all 16 active & captured nodes)
                                     │
                                     ▼
                    [Cohesion Current drags next node]
                (Spring pulls fluid, fluid drags remaining nodes)
                                     │
                                     ▼
                    [Inescapable Swallowing Cascade]
```

### Emergent Swallowing Mechanics
The portal has no long-range "cheat" gravity that pulls things through walls. It is a purely local touch trigger. 
*   **The Chain Reaction:** Once a single node of the slime touches the portal, it is captured and locked at the center of the vortex.
*   **The Pull of the Mass:** Because the global Center of Mass (`com`) is calculated across **all 16 nodes** (including captured ones), capturing a node immediately shifts the `com` towards the portal.
*   **The Cascade:** This shift pulls the ideal target positions of the remaining active nodes closer, injecting a strong physical current into the fluid. The fluid flows towards the portal, naturally dragging the next nearest nodes into the horizon. 
*   **Tactical Navigation:** The slime is swallowed piece-by-piece. If some nodes are stuck behind a wall corner, the fluid flow will be blocked, and the swallowing will halt. You must actively steer the remaining free nodes around the corner to let the portal finish inhaling the rest of the body.

---

## 🎮 Game Loop & Controls

*   **L:** Toggle Fluid Velocity/Tension Lines (Hidden by default for maximum immersion).
*   **K:** Toggle Electrode Sensors & UI Markers (Hidden by default).
*   **Space:** Compress the True Qi (Modulate slime scale/node radius).
*   **WASD / Arrow Keys / Joystick:** Manual override to inject directional force vectors into the fluid.

*Designed for high-fidelity mesoscopic cortical dynamics mapping. AGPL v3 licensed.*

