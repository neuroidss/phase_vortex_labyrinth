# AGENTS.md

# Neurofeedback Guidelines
- **CRITICAL**: FORBIDDEN to add any internal delay, artificial buffering, inertia, camera smoothing, or velocity interpolation (`lerp()`) to user inputs when implementing Neurofeedback or control features.
- Neurofeedback MUST have 0 milliseconds of artificial delay.
- The interface must react instantaneously to the raw thoughts/inputs. If there is noise, let the noise show, but never obscure the user's control with game-engine smoothing or camera inertia. 

# Interaction Constraints
- Do not artificially manipulate elements when the user has input explicitly mapped. "Пилюля" (the pill/ball) must not pulse or move on its own with a sine wave (`Math.sin(time)`) if the user is supposed to control it.
- If the mind is idle and outputting `0`, the element MUST BE STILL.
- DO NOT ADD Game-like visual flair like automatic rotating cameras, wobbling animations, or sin waves on controlled objects that ruin the perception of neurofeedback.

# NO AI AUTOPILOT / NO VECTOR BLENDING
- **STRICTLY FORBIDDEN**: Under any circumstances, do NOT implement AI autopilot, automated steering, path nudging, or blending of shortest-path vectors with user intents.
- The AI must act as a **purely passive observer and state diagnostics layer**. It only measures and registers when the human's own subconscious/parallel mind takes autonomous biological control (represented by `mind_nav_auto` state).
- The player must have 100% direct neurological or manual control of the physical translation at all times.

# FILE OUTPUT CONSTRAINTS
- **CRITICAL / MANDATORY**: NEVER output, print or display files that you did *not* modify during the current turn.
- ONLY output the complete code of the files that you are **actively changing right now** in this specific turn. Do not output unmodified helper files, templates, or configuration files under any circumstances.

# General AI Rules
- Do NOT ignore the user's request for zero latency or neurofeedback principles.
- Use direct physical bounds clamping for arenas (walls) to prevent avatar from falling out of bounds, rather than relying strictly on impulses or teleporting only when they've fallen infinitely.

# Axis & Sign Constraints
- **CRITICAL**: NEVER REMOVE THE SIGN from axes (e.g., do not use `Math.abs` to create half-axes where 0..1 is magnitude). It is **FORBIDDEN** to create half-axes, because half-axes destroy the phase/polarity mappings needed for coherence, which relies on full bipolar (-1 to 1) vectors. Always preserve the original sign so the physics and audio mappings can utilize the full `-1 to +1` continuous space correctly.
