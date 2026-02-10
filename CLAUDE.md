# Quantum Cracker - Claude Code Reference

## Project Overview
Harmonic Spherical Compiler -- software to prove quantum superposition and its relational imprint on observable metrics. Uses 78 MHz resonance on a spherical voxel grid to resolve 256-bit keys.

## Project Location
- **Path:** /Users/kjm/quantum-cracker
- **Type:** Python project (NumPy, SciPy, Qiskit, PyOpenGL)

## Tech Stack
- **Language:** Python 3.12+ (tested 3.14)
- **Quantum:** Qiskit + Qiskit Aer
- **Scientific:** NumPy, SciPy (sph_harm_y), Matplotlib
- **3D Rendering:** PyOpenGL, GLFW, pyrr
- **Testing:** pytest (174 tests)
- **Package Management:** pip + pyproject.toml (src layout)

## Project Structure
```
src/quantum_cracker/
    __main__.py              # CLI entry point
    core/
        key_interface.py     # 256-bit key input (hex/bin/int/bytes/CLI)
        voxel_grid.py        # 78^3 spherical voxel grid + SH decomposition
        rip_engine.py        # 256-thread expansion simulator
        harmonic_compiler.py # 78 MHz resonance + peak extraction
    analysis/
        metrics.py           # Observable metric extraction
        validation.py        # Statistical validation + confidence intervals
    visualization/
        plots.py             # 5 Matplotlib plot types
        renderer.py          # PyOpenGL 3D renderer
        shaders.py           # GLSL shader sources
    utils/
        constants.py         # Physical constants (GRID_SIZE=78, etc.)
        types.py             # Dataclass definitions
        math_helpers.py      # Coordinate transforms, Fibonacci sphere
```

## Commands
- **Run tests:** `.venv/bin/pytest`
- **Simulate:** `python -m quantum_cracker simulate --random --steps 100 --csv`
- **3D View:** `python -m quantum_cracker visualize --random`
- **Install:** `pip install -e ".[dev]"`
- **Preflight:** `bash scripts/preflight.sh`
- **Smoke test:** `bash scripts/smoke-test.sh`

## Key Technical Notes
- SciPy 1.17+: use `sph_harm_y(l, m, theta, phi)` NOT old `sph_harm(m, l, phi, theta)`
- PyOpenGL on macOS needs `OPENGL_FORWARD_COMPAT = True` for 3.3 core profile
- Expansion rate is fractional (0.01/tick), not Planck length (float precision issue)
- Grid size default 78 for production, 15-20 for fast testing

## Development Workflow
1. **Start session:** Run `bash scripts/preflight.sh`
2. **Starting a feature:** Update "Current Session" in memory MEMORY.md
3. **At breakpoints:** WIP commits
4. **After changes:** Run `bash scripts/smoke-test.sh`
5. **On finishing:** Update PROGRESS.md + REBUILD.md + DECISIONS.md, clean commit
6. **Session dies:** Next session reads memory + `git status` + `git diff`

## User Preferences
- No emojis in code or output
- CSV reports to ~/Desktop
- Commit and push after features
- Terminal/CLI workflows preferred
- Co-author: `Co-Authored-By: Claude <noreply@anthropic.com>`
