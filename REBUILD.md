# Quantum Cracker - Rebuild Manual

Complete instructions to rebuild this project from scratch on a new machine.

---

## Prerequisites
- Python 3.12+ (tested on 3.14)
- git
- pip
- OpenGL support (for 3D renderer)

## Steps

### 1. Clone the repo
```bash
git clone <repo-url> ~/quantum-cracker
cd ~/quantum-cracker
```

### 2. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -e ".[dev]"
```

### 4. Verify installation
```bash
pytest
python -m quantum_cracker --version
```

### 5. Run a simulation
```bash
# Random key, 100 steps, grid size 20, with CSV export
python -m quantum_cracker simulate --random --steps 100 --grid-size 20 --csv

# Specific key
python -m quantum_cracker simulate --key ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00 --steps 200

# Interactive key input
python -m quantum_cracker simulate
```

### 6. Launch 3D renderer
```bash
python -m quantum_cracker visualize --random
# Controls: SPACE=pause, R=reset, +/-=speed, 1/2=toggle layers, ESC=close
```

### 7. Set up Claude Code memory
```bash
# Memory files at ~/.claude/projects/-Users-kjm-quantum-cracker/memory/
# Auto-created on first Claude Code session in this directory
```

## macOS Notes
- PyOpenGL requires OpenGL 3.3 core profile (GLFW sets OPENGL_FORWARD_COMPAT)
- If glfw fails, try: `brew install glfw` (though pip package usually works)

## Dependencies
Core: numpy, scipy, qiskit, qiskit-aer, matplotlib, PyOpenGL, glfw, pyrr
Dev: pytest, pytest-cov, mypy, ruff, jupyter, ipykernel

## Project Structure
```
src/quantum_cracker/
    core/
        key_interface.py       -- 256-bit key input (hex/bin/int/bytes)
        voxel_grid.py          -- 78^3 spherical voxel grid
        rip_engine.py          -- 256-thread expansion simulator
        harmonic_compiler.py   -- 78 MHz resonance + peak extraction
    analysis/
        metrics.py             -- observable metric extraction
        validation.py          -- statistical validation
    visualization/
        plots.py               -- matplotlib 2D plots (5 types)
        renderer.py            -- PyOpenGL 3D renderer
        shaders.py             -- GLSL shader sources
    utils/
        constants.py           -- physical constants
        types.py               -- dataclass definitions
        math_helpers.py        -- coordinate transforms
    __main__.py                -- CLI entry point
```
