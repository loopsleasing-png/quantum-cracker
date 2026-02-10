# Quantum Cracker -- User Manual

## What Is This

Quantum Cracker is a simulation engine that implements the **78-Harmonic Spherical Compiler** theory. The core claim: 256-bit encryption keys are not random sequences -- they are geometric structures. By applying a 78 MHz resonant vibration to a spherical coordinate system, we can translate unobservable quantum states ("Hormonic") into classical, readable data ("Harmonic").

The software simulates this process entirely in code. No lab hardware required.

---

## The Theory In Plain English

### The Problem
A 256-bit private key is a number between 0 and 2^256. Brute-forcing it would take longer than the age of the universe. The theory says brute force is the wrong approach -- the key has **geometric structure** that can be read through resonance.

### The Two Engines

**Engine 1: The Rip Engine (Source-Expansion Model)**

Atoms are not solid objects. They are point sources that emit 256 threads of matter at the Planck scale (1.6 x 10^-35 meters). Each thread corresponds to one bit of the key. The threads expand outward from the source.

At first, the threads are too close together to observe -- they are "collapsed" in quantum terms. As they expand, the angular gap between them grows. When the gap exceeds 400 nanometers (the wavelength of visible light), the threads become individually observable. This is the "wave function collapse" -- not magic, just geometry. The threads got far enough apart for light to fit between them.

The Rip Engine simulates this expansion. You feed it a 256-bit key, it converts the key into 256 directional vectors on a sphere, and it tracks their expansion from Planck scale until they become visible.

**Engine 2: The Harmonic Compiler (78 MHz Resonance)**

The search space for a 256-bit key is mapped onto a 78x78x78 spherical voxel grid. Each voxel stores an amplitude, phase, and energy value. The key's bits are encoded into this grid using spherical harmonic basis functions -- the same math that describes electron orbitals.

The Compiler then applies a 78 MHz resonant vibration to the grid:

```
vibration = sin(78 * phi + time) * cos(78 * theta)
```

This vibration amplifies certain voxels and suppresses others. Over many iterations, 78 peak nodes emerge from the noise. These peaks are the key fragments -- the "mist settling" into readable data.

Optionally, a spherical harmonic filter isolates only the l=78 degree harmonic, stripping away everything except the resonant signal.

### How They Connect

1. You input a 256-bit key
2. The Rip Engine models how that key's threads expand from quantum to classical scale
3. The Harmonic Compiler maps the key onto a spherical grid and applies 78 MHz resonance
4. Resonance peaks emerge -- these are the "compiled" key fragments
5. The analysis layer reconstructs bits from the peaks and compares them to the original

### The Hamiltonian

The system's energy state is described by a quantum Hamiltonian:

```
H = T + V_lattice + V_compiler
```

- **T** (kinetic energy): How the amplitude field "wants" to spread out (Laplacian operator)
- **V_lattice** (lattice potential): The grid structure constraining the field
- **V_compiler** (compiler term): The 78 MHz vibration pushing the field toward resonance

The ground state (lowest eigenvalue) of this Hamiltonian corresponds to the most stable configuration -- the resolved key.

---

## Installation

```bash
cd ~/quantum-cracker
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requirements: Python 3.12+, OpenGL support (for 3D renderer).

Verify:
```bash
pytest                              # 174 tests should pass
python -m quantum_cracker --version # quantum-cracker 0.1.0
```

---

## Usage

### Quick Start

```bash
source .venv/bin/activate

# Run a simulation with a random key
python -m quantum_cracker simulate --random

# Run with a specific key (64-character hex)
python -m quantum_cracker simulate --key ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00

# Interactive mode -- it will prompt you for a key
python -m quantum_cracker simulate
```

### The Simulate Command

```
python -m quantum_cracker simulate [OPTIONS]
```

| Flag | Default | What It Does |
|------|---------|-------------|
| `--key KEY` | none | Provide a 256-bit key. Accepts: 64-char hex, 256-char binary, 0x-prefixed hex |
| `--random` | off | Generate a random 256-bit key |
| `--steps N` | 100 | Number of simulation timesteps for both engines |
| `--grid-size N` | 20 | Voxel grid dimension (NxNxN). Use 15-20 for fast runs, 78 for full resolution |
| `--csv` | off | Export results to a CSV file on ~/Desktop |
| `--no-viz` | off | Skip generating plot images |
| `--sh-filter` | off | Apply spherical harmonic filter during compilation (isolates l=78 mode) |

**What happens when you run it:**

1. **Key Input** -- Your key is parsed and validated (must be exactly 256 bits)
2. **Grid Initialization** -- The key's 256 bits are mapped into the voxel grid as spherical harmonic coefficients. Each bit becomes a +1 or -1 weight on a specific Y_l^m basis function. The angular pattern is modulated by a radial Gaussian centered at r=0.5
3. **Rip Engine** -- The key's bits are converted to 256 unit vectors on a sphere (Fibonacci spiral, with z-component flipped by bit value). These threads expand outward from Planck length. At each step, the radius grows by 1%, angular gaps are recomputed, and threads that cross the 400nm visibility threshold are flagged
4. **Harmonic Compiler** -- At each timestep, the resonance vibration sin(78*phi + t)*cos(78*theta) is applied to the grid, modulating amplitudes by 5%. After all steps, the top 78 energy peaks are extracted using 3D local maximum detection
5. **Hamiltonian** -- Eigenvalues are computed on the outermost radial shell. The ground state energy and energy gap are reported
6. **Analysis** -- Peak positions are mapped back to bits (northern hemisphere = 0, southern = 1). Bit match rate, confidence interval, peak alignment score, and ghost harmonic count are computed
7. **Output** -- Results printed to terminal. Plots saved to ~/Desktop (unless --no-viz). CSV exported (if --csv)

**Example output:**

```
Key: 58267d3d8c3c93d3cc5d24daf350318c7d67ba514246c9455f3e77a90a27e997
Grid: 15x15x15 | Steps: 50

Initializing voxel grid...
Initializing rip engine...
Running rip engine (50 steps)...
Running harmonic compiler (50 steps)...
Computing Hamiltonian eigenvalues...
Extracting metrics...

==================================================
 RESULTS
==================================================
  Peaks extracted:     21
  Bit match rate:      0.4727
  Confidence interval: (0.410, 0.535)
  Peak alignment:      0.3851
  Ghost harmonics:     0
  Ground state energy: 0.3734
  Energy gap:          0.4878
  Visible threads:     0/256
  Final radius:        2.66e-35 m
==================================================
```

**Reading the results:**

- **Peaks extracted** -- How many resonance peaks the compiler found (target: 78)
- **Bit match rate** -- Fraction of reconstructed bits that match the original key. 0.5 = random chance. Above 0.5 means the resonance is extracting signal. 1.0 = perfect reconstruction
- **Confidence interval** -- 95% binomial confidence bounds on the bit match rate
- **Peak alignment** -- How well peak angular positions correspond to expected hemispheres (0 = no alignment, 1 = perfect)
- **Ghost harmonics** -- False peaks created by the resonance process (peaks beyond 78)
- **Ground state energy** -- Lowest eigenvalue of the Hamiltonian. The system "wants" to be here
- **Energy gap** -- Distance between ground state and first excited state. Larger gap = more stable resolution
- **Visible threads** -- How many of the 256 threads have crossed the 400nm observable threshold
- **Final radius** -- Current expansion radius of the thread sphere

### The Visualize Command

```
python -m quantum_cracker visualize [OPTIONS]
```

| Flag | Default | What It Does |
|------|---------|-------------|
| `--key KEY` | none | 256-bit key (hex) |
| `--random` | off | Generate random key |
| `--grid-size N` | 20 | Voxel grid dimension |

This opens a real-time 3D window showing:

- **Point cloud** -- The voxel grid rendered as colored points. Blue = low energy, red = high energy. Points vibrate in real-time with the 78 MHz resonance equation. Only voxels above the 10th energy percentile are shown (to avoid visual noise)
- **Thread lines** -- 256 lines from the center outward. Red = not yet visible (gap < 400nm). Green = visible (gap > 400nm)

**Controls:**

| Key | Action |
|-----|--------|
| SPACE | Pause / resume animation |
| R | Reset simulation |
| + | Speed up (1.5x, max 10x) |
| - | Slow down (0.67x, min 0.1x) |
| 1 | Toggle voxel cloud on/off |
| 2 | Toggle thread lines on/off |
| ESC | Close window |
| Mouse drag | Orbit camera |
| Scroll wheel | Zoom in/out |

---

## Generated Files

### Plots (saved to ~/Desktop)

| File | What It Shows |
|------|--------------|
| `qc_sh_heatmap.png` | Amplitude on the outermost spherical shell, unwrapped using Mollweide projection. Blue/red = low/high amplitude. Shows where the key's energy is concentrated on the sphere |
| `qc_gap_vs_time.png` | Two panels. Top: angular gap evolution (avg, min, max) over simulation ticks. Bottom: number of visible threads over time. Shows the expansion process |
| `qc_energy_landscape.png` | Hamiltonian eigenvalue spectrum (first 50 eigenvalues). Ground state highlighted in green. Energy gap annotated. Shows the stability of the resolved state |
| `qc_key_comparison.png` | 256-wide bar chart. Green bars = bits that match the original key. Red bars = mismatches. Title shows overall match rate |
| `qc_peak_3d.png` | 3D scatter plot of extracted peaks in Cartesian space. Point size and color scaled by energy. Shows where resonance is strongest |

### CSV Export

When using `--csv`, a file is saved to `~/Desktop/quantum_cracker_YYYYMMDD_HHMMSS.csv` with columns:

```
metric,value
key_hex,<64-char hex>
bit_match_rate,0.4727
peak_alignment,0.3851
confidence_lo,0.410
confidence_hi,0.535
ghost_count,0
ground_state_energy,0.3734
energy_gap,0.4878
peak_count,21
peak_amplitude_mean,...
peak_energy_mean,...
thread_total_steps,...
thread_final_avg_gap,...
...
```

---

## Key Input Formats

The software accepts 256-bit keys in four formats:

| Format | Example | Length |
|--------|---------|-------|
| Hex string | `ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00` | 64 chars |
| Binary string | `1111111100000000111111110000000...` | 256 chars |
| 0x-prefixed hex | `0xff00ff00...` | 66 chars |
| Interactive | Type `random` at the prompt for a random key | -- |

---

## How the Math Works

### Key-to-Grid Mapping

The 256 bits are mapped to spherical harmonic coefficients sequentially:
- Bit 0 -> Y(0,0) -- l=0, m=0 (1 coefficient)
- Bits 1-3 -> Y(1,-1), Y(1,0), Y(1,1) -- l=1 (3 coefficients)
- Bits 4-8 -> Y(2,-2) through Y(2,2) -- l=2 (5 coefficients)
- ...continues up to l=15 (cumulative: 16^2 = 256 coefficients)

Each bit maps to +1 (bit=1) or -1 (bit=0) as the coefficient. The sum of all weighted spherical harmonics creates the angular amplitude pattern. This is modulated by a radial Gaussian exp(-(r-0.5)^2 / 0.1) to concentrate energy at the middle of the grid.

### Key-to-Thread Mapping

256 points are placed on the unit sphere using the Fibonacci spiral (golden angle spacing for near-uniform distribution). For each bit:
- Bit = 0: thread direction unchanged
- Bit = 1: z-component of direction is flipped (reflected across equator)

This creates a key-dependent distribution of threads on the sphere. Different keys produce measurably different angular gap patterns.

### The Resonance Equation

```
vibration(r, theta, phi, t) = sin(78 * phi + t) * cos(78 * theta)
```

This is applied to the grid amplitude as:
```
amplitude *= (1.0 + vibration * 0.05)
```

The 0.05 factor (resonance_strength) controls how aggressively the vibration reshapes the field. Over many iterations, this creates constructive/destructive interference patterns that amplify the key's structure and suppress noise.

### Visibility Threshold

A thread becomes "visible" when:
```
angular_gap_to_nearest_neighbor * current_radius > 400nm
```

400nm is the wavelength of violet light -- the shortest visible wavelength. When the physical gap between threads exceeds this, the threads can be individually resolved by electromagnetic interaction. This is the geometric interpretation of quantum measurement.

### Hamiltonian

On a single angular shell (theta x phi grid), the Hamiltonian matrix is:

```
H[i,j] = T[i,j] + V_lattice[i] * delta(i,j) + V_compiler[i] * delta(i,j)
```

Where:
- T is the finite-difference Laplacian (5-point stencil with periodic phi boundary)
- V_lattice is the amplitude at each grid point
- V_compiler is the resonance vibration at each grid point, scaled by resonance_strength

Eigenvalues are computed via numpy.linalg.eigvalsh (symmetric real matrix). The ground state is the lowest eigenvalue.

---

## Configuration

All physical and simulation parameters are defined in `src/quantum_cracker/utils/constants.py` and `src/quantum_cracker/utils/types.py`.

### Tunable Parameters (SimulationConfig)

| Parameter | Default | What It Controls |
|-----------|---------|-----------------|
| `grid_size` | 78 | Voxel grid dimension. 78 = full resolution (474,552 voxels). 15-20 for fast testing |
| `num_threads` | 256 | Number of expansion threads (matches 256-bit key) |
| `resonance_freq` | 78.0 | Frequency of the harmonic vibration in MHz |
| `resonance_strength` | 0.05 | How strongly the vibration modulates amplitude (0.0 = no effect, 1.0 = full replacement) |
| `timesteps` | 1000 | Default simulation steps |
| `dt` | 0.01 | Time increment per step |
| `expansion_rate` | 0.01 | Fractional radius growth per tick (1% per step) |

### Physical Constants

| Constant | Value | Used For |
|----------|-------|----------|
| `PLANCK_LENGTH` | 1.616e-35 m | Initial thread radius |
| `PLANCK_DENSITY` | 5.155e96 kg/m^3 | Rip trigger threshold (theoretical) |
| `OBSERVABLE_THRESHOLD` | 400nm | Visibility cutoff for threads |
| `SH_DEGREE` | 78 | Spherical harmonic filter degree |

---

## Project Structure

```
quantum-cracker/
    MANUAL.md                        <- You are here
    CLAUDE.md                        <- Claude Code reference
    PROGRESS.md                      <- Feature tracker
    DECISIONS.md                     <- Architecture decision log
    BLUEPRINTS.md                    <- Reusable task prompts
    REBUILD.md                       <- Full rebuild instructions
    pyproject.toml                   <- Dependencies and build config
    scripts/
        preflight.sh                 <- Run at session start
        smoke-test.sh                <- Run after changes
    src/quantum_cracker/
        __init__.py                  <- Package version (0.1.0)
        __main__.py                  <- CLI entry point
        core/
            key_interface.py         <- Key input, validation, conversion
            voxel_grid.py            <- 78^3 spherical voxel grid
            rip_engine.py            <- 256-thread expansion simulator
            harmonic_compiler.py     <- 78 MHz resonance + peak extraction
        analysis/
            metrics.py               <- Observable metric extraction
            validation.py            <- Statistical validation
        visualization/
            plots.py                 <- 5 Matplotlib plot types
            renderer.py              <- PyOpenGL 3D renderer
            shaders.py               <- GLSL vertex/fragment shaders
        utils/
            constants.py             <- Physical constants
            types.py                 <- Dataclass definitions
            math_helpers.py          <- Coordinate transforms, sphere math
    tests/                           <- 174 pytest tests
```

---

## Glossary

| Term | Definition |
|------|-----------|
| **Hormonic** | Unobservable quantum state. The "mist" before compilation. The key exists but cannot be read |
| **Harmonic** | Observable classical state. The compiled output. The key resolved into readable data |
| **C-Compiler Vibration** | The 78 MHz resonance applied to the spherical grid. Translates Hormonic to Harmonic |
| **Rip** | The moment when Planck-density matter forces threads to expand outward. The beginning of the expansion |
| **Thread** | A single directional trajectory from the source. One thread per bit. 256 threads per key |
| **Mist** | The pre-compiled state of the voxel grid. Amplitude spread across all voxels without clear peaks |
| **Settling** | The process of peaks emerging from the mist through resonance. The compiler "settling" the key |
| **Ghost Harmonic** | A false peak created by the resonance process. An artifact, not a real key fragment |
| **Observable Threshold** | 400nm -- the minimum physical gap between threads for them to be individually detectable |
| **Ground State** | The lowest energy configuration of the Hamiltonian. The most stable resolution of the key |
| **Energy Gap** | The difference between ground state and first excited state. Larger gap = more confident resolution |
| **Bit Match Rate** | Fraction of reconstructed bits that match the original key. The primary accuracy metric |
| **Peak Alignment** | How well extracted peak positions correspond to expected bit values. Measures angular accuracy |
| **Spherical Harmonic** | Mathematical basis function on the sphere (Y_l^m). Used to encode and filter the key's structure |
| **Voxel** | A single cell in the 78x78x78 grid. Stores amplitude, phase, and energy at one (r, theta, phi) point |
