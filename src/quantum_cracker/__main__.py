"""Main entry point: python -m quantum_cracker"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

from quantum_cracker import __version__
from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.core.voxel_grid import SphericalVoxelGrid
from quantum_cracker.core.rip_engine import RipEngine
from quantum_cracker.core.harmonic_compiler import HarmonicCompiler
from quantum_cracker.analysis.metrics import MetricExtractor
from quantum_cracker.analysis.validation import Validator
from quantum_cracker.utils.types import SimulationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantum-cracker",
        description="Harmonic Spherical Compiler -- resolve 256-bit keys via 78 MHz resonance",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command")

    # simulate
    sim = sub.add_parser("simulate", help="Run full simulation pipeline")
    sim.add_argument("--key", type=str, help="256-bit key (hex, binary, or int)")
    sim.add_argument("--random", action="store_true", help="Generate random key")
    sim.add_argument("--steps", type=int, default=100, help="Simulation timesteps")
    sim.add_argument("--grid-size", type=int, default=20, help="Voxel grid size (default 20)")
    sim.add_argument("--no-viz", action="store_true", help="Skip plot generation")
    sim.add_argument("--csv", action="store_true", help="Export results CSV to ~/Desktop")
    sim.add_argument("--sh-filter", action="store_true", help="Apply SH filter during compilation")

    # visualize
    viz = sub.add_parser("visualize", help="Launch 3D renderer")
    viz.add_argument("--key", type=str, help="256-bit key (hex)")
    viz.add_argument("--random", action="store_true", help="Generate random key")
    viz.add_argument("--grid-size", type=int, default=20, help="Voxel grid size")

    return parser


def get_key(args: argparse.Namespace) -> KeyInput:
    """Resolve key from args."""
    if getattr(args, "random", False):
        key = KeyInput.random()
        print(f"Generated random key: {key.as_hex}")
        return key
    elif getattr(args, "key", None):
        return KeyInput(args.key)
    else:
        return KeyInput.from_cli()


def run_simulation(args: argparse.Namespace) -> None:
    """Full pipeline: key -> engines -> simulate -> analyze -> report."""
    key = get_key(args)
    grid_size = args.grid_size
    steps = args.steps

    print(f"Key: {key.as_hex}")
    print(f"Grid: {grid_size}x{grid_size}x{grid_size} | Steps: {steps}")
    print()

    # Initialize
    config = SimulationConfig(
        grid_size=grid_size,
        timesteps=steps,
    )

    print("Initializing voxel grid...")
    grid = SphericalVoxelGrid(size=grid_size)
    grid.initialize_from_key(key)

    print("Initializing rip engine...")
    engine = RipEngine(config=config)
    engine.initialize_from_key(key)

    # Run rip engine
    print(f"Running rip engine ({steps} steps)...")
    rip_history = engine.run(steps)

    # Run harmonic compiler
    print(f"Running harmonic compiler ({steps} steps)...")
    compiler = HarmonicCompiler(grid, config=config)
    peaks = compiler.compile(
        num_steps=steps,
        dt=0.01,
        apply_sh_filter=args.sh_filter,
    )

    # Hamiltonian
    print("Computing Hamiltonian eigenvalues...")
    eigenvalues = compiler.compute_hamiltonian_eigenvalues()

    # Analysis
    print("Extracting metrics...")
    extractor = MetricExtractor(peaks, rip_history)
    report = extractor.full_report()
    extracted_bits = extractor.peaks_to_key_bits()

    # Validation
    validator = Validator(key, extracted_bits)
    validation = validator.summary(
        total_peaks=len(peaks),
        peaks_theta=[p.theta for p in peaks],
    )

    # Output
    print()
    print("=" * 50)
    print(" RESULTS")
    print("=" * 50)
    print(f"  Peaks extracted:     {len(peaks)}")
    print(f"  Bit match rate:      {validation['bit_match_rate']:.4f}")
    print(f"  Confidence interval: ({validation['confidence_interval'][0]:.3f}, {validation['confidence_interval'][1]:.3f})")
    print(f"  Peak alignment:      {validation['peak_alignment']:.4f}")
    print(f"  Ghost harmonics:     {validation['ghost_count']}")
    print(f"  Ground state energy: {eigenvalues[0]:.4f}")
    if len(eigenvalues) > 1:
        print(f"  Energy gap:          {eigenvalues[1] - eigenvalues[0]:.4f}")
    print(f"  Visible threads:     {engine.num_visible}/{engine.num_threads}")
    print(f"  Final radius:        {engine.radius:.2e} m")
    print("=" * 50)

    # CSV export
    if args.csv:
        export_csv(key, report, validation, eigenvalues)

    # Plots
    if not args.no_viz:
        from quantum_cracker.visualization.plots import PlotSuite

        print("\nGenerating plots...")
        plots = PlotSuite()
        plots.spherical_harmonic_heatmap(grid)
        plots.thread_gap_vs_time(rip_history)
        plots.energy_landscape(eigenvalues)
        plots.key_comparison(key, extracted_bits)
        plots.peak_distribution_3d(peaks)
        print("Plots saved to ~/Desktop/")


def export_csv(
    key: KeyInput,
    report: dict,
    validation: dict,
    eigenvalues,
) -> None:
    """Write results to ~/Desktop/quantum_cracker_<timestamp>.csv"""
    desktop = os.path.expanduser("~/Desktop")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(desktop, f"quantum_cracker_{timestamp}.csv")

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["key_hex", key.as_hex])
        writer.writerow(["bit_match_rate", validation["bit_match_rate"]])
        writer.writerow(["peak_alignment", validation["peak_alignment"]])
        writer.writerow(["confidence_lo", validation["confidence_interval"][0]])
        writer.writerow(["confidence_hi", validation["confidence_interval"][1]])
        writer.writerow(["ghost_count", validation["ghost_count"]])
        writer.writerow(["ground_state_energy", float(eigenvalues[0])])
        if len(eigenvalues) > 1:
            writer.writerow(["energy_gap", float(eigenvalues[1] - eigenvalues[0])])
        for k, v in report.get("peak_stats", {}).items():
            writer.writerow([f"peak_{k}", v])
        for k, v in report.get("thread_stats", {}).items():
            writer.writerow([f"thread_{k}", v])

    print(f"Results exported to {filepath}")


def run_visualize(args: argparse.Namespace) -> None:
    """Launch 3D renderer."""
    key = get_key(args)
    grid_size = args.grid_size

    print(f"Key: {key.as_hex}")
    print(f"Grid: {grid_size}x{grid_size}x{grid_size}")
    print("Launching 3D renderer...")
    print("  SPACE=pause  R=reset  +/-=speed  1/2=toggle  ESC=close")

    grid = SphericalVoxelGrid(size=grid_size)
    grid.initialize_from_key(key)

    engine = RipEngine()
    engine.initialize_from_key(key)

    from quantum_cracker.visualization.renderer import QuantumRenderer

    renderer = QuantumRenderer(grid, engine)
    renderer.run()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "simulate":
        run_simulation(args)
    elif args.command == "visualize":
        run_visualize(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
