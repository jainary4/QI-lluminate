# Quantum Illumination POVM Simulation

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance simulation framework for **discrete-variable quantum illumination (QI)** using high-dimensional Bell states and physically realizable interferometric receivers.

This project implements the quantum illumination protocol introduced in:

- Pannu, Helmy, and El Gamal (2024)
- *Towards Error-Free Quantum Target Finding: When Sequential Detection Meets High-Dimensional Entanglement*

The framework focuses on scalable simulation of entangled target-detection systems under realistic thermal noise, together with the construction of experimentally motivated POVM (positive operator-valued measure) receivers using beam splitters, spatial light modulators, and photon-number-resolving detectors.

---

# Overview

Quantum illumination is an entanglement-enhanced sensing protocol designed for detecting weakly reflecting targets embedded in bright thermal noise. Unlike classical radar or lidar systems, QI exploits residual quantum correlations between a transmitted signal beam and a retained idler beam to improve target discrimination performance.

This repository provides:

- Efficient simulation of the **target-present** and **target-absent** quantum states,
- Monte Carlo generation of reduced density matrices,
- Automatic exploitation of conserved photon-number symmetries,
- Construction of a physically realizable interferometric POVM,
- Log-likelihood based detection and error analysis,
- Infrastructure for scalable CPU/GPU acceleration.

The implementation is written entirely from first principles using **NumPy/SciPy**, without quantum-computing frameworks, allowing full transparency and direct control over the underlying quantum-optical simulation pipeline.

---

# Key Features

## Efficient Quantum-State Generation

The target-present state generation exploits:

- Beam-splitter locality,
- Tensor-product factorization,
- Partial-trace commutativity,
- Classical Monte Carlo sampling of the thermal environment,

to reduce the computational scaling from exponential:

```math
\mathcal{O}(d^{6M})
```

to polynomial complexity:

```math
\mathcal{O}(K\,M^2\,d^3)
```

where:

- \(M\) = number of mode pairs,
- \(d = N_{\max}+1\) is the local Fock-space dimension,
- \(K\) = Monte Carlo samples.

---

## Block-Diagonal Hilbert-Space Decomposition

The framework automatically identifies and exploits the conserved total return photon number:

```math
N_C = \sum_j n_j
```

which decomposes the full idler-return Hilbert space into independent sectors.

This block-diagonal structure:

- dramatically reduces computational cost,
- simplifies POVM construction,
- ensures physical consistency of the receiver model,
- enables scalable lookup-table generation.

---

## Physically Realizable Interferometric POVM

The repository implements a realistic receiver architecture based on:

- Spatial light modulators (SLMs),
- 50:50 beam-splitter interferometers,
- Photon-number-resolving detectors (PNRDs),
- Log-likelihood ratio hypothesis testing.

The POVM maps detector click patterns to:

- \(P(H_0)\),
- \(P(H_1)\),
- Log-likelihood scores,
- Detection decisions.

---

## Parallel and Scalable

The simulation framework supports:

- Outer/inner multiprocessing,
- Local beam-splitter amplitude caching,
- Monte Carlo parallelism,
- Memory-efficient tensor construction,
- Future CUDA/CuPy GPU acceleration.

Current implementations support moderate mode numbers, while the architecture is designed to scale toward:

```math
M \sim 10^3
```

modes.

---

# Repository Structure

```text
quantum-illumination-povm/
├── README.md
├── requirements.txt
├── src/
│
├── states/
│   ├── target_present_efficient.py
│   │     # Monte-Carlo target-present state generation (GPU-ready)
│   │
│   ├── target_present_optimised.py
│   │     # CPU-optimized implementation
│   │
│   ├── target_absent.py
│   │     # Analytical target-absent state construction
│   │
│   └── Target_Absent_Block_diagonal.py
│         # Block-diagonal target-absent builder
│
├── POVM/
│   ├── POVM_interferometric.py
│   │     # Interferometric POVM lookup-table construction
│   │
│   └── POVM_Detection.py
│         # Detection simulation and error analysis
│
├── utils/
│   ├── Local_BS_table_generator.py
│   │     # Precompute local beam-splitter amplitudes
│   │
│   ├── photon_number_bell_state.py
│   │     # Bell-state utilities
│   │
│   └── scaling_analysis.py
│         # Runtime and scaling benchmarks
│
└── .gitignore
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/jainary4/quantum-illumination-povm.git
cd quantum-illumination-povm
```

---

## 2. Create a Virtual Environment (Recommended)

```bash
python3 -m venv env
source env/bin/activate
```

Windows:

```bash
env\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include:

- `numpy`
- `scipy`
- `matplotlib`
- `cupy-cuda12x` (optional GPU acceleration)
- `modal` (optional cloud storage utilities)

For CPU-only simulations, `cupy` may be omitted.

---

# Quick Start

## 1. Generate Local Beam-Splitter Amplitude Tables

```python
from src.utils.Local_BS_table_generator import build_local_BS_table

local_BS = build_local_BS_table(N_max=2)
```

This precomputes all local 50:50 beam-splitter amplitudes in truncated Fock space for reuse during simulation.

---

## 2. Generate Quantum States

### Target-Present State (Monte Carlo)

```python
from src.states.target_present_efficient import run_full_pipeline

metadata = run_full_pipeline(
    M=2,
    Kappa=0.01,
    Nbar=0.5,
    Nmax=2,
    K_samples=1000
)
```

---

### Target-Absent State (Block-Diagonal)

```python
from src.states.Target_Absent_Block_diagonal import (
    generate_target_absent_blocks
)

rho_abs_blocks = generate_target_absent_blocks(
    M=2,
    Nmax=2,
    Nbar=0.5
)
```

---

## 3. Build the Interferometric POVM Lookup Table

```python
from src.POVM.POVM_interferometric import (
    build_global_lookup_table
)

global_lookup = build_global_lookup_table(
    M=2,
    N_max=2,
    local_BS=local_BS,
    rho_pres_blocks=rho_pres_blocks,
    rho_abs_blocks=rho_abs_blocks,
    outer_workers=4,
    inner_workers=1,
    use_slm=False
)
```

The lookup table maps every detector click pattern to:

- Log-likelihood ratio,
- \(P(H_0)\),
- \(P(H_1)\),
- Detection decision statistics.

---

## 4. Simulate Detection

```python
from src.POVM.POVM_Detection import DetectionSimulator

sim = DetectionSimulator(global_lookup)

Pe = sim.estimate_error_probability(
    n_trials=10000
)

print(f"Estimated error probability: {Pe:.4f}")
```

---

# Physical Background

## Quantum Illumination

Quantum illumination is a quantum sensing protocol in which an entangled signal-idler pair is generated:

- The **signal** probes a target region,
- The **idler** is stored locally as a quantum reference.

If a weakly reflecting target is present, a small fraction of the signal returns mixed with bright thermal noise. Although environmental interactions destroy most of the original entanglement, residual quantum correlations survive and can still improve target detection.

---

## Beam-Splitter Interaction

The target interaction is modeled as a beam splitter of reflectivity \(\eta\):

```math
\hat{c}= \sqrt{\eta}\,\hat{a} + \sqrt{1-\eta}\,\hat{b}
```

where:

- \(\hat{a}\) = signal mode,
- \(\hat{b}\) = thermal environment mode,
- \(\hat{c}\) = collected return mode.

The simulation evolves these interactions locally for every mode pair.

---

## High-Dimensional Bell States

The transmitted entangled state is an \(M\)-mode Bell state:

```math
|\psi_1\rangle =
\frac{1}{\sqrt{M}}
\sum_{k=1}^{M}
|e_k\rangle_I \otimes |e_k\rangle_S
```

This state distributes a single photon coherently across many modes, allowing noise resilience to improve with increasing mode number \(M\).

---

## Detection Strategy

The interferometric receiver converts surviving quantum coherences into measurable photon-number correlations using:

- Interference,
- Beam splitting,
- Phase conditioning,
- Photon-number detection.

A log-likelihood ratio classifier is then used to distinguish between:

- \(H_0\): target absent,
- \(H_1\): target present.

---

# Research Goals

This repository is part of an ongoing research effort focused on:

- Scalable discrete-variable quantum illumination,
- Realistic optical POVM construction,
- Quantum-enhanced radar/lidar architectures,
- GPU-accelerated many-mode simulation,
- Sequential quantum detection protocols,
- Error-exponent benchmarking against fundamental quantum limits.

---

# References

1. Pannu, A., Helmy, A. S., & El Gamal, H. (2024).  
   *Quantum Illumination with High-Dimensional Bell States*.  
   Physical Review A, 110, L050603.

2. *Towards Error-Free Quantum Target Finding: When Sequential Detection Meets High-Dimensional Entanglement*

3. Tan, S.-H., et al. (2008).  
   *Quantum Illumination with Gaussian States*.  
   Physical Review Letters, 101, 253601.

4. Shapiro, J. H. (2020).  
   *The Quantum Illumination Story*.  
   IEEE Aerospace and Electronic Systems Magazine, 35(4), 8–20.

5. Nielsen, M. A., & Chuang, I. L. (2010).  
   *Quantum Computation and Quantum Information*.  
   Cambridge University Press.

---

# Future Work

Planned extensions include:

- Full GPU/CUDA acceleration with CuPy,
- Adaptive and sequential POVM strategies,
- Error-exponent benchmarking,
- Multi-target discrimination,
- Experimental optical receiver modeling,
- Continuous-variable/discrete-variable hybrid protocols,
- Hardware-oriented detector simulation.

---

# Citation

If you use this repository in academic work, please cite:

```bibtex
@article{pannu2024qi,
  title={Quantum Illumination with High-Dimensional Bell States},
  author={Pannu, Armanpreet and Helmy, Amr S. and El Gamal, Hesham},
  journal={Physical Review A},
  volume={110},
  pages={L050603},
  year={2024}
}
```

---

# License

This project is licensed under the MIT License.

See the `LICENSE` file for details.

---

# Author

Aryan Jain  
University of Toronto

Developed as part of research in quantum sensing, quantum optics, and computational quantum simulation.