# Fractal and Chaotic Activation Functions in Echo State Networks

Companion code for the paper **"Fractal and Chaotic Activation Functions in Echo State Networks: Preprocessing Topology Governs the Echo State Property"** (arXiv:[2512.14675](https://arxiv.org/abs/2512.14675)).

This repo contains the reservoir computing implementation, activation functions, and analysis scripts used to produce the paper's results and figures.

## Contents

| File | Description |
|---|---|
| `activations.py` | Fractal, chaotic, and standard activation functions used in the study |
| `ReservoirComputer.py` | Leaky echo state network (ESN) implementation |
| `d_esp_verify.py` | Statistical verification of the Echo State Property (ESP) |
| `parameter_sweep.py` | Parameter sweeps over reservoir/activation settings, including extended and multi-seed robustness runs |
| `mandelbrot_attractor_analysis.py` | Attractor analysis (Figure 11) |
| `eigenvalue_verification.py` | Spectral radius / eigenvalue confirmation (Figure 9) |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Each script can be run directly, e.g.:

```bash
python eigenvalue_verification.py
```

`d_esp_verify.py` and `parameter_sweep.py` support `QUICK_TEST`, `EXTENDED_TIMESTEPS`, and `EXTENDED_SWEEP` / `MULTISEED_SWEEP` flags for faster iteration vs. full reproduction of paper results. Note that full runs (particularly `d_esp_verify.py`) are slow — expect a long runtime if you run the extended/full settings rather than the quick-test mode.

## Citation

If you use this code, please cite the preprint:

```bibtex
@misc{chipera2026fractal,
  title={Fractal and Chaotic Activation Functions in Echo State Networks: Preprocessing Topology Governs the Echo State Property},
  author={Chipera, Rae and Du, Jenny and Tsapara, Irene},
  year={2026},
  eprint={2512.14675},
  archivePrefix={arXiv}
}
```

*(Citation will be updated once the journal version is published.)*

## License

MIT — see [LICENSE](LICENSE).
