# High-Precision Fidelity Estimation with Common Randomized Measurements

[![arXiv](https://img.shields.io/badge/arXiv-2511.22509-b31b1b.svg)](https://arxiv.org/abs/2511.22509)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

Efficient fidelity estimation of multiqubit quantum states is crucial to quantum information processing. However, existing protocols require order $1/\epsilon^2$ different circuits to estimate the infidelity $\epsilon$ to multiplicative precision, posing a major bottleneck for high-precision fidelity estimation.

We prove that Clifford-based common randomized measurement (CRM) shadow estimation achieves a quadratic reduction to $1/\epsilon$ circuits. This follows from tight variance bounds for arbitrary observables under Clifford CRM, controlled by the deviation between the true state and a chosen prior.

Under dominant noise models—including depolarizing and Pauli noise—the circuit cost becomes independent of both $\epsilon$ and the number of qubits, and a single circuit often suffices for intermediate and large systems. We further show that experimentally simpler Clifford measurements outperform 4-design measurements in many practical scenarios, while both offer exponential advantages over Pauli measurements.

## Repository Overview

The code in this repository is used to:

- Compute quantum-state fidelities, cross characteristic functions, and twisted cross characteristic functions.
- Simulate these quantities under different noise models.
- Evaluate the variance and circuit cost of different estimation protocols.
- Generate the numerical data and plots discussed in the paper.

Paper: [High-Precision Fidelity Estimation with Common Randomized Measurements](https://arxiv.org/abs/2511.22509)

## Requirements

The project is written in Python. We recommend using `conda` or `venv` to create an isolated environment.

Main dependencies:

- Python >= 3.11
- NumPy
- SciPy
- Matplotlib
- Pandas
- PaddlePaddle
- Paddle Quantum
- mpi4py

Some scripts only require NumPy and Matplotlib. PaddlePaddle, Paddle Quantum, and mpi4py are needed only for the corresponding numerical or MPI calculations.

## Output Directory

The plotting scripts in `upload_git` use a user-defined output directory:

```python
folder_path = rf"xxxx"  # User defined
```

Replace `xxxx` with the directory in which the generated figures and numerical data should be saved. For example:

```python
folder_path = rf"./figures"
```

The directory is created automatically if it does not already exist.

## Figure–Code Mapping

| Figure / Output | Corresponding Code | Notes |
|---|---|---|
| `gain_combined_n10.png/.pdf` | [`upload_git/plot_gain_combined.py`](./upload_git/plot_gain_combined.py) | Combined analytical gain $\Lambda(R,x,\epsilon)$. |
| `gain_combined_ghz_z1noise_n10.png/.pdf` | [`upload_git/plot_gain_combined_ghz_z1noise.py`](./upload_git/plot_gain_combined_ghz_z1noise.py) | GHZ target state under single-qubit $Z_1$ Pauli noise. |
| `gain_combined_coherent_s77_numerical.png/.pdf` | [`upload_git/plot_gain_coherent_skk_numerical.py`](./upload_git/plot_gain_coherent_skk_numerical.py) | Numerical Clifford CRM calculation under coherent noise. |
| `lambda_vs_M2_two_panels_short_ticks_legend_reversed.png/.pdf` | [`upload_git/lambda_vs_M2_two_panels_short_ticks_legend_reversed.py`](./upload_git/lambda_vs_M2_two_panels_short_ticks_legend_reversed.py) | Optimized gain versus stabilizer Rényi entropy $M_2(\sigma)$. |
| [Figure 2](./figures/fig2.png) | `upperbound_k_0808.py`<br>`4design_upperbound_k.py`<br>`4design_and_Clifford_sametime(0304coherent).py` | — |
| [Figure 3](./figures/fig3.png) | `runmpi.py` | Pauli CRM is simulated numerically. Clifford and 4-design CRM are evaluated analytically. |
| [Figure 4](./figures/fig4.png) | `single_pauli_error.py` | Set `fdesign` to select Clifford or 4-design CRM. |
| [Figure S1](./figures/figS1.png) | `sampling0820.py` | Clifford and 4-design CRM under Pauli noise. |
| [Figure S2](./figures/figS2.png) | `sampling0820_unitary.py` | Clifford and 4-design CRM under coherent noise. |
| [Figure S3](./figures/figS3.png) | `upperboundterms_k_depolar.py` | — |
| [Figure S4](./figures/figS4.png) | `unoise_1022vio.py` | — |
| [Figure S5](./figures/figS5.png) | `upperbound_k_0808.py` | Set the parameters to the values used for Figure S5. |
| [Figure S6](./figures/figS6.png) | `A_Small_repetition_upperbound_k.py` | — |
| [Figure S7](./figures/figS7.png) | `upperboundterms_k copy 2.py` | — |
| [Figure S8](./figures/figS8.png) | `ghz_diffferent_n.py` | — |
| [Figure S9](./figures/figS9.png) | Eqs. (2), (165), and (166) | Calculated directly from the analytical expressions. |
| [Figure S10](./figures/figS10.png) | `PauliCRMTFIM_EDGS.py` | Pauli CRM is evaluated numerically. Clifford and 4-design CRM are evaluated analytically. |
| [Figure S11](./figures/figS11.png) | `PauliCRMTFIM_EDGS.py` | Pauli CRM is evaluated numerically. Clifford and 4-design CRM are evaluated analytically. |

## Running the Additional Plotting Scripts

### Combined analytical gain

```bash
python upload_git/plot_gain_combined.py
```

This script generates:

- `gain_combined_n10.png`
- `gain_combined_n10.pdf`

### GHZ state under single-qubit Pauli noise

```bash
python upload_git/plot_gain_combined_ghz_z1noise.py
```

This script generates:

- `gain_combined_ghz_z1noise_n10.png`
- `gain_combined_ghz_z1noise_n10.pdf`

### Product magic state under coherent noise

```bash
python upload_git/plot_gain_coherent_skk_numerical.py
```

This script generates:

- `gain_combined_coherent_s77_numerical.png`
- `gain_combined_coherent_s77_numerical.pdf`
- Numerical variance and gain data in `coherent_s77_data`

### Optimized gain versus stabilizer Rényi entropy

```bash
python upload_git/lambda_vs_M2_two_panels_short_ticks_legend_reversed.py
```

This script generates:

- `lambda_vs_M2_two_panels_short_ticks_legend_reversed.png`
- `lambda_vs_M2_two_panels_short_ticks_legend_reversed.pdf`

## Notes

The scripts primarily generate numerical data and plots. Reproducing a specific figure may require setting the parameters to the values used in the corresponding section of the paper.

The repository reflects the updated version of the work and may differ from earlier versions of the arXiv manuscript.

## Citation

If you use this code, please cite:

```bibtex
@misc{yang2025highprecisionfidelityestimationcommon,
    title         = {High-Precision Fidelity Estimation with Common Randomized Measurements},
    author        = {Zhongyi Yang and Datong Chen and Zihao Li and Huangjun Zhu},
    year          = {2025},
    eprint        = {2511.22509},
    archivePrefix = {arXiv},
    primaryClass  = {quant-ph},
    url           = {https://arxiv.org/abs/2511.22509}
}
```

## License

This project is released under the MIT License.
