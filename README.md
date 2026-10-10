# Cross-validated feature selection using LS-SVMs

**English** | [日本語](README.ja.md)

Code for reproducing the experiments in

> M. Mori, S. Ikeda, R. Tamura, Y. Takano, and R. Miyashiro.
> *Cross-validated feature selection using least-squares support vector machines for binary classification.*

## Quick start

Requirements:

- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- [Gurobi Optimizer](https://www.gurobi.com) 12.0.3 with a valid license
- [poethepoet](https://poethepoet.natn.io/)

```bash
uv sync
uv tool install poethepoet
```

## Commands

| Command | Description |
|---|---|
| `poe run --config configs/<condition>.yml` | Run all methods for one experimental condition and save the results under `results/` |
| `poe refit results/<result_dir>` | Refit a common linear SVM on the selected features and compute the AUC |
| `poe figures results/<result_dir>` | Draw the AUC, F1 score, and subset size panels for each SNR |
| `poe table results/<dir_1> ... results/<dir_4>` | Build the computation-time table from the four conditions |

## Reproducing the experiments

The four experimental conditions in the paper correspond to the configuration files in `configs/`.

```bash
poe run --config configs/p10_theta5_rho_0.35.yml
poe run --config configs/p10_theta5_rho_0.70.yml
poe run --config configs/p20_theta10_rho_0.35.yml
poe run --config configs/p20_theta10_rho_0.70.yml
```

Then, for each result directory, run `poe refit` and `poe figures`, and build the table with `poe table`.

## License

MIT License (see `LICENSE`).
