# 最小二乗SVMを用いた交差確認に基づく特徴量選択

[English](README.md) | **日本語**

次の論文の実験を再現するためのコードです．

> M. Mori, S. Ikeda, R. Tamura, Y. Takano, and R. Miyashiro.
> *Cross-validated feature selection using least-squares support vector machines for binary classification.*

## クイックスタート

必要な環境:

- Python 3.13 と [uv](https://docs.astral.sh/uv/)
- [Gurobi Optimizer](https://www.gurobi.com) 12.0.3 と有効なライセンス
- [poethepoet](https://poethepoet.natn.io/)

```bash
uv sync
uv tool install poethepoet
```

## コマンド

| コマンド | 内容 |
|---|---|
| `poe run --config configs/<条件>.yml` | 1つの実験条件で全手法を実行し，結果を `results/` に保存 |
| `poe refit results/<result_dir>` | 選択された特徴量で共通の線形SVMを再学習し，AUC を計算 |
| `poe figures results/<result_dir>` | SNR ごとに AUC・F1 スコア・選択特徴数の図を作成 |
| `poe table results/<dir_1> ... results/<dir_4>` | 4つの条件から計算時間の表を作成 |

## 実験の再現

論文の4つの実験条件は，`configs/` の設定ファイルに対応しています．

```bash
poe run --config configs/p10_theta5_rho_0.35.yml
poe run --config configs/p10_theta5_rho_0.70.yml
poe run --config configs/p20_theta10_rho_0.35.yml
poe run --config configs/p20_theta10_rho_0.70.yml
```

その後，各結果ディレクトリで `poe refit` と `poe figures` を実行し，`poe table` で表を作成します．

## ライセンス

MIT License（`LICENSE` を参照）．
