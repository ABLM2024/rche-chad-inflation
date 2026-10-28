# Robust probabilistic inflation forecasting for Chad (RCHE)

Code, data and results for:

> I. Dagal, A. L. Mahamat, *Robust Probabilistic Inflation Forecasting from 34 Annual Observations: Online Hedge Ensembles and Adaptive Conformal Prediction in Chad* (manuscript submitted, 2026).

A robust conformal Hedge ensemble (RCHE) combines six robust or regularised learners (historical median, two Huber regressions, support vector regression, Gaussian process regression, random forest) with online Hedge weights and wraps the combination in a conformal predictive distribution with adaptive conformal intervals. It is evaluated by rolling origin over 2005–2024 on annual data for 1991–2024.

## Requirements
Python ≥ 3.10 with numpy, pandas, scipy, scikit-learn, statsmodels and matplotlib:

    pip install numpy pandas scipy scikit-learn statsmodels matplotlib

## Reproducing the results
From the `work` folder:

| Step | Command | Output |
|---|---|---|
| 1 | `python3 model.py` | `pred_base.csv`, `fc2025_base.json` – rolling-origin forecasts 1997–2024 and the 2025 forecast |
| 2 | `python3 model.py '{"label":"winsor","winsor":true}'` (and the `eta`, `cap`, `gamma` variants) | robustness variants `pred_*.csv` |
| 3 | `python3 evaluate.py base 2005` | `eval_base_2005.json` – point, interval, density and event scores, Diebold–Mariano tests, model confidence set |
| 4 | `python3 figs.py` | `figs/*.png` |

The result files of every step are included. Steps 1 and 3 were re-run from this repository and reproduce `pred_base.csv`, `fc2025_base.json` and `eval_base_2005.json` exactly (about 20 s).

From the repository root, `python3 sensitivity_2020.py` re-runs steps 1 and 3 with the revised 2020 value of the IMF October 2026 vintage and prints the main metrics next to the published ones.

Figure files and paper figures: `fig1`–`fig4` and `fig7`–`fig9` are Figs. 1–4 and 7–9; `fig6_point.png` is Fig. 5 and `fig5_weights.png` is Fig. 6.

## Data
- `data/chad.csv` – annual inflation (%, average consumer prices, IMF World Economic Outlook, indicator PCPIPCH), unemployment rate (% of labour force, modelled ILO estimate, World Bank WDI SL.UEM.TOTL.ZS) and misery index (sum of the two), 1991–2024.
- `data/imf_weo_oct2026_PCPIPCH_TCD.csv` – the IMF DataMapper series for Chad (October 2026 vintage) used to check the inflation series; it agrees in 33 of 34 years, the exception being 2020 (4.5% in the analysed series, −2.7% after the latest revision).

## Licence
Code: MIT (`LICENSE`). Data compiled from the IMF and the World Bank remain subject to their terms of use.
