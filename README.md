# Formula 1 Race Prediction with Bayesian Simulation

This project predicts Formula 1 race finishing order using FastF1 data, feature engineering, a PyMC hierarchical Bayesian model, and Monte Carlo simulation.

The main idea is simple: a regression-style model can estimate how strong each driver looks, but an F1 race result has to be a valid ranking. There can only be one P1, one P2, and so on. I used the Bayesian model to estimate performance and uncertainty, then used simulation to turn those estimates into realistic 1-20 race rankings.

The best place to read the modeling logic is `F1ModelExplain.ipynb`. The trained model is saved in `model/f1_trace.nc`.

## What This Project Does

- Collects and prepares 2025 F1 race data with FastF1.
- Builds driver, team, track-type, recent-form, and DNF-risk features.
- Trains a hierarchical Bayesian model in PyMC.
- Checks posterior convergence with ArviZ.
- Converts model outputs into valid race rankings through Monte Carlo simulation.
- Evaluates predictions on Qatar GP and Abu Dhabi GP.
- Uses the Abu Dhabi prediction to project the 2025 championship standings.

## Why I Used This Approach

Race prediction has two problems that a basic model does not handle well.

First, drivers are not independent. If one driver finishes first, no other driver can also finish first. Predicting each driver's finishing position separately can produce probabilities that do not make sense as a full race result.

Second, F1 performance is layered. A driver result depends on the driver, the car, the grid position, the circuit type, recent form, and reliability. A hierarchical Bayesian model gives a cleaner way to separate some of these effects while still keeping uncertainty in the final prediction.

## Modeling Setup

For driver `d` in race `r`, the model estimates finishing position as:

```text
y[d,r] ~ Normal(mu[d,r], sigma_race^2)

mu[d,r] =
    alpha
  + beta_team[team[d]]
  + gamma_driver[d]
  + eta_grid * GridPosition[d,r]
  + delta_track[track[r]] * adaptation[d]
  + epsilon_trend * recent_form[d]
  + zeta_dnf * DNF_risk[d]
```

In plain English:

| Term | Meaning |
| --- | --- |
| `alpha` | Baseline finishing position |
| `beta_team` | Team or constructor strength |
| `gamma_driver` | Driver ability after accounting for the car |
| `eta_grid` | Starting grid position effect |
| `delta_track` | Track-type effect for high-speed, balanced, and technical circuits |
| `epsilon_trend` | Recent performance trend |
| `zeta_dnf` | Penalty for DNF risk |
| `sigma_race` | Race noise that the features do not explain |

Lower finishing position is better, so negative effects usually mean stronger performance.

One modeling detail I had to handle carefully: `GridPosition` and `QualifyingPosition` were highly correlated, about `0.974` in the processed data. I avoided relying on both as separate strong signals because that would double-count almost the same information.

## Ranking Simulation

The PyMC model produces continuous predicted finishing scores. Those scores are useful, but they are not yet a race result. To get a valid ranking, I used this process:

1. Sample one set of parameters from the posterior.
2. Generate one latent performance score for each driver.
3. Sort the drivers by that score.
4. Assign unique finishing positions from P1 to P20.
5. Repeat many times.
6. Summarize win probability, podium probability, Top 10 probability, expected position, mode position, and uncertainty intervals.

For the Abu Dhabi GP prediction, the notebook runs 500,000 simulations. The final ranking is sorted by:

```text
Mode -> Mean -> P_Win descending -> Q25 -> CI_2.5
```

## Results

### Model Training

The final model was trained in `F1ModelBuild.ipynb` with 4 chains, 2,000 tuning steps, and 4,000 posterior draws per chain.

| Metric | Result |
| --- | ---: |
| Maximum R-hat | 1.0047 |
| Minimum ESS | 455 |
| Divergences | 0 |
| Posterior predictive R2 | 0.490 |
| Posterior predictive MAE | 3.12 positions |

The convergence diagnostics looked acceptable for the project goal. The model is not perfect, but it gave a usable posterior for simulation and race-level evaluation.

### Qatar GP

| Metric | Result |
| --- | ---: |
| MAE | 3.80 positions |
| RMSE | 4.89 positions |
| Spearman correlation | 0.641 |
| Top 5 hit rate | 4/5 |
| Top 10 hit rate | 8/10 |

The model captured most of the front half of the field, but missed the winner. It predicted `O PIASTRI`, while the actual winner was `M VERSTAPPEN`.

### Abu Dhabi GP

| Metric | Result |
| --- | ---: |
| MAE | 3.10 positions |
| RMSE | 4.40 positions |
| Exact rank predictions | 4/20 |
| Within +/-1 position | 10/20 |
| Top 3 driver-set hit rate | 3/3 |
| Top 10 driver-set hit rate | 6/10 |

Predicted Abu Dhabi GP Top 5:

1. `M VERSTAPPEN`
2. `L NORRIS`
3. `O PIASTRI`
4. `G RUSSELL`
5. `C LECLERC`

The strongest result here was the podium group. The model identified the correct Top 3 drivers, even though exact ordering still had uncertainty.

### Championship Projection

Using the Abu Dhabi simulation result, the projected 2025 championship Top 3 were:

| Rank | Driver | Projected Points |
| ---: | --- | ---: |
| 1 | `L NORRIS` | 426 |
| 2 | `M VERSTAPPEN` | 421 |
| 3 | `O PIASTRI` | 407 |

## Visuals

### Data Exploration

![F1 data exploration](f1_data_exploration.png)

This figure shows the relationship between qualifying and race result, average constructor performance, DNF rate by constructor, and the track-type mix in the dataset.

### Team Tier Priors

![Team tier distributions](team_tiers.png)

This figure shows how team tiers were used to guide the team-strength prior. It is a simple prior, but it helps the model start with a reasonable view of the field.

### Qatar GP Position Distribution

![Qatar GP predicted position distribution](qatar_position_distribution.png)

The dark points show the final predicted rankings. The vertical ranges show uncertainty from the simulation.

### Abu Dhabi GP Position Distribution

![Abu Dhabi GP predicted position distribution](abu_dhabi_position_distribution.png)

This figure shows the simulated finishing range for each Abu Dhabi GP driver.

## Repository Structure

```text
.
|-- F1DataGet.ipynb
|-- F1ModelBuild.ipynb
|-- F1ModelExplain.ipynb
|-- F1MTKL.ipynb
|-- F1 Predict Qatar.ipynb
|-- F1 Predict ABU DHABI.ipynb
|-- F1 Final predict.ipynb
|-- model build.ipynb
|-- data get.ipynb
|-- data clean.ipynb
|-- data/
|   |-- f1_multi_season_results.csv
|   |-- f1_race_data_cleaned.csv
|   |-- driver_features.csv
|   |-- drivers_info.csv
|   |-- teams_info.csv
|   |-- qatar_ready.csv
|   |-- qatar_predict_menka.csv
|   |-- abu_dhabi_ready.csv
|   |-- abu_dhabi_predict_menka.csv
|   `-- final_2025_championship_prediction.csv
|-- model/
|   `-- f1_trace.nc
|-- f1_data_exploration.png
|-- team_tiers.png
|-- qatar_position_distribution.png
`-- abu_dhabi_position_distribution.png
```

## Main Notebooks

| Notebook | Purpose |
| --- | --- |
| `F1DataGet.ipynb` | Pulls race data with FastF1 and builds the base dataset. |
| `F1ModelBuild.ipynb` | Trains the final PyMC model and saves the posterior trace. |
| `F1ModelExplain.ipynb` | Explains the model, priors, diagnostics, and final results. |
| `F1MTKL.ipynb` | Tests ranking logic, strategy weighting, and Qatar validation. |
| `F1 Predict Qatar.ipynb` | Runs and evaluates Qatar GP predictions. |
| `F1 Predict ABU DHABI.ipynb` | Runs Abu Dhabi GP predictions and championship projection. |
| `model build.ipynb` | Earlier modeling experiment. |

## Data Artifacts

| File | Description |
| --- | --- |
| `data/f1_multi_season_results.csv` | Aggregated FastF1 race-level data. |
| `data/f1_race_data_cleaned.csv` | Cleaned model-ready race data. |
| `data/driver_features.csv` | Driver-level features, including points, average position, recent form, track-type averages, and DNF rate. |
| `data/teams_info.csv` | Constructor points, rank, and tier information. |
| `data/qatar_ready.csv` | Qatar GP prediction input. |
| `data/qatar_predict_menka.csv` | Qatar GP simulation output. |
| `data/abu_dhabi_ready.csv` | Abu Dhabi GP prediction input. |
| `data/abu_dhabi_predict_menka.csv` | Abu Dhabi GP simulation output. |
| `model/f1_trace.nc` | Saved ArviZ NetCDF posterior trace. |

## How to Run

Run notebooks from the project root so relative paths resolve correctly.

Install dependencies:

```bash
pip install fastf1 pandas numpy scipy matplotlib seaborn pymc arviz scikit-learn tqdm joblib jupyter
```

Start Jupyter:

```bash
jupyter notebook
```

Recommended order:

1. `F1DataGet.ipynb`
2. `F1ModelBuild.ipynb`
3. `F1ModelExplain.ipynb`
4. `F1 Predict Qatar.ipynb`
5. `F1 Predict ABU DHABI.ipynb`
6. `F1MTKL.ipynb`

If you only want to review the completed work, start with `F1ModelExplain.ipynb` and use the saved trace in `model/f1_trace.nc`.

## What I Would Improve Next

- Add lap-level pace and tire strategy instead of relying mostly on race-level features.
- Include weather, safety cars, penalties, pit stops, and tire degradation.
- Model constructor-season effects more explicitly.
- Test the model on more seasons to reduce overfitting to one season's field.
- Build a small dashboard where users can change grid order and simulate a race live.

## Main Takeaway

The model is strongest at identifying front-of-field and Top 10 drivers, while exact ranking in the midfield is still noisy. That matches the nature of F1: the front runners are more stable, while the midfield is more sensitive to incidents, strategy, and small race-day changes.

The useful part of this project is not just the final ranking. It is the full workflow: data collection, feature design, Bayesian modeling, convergence checks, simulation under ranking constraints, and honest evaluation against actual race results.
