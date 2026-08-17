# NFL-Total-Points-Prediction

Predicting combined game totals from team performance trends, and testing whether a model can beat the Vegas betting line.

## Overview

Sportsbooks set point totals using their own models, insider information, and market pressure from bettors, making the Vegas line one of the most efficient predictive benchmarks in sports. This project builds a full pipeline, from raw play-by-play data to trained models, to predict the combined total points scored in an NFL game, then honestly benchmarks the result against the actual Vegas closing line to see whether the model adds any real predictive edge.

The resulting analysis can help:

- **Sports bettors/analysts** understand how hard it actually is to beat a well-calibrated market line
- **Data practitioners** see a full, realistic ML pipeline: data collection, cleaning, feature engineering, EDA, and multi-model comparison
- **Anyone curious about sports analytics** understand what actually drives scoring (recent offensive/defensive efficiency, weather, dome vs. outdoor)

## Data

Data collected directly via the `nfl_data_py` package (sourced from `nflfastR`), covering the 2000-2023 regular seasons:

- **Schedules** - game-level info: teams, scores, rest days, stadium roof type, weather, and the Vegas total line
- **Play-by-play (EPA)** - Expected Points Added per play, used to measure offensive and defensive efficiency per team per game

Both are cached locally as `.parquet` files at each pipeline stage (raw, cleaned, and feature-engineered).

## Approach

1. **Data collection** - pulled schedules and play-by-play data for 24 seasons, filtered to regular-season games only.
2. **Data cleaning** - dropped games with missing scores, flagged dome/closed-roof games, and filled missing weather (temp/wind) using dome-appropriate defaults (72°F, 0 wind) or the outdoor median.
3. **Feature engineering** - calculated offensive and defensive EPA per team per game, then built prior-4-game rolling averages and season-to-date averages (both properly shifted to avoid data leakage), merged onto both home and away teams. Filtered to week 4+ so every game has enough prior data to generate meaningful rolling features.
4. **Exploratory analysis** - visualized scoring trends by season (noting the scoring increase after ~2004 rule changes), dome vs. outdoor scoring distributions, a feature correlation heatmap, and the overall points distribution.
5. **Modeling** - trained on 2000-2021, held out 2022-2023 as the test set, and compared Linear Regression, Ridge, Random Forest, and Gradient Boosting.
6. **Benchmarking** - compared every model's test performance directly against the actual Vegas total line for the same games.

## Results

| Model | CV RMSE | Test RMSE | Test MAE |
|---|---|---|---|
| Linear Regression | 13.87 | 13.41 | 10.57 |
| Ridge Regression | 13.87 | 13.41 | 10.57 |
| Random Forest | 13.95 | 13.60 | 10.67 |
| Gradient Boosting | 14.27 | 13.63 | 10.68 |
| **Vegas Line (benchmark)** | - | **13.18** | **10.32** |

Every model landed in a tight, similar range (Test RMSE 13.41-13.63), with simple Linear/Ridge Regression performing just as well as the more complex tree-based models. None of them beat the Vegas line, which posted a lower RMSE and MAE than every model tested. This is an honest and expected result: Vegas lines incorporate information (injuries, weather updates, sharp betting action) that a model built purely on rolling EPA and rest days can't see, and consistently beating a mature betting market is a genuinely hard problem. Feature importance showed away-team season-long offensive EPA as the single strongest predictor, ahead of short-term rolling defensive metrics - suggesting sustained offensive quality matters more than recent hot/cold streaks.

## Tech Stack

- Python, pandas, NumPy
- nfl_data_py
- scikit-learn (Linear Regression, Ridge, Random Forest, Gradient Boosting)
- matplotlib, seaborn
- joblib (model persistence)

## Future Improvements

- Incorporate injury reports and betting line movement as features, closer to what Vegas actually uses
- Test whether the model can identify specific games where it disagrees meaningfully with Vegas, rather than only comparing aggregate error
- Add quarterback-specific EPA, since QB play drives scoring more than team-wide averages capture
- Try ensembling the model's prediction with the Vegas line itself as a blended estimate

## Author

Omar Rodriguez Arellano

Part of my [Data Science Portfolio](https://omaralexrod.github.io)
