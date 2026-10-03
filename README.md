# Marine microplastic concentration prediction

Semester project by Arnab Chakraborty and Gourab Adhikari. `microplastics.ipynb` predicts microplastic
concentration in sea water from the NOAA NCEI Marine Microplastics database. It compares nine models
under random, study-grouped and chronological validation, tunes CatBoost with a genetic algorithm and
explains the result with SHAP.

## Running

```sh
uv sync
uv run jupyter lab microplastics.ipynb
```

No GPU is needed. A full run takes about an hour on 4 cores, most of it the GA and random searches. Their
results are cached in `artifacts/` and reused while the search configuration is unchanged, so later runs
take about 15 minutes.

`microplastics.py` is the same notebook in jupytext percent format, paired with the `.ipynb`. Edit
either one and jupytext keeps them in sync. It is also easier to read in a diff.
