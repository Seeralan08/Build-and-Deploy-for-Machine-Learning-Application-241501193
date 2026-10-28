# ChargePulse | India EV Charging Demand

**BDML mini project by Seeralan M (Reg no: 241501193)**

A local-first machine-learning demonstration that estimates charging sessions per station-hour across six illustrative Indian city clusters. It includes a reproducible dataset, a time-ordered holdout evaluation, a small Streamlit interface, and generated analysis charts.

## Data honesty and scope

`data/india_ev_charging_demand_synthetic.csv` contains 10,800 **synthetic** station-hour rows generated deterministically for this project. The values are simulated to demonstrate a realistic analytics workflow; they are not CPCB, utility, charger-operator, or field observations. This project makes no claim of real-world predictive validity. The seed and generation rules are in the source script.

## Run locally

1. Install Python 3.10 or later.
2. In this folder run `python -m pip install -r requirements.txt`.
3. Reproduce metrics with `python run_analysis.py`.
4. Launch the interactive demo with `streamlit run app.py` and open the local address it prints.

After packages are installed, analysis and app run without an internet connection. The first install requires package access.

## Model

A NumPy least-squares linear regression uses hour, weekend flag, temperature, rainfall, port count, event flag, and cluster coordinates. The first 80% of timestamp-ordered observations train the model; the final 20% are held out. MAE/RMSE/R² are reported beside a training-mean baseline. This split is intentionally time-aware, but the simulated process is simple and results should be interpreted as a teaching demonstration.

## Files

- `data/india_ev_charging_demand_synthetic.csv` — complete dataset (10,800 rows)
- `run_analysis.py` — reproducible metric calculation
- `app.py` — interactive local Streamlit demo
- `metrics.json` — recorded evaluation values
- `assets/` — plots generated from the same dataset and predictions
- `../BDML_Mini_Project_241501193.pdf` — formatted project report
