# ml-testing-streamlit-final

Streamlit app for **ML testing & data preprocessing** using your aggregation logic and model artifacts.

## Features
- Upload multiple files (Beneficiary, Inpatient, Outpatient, Test).
- Provider-level aggregation exactly as per your provided notebook.
- Auto plots: claim distribution, top providers by claims, PotentialFraud counts (if available).
- Preprocessing via metadata/encoders to ensure numeric-only matrix for XGBoost.
- Predictions with optimized threshold from `fraud_detection_metadata.pkl`.
- Download aggregated & predictions as CSV.

## Run
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Upload the 4 CSVs from `data/sample/` to test the pipeline quickly.
