# Student Performance Predictor

This workspace contains a Streamlit app that uses Logistic Regression to predict whether a student will "Pass" or "Fail" based on scores and demographic features.

Files added:
- [app.py](app.py) — Streamlit application.
- [requirements.txt](requirements.txt) — Python dependencies.

Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place your encoded CSV next to `app.py`. The file must contain a `Pass/Fail` column and the encoded feature columns. By default the app looks for `StudentsPerformance_final_encoded.csv`.

3. Run the app:

```bash
streamlit run app.py
```

If your CSV has a different name or path, update the `load_data()` call in `app.py` or place the file in the same folder.
