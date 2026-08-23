# Student Performance Predictor

Put your CSV file at `data/students.csv`.

Open a terminal in this folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.eda
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\streamlit.exe run app.py
```

If the result column in the CSV is not named `pass_fail`, update `TARGET_COLUMN` in `src/config.py`.
