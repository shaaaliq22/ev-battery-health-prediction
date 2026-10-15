# EV Battery Health Prediction — Streamlit App

## Run locally in VS Code

1. Install Python 3.10+ and Visual Studio Code.
2. Open this folder in VS Code.
3. Open the terminal in VS Code.
4. Create and activate a virtual environment (recommended).

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Windows PowerShell
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually http://localhost:8501. Open it in your browser.

## Using the app
1. Upload a CSV containing battery data.
2. Select the State of Health (SoH) target column.
3. Select numeric input features such as voltage, current, temperature, State of Charge, or charge cycles if they exist in your dataset.
4. Click **Train model**.
5. Review MAE, RMSE, R² and plots.
6. Enter measurements to get a prediction.

## Important
- Use a real dataset and verify its column names, units, source, and license.
- Do not include SoH itself or a direct duplicate of SoH among the input features.
- The app is an educational demonstration, not a certified battery diagnostic.
- A random row split can overestimate performance if repeated rows from the same battery occur in both train and test sets.
