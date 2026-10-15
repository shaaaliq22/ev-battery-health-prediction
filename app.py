import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="EV Battery Health Prediction", page_icon="🔋", layout="wide")

st.title("🔋 Electric Vehicle Battery Health Prediction")
st.write(
    "Predict battery State of Health (SoH) using a Random Forest Regressor. "
    "Upload a CSV dataset, choose the target and input features, train the model, "
    "and then try a prediction."
)
st.info(
    "Educational project only: predictions are not a certified battery diagnostic or safety assessment. "
    "Use a real, documented dataset and do not report results until the model has been trained."
)

with st.sidebar:
    st.header("Model settings")
    test_size = st.slider("Test data (%)", min_value=10, max_value=40, value=20, step=5)
    n_estimators = st.slider("Number of trees", min_value=50, max_value=500, value=200, step=50)
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

uploaded_file = st.file_uploader("Upload your battery dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.markdown("### Getting started")
    st.markdown(
        "1. Download the battery dataset CSV from its source.\n"
        "2. Upload it above.\n"
        "3. Select the State of Health target column and numeric input features.\n"
        "4. Click **Train model** to see evaluation results."
    )
    st.caption("Suggested dataset source: https://www.kaggle.com/dsv/10969913")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

st.subheader("1. Dataset preview")
c1, c2 = st.columns(2)
c1.metric("Rows", f"{raw_df.shape[0]:,}")
c2.metric("Columns", f"{raw_df.shape[1]:,}")
st.dataframe(raw_df.head(20), use_container_width=True)

with st.expander("Inspect data quality"):
    st.write("Column names and data types")
    info_df = pd.DataFrame({
        "Column": raw_df.columns.astype(str),
        "Data type": [str(t) for t in raw_df.dtypes],
        "Missing values": [int(raw_df[c].isna().sum()) for c in raw_df.columns],
    })
    st.dataframe(info_df, use_container_width=True)
    st.write("Duplicate rows:", int(raw_df.duplicated().sum()))

st.subheader("2. Select target and features")
target_candidates = ["SoH", "SOH", "State_of_Health", "State of Health", "Battery_SoH", "Battery_Health", "soh"]
default_target = next((c for c in target_candidates if c in raw_df.columns), raw_df.columns[-1])

target = st.selectbox(
    "Target column (the value to predict)",
    options=list(raw_df.columns),
    index=list(raw_df.columns).index(default_target),
    help="Choose the column containing battery State of Health (SoH)."
)

numeric_columns = [
    c for c in raw_df.columns
    if c != target and pd.api.types.is_numeric_dtype(pd.to_numeric(raw_df[c], errors="coerce"))
]
suggested_features = [
    c for c in numeric_columns
    if any(token in str(c).lower() for token in
           ["soc", "voltage", "current", "temperature", "temp", "cycle", "capacity", "resistance", "charge"])
]
if not suggested_features:
    suggested_features = numeric_columns

features = st.multiselect(
    "Input features (select measurements used to predict SoH)",
    options=numeric_columns,
    default=suggested_features,
    help="Do not include the target itself or a direct duplicate of it as an input feature."
)

if target in features:
    st.error("The target column cannot also be an input feature.")
    st.stop()
if len(features) == 0:
    st.warning("Select at least one numeric input feature.")
    st.stop()

# Build clean modelling dataframe from the selected columns.
model_df = raw_df[features + [target]].copy()
for col in model_df.columns:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
model_df = model_df.replace([np.inf, -np.inf], np.nan)
model_df = model_df.dropna(subset=[target]).drop_duplicates()

# Convert fractional SoH (e.g., 0.92) to percentage (92), if appropriate.
if len(model_df) and model_df[target].quantile(0.95) <= 1.5:
    model_df[target] = model_df[target] * 100

# Keep a plausible range for percentage-based SoH.
model_df = model_df[model_df[target].between(0, 150)]

if len(model_df) < 20:
    st.error(
        f"Only {len(model_df)} usable rows remain. Check the target column and dataset; "
        "at least 20 rows are recommended for this demonstration."
    )
    st.stop()

st.write(f"Usable rows after basic preprocessing: **{len(model_df):,}**")
st.write(f"Selected target: **{target}**")
st.write("Selected features:", ", ".join(features))

st.subheader("3. Exploratory data analysis")
left, right = st.columns(2)
with left:
    fig, ax = plt.subplots()
    ax.hist(model_df[target].dropna(), bins=25, edgecolor="black")
    ax.set_title("State of Health Distribution")
    ax.set_xlabel("SoH (%)")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

with right:
    corr = model_df[features + [target]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=75, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Feature Correlation Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("4. Train and evaluate model")
st.caption(
    "This uses a random row split. If the dataset contains repeated measurements from the same battery, "
    "a battery-ID-based or time-based split may provide a more realistic evaluation."
)

if st.button("Train model", type="primary", use_container_width=True):
    X = model_df[features]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=int(random_state)
    )

    trained_model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(
            n_estimators=int(n_estimators),
            random_state=int(random_state),
            n_jobs=-1,
            min_samples_leaf=2
        ))
    ])

    trained_model.fit(X_train, y_train)
    predictions = trained_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)

    st.session_state["trained_model"] = trained_model
    st.session_state["feature_names"] = list(features)
    st.session_state["target_name"] = target
    st.session_state["feature_medians"] = X_train.median(numeric_only=True).to_dict()
    st.session_state["metrics"] = {"MAE": mae, "RMSE": rmse, "R²": r2}
    st.session_state["actual"] = np.asarray(y_test)
    st.session_state["predicted"] = np.asarray(predictions)
    st.session_state["importance"] = pd.Series(
        trained_model.named_steps["regressor"].feature_importances_,
        index=features
    ).sort_values(ascending=False)
    st.session_state["train_rows"] = len(X_train)
    st.session_state["test_rows"] = len(X_test)

if "trained_model" in st.session_state:
    st.success("Model trained successfully.")
    metrics = st.session_state["metrics"]
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f'{metrics["MAE"]:.3f}')
    m2.metric("RMSE", f'{metrics["RMSE"]:.3f}')
    m3.metric("R² score", f'{metrics["R²"]:.3f}')
    st.caption(
        f'Training rows: {st.session_state["train_rows"]:,} · '
        f'Test rows: {st.session_state["test_rows"]:,}. '
        "MAE and RMSE are in SoH percentage points when the target is represented as a percentage."
    )

    actual = st.session_state["actual"]
    predicted = st.session_state["predicted"]
    p1, p2 = st.columns(2)

    with p1:
        fig, ax = plt.subplots()
        ax.scatter(actual, predicted, alpha=0.65)
        low = min(float(np.min(actual)), float(np.min(predicted)))
        high = max(float(np.max(actual)), float(np.max(predicted)))
        ax.plot([low, high], [low, high], linestyle="--")
        ax.set_xlabel("Actual SoH (%)")
        ax.set_ylabel("Predicted SoH (%)")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)
        plt.close(fig)

    with p2:
        residuals = actual - predicted
        fig, ax = plt.subplots()
        ax.scatter(predicted, residuals, alpha=0.65)
        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Predicted SoH (%)")
        ax.set_ylabel("Residual (actual - predicted)")
        ax.set_title("Residual Plot")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Feature importance")
    importance = st.session_state["importance"]
    fig, ax = plt.subplots()
    importance.sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Random Forest Feature Importance")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write(f"Most important feature in this fitted model: **{importance.index[0]}**")

    st.subheader("5. Predict SoH for a new measurement")
    st.write("Enter values in the same units and ranges used in the training dataset.")
    with st.form("prediction_form"):
        user_values = {}
        cols = st.columns(2)
        for i, feature in enumerate(st.session_state["feature_names"]):
            default_value = st.session_state["feature_medians"].get(feature, 0.0)
            if pd.isna(default_value):
                default_value = 0.0
            with cols[i % 2]:
                user_values[feature] = st.number_input(
                    f"{feature}",
                    value=float(default_value),
                    format="%.4f",
                    key=f"input_{feature}"
                )
        submitted = st.form_submit_button("Predict battery health", type="primary")

    if submitted:
        input_df = pd.DataFrame([user_values], columns=st.session_state["feature_names"])
        predicted_soh = float(st.session_state["trained_model"].predict(input_df)[0])
        st.metric("Predicted State of Health", f"{predicted_soh:.2f}%")
        if predicted_soh < 80:
            st.warning("The prediction is below the commonly used 80% reference threshold. This is not a safety diagnosis.")
        else:
            st.info("The prediction is at or above the commonly used 80% reference threshold. This is not a safety diagnosis.")

    # Download held-out predictions and metrics for use in the report.
    result_df = pd.DataFrame({"Actual_SoH": actual, "Predicted_SoH": predicted, "Residual": actual - predicted})
    st.download_button(
        "Download test predictions (CSV)",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="ev_battery_test_predictions.csv",
        mime="text/csv"
    )
    metrics_df = pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})
    st.download_button(
        "Download evaluation metrics (CSV)",
        data=metrics_df.to_csv(index=False).encode("utf-8"),
        file_name="ev_battery_metrics.csv",
        mime="text/csv"
    )

st.divider()
st.caption("Project: Machine Learning Application — Assignment 2 | Electric Vehicle Battery Health Prediction")
