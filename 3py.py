import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction App")
st.write(
    "This app trains a Machine Learning model automatically using the built-in "
    "California Housing dataset and predicts house prices."
)

# ------------------------------
# Load Dataset
# ------------------------------
@st.cache_data
def load_data():
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target
    return X, y

X, y = load_data()

# ------------------------------
# Train Model
# ------------------------------
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    score = r2_score(y_test, pred)

    return model, score

model, score = train_model()

st.success(f"Model trained successfully! R² Score: {score:.3f}")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("Enter House Details")

MedInc = st.sidebar.slider("Median Income", 0.5, 15.0, 3.5)
HouseAge = st.sidebar.slider("House Age", 1, 52, 25)
AveRooms = st.sidebar.slider("Average Rooms", 1.0, 15.0, 5.0)
AveBedrms = st.sidebar.slider("Average Bedrooms", 0.5, 5.0, 1.0)
Population = st.sidebar.slider("Population", 100, 35000, 1000)
AveOccup = st.sidebar.slider("Average Occupancy", 1.0, 10.0, 3.0)
Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 34.0)
Longitude = st.sidebar.slider("Longitude", -125.0, -114.0, -118.0)

input_data = pd.DataFrame({
    "MedInc": [MedInc],
    "HouseAge": [HouseAge],
    "AveRooms": [AveRooms],
    "AveBedrms": [AveBedrms],
    "Population": [Population],
    "AveOccup": [AveOccup],
    "Latitude": [Latitude],
    "Longitude": [Longitude]
})

# ------------------------------
# Prediction
# ------------------------------
st.subheader("Input Values")
st.dataframe(input_data)

if st.button("Predict House Price"):

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated House Price: ${prediction*100000:,.0f}"
    )

# ------------------------------
# Feature Importance
# ------------------------------
st.subheader("Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(
    importance.set_index("Feature")
)

# ------------------------------
# Dataset Preview
# ------------------------------
with st.expander("View Dataset"):
    st.dataframe(X.head())

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.write("Developed using Streamlit + Scikit-Learn")