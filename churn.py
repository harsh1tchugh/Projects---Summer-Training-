import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="🏦",
    layout="wide"
)

# -------------------------------------------------------
# CUSTOM CSS (Goldenrod Theme)
# -------------------------------------------------------
st.markdown("""
<style>

/* Main Title */
h1{
    text-align:center;
    color:#DAA520;
}

[data-testid="stMetric"]{
    background:#F8F9FA;
    border-radius:12px;
    padding:15px;
}

[data-testid="stMetric"] *{
    color:goldenrod !important;
}

/* Buttons */
.stButton>button{
    width:100%;
    height:50px;
    border-radius:12px;
    background:#DAA520;
    color:black;
    font-size:18px;
    font-weight:bold;
}

/* Button Hover */
.stButton>button:hover{
    background:#c89b1a;
    color:grey;
}

/* Progress Bar */
.stProgress > div > div > div > div{
    background:#DAA520;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# TITLE
# -------------------------------------------------------
st.title("🏦 Customer Churn Prediction Dashboard")

st.write(
    "Predict whether a customer will stay with the bank or churn using Machine Learning."
)

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"D:\devin\SUmmer training\Churn_Modelling.csv")
    return df

df = load_data()
# -------------------------------------------------------
# PREPROCESSING
# -------------------------------------------------------
data = df.copy()

data.drop(
    ["RowNumber", "CustomerId", "Surname"],
    axis=1,
    inplace=True
)

geo_encoder = LabelEncoder()
gender_encoder = LabelEncoder()

data["Geography"] = geo_encoder.fit_transform(data["Geography"])
data["Gender"] = gender_encoder.fit_transform(data["Gender"])

X = data.drop("Exited", axis=1)
y = data["Exited"]

# -------------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------------
@st.cache_resource
def train_model():

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    return model, accuracy

model, accuracy = train_model()

# -------------------------------------------------------
# KPI CARDS
# -------------------------------------------------------
retained = (df["Exited"] == 0).sum()
churned = (df["Exited"] == 1).sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🎯 Accuracy", f"{accuracy*100:.2f}%")

with col2:
    st.metric("👥 Customers", len(df))

with col3:
    st.metric("✅ Retained", retained)

with col4:
    st.metric("❌ Churned", churned)
    # -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.header("🧾 Customer Information")

CreditScore = st.sidebar.slider(
    "Credit Score",
    300,
    900,
    650
)

Geography = st.sidebar.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

Gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

Age = st.sidebar.slider(
    "Age",
    18,
    92,
    35
)

Tenure = st.sidebar.slider(
    "Tenure",
    0,
    10,
    5
)

Balance = st.sidebar.number_input(
    "Balance (£)",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

NumOfProducts = st.sidebar.slider(
    "Number of Products",
    1,
    4,
    1
)

HasCrCard = st.sidebar.selectbox(
    "Has Credit Card",
    ["Yes", "No"]
)

IsActiveMember = st.sidebar.selectbox(
    "Active Member",
    ["Yes", "No"]
)

EstimatedSalary = st.sidebar.number_input(
    "Estimated Salary (£)",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

# -------------------------------------------------------
# ENCODE INPUTS
# -------------------------------------------------------
geo = geo_encoder.transform([Geography])[0]
gender = gender_encoder.transform([Gender])[0]

credit_card = 1 if HasCrCard == "Yes" else 0
active_member = 1 if IsActiveMember == "Yes" else 0

# -------------------------------------------------------
# CREATE INPUT DATAFRAME
# -------------------------------------------------------
input_data = pd.DataFrame({

    "CreditScore": [CreditScore],
    "Geography": [geo],
    "Gender": [gender],
    "Age": [Age],
    "Tenure": [Tenure],
    "Balance": [Balance],
    "NumOfProducts": [NumOfProducts],
    "HasCrCard": [credit_card],
    "IsActiveMember": [active_member],
    "EstimatedSalary": [EstimatedSalary]

})

# -------------------------------------------------------
# CUSTOMER DETAILS
# -------------------------------------------------------
st.subheader("📋 Customer Details")

display_df = pd.DataFrame({

    "Credit Score": [CreditScore],
    "Geography": [Geography],
    "Gender": [Gender],
    "Age": [Age],
    "Tenure": [Tenure],
    "Balance (£)": [Balance],
    "Products": [NumOfProducts],
    "Credit Card": [HasCrCard],
    "Active Member": [IsActiveMember],
    "Estimated Salary (£)": [EstimatedSalary]

})

st.dataframe(
    display_df,
    use_container_width=True
)
# -------------------------------------------------------
# PREDICTION
# -------------------------------------------------------

st.markdown("---")

if st.button("🔍 Predict Customer Churn"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    retained_probability = float(probability[0])
    churn_probability = float(probability[1])

    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.error("❌ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to be RETAINED")

    st.markdown("---")

    st.subheader("📈 Churn Probability")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Retained Probability",
            f"{retained_probability*100:.2f}%"
        )

    with col2:

        st.metric(
            "Churn Probability",
            f"{churn_probability*100:.2f}%"
        )

    st.write("### Churn Risk")

    st.progress(churn_probability)

    st.markdown(
        f"""
        <h2 style='text-align:center;
        color:navy;
        font-weight:bold;'>
        {churn_probability*100:.2f}%
        </h2>
        """,
        unsafe_allow_html=True
    )

    if churn_probability >= 0.80:

        st.error("🔴 High Risk Customer")

    elif churn_probability >= 0.50:

        st.warning("🟠 Moderate Risk Customer")

    else:

        st.success("🟢 Low Risk Customer")

    st.markdown("---")