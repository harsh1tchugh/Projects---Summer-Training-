import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
 
DATA_PATH = "creditcard.csv"          # <-- change if your file lives elsewhere
 
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection — Deep Learning")
st.caption("Neural network trained on anonymized transaction data (Time, V1-V28, Amount → Class)")
 
 
# ---------------------------------------------------------
# 1. Load data + train model ONCE, cache it
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading data and training model (first run only)...")
def load_and_train():
    df = pd.read_csv(DATA_PATH)
 
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values
    feature_names = df.drop(columns=["Class"]).columns.tolist()
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
 
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
 
    model = keras.Sequential([
        layers.Input(shape=(X_train_s.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"]
    )
 
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=5, restore_best_weights=True
    )
 
    model.fit(
        X_train_s, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=2048,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=0
    )
 
    y_test_prob = model.predict(X_test_s, verbose=0).ravel()
 
    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "X_test": X_test,
        "X_test_scaled": X_test_s,
        "y_test": y_test,
        "y_test_prob": y_test_prob,
        "df": df,
    }
 
 
data = load_and_train()
model = data["model"]
scaler = data["scaler"]
feature_names = data["feature_names"]
df = data["df"]
 
# ---------------------------------------------------------
# 2. Sidebar - choose / configure a transaction
# ---------------------------------------------------------
st.sidebar.header("⚙️ Transaction Settings")
 
mode = st.sidebar.radio(
    "How do you want to pick a transaction?",
    ["Random sample from dataset", "Manual entry"]
)
 
if "current_txn" not in st.session_state:
    st.session_state.current_txn = None
    st.session_state.current_label = None
 
if mode == "Random sample from dataset":
    sample_type = st.sidebar.selectbox(
        "Sample type", ["Any transaction", "Known Normal", "Known Fraud"]
    )
 
    if st.sidebar.button("🎲 Pick Random Transaction"):
        if sample_type == "Known Normal":
            pool = df[df["Class"] == 0]
        elif sample_type == "Known Fraud":
            pool = df[df["Class"] == 1]
        else:
            pool = df
 
        row = pool.sample(1).iloc[0]
        st.session_state.current_txn = row[feature_names].values.astype(float)
        st.session_state.current_label = int(row["Class"])
 
    st.sidebar.caption("Click the button to load a transaction, then Detect Fraud below.")
 
else:
    st.sidebar.caption("Adjust the key fields. Other PCA features (V1-V28) default to 0 "
                        "(dataset average) unless expanded below.")
 
    time_val = st.sidebar.slider("Time (seconds since first txn)", 0, 172792, 50000)
    amount_val = st.sidebar.slider("Amount ($)", 0.0, 5000.0, 100.0, step=1.0)
 
    v_values = {}
    with st.sidebar.expander("Advanced: PCA features (V1-V28)"):
        for v in [f"V{i}" for i in range(1, 29)]:
            v_values[v] = st.slider(v, -30.0, 30.0, 0.0, step=0.1, key=f"slider_{v}")
 
    manual_row = {"Time": time_val}
    manual_row.update(v_values)
    manual_row["Amount"] = amount_val
 
    if st.sidebar.button("✅ Use This Transaction"):
        st.session_state.current_txn = np.array([manual_row[f] for f in feature_names], dtype=float)
        st.session_state.current_label = None  # unknown ground truth for manual entries
 
st.sidebar.divider()
detect_clicked = st.sidebar.button("🔍 Detect Fraud", type="primary", use_container_width=True)
 
# ---------------------------------------------------------
# 3. Main area - show chosen transaction + prediction
# ---------------------------------------------------------
if st.session_state.current_txn is None:
    st.info("👈 Use the sidebar to pick a random transaction or enter one manually, then click **Detect Fraud**.")
else:
    txn = st.session_state.current_txn
    st.subheader("Selected Transaction")
    txn_df = pd.DataFrame([txn], columns=feature_names)
    st.dataframe(txn_df, use_container_width=True)
 
    if detect_clicked:
        txn_scaled = scaler.transform(txn.reshape(1, -1))
        prob = float(model.predict(txn_scaled, verbose=0).ravel()[0])
        pred_label = 1 if prob > 0.5 else 0
 
        st.divider()
        st.subheader("Prediction Result")
 
        col1, col2, col3 = st.columns(3)
        with col1:
            if pred_label == 1:
                st.error("🚨 FRAUD DETECTED")
            else:
                st.success("✅ NORMAL TRANSACTION")
        with col2:
            st.metric("Fraud Probability", f"{prob*100:.2f}%")
        with col3:
            if st.session_state.current_label is not None:
                truth = "Fraud" if st.session_state.current_label == 1 else "Normal"
                st.metric("Actual Label (dataset)", truth)
 
        # -----------------------------------------------------
        # 4. Charts / visuals at the bottom
        # -----------------------------------------------------
        st.divider()
        st.subheader("📊 Visuals")
 
        c1, c2 = st.columns(2)
 
        with c1:
            st.markdown("**Fraud Probability**")
            prob_df = pd.DataFrame({
                "Outcome": ["Normal", "Fraud"],
                "Probability": [1 - prob, prob]
            })
            st.bar_chart(prob_df.set_index("Outcome"))
 
        with c2:
            st.markdown("**Model Performance on Test Set — ROC Curve**")
            fpr, tpr, _ = roc_curve(data["y_test"], data["y_test_prob"])
            auc_score = roc_auc_score(data["y_test"], data["y_test_prob"])
 
            fig, ax = plt.subplots(figsize=(4, 3.2))
            ax.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            st.pyplot(fig)
 
        st.markdown("**Confusion Matrix — Test Set**")
        y_test_pred = (data["y_test_prob"] > 0.5).astype(int)
        cm = confusion_matrix(data["y_test"], y_test_pred)
 
        fig2, ax2 = plt.subplots(figsize=(4, 3.2))
        im = ax2.imshow(cm, cmap="Blues")
        ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Normal", "Fraud"])
        ax2.set_yticks([0, 1]); ax2.set_yticklabels(["Normal", "Fraud"])
        ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax2.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        st.pyplot(fig2)
 
