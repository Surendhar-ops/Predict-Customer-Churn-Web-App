import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📉 Customer Churn Prediction Web App")

# Load pickled model and encoders
def load_model():
    with open("customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model_data, encoders

model_data, encoders = load_model()
model = model_data["model"]
feature_names = model_data["features_names"]

# Session state for data and flow control
if "df" not in st.session_state:
    st.session_state.df = None
if "operations_done" not in st.session_state:
    st.session_state.operations_done = False

# Utility functions
def data_understanding(data):
    st.write("### Data Understanding")
    st.write("**Shape of the data:**", data.shape)
    st.write("**Data Info:**")
    buffer = StringIO()
    data.info(buf=buffer)
    st.text(buffer.getvalue())
    st.write("**Missing Values:**")
    st.write(data.isnull().sum())
    st.write("**Unique Values per Column:**")
    for col in data.columns:
        st.markdown(f"**{col}**: {data[col].unique()}")


def data_operations_menu(data):
    st.write("### Data Operations")

    if "temp_df" not in st.session_state:
        st.session_state.temp_df = data.copy()

    temp_df = st.session_state.temp_df

    col_to_drop = st.selectbox("Select a column to drop", options=temp_df.columns, key="drop")
    if st.button("Drop Column"):
        temp_df = temp_df.drop(columns=[col_to_drop])
        st.session_state.temp_df = temp_df
        st.success(f"Dropped column: {col_to_drop}")

    col_to_convert = st.selectbox("Select column to change type", options=temp_df.columns, key="convert")
    new_type = st.selectbox("Select new data type", options=["int", "float", "str"], key="dtype")
    if st.button("Convert Column Type"):
        try:
            temp_df[col_to_convert] = pd.to_numeric(temp_df[col_to_convert], errors='coerce') if new_type in ['int', 'float'] else temp_df[col_to_convert].astype(str)
            temp_df[col_to_convert] = temp_df[col_to_convert].astype(new_type)
            st.session_state.temp_df = temp_df
            st.success(f"Converted {col_to_convert} to {new_type}")
        except Exception as e:
            st.warning(f"Conversion failed: {e}")

    if st.button("Save Changes and Continue to EDA"):
        st.session_state.df = temp_df.copy()
        st.session_state.operations_done = True
        st.success("Changes saved. Proceed to EDA and Visualization.")

    return temp_df


def EDA(df):
    st.write("### Exploratory Data Analysis")
    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", df.columns.tolist())
    st.write("**Describe:**")
    st.write(df.describe())

# Sidebar for dataset upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (.csv)", type=["csv"])

if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.session_state.operations_done = False
    st.success("Dataset uploaded and loaded successfully.")

# Display UI
if st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("Raw Dataset")
    st.write(df.head())

    with st.expander("📋 Data Understanding"):
        data_understanding(df)

    if not st.session_state.operations_done:
        with st.expander("🧰 Data Operations"):
            data_operations_menu(df)

    if st.session_state.operations_done:
        df = st.session_state.df.copy()  # ensure we're using the updated version

        with st.expander("📈 Exploratory Data Analysis"):
            EDA(df)

        st.subheader("Target Distribution")
        if "Churn" in df.columns:
            st.bar_chart(df["Churn"].value_counts())

        df = df.select_dtypes(include=["number", "object"]).copy()  # Ensure correct types post conversion

        num_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col != "Churn"]
        st.subheader("Numerical Feature Distributions")
        selected_num_cols = st.multiselect("Select numerical features to visualize", options=num_cols, default=num_cols)
        for col in selected_num_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            plt.axvline(df[col].mean(), color='r', linestyle='--', label='Mean')
            plt.axvline(df[col].median(), color='g', linestyle='-', label='Median')
            plt.legend()
            st.pyplot(fig)

        st.subheader("Boxplots")
        for col in num_cols:
            if df[col].nunique() > 1:
                fig, ax = plt.subplots()
                sns.boxplot(y=df[col].dropna(), ax=ax)
                st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

        st.subheader("Categorical Feature Counts")
        cat_cols = [col for col in df.columns if df[col].dtype == 'object']
        selected_cat_cols = st.multiselect("Select categorical features to visualize", options=cat_cols, default=cat_cols)
        for col in selected_cat_cols:
            fig, ax = plt.subplots()
            sns.countplot(x=col, data=df, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)


st.header("🔮 Predict Churn for a New Customer")
input_data = {}

fields = {
    'gender': ["Male", "Female"],
    'SeniorCitizen': [0, 1],
    'Partner': ["Yes", "No"],
    'Dependents': ["Yes", "No"],
    'tenure': 1,
    'PhoneService': ["Yes", "No"],
    'MultipleLines': ["Yes", "No", "No phone service"],
    'InternetService': ["DSL", "Fiber optic", "No"],
    'OnlineSecurity': ["Yes", "No", "No internet service"],
    'OnlineBackup': ["Yes", "No", "No internet service"],
    'DeviceProtection': ["Yes", "No", "No internet service"],
    'TechSupport': ["Yes", "No", "No internet service"],
    'StreamingTV': ["Yes", "No", "No internet service"],
    'StreamingMovies': ["Yes", "No", "No internet service"],
    'Contract': ["Month-to-month", "One year", "Two year"],
    'PaperlessBilling': ["Yes", "No"],
    'PaymentMethod': ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    'MonthlyCharges': 1.0,
    'TotalCharges': 1.0
}

with st.form("prediction_form"):
    for key, value in fields.items():
        if isinstance(value, list):
            input_data[key] = st.selectbox(key, value)
        elif isinstance(value, int):
            input_data[key] = st.slider(key, 0, 72, value)
        elif isinstance(value, float):
            input_data[key] = st.number_input(key, step=0.1)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([input_data])
    for column, encoder in encoders.items():
        input_df[column] = encoder.transform(input_df[column])
    prediction = model.predict(input_df)
    pred_proba = model.predict_proba(input_df)

    st.subheader("Prediction Result")
    st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
    st.write(f"Probability of Churn: {pred_proba[0][1]:.2f}")
