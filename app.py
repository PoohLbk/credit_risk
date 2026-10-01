import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
import streamlit as st

# Set Page Config
st.set_page_config(
    page_title="Credit Risk Assessment AI", page_icon="💳", layout="centered"
)


# 1. Train and Cache Model
@st.cache_resource
def train_model():
    # Load dataset
    df = pd.read_csv("credit_risk_dataset.csv")

    # Clean missing values
    df["person_emp_length"] = df["person_emp_length"].fillna(
        df["person_emp_length"].median()
    )
    df["loan_int_rate"] = df["loan_int_rate"].fillna(
        df["loan_int_rate"].median()
    )

    # Prepare Encoders
    encoders = {}
    cat_cols = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ]

    for col in cat_cols:
        oe = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        df[col] = oe.fit_transform(df[[col]])
        encoders[col] = oe

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    # Train Model
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X, y)

    return model, encoders


model, encoders = train_model()

# 2. UI Layout
st.title("💳 ระบบประเมินความเสี่ยงการเบี้ยวหนี้ (Credit Risk AI)")
st.write(
    "กรอกข้อมูลผู้กู้ยืมเพื่อประเมินโอกาสการค้างชำระหนี้ด้วย Machine Learning"
)

st.divider()

# Input Form
with st.form("loan_form"):
    st.subheader("👤 ข้อมูลส่วนตัวและสัญญากู้ยืม")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("อายุ (ปี)", min_value=18, max_value=100, value=25)
        income = st.number_input(
            "รายได้ต่อปี ($USD)", min_value=1000, value=35000, step=1000
        )
        emp_length = st.number_input(
            "อายุการทำงาน (ปี)", min_value=0, max_value=60, value=3
        )
        home_ownership = st.selectbox(
            "สถานะที่อยู่อาศัย", ["RENT", "MORTGAGE", "OWN", "OTHER"]
        )

    with col2:
        loan_amnt = st.number_input(
            "วงเงินขอกู้ ($USD)", min_value=500, value=10000, step=500
        )
        loan_intent = st.selectbox(
            "วัตถุประสงค์การกู้",
            [
                "PERSONAL",
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "HOMEIMPROVEMENT",
                "DEBTCONSOLIDATION",
            ],
        )
        loan_grade = st.selectbox(
            "เกรดสินเชื่อ (Loan Grade)", ["A", "B", "C", "D", "E", "F", "G"]
        )
        loan_int_rate = st.number_input(
            "อัตราดอกเบี้ย (%)",
            min_value=5.0,
            max_value=30.0,
            value=11.0,
            step=0.1,
        )

    cb_default = st.radio(
        "เคยมีประวัติค้างชำระหนี้ในระบบเครดิตหรือไม่?", ["N", "Y"]
    )
    cred_hist_length = st.slider("ประวัติเครดิต (ปี)", 1, 30, 4)

    submit_btn = st.form_submit_button("🔍 ประเมินผลความเสี่ยง (Evaluate Risk)")

# 3. Calculation & Display Output
if submit_btn:
    # Feature Engineering (คำนวณอัตราส่วนภาระหนี้ต่อรายได้)
    loan_percent_income = loan_amnt / income if income > 0 else 0.0

    # Build Feature DataFrame
    input_data = pd.DataFrame(
        [
            {
                "person_age": age,
                "person_income": income,
                "person_home_ownership": home_ownership,
                "person_emp_length": emp_length,
                "loan_intent": loan_intent,
                "loan_grade": loan_grade,
                "loan_amnt": loan_amnt,
                "loan_int_rate": loan_int_rate,
                "loan_percent_income": loan_percent_income,
                "cb_person_default_on_file": cb_default,
                "cb_person_cred_hist_length": cred_hist_length,
            }
        ]
    )

    # Encode Categorical features
    for col in [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ]:
        input_data[col] = encoders[col].transform(input_data[[col]])

    # Predict Probability
    default_prob = model.predict_proba(input_data)[0][1] * 100

    st.divider()
    st.subheader("📊 ผลการประเมินความเสี่ยง")

    # Display Percentage
    st.metric(
        label="โอกาสที่ผู้กู้จะเบี้ยวหนี้ (Default Probability)",
        value=f"{default_prob:.2f}%",
    )
    st.write(
        f"**สัดส่วนภาระหนี้ต่อรายได้ (Loan-to-Income Ratio):** {loan_percent_income * 100:.2f}%"
    )

    # Risk Level Categorization & Warning
    if default_prob >= 50:
        st.error("🚨 **ความเสี่ยงสูงมาก (High Risk):** อนุมัติยาก / ควรปฏิเสธ")
    elif default_prob >= 25:
        st.warning(
            "⚠️ **ความเสี่ยงปานกลาง (Medium Risk):** ควรขอหลักประกันหรือค้ำประกันเพิ่ม"
        )
    else:
        st.success("✅ **ความเสี่ยงต่ำ (Low Risk):** ผ่านเกณฑ์เบื้องต้น")
