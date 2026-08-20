
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import pandas as pd
import numpy as np

st.title("My Data Analytics App")

st.write("Streamlit is working!")

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Telecom Churn Prediction System",
    page_icon="📡",
    layout="wide"
)


# =========================================================
# LOAD SAVED MODEL, SCALER AND FEATURE COLUMNS
# =========================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    return model, scaler, feature_columns


model, scaler, feature_columns = load_artifacts()


# =========================================================
# TITLE
# =========================================================

st.title("📡 Telecom Customer Churn Prediction System")

st.write(
    """
    This application predicts whether a telecom customer is likely to
    *churn (leave the service)* or *remain with the company*.

    The prediction is based on customer usage, charges, satisfaction,
    support activity and service information.
    """
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Customer Information")

st.sidebar.write(
    "Enter the customer's information below to generate a churn prediction."
)


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    tenure_months = st.number_input(
        "Tenure (Months)",
        min_value=1,
        max_value=120,
        value=12,
        step=1
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0,
        step=10.0
    )


with col2:

    satisfaction_score = st.slider(
        "Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3
    )

    num_support_tickets = st.number_input(
        "Number of Support Tickets",
        min_value=0,
        max_value=50,
        value=2,
        step=1
    )

    contract_type = st.selectbox(
        "Contract Type",
        options=[
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )


with col3:

    internet_service = st.selectbox(
        "Internet Service",
        options=[
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    payment_method = st.selectbox(
        "Payment Method",
        options=[
            "Electronic check",
            "Mailed check",
            "Bank transfer",
            "Credit card"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        options=[
            "No",
            "Yes"
        ]
    )

    streaming_service = st.selectbox(
        "Streaming Service",
        options=[
            "No",
            "Yes"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        options=[
            "No",
            "Yes"
        ]
    )


st.divider()


# =========================================================
# CONVERT BINARY INPUTS TO 0 AND 1
# =========================================================

tech_support_value = 1 if tech_support == "Yes" else 0

streaming_service_value = 1 if streaming_service == "Yes" else 0

paperless_billing_value = 1 if paperless_billing == "Yes" else 0


# =========================================================
# FEATURE ENGINEERING
# =========================================================

# Average monthly value
avg_monthly_value = total_charges / tenure_months


# Tenure group
if tenure_months <= 12:
    tenure_group = "new"

elif tenure_months <= 36:
    tenure_group = "mid"

else:
    tenure_group = "long"


# High risk customer flag
high_risk = int(
    (satisfaction_score <= 2)
    and
    (num_support_tickets >= 3)
)


# Charges per support ticket
charges_per_ticket = (
    total_charges / (num_support_tickets + 1)
)


# Engagement score
engagement_score = (
    tenure_months * 0.4
    +
    satisfaction_score * 0.3
    -
    num_support_tickets * 0.3
)


# =========================================================
# DISPLAY ENGINEERED FEATURES
# =========================================================

st.subheader("⚙️ Customer Risk Indicators")

risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)


with risk_col1:
    st.metric(
        "Average Monthly Value",
        f"{avg_monthly_value:,.2f}"
    )


with risk_col2:
    st.metric(
        "Charges per Ticket",
        f"{charges_per_ticket:,.2f}"
    )


with risk_col3:
    st.metric(
        "Engagement Score",
        f"{engagement_score:,.2f}"
    )


with risk_col4:

    if high_risk == 1:
        st.metric(
            "High Risk Flag",
            "HIGH RISK"
        )

    else:
        st.metric(
            "High Risk Flag",
            "NORMAL"
        )


st.divider()


# =========================================================
# CREATE RAW DATAFRAME
# =========================================================

input_data = pd.DataFrame({

    "tenure_months": [tenure_months],

    "monthly_charges": [monthly_charges],

    "total_charges": [total_charges],

    "satisfaction_score": [satisfaction_score],

    "num_support_tickets": [num_support_tickets],

    "tech_support": [tech_support_value],

    "streaming_service": [streaming_service_value],

    "paperless_billing": [paperless_billing_value],

    "contract_type": [contract_type],

    "internet_service": [internet_service],

    "payment_method": [payment_method],

    "avg_monthly_value": [avg_monthly_value],

    "tenure_group": [tenure_group],

    "high_risk": [high_risk],

    "charges_per_ticket": [charges_per_ticket],

    "engagement_score": [engagement_score]
})


# =========================================================
# ONE-HOT ENCODING
# =========================================================

input_data = pd.get_dummies(
    input_data,
    columns=[
        "contract_type",
        "internet_service",
        "payment_method",
        "tenure_group"
    ]
)


# =========================================================
# ALIGN FEATURES WITH TRAINING DATA
# =========================================================

# Make sure the prediction dataframe has exactly
# the same columns used during model training.

input_data = input_data.reindex(
    columns=feature_columns,
    fill_value=0
)


# =========================================================
# SCALE NUMERICAL FEATURES
# =========================================================

num_cols = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "satisfaction_score",
    "num_support_tickets",
    "avg_monthly_value",
    "charges_per_ticket",
    "engagement_score"
]


input_data[num_cols] = scaler.transform(
    input_data[num_cols]
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.subheader("🔮 Churn Prediction")

predict_button = st.button(
    "Predict Customer Churn",
    type="primary",
    use_container_width=True
)


# =========================================================
# MAKE PREDICTION
# =========================================================

if predict_button:

    prediction = model.predict(input_data)[0]

    # Probability of churn
    probability = model.predict_proba(input_data)[0][1]

    churn_probability = probability * 100

    st.divider()

    # =====================================================
    # CHURN RESULT
    # =====================================================

    if prediction == 1:

        st.error(
            "⚠️ HIGH CHURN RISK"
        )

        st.subheader(
            "This customer is predicted to churn."
        )

        st.write(
            f"Estimated probability of churn: "
            f"*{churn_probability:.2f}%*"
        )

        st.warning(
            """
            *Recommended business action:*

            Consider contacting the customer with a targeted
            retention strategy. Possible actions include improving
            customer support, offering a suitable plan or providing
            a personalized retention incentive.
            """
        )

    else:

        st.success(
            "✅ LOW CHURN RISK"
        )

        st.subheader(
            "This customer is predicted to remain."
        )

        st.write(
            f"Estimated probability of churn: "
            f"*{churn_probability:.2f}%*"
        )

        st.info(
            """
            *Recommended business action:*

            Continue providing good service and monitor the
            customer's satisfaction and engagement over time.
            """
        )


    # =====================================================
    # PROBABILITY VISUALIZATION
    # =====================================================

    st.subheader("📊 Prediction Probability")

    probability_data = pd.DataFrame(
        {
            "Outcome": [
                "Remain",
                "Churn"
            ],
            "Probability": [
                1 - probability,
                probability
            ]
        }
    )

    probability_data["Probability"] = (
        probability_data["Probability"] * 100
    )

    st.bar_chart(
        probability_data.set_index("Outcome")
    )


    # =====================================================
    # CUSTOMER SUMMARY
    # =====================================================

    st.subheader("📋 Customer Summary")

    summary_col1, summary_col2 = st.columns(2)


    with summary_col1:

        st.write("*Customer Profile*")

        st.write(
            f"""
            - *Tenure:* {tenure_months} months
            - *Contract:* {contract_type}
            - *Internet Service:* {internet_service}
            - *Payment Method:* {payment_method}
            """
        )


    with summary_col2:

        st.write("*Customer Risk Information*")

        st.write(
            f"""
            - *Monthly Charges:* {monthly_charges:,.2f}
            - *Total Charges:* {total_charges:,.2f}
            - *Satisfaction Score:* {satisfaction_score}/5
            - *Support Tickets:* {num_support_tickets}
            - *High Risk Flag:* {"Yes" if high_risk == 1 else "No"}
            """
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Telecom Customer Churn Prediction System | "
    "Machine Learning Classification Application"
)