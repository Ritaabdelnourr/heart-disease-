import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Open Heart Surgery Dashboard",
    layout="wide",
)
st.sidebar.header("Filters – Open Heart Surgery")

# ── Load & preprocess ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("heart disease.xlsx")  # or .csv if you converted
    # Only keep those who had open-heart surgery for the two reasons
    df = df[df["Surgery"].isin(["cardiovascular disease", "valvular disease"])]
    # Ensure date is datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

df = load_data()

# ── Sidebar widgets ──────────────────────────────────────────────────────────
min_date, max_date = st.sidebar.date_input(
    "Date range",
    value=(df["Date"].min(), df["Date"].max()),
)
res_choices = st.sidebar.multiselect(
    "Residence",
    options=sorted(df["Residence"].unique()),
    default=sorted(df["Residence"].unique()),
)

# ── Filtered frame ────────────────────────────────────────────────────────────
df_f = df[
    (df["Date"] >= pd.to_datetime(min_date)) &
    (df["Date"] <= pd.to_datetime(max_date)) &
    (df["Residence"].isin(res_choices))
].copy()

# ── Top KPI row ───────────────────────────────────────────────────────────────
st.title("🚑 Open-Heart Surgery Cohort Dashboard")
c1, c2, c3 = st.columns(3)
c1.metric("Patients", f"{len(df_f):,}")
c2.metric("Smokers (%)", f"{df_f['Smoker'].mean()*100:.1f}")
c3.metric("HTN (%)",      f"{df_f['HTN'].mean()*100:.1f}")

# ── 2×2 grid of distributions ─────────────────────────────────────────────────
r1c1, r1c2 = st.columns(2)
with r1c1:
    # 1. Gender split
    fig = px.histogram(
        df_f, x="Sex", color="Sex",
        title="Gender"
    )
    fig.update_layout(height=300, margin=dict(t=30,b=10,l=10,r=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with r1c2:
    # 2. Smoker vs Non-smoker
    fig = px.pie(
        df_f, names="Smoker",
        title="Smoker vs. Non-Smoker",
        hole=0.4
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=300, margin=dict(t=30,b=10,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)

r2c1, r2c2 = st.columns(2)
with r2c1:
    # 3. Age distribution
    fig = px.box(
        df_f, y="Age",
        title="Age Distribution"
    )
    fig.update_layout(height=300, margin=dict(t=30,b=10,l=10,r=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with r2c2:
    # 4. Residence breakdown
    fig = px.bar(
        df_f["Residence"].value_counts().reset_index().rename(columns={"index":"Residence","Residence":"Count"}),
        x="Residence", y="Count",
        title="By Residence"
    )
    fig.update_layout(height=300, margin=dict(t=30,b=10,l=10,r=10), xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ── Prediction panel ──────────────────────────────────────────────────────────
st.subheader("Predict Bleeding Post-Surgery (HTN → Bleeding)")

# Prepare data
X = df_f[["HTN"]].astype(int)     # feature
y = df_f["Bleeding"].astype(int)  # target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model = LogisticRegression(solver="liblinear").fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Metrics
acc = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Display
st.markdown(f"**Accuracy:** {acc:.2f} &nbsp;&nbsp;&nbsp; **AUC:** {roc_auc:.2f}")

fig_roc = px.area(
    x=fpr, y=tpr,
    title="ROC Curve",
    labels={"x":"False Positive Rate", "y":"True Positive Rate"},
)
fig_roc.add_shape(
    type="line", line=dict(dash="dash"),
    x0=0, x1=1, y0=0, y1=1
)
fig_roc.update_layout(height=300, margin=dict(t=30,b=10,l=10,r=10), showlegend=False)
st.plotly_chart(fig_roc, use_container_width=True)
