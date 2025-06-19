import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Open Heart Surgery Dashboard",
    layout="wide",
)
st.sidebar.header("Filters – Open Heart Surgery")

# ── Load & preprocess data ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # adjust the file name/extension as needed
    df = pd.read_excel("heart disease.xlsx")  
    # keep only the two surgery reasons
    df = df[df["Surgery"].isin(["cardiovascular disease", "valvular disease"])]
    # parse to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

df = load_data()

# ── Sidebar widgets ────────────────────────────────────────────────────────────
# convert to native Python dates for st.date_input
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

res_choices = st.sidebar.multiselect(
    "Residence",
    options=sorted(df["Residence"].dropna().unique()),
    default=sorted(df["Residence"].dropna().unique()),
)

# ── Filter dataframe ───────────────────────────────────────────────────────────
df_f = df[
    (df["Date"].dt.date >= start_date) &
    (df["Date"].dt.date <= end_date) &
    (df["Residence"].isin(res_choices))
].copy()

# ── Title & KPIs ───────────────────────────────────────────────────────────────
st.title("🚑 Open-Heart Surgery Cohort Dashboard")
c1, c2, c3 = st.columns(3)
c1.metric("Patients", f"{len(df_f):,}")
c2.metric("Smokers (%)", f"{df_f['Smoker'].mean() * 100:.1f}")
c3.metric("HTN (%)",      f"{df_f['HTN'].mean() * 100:.1f}")

# ── 2×2 Grid of Distributions ─────────────────────────────────────────────────
r1c1, r1c2 = st.columns(2)
with r1c1:
    fig = px.histogram(
        df_f, x="Sex", color="Sex", title="Gender"
    )
    fig.update_layout(height=300, margin=dict(t=30, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with r1c2:
    fig = px.pie(
        df_f, names="Smoker", hole=0.4, title="Smoker vs. Non-Smoker"
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=300, margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

r2c1, r2c2 = st.columns(2)
with r2c1:
    fig = px.box(
        df_f, y="Age", title="Age Distribution"
    )
    fig.update_layout(height=300, margin=dict(t=30, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with r2c2:
    counts = df_f["Residence"].value_counts().reset_index()
    counts.columns = ["Residence", "Count"]
    fig = px.bar(
        counts, x="Residence", y="Count", title="By Residence"
    )
    fig.update_layout(
        height=300,
        margin=dict(t=30, b=10, l=10, r=10),
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Predict Bleeding (HTN → Bleeding) ──────────────────────────────────────────
st.subheader("Predict Bleeding Post-Surgery (HTN → Bleeding)")

# prepare features & target
X = df_f[["HTN"]].astype(int)
y = df_f["Bleeding"].astype(int)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)
model = LogisticRegression(solver="liblinear")
model.fit(X_train, y_train)

# predictions & metrics
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
acc     = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc     = auc(fpr, tpr)

# display metrics
st.markdown(f"**Accuracy:** {acc:.2f} &nbsp;&nbsp; **AUC:** {roc_auc:.2f}")

# ROC curve
fig_roc = px.area(
    x=fpr, y=tpr,
    title="ROC Curve",
    labels={"x": "False Positive Rate", "y": "True Positive Rate"},
)
fig_roc.add_shape(
    type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1
)
fig_roc.update_layout(height=300, margin=dict(t=30, b=10, l=10, r=10), showlegend=False)
st.plotly_chart(fig_roc, use_container_width=True)
