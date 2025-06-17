# ---------------------------------------------------------------
#  Open-Heart Surgery Dashboard  (single-file version)
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path  
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config("Open-Heart Surgery Dashboard", "❤️", layout="wide")

# ---------- 1. Load the cleaned public dataset ------------------
@st.cache_data
@st.cache_data
def load_data():
    # file lives in the same directory as Home.py
    data_path = Path(__file__).parent / "heart disease.xlsx"
    return pd.read_excel(data_path)


df = load_data()

# ---------- 2. Simple logistic-reg model (train on the fly) -----
@st.cache_resource
def train_model(data: pd.DataFrame):
    htn = data[data["hypertension"] == 1]
    X = pd.get_dummies(htn[["age", "sex", "smoker"]], drop_first=True)
    y = htn["postop_bleeding"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    auc = clf.score(X_test, y_test)  # quick proxy
    return clf, auc

model, auc = train_model(df)

# ---------- 3. Sidebar navigation --------------------------------
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Demographics", "Bleeding predictor"],
    help="Choose a section of the dashboard",
)

# ---------- 4. Pages ---------------------------------------------
if page == "Overview":
    st.title("Open-Heart Surgery – Cohort Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total cases", len(df))
    col2.metric("Female %", f"{100*(df['sex']=='F').mean():.1f}%")
    col3.metric("Smokers %", f"{100*df['smoker'].mean():.1f}%")
    col4.metric("Hypertensive %", f"{100*df['hypertension'].mean():.1f}%")

    st.markdown("#### Surgeries per year")
    st.plotly_chart(
        px.histogram(
            df, x="surgery_year", nbins=len(df["surgery_year"].unique()),
            labels={"surgery_year": "Year", "count": "Cases"}
        ),
        use_container_width=True,
    )

elif page == "Demographics":
    st.title("Demographic breakdown")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Sex", "Age bands", "Residence (top 15)", "Smoking"]
    )

    with tab1:
        st.plotly_chart(
            px.histogram(df, x="sex", color="sex", text_auto=True),
            use_container_width=True,
        )

    with tab2:
        order = ["<40","40-49","50-59","60-69","70-79","80+"]
        st.plotly_chart(
            px.histogram(df, x="age_band", category_orders={"age_band": order}),
            use_container_width=True,
        )

    with tab3:
        top15 = df["residence"].value_counts().nlargest(15).reset_index()
        top15.columns = ["residence", "count"]
        st.plotly_chart(
            px.bar(top15, x="residence", y="count", text="count"),
            use_container_width=True,
        )

    with tab4:
        st.plotly_chart(
            px.histogram(df, x="smoker", color="smoker", text_auto=True),
            use_container_width=True,
        )

elif page == "Bleeding predictor":
    st.title("🩸 Predict post-operative bleeding\n*(hypertensive patients only)*")

    with st.form("predict_form"):
        sex    = st.selectbox("Sex", ["M", "F"])
        age    = st.slider("Age", 18, 100, 60)
        smoker = st.selectbox("Smoker", ["no", "yes"])
        submitted = st.form_submit_button("Estimate risk")

    if submitted:
        row = pd.DataFrame(
            {
                "age": [age],
                "sex_M": [1 if sex == "M" else 0],  # matches get_dummies
                "smoker": [1 if smoker == "yes" else 0],
            }
        )
        prob = model.predict_proba(row)[0, 1]
        st.metric("Predicted bleeding probability", f"{prob*100:.1f}%")
        st.caption(f"Model AUC on hold-out set: **{auc:.2f}**  "
                   "(logistic regression with age, sex, smoker)")

st.sidebar.markdown("---")
st.sidebar.caption("Data source: heart disease.xlsx  •  "
                   "Dashboard built with Streamlit")
