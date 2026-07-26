import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="MovieIQ | Predictive Analytics on Film Success",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Load Data & Model
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("movies_clean.csv")

@st.cache_resource
def load_model():
    return joblib.load("rf_model.joblib")

df = load_data()
model = load_model()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("🎬 MovieIQ Filters")
st.sidebar.markdown("---")

all_genres = sorted(df["primary_genre"].dropna().unique())
selected_genres = st.sidebar.multiselect(
    "Select Genres",
    options=all_genres,
    default=all_genres[:6]
)

min_vote = st.sidebar.slider("Minimum Vote Average", 0.0, 10.0, 5.0, 0.1)

filtered = df[
    (df["primary_genre"].isin(selected_genres)) &
    (df["vote_average"] >= min_vote)
].copy()

st.sidebar.markdown(f"**Movies shown:** `{len(filtered)}`")
st.sidebar.markdown("---")
st.sidebar.info("A movie is considered **successful** when Revenue > Budget.")

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🎬 MovieIQ — Predictive Analytics on Film Success")
st.markdown("Predict whether a movie will be commercially successful using budget, popularity, runtime and ratings.")
st.markdown("---")

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview & Insights",
    "📈 Exploratory Analysis",
    "🧪 Statistical Tests",
    "🤖 Model Performance",
    "🎯 Predict Success",
    "📝 Conclusion & Recommendations"
])

# ====================== TAB 1: OVERVIEW ======================
with tab1:
    st.subheader("Project Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Movies", f"{len(df):,}")
    col2.metric("Success Rate", f"{df['success'].mean():.1%}")
    col3.metric("Avg Budget", f"${df['budget'].mean()/1e6:.1f}M")
    col4.metric("Avg Revenue", f"${df['revenue'].mean()/1e6:.1f}M")

    st.markdown("### Key Insights from the Data")
    st.markdown("""
    - Higher budgets generally lead to higher revenue, but many high-budget films still fail.
    - **Popularity** and **Vote Average** show the strongest relationship with commercial success.
    - Certain genres (Animation, Adventure, Science Fiction) tend to have higher success rates.
    - Low-budget hits and high-budget flops both exist — budget alone is not enough.
    - The Random Forest model uses only pre-release features (no revenue leakage).
    """)

    st.image("assets/budget_vs_revenue.png", caption="Budget vs Revenue (Success colored)", width="stretch")

# ====================== TAB 2: EDA ======================
with tab2:
    st.subheader("Exploratory Data Analysis")
    st.markdown("Each section below shows the chart **and** the key insight derived from it.")

    # ---------- 1. Core Relationships ----------
    st.markdown("### 1. Core Relationships")
    c1, c2 = st.columns(2)
    with c1:
        st.image("assets/budget_vs_revenue.png", caption="Budget vs Revenue", width="stretch")
    with c2:
        st.image("assets/07_popularity_vs_budget.png", caption="Popularity vs Budget", width="stretch")

    st.info("""
    **Insights**
    - There is a clear positive relationship between budget and revenue (higher budgets tend to generate higher revenue).
    - However, many high-budget films still fall below the break-even line (blue points) → budget alone does not guarantee success.
    - Popularity shows a weaker but visible positive trend with budget.
    """)

    st.image("assets/correlation_heatmap.png", caption="Correlation Heatmap", width="stretch")
    st.info("""
    **Insights from Correlation**
    - Strongest correlation is between **Budget and Revenue** (0.76) — expected.
    - Success has moderate correlation with Revenue (0.37) because success is defined from it.
    - Popularity, runtime and vote_average have very weak linear correlation with success → non-linear relationships may exist (this is why Random Forest is useful).
    - **Important**: We must never use Revenue as a feature (data leakage).
    """)

    # ---------- 2. Distributions ----------
    st.markdown("### 2. Feature Distributions")
    st.image("assets/01_distributions.png", width="stretch")
    st.image("assets/02_log_distributions.png", width="stretch")
    st.info("""
    **Insights**
    - Budget and Revenue are heavily right-skewed → most movies have moderate budgets/revenues, while a few blockbusters pull the mean up.
    - Log transformation makes the distributions more normal and easier to interpret.
    - Runtime is roughly normal (centered around 120–130 minutes).
    - Vote average is slightly left-skewed with most movies rated between 5 and 8.
    """)

    # ---------- 3. What Drives Success ----------
    st.markdown("### 3. What Drives Success?")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image("assets/03_success_by_budget.png", width="stretch")
    with c2:
        st.image("assets/04_success_by_vote.png", width="stretch")
    with c3:
        st.image("assets/06_success_by_runtime.png", width="stretch")

    st.image("assets/12_heatmap_vote_budget.png", caption="Success Rate: Vote Average × Budget Level", width="stretch")
    st.info("""
    **Insights**
    - Success rate stays high (≈78–83%) across all budget levels → even low-budget films can succeed in this dataset.
    - Vote average does not show a strong monotonic increase in success rate.
    - Runtime has almost no impact on success rate.
    - The heatmap confirms that combinations of budget and vote average do not create dramatically different success probabilities.
    """)

    # ---------- 4. Genre Deep Dive ----------
    st.markdown("### 4. Genre Deep Dive")
    st.image("assets/genre_trends.png", width="stretch")
    st.image("assets/08_genre_volume_success.png", width="stretch")
    st.info("""
    **Insights**
    - Romance, Adventure, Science Fiction, Animation and Comedy are the most common genres.
    - Success rates across top genres are surprisingly similar (mostly 75–82%).
    - No single genre dominates commercial success in this dataset → genre alone is a weak predictor (confirmed by the Chi-Square test).
    """)

    # ---------- 5. Distributional Comparison ----------
    st.markdown("### 5. Distributional Comparison (Success vs Failure)")
    st.image("assets/features_vs_success.png", width="stretch")
    st.image("assets/09_violin_plots.png", width="stretch")
    st.info("""
    **Insights**
    - Successful movies tend to have slightly higher popularity and vote average (visible in the box/violin plots).
    - Runtime distributions of successful and unsuccessful movies largely overlap → runtime is not a strong differentiator.
    - The violin plots show that the difference in popularity is more pronounced than the difference in vote average.
    """)

    # ---------- 6. Extremes & ROI ----------
    st.markdown("### 6. Extremes & ROI")
    st.image("assets/10_flops_vs_hits.png", width="stretch")
    st.image("assets/05_roi_by_success.png", width="stretch")
    st.info("""
    **Insights**
    - High-budget flops exist (109 movies) — even large budgets can lose money.
    - Low-budget hits are more common (403 movies) — efficient films can deliver strong returns.
    - Successful movies show clearly higher ROI (Revenue ÷ Budget). The median ROI for successful films is well above 1.0, while failed films stay below 1.0.
    """)

    # ---------- 7. Multivariate ----------
    st.markdown("### 7. Multivariate View")
    st.image("assets/11_pairplot.png", width="stretch")
    st.info("""
    **Insights**
    - No single pair of features cleanly separates successful from unsuccessful movies.
    - This supports the use of a non-linear model (Random Forest) that can capture complex interactions between budget, popularity, runtime and vote average.
    """)

# ====================== TAB 3: STATISTICAL TESTS ======================
with tab3:
    st.subheader("Statistical Hypothesis Testing")

    st.markdown("""
    **1. Independent T-Test (Popularity)**  
    - **H₀**: Mean popularity is the same for successful and unsuccessful movies.  
    - **Result**: t = 2.062, p = 0.0397  
    - **Conclusion**: Reject H₀ (p < 0.05) → successful movies have significantly higher popularity.

    **2. Chi-Square Test (Primary Genre)**  
    - **H₀**: Genre and success are independent.  
    - **Result**: χ² = 1.8, p = 0.995  
    - **Conclusion**: **Fail to reject H₀** → no statistically significant association between primary genre and success in this dataset.

    **Interpretation of p-value**  
    A p-value is the probability of observing the data (or more extreme) if the null hypothesis is true.  
    We used the conventional significance level **α = 0.05**.
    """)

# ====================== TAB 4: MODEL PERFORMANCE ======================
with tab4:
    st.subheader("Random Forest Model Performance")

    st.markdown("""
    **Features used**: `budget`, `popularity`, `runtime`, `vote_average`  
    **Excluded**: `title` (unique), `revenue` (data leakage)  
    **Train/Test split**: 80/20 stratified  
    **Class weight**: balanced (handles 80.7% success rate)
    """)

    # Real metrics from your last run
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", "0.645")
    m2.metric("Precision", "0.811")
    m3.metric("Recall", "0.731")

    st.markdown("""
    **Confusion Matrix (Test set)**  
    - True Negatives (correct failures): 22  
    - False Positives: 55  
    - False Negatives: 87  
    - True Positives (correct successes): 236  
    """)

    st.image("assets/feature_importance.png", caption="Feature Importance", width="stretch")

    st.info("Feature importance ranks **Popularity ≈ Budget ≈ Vote Average** as the strongest predictors. "
            "The model is decent at detecting successes but weaker at detecting failures due to class imbalance.")
    
# ====================== TAB 5: PREDICT ======================
with tab5:
    st.subheader("Will this movie succeed?")
    st.markdown("Enter the planned characteristics of a new film:")

    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input("Budget (USD)", min_value=1_000_000, value=40_000_000, step=1_000_000)
        popularity = st.number_input("Expected Popularity (TMDB scale)", min_value=0.0, value=25.0, step=1.0)
    with col2:
        runtime = st.number_input("Runtime (minutes)", min_value=60, value=115, step=5)
        vote_avg = st.number_input("Expected Vote Average (0–10)", min_value=0.0, max_value=10.0, value=6.8, step=0.1)

    if st.button("🚀 Predict Success", type="primary", width="stretch"):
        X_new = pd.DataFrame(
            [[budget, popularity, runtime, vote_avg]],
            columns=["budget", "popularity", "runtime", "vote_average"]
        )
        pred = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0][1]

        st.markdown("---")
        if pred == 1:
            st.success(f"✅ **SUCCESS** predicted  \nProbability of success: **{proba:.1%}**")
        else:
            st.error(f"❌ **NOT SUCCESS** predicted  \nProbability of success: only **{proba:.1%}**")

        st.progress(proba)
        st.caption("Model is a Random Forest trained on historical TMDB data. Results are probabilistic estimates, not guarantees.")

# ====================== TAB 6: CONCLUSION ======================
with tab6:
    st.subheader("Conclusion & Recommendations")

    st.markdown("### Overall Conclusion")
    st.markdown("""
    MovieIQ successfully demonstrates an end-to-end predictive analytics workflow on film success.  
    Using only four pre-release features (budget, popularity, runtime, vote average), the Random Forest model achieves reasonable performance, with popularity and budget emerging as the most important drivers.

    However, the high success rate in the dataset (80.7%) and the relatively weak statistical association of genre with success indicate that commercial success is influenced by many factors not captured here (marketing spend, star power, release timing, competition, etc.).
    """)

    st.markdown("### Key Findings")
    st.markdown("""
    1. **Budget and Revenue** are strongly related, but high budget does not guarantee success.
    2. **Popularity** is the strongest single predictor among the features we used.
    3. **Genre** shows no statistically significant association with success in this dataset.
    4. The model is better at identifying successful movies than failures (due to class imbalance).
    5. Low-budget hits and high-budget flops both exist — efficiency and audience reception matter.
    """)

    st.markdown("### Recommendations for a Studio")
    st.markdown("""
    - Treat MovieIQ’s prediction as a **directional signal**, not a final decision.
    - Focus more on building audience interest (popularity) early than on simply increasing budget.
    - Combine this model with additional data: marketing budget, cast strength, release window, and competitor analysis.
    - Use the ROI view to identify efficient (high return) projects rather than only chasing high absolute revenue.
    """)

    st.markdown("### Limitations & Future Improvements")
    st.markdown("""
    **Limitation**  
    The model only uses four numeric features and ignores important drivers such as marketing spend, star power, release date competition, and script quality.

    **Improvements with more time**
    - Add multi-hot encoding for genres and cast/crew features.
    - Collect external data (marketing budget, social media buzz).
    - Use a temporal train-test split (train on older films, test on recent ones).
    - Experiment with more advanced models or cost-sensitive learning to better handle the minority class (flops).
    """)

    st.success("If a studio asked “Will our next film succeed?”, I would treat MovieIQ’s answer as a useful but incomplete signal and recommend combining it with domain expertise and additional data sources.")
# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("MovieIQ • Student Project • Data source: TMDB 5000 Movies • Random Forest Classifier")