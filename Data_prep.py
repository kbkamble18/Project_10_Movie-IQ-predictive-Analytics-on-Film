import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

Path("assets").mkdir(exist_ok=True)
df = pd.read_csv("movies_clean.csv")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# Exploratory Data Analysis

# 1. Load
df = pd.read_csv("movies.csv")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(df[["budget", "revenue", "popularity", "runtime", "vote_average"]].describe())

# 2. Missing / zero handling
print("\nMissing values:")
print(df.isnull().sum())
print("\nZero budget:", (df.budget == 0).sum())
print("Zero revenue:", (df.revenue == 0).sum())

# Zeros almost always mean “unknown / not reported”. Keeping them would create false failures.
df = df[(df.budget > 0) & (df.revenue > 0)].copy()
df = df.dropna(subset=["runtime", "vote_average", "popularity"])
print(f"\nAfter cleaning: {df.shape[0]} rows")

# 3. Target
df["success"] = (df.revenue > df.budget).astype(int)
print(f"\nSuccess rate: {df.success.mean():.1%}")
print("Class balance:\n", df.success.value_counts(normalize=True))

# 4. Genres processing (TMDB stores JSON-like strings)
def extract_genres(x):
    try:
        return [g["name"] for g in ast.literal_eval(x)]
    except:
        return []

df["genre_list"] = df["genres"].apply(extract_genres)
df["primary_genre"] = df["genre_list"].apply(lambda x: x[0] if x else "Unknown")

# Keep a clean working copy
df.to_csv("movies_clean.csv", index=False)
print("\nSaved movies_clean.csv")

df = pd.read_csv("movies_clean.csv")
df["genre_list"] = df["genre_list"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# 1. Budget vs Revenue
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="budget", y="revenue", hue="success", alpha=0.6)
plt.xscale("log"); plt.yscale("log")
plt.title("Budget vs Revenue (log scale)")
plt.savefig("assets/budget_vs_revenue.png", dpi=150, bbox_inches="tight")
plt.show()
# Observation: clear positive relationship – higher budgets tend to earn higher revenue,
# but many high-budget films still fail (points below the diagonal).

# 2. Genre trends
genre_counts = df.explode("genre_list")["genre_list"].value_counts().head(10)
genre_success = (df.explode("genre_list")
                   .groupby("genre_list")["success"]
                   .mean()
                   .sort_values(ascending=False)
                   .head(10))

fig, ax = plt.subplots(1,2, figsize=(14,5))
genre_counts.plot(kind="bar", ax=ax[0], color="steelblue")
ax[0].set_title("Most common genres")
genre_success.plot(kind="bar", ax=ax[1], color="seagreen")
ax[1].set_title("Highest success rate by genre")
plt.tight_layout()
plt.savefig("assets/genre_trends.png", dpi=150, bbox_inches="tight")
plt.show()

# 3. Popularity / runtime / vote_average vs success
fig, axes = plt.subplots(1,3, figsize=(15,4))
for ax, col in zip(axes, ["popularity", "runtime", "vote_average"]):
    sns.boxplot(data=df, x="success", y=col, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.savefig("assets/features_vs_success.png", dpi=150, bbox_inches="tight")
plt.show()
# vote_average and popularity show the strongest visual association with success.

# 4. Correlation heatmap
num_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "success"]
corr = df[num_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("assets/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
# Strong pairs: budget–revenue, popularity–revenue.  
# Concern for modelling: we must NOT use revenue as a feature (data leakage).

#Statistical Testing

from scipy import stats

# 1. T-Test (popularity)
success_pop = df.loc[df.success == 1, "popularity"]
fail_pop    = df.loc[df.success == 0, "popularity"]

t_stat, p_val = stats.ttest_ind(success_pop, fail_pop, equal_var=False)
print("\nT-Test on popularity")
print("H0: mean popularity is the same for successful and unsuccessful movies")
print(f"t = {t_stat:.3f}, p = {p_val:.4e}")
if p_val < 0.05:
    print("Conclusion: reject H0 (p < 0.05) → successful movies have significantly higher popularity")
else:
    print("Conclusion: fail to reject H0 → no significant difference in popularity")

# 2. Chi-Square (primary_genre) – FIXED CONCLUSION
contingency = pd.crosstab(df["primary_genre"], df["success"])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print("\nChi-Square test (primary_genre vs success)")
print("H0: genre and success are independent")
print(f"chi2 = {chi2:.1f}, p = {p:.4e}")
if p < 0.05:
    print("Conclusion: reject H0 → genre is associated with success")
else:
    print("Conclusion: fail to reject H0 → no significant association between genre and success")

# 3. What a p-value means
# A p-value is the probability of observing the data (or more extreme) if the null hypothesis is true.
# We used α = 0.05 (conventional 5 % significance level).

# Predictive Modeling (Random Forest)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import joblib

# 1. Features & target
features = ["budget", "popularity", "runtime", "vote_average"]
X = df[features]
y = df["success"]
# Excluded: title (text, unique), revenue (leakage), genres (would need multi-hot; primary_genre optional)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# 3. Train
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# How a Random Forest works (short explanation)
# It builds many decision trees on random subsets of rows and features.
# Each tree votes; the majority vote is the final prediction.
# This reduces overfitting compared with a single tree.

# 4. Evaluate
y_pred = rf.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)

print("Accuracy :", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred), 3))
print("Recall   :", round(recall_score(y_test, y_pred), 3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 5. Feature importance
imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importance:\n", imp)
imp.plot(kind="bar", title="Feature Importance")
plt.savefig("assets/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
# Typically budget & popularity dominate – agrees with EDA and t-test.

# Save model for the app
joblib.dump(rf, "rf_model.joblib")
print("Model saved → rf_model.joblib")

# ----------------------------------------------------------
# ------------ 12 EDA charts -------------------------------
# 1. Distribution of all numeric features
# ----------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
num_cols = ["budget", "revenue", "popularity", "runtime", "vote_average"]
for ax, col in zip(axes.flatten(), num_cols + [None]):
    if col is None:
        ax.axis("off")
        continue
    sns.histplot(df[col], kde=True, ax=ax, bins=40)
    ax.set_title(f"Distribution of {col}")
axes[1,2].axis("off")
plt.tight_layout()
plt.savefig("assets/01_distributions.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------
# 2. Log-scale distributions (budget & revenue are skewed)
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(np.log10(df["budget"]), kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("log10(Budget)")
sns.histplot(np.log10(df["revenue"]), kde=True, ax=axes[1], color="seagreen")
axes[1].set_title("log10(Revenue)")
plt.tight_layout()
plt.savefig("assets/02_log_distributions.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------
# 3. Success rate by Budget bins
# ----------------------------------------------------------
df["budget_bin"] = pd.qcut(df["budget"], q=5, labels=["Very Low", "Low", "Medium", "High", "Very High"])
budget_success = df.groupby("budget_bin", observed=True)["success"].mean()

plt.figure(figsize=(9, 5))
budget_success.plot(kind="bar", color="coral", edgecolor="black")
plt.title("Success Rate by Budget Quintile")
plt.ylabel("Success Rate")
plt.xticks(rotation=0)
plt.ylim(0, 1)
for i, v in enumerate(budget_success):
    plt.text(i, v + 0.02, f"{v:.1%}", ha="center")
plt.tight_layout()
plt.savefig("assets/03_success_by_budget.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------
# 4. Success rate by Vote Average bins
# ----------------------------------------------------------
df["vote_bin"] = pd.cut(df["vote_average"], bins=[0, 5, 6, 7, 8, 10],
                        labels=["≤5", "5-6", "6-7", "7-8", "8+"])
vote_success = df.groupby("vote_bin", observed=True)["success"].mean()

plt.figure(figsize=(8, 5))
vote_success.plot(kind="bar", color="mediumpurple", edgecolor="black")
plt.title("Success Rate by Vote Average")
plt.ylabel("Success Rate")
plt.xticks(rotation=0)
plt.ylim(0, 1)
for i, v in enumerate(vote_success):
    plt.text(i, v + 0.02, f"{v:.1%}", ha="center")
plt.tight_layout()
plt.savefig("assets/04_success_by_vote.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------
# 5. ROI (Revenue / Budget) analysis
# ----------------------------------------------------------
df["roi"] = df["revenue"] / df["budget"]
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="success", y="roi", showfliers=False)
plt.title("ROI (Revenue ÷ Budget) by Success")
plt.yscale("log")
plt.tight_layout()
plt.savefig("assets/05_roi_by_success.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------
# 6. Runtime categories vs Success
# ----------------------------------------------------------
df["runtime_cat"] = pd.cut(df["runtime"], bins=[0, 90, 120, 150, 300],
                           labels=["≤90 min", "90-120", "120-150", "150+"])
rt_success = df.groupby("runtime_cat", observed=True)["success"].mean()

plt.figure(figsize=(8, 5))
rt_success.plot(kind="bar", color="teal", edgecolor="black")
plt.title("Success Rate by Runtime Category")
plt.ylabel("Success Rate")
plt.xticks(rotation=0)
plt.ylim(0, 1)
for i, v in enumerate(rt_success):
    plt.text(i, v + 0.02, f"{v:.1%}", ha="center")
plt.tight_layout()
plt.savefig("assets/06_success_by_runtime.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------
# 7. Popularity vs Budget colored by Success
# ----------------------------------------------------------
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x="budget", y="popularity", hue="success",
                alpha=0.6, palette={0: "steelblue", 1: "darkorange"})
plt.xscale("log")
plt.title("Popularity vs Budget (colored by Success)")
plt.tight_layout()
plt.savefig("assets/07_popularity_vs_budget.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------
# 8. Top 10 genres by volume + success rate (dual view)
# ----------------------------------------------------------
genre_stats = (df.explode("genre_list")
                 .groupby("genre_list")
                 .agg(count=("success", "size"),
                      success_rate=("success", "mean"))
                 .sort_values("count", ascending=False)
                 .head(12))

fig, ax1 = plt.subplots(figsize=(11, 6))
ax2 = ax1.twinx()
genre_stats["count"].plot(kind="bar", ax=ax1, color="steelblue", alpha=0.7, width=0.4, position=1)
genre_stats["success_rate"].plot(kind="bar", ax=ax2, color="seagreen", alpha=0.7, width=0.4, position=0)
ax1.set_ylabel("Number of Movies")
ax2.set_ylabel("Success Rate")
ax1.set_title("Top Genres – Volume vs Success Rate")
ax1.set_xticklabels(genre_stats.index, rotation=45, ha="right")
plt.tight_layout()
plt.savefig("assets/08_genre_volume_success.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------
# 9. Violin plots – deeper look at distributions by success
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, ["popularity", "runtime", "vote_average"]):
    sns.violinplot(data=df, x="success", y=col, ax=ax, inner="quartile")
    ax.set_title(col)
plt.suptitle("Feature Distributions by Success (Violin)", y=1.02)
plt.tight_layout()
plt.savefig("assets/09_violin_plots.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------
# 10. High-budget flops vs Low-budget hits
# ----------------------------------------------------------
high_budget_flops = df[(df["budget"] > df["budget"].quantile(0.75)) & (df["success"] == 0)]
low_budget_hits  = df[(df["budget"] < df["budget"].quantile(0.25)) & (df["success"] == 1)]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.scatterplot(data=high_budget_flops, x="budget", y="revenue", ax=axes[0], color="crimson", alpha=0.7)
axes[0].set_title(f"High-Budget Flops (n={len(high_budget_flops)})")
axes[0].set_xscale("log"); axes[0].set_yscale("log")

sns.scatterplot(data=low_budget_hits, x="budget", y="revenue", ax=axes[1], color="green", alpha=0.7)
axes[1].set_title(f"Low-Budget Hits (n={len(low_budget_hits)})")
axes[1].set_xscale("log"); axes[1].set_yscale("log")
plt.tight_layout()
plt.savefig("assets/10_flops_vs_hits.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------
# 11. Pairplot of key features (sample for speed)
# ----------------------------------------------------------
sample = df.sample(min(800, len(df)), random_state=42)
sns.pairplot(sample, vars=["budget", "popularity", "runtime", "vote_average"],
             hue="success", corner=True, plot_kws={"alpha": 0.5, "s": 20})
plt.savefig("assets/11_pairplot.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------
# 12. Success rate heatmap – Vote Average × Budget bins
# ----------------------------------------------------------
pivot = df.pivot_table(values="success", index="vote_bin", columns="budget_bin",
                       aggfunc="mean", observed=True)
plt.figure(figsize=(9, 6))
sns.heatmap(pivot, annot=True, fmt=".0%", cmap="YlGnBu", cbar_kws={"label": "Success Rate"})
plt.title("Success Rate: Vote Average × Budget Level")
plt.tight_layout()
plt.savefig("assets/12_heatmap_vote_budget.png", bbox_inches="tight")
plt.close()

print("✅ All 12 EDA charts saved to assets/")