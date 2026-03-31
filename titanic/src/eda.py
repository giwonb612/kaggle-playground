"""
EDA chart functions — each returns a plotly Figure.
Usable in Jupyter notebooks and the static HTML report.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

PALETTE = {"survived": "#38bdf8", "died": "#f87171", "neutral": "#a78bfa"}
DARK_BG = "#0f172a"
CARD_BG = "#1e293b"
TEXT_COLOR = "#e2e8f0"
GRID_COLOR = "#334155"

LAYOUT_BASE = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT_COLOR, family="Segoe UI, sans-serif"),
    margin=dict(t=60, b=40, l=40, r=20),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

AXIS_BASE = dict(
    gridcolor=GRID_COLOR,
    zerolinecolor=GRID_COLOR,
    showgrid=True,
)


def _apply_layout(fig, title="", height=420):
    fig.update_layout(title=dict(text=title, font=dict(size=16, color="#38bdf8")),
                      height=height, **LAYOUT_BASE)
    fig.update_xaxes(**AXIS_BASE)
    fig.update_yaxes(**AXIS_BASE)
    return fig


def _extract_title(name: str) -> str:
    title = name.split(",")[1].split(".")[0].strip()
    rare = {"Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
            "Rev", "Sir", "Jonkheer", "Dona"}
    replacements = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    title = replacements.get(title, title)
    return "Rare" if title in rare else title


# ── 1. Survival Overview ────────────────────────────────────────────────────
def plot_survival_overview(df: pd.DataFrame) -> go.Figure:
    counts = df["Survived"].value_counts().sort_index()
    labels = ["Died", "Survived"]
    colors = [PALETTE["died"], PALETTE["survived"]]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=counts.values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=DARK_BG, width=3)),
        textfont=dict(size=14),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    total = len(df)
    survived = int(counts.get(1, 0))
    fig.add_annotation(text=f"<b>{survived/total:.1%}</b><br>survived",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=16, color=PALETTE["survived"]))
    _apply_layout(fig, "Overall Survival Rate", height=380)
    fig.update_layout(showlegend=True)
    return fig


# ── 2. Survival by Pclass ────────────────────────────────────────────────────
def plot_survival_by_pclass(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby("Pclass")["Survived"].agg(["mean", "sum", "count"]).reset_index()
    grp.columns = ["Pclass", "Rate", "Survived", "Total"]
    grp["Pclass_Label"] = grp["Pclass"].map({1: "1st", 2: "2nd", 3: "3rd"})

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Survival Rate", "Passenger Count"],
                        horizontal_spacing=0.12)

    fig.add_trace(go.Bar(x=grp["Pclass_Label"], y=grp["Rate"],
                         marker_color=PALETTE["survived"], name="Rate",
                         text=[f"{r:.1%}" for r in grp["Rate"]],
                         textposition="outside",
                         hovertemplate="Pclass %{x}<br>Survival Rate: %{y:.1%}<extra></extra>"),
                  row=1, col=1)

    fig.add_trace(go.Bar(x=grp["Pclass_Label"], y=grp["Survived"],
                         name="Survived", marker_color=PALETTE["survived"],
                         hovertemplate="%{y} survived<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Bar(x=grp["Pclass_Label"], y=grp["Total"] - grp["Survived"],
                         name="Died", marker_color=PALETTE["died"],
                         hovertemplate="%{y} died<extra></extra>"), row=1, col=2)

    fig.update_layout(barmode="stack", **LAYOUT_BASE, height=400,
                      title=dict(text="Survival by Passenger Class",
                                 font=dict(size=16, color="#38bdf8")))
    fig.update_xaxes(**AXIS_BASE)
    fig.update_yaxes(**AXIS_BASE)
    return fig


# ── 3. Survival by Sex ───────────────────────────────────────────────────────
def plot_survival_by_sex(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby("Sex")["Survived"].agg(["mean", "sum", "count"]).reset_index()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Survival Rate by Sex", "Count by Sex"],
                        horizontal_spacing=0.12)

    colors = ["#f472b6", "#60a5fa"]
    for i, row in grp.iterrows():
        fig.add_trace(go.Bar(x=[row["Sex"].capitalize()], y=[row["mean"]],
                             marker_color=colors[i], name=row["Sex"].capitalize(),
                             text=[f"{row['mean']:.1%}"], textposition="outside",
                             showlegend=False,
                             hovertemplate=f"{row['Sex']}: {row['mean']:.1%}<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=[row["Sex"].capitalize()],
                             y=[row["sum"]], marker_color=PALETTE["survived"],
                             name="Survived", showlegend=(i == 0),
                             hovertemplate="%{y} survived<extra></extra>"), row=1, col=2)
        fig.add_trace(go.Bar(x=[row["Sex"].capitalize()],
                             y=[row["count"] - row["sum"]], marker_color=PALETTE["died"],
                             name="Died", showlegend=(i == 0),
                             hovertemplate="%{y} died<extra></extra>"), row=1, col=2)

    fig.update_layout(barmode="group", **LAYOUT_BASE, height=400,
                      title=dict(text="Survival by Sex (Women & Children First)",
                                 font=dict(size=16, color="#38bdf8")))
    fig.update_xaxes(**AXIS_BASE)
    fig.update_yaxes(**AXIS_BASE)
    return fig


# ── 4. Age Distribution ──────────────────────────────────────────────────────
def plot_age_distribution(df: pd.DataFrame) -> go.Figure:
    survived = df[df["Survived"] == 1]["Age"].dropna()
    died = df[df["Survived"] == 0]["Age"].dropna()

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=died, name="Died", nbinsx=30,
                               marker_color=PALETTE["died"], opacity=0.7,
                               hovertemplate="Age: %{x}<br>Count: %{y}<extra></extra>"))
    fig.add_trace(go.Histogram(x=survived, name="Survived", nbinsx=30,
                               marker_color=PALETTE["survived"], opacity=0.7,
                               hovertemplate="Age: %{x}<br>Count: %{y}<extra></extra>"))
    fig.add_vline(x=16, line_dash="dash", line_color="#fbbf24",
                  annotation_text="Age 16", annotation_font_color="#fbbf24")
    fig.update_layout(barmode="overlay", **LAYOUT_BASE, height=420,
                      title=dict(text="Age Distribution by Survival",
                                 font=dict(size=16, color="#38bdf8")))
    fig.update_xaxes(title_text="Age", **AXIS_BASE)
    fig.update_yaxes(title_text="Count", **AXIS_BASE)
    return fig


# ── 5. Survival by Age Group ─────────────────────────────────────────────────
def plot_survival_by_age_group(df: pd.DataFrame) -> go.Figure:
    tmp = df.copy()
    tmp["AgeGroup"] = pd.cut(tmp["Age"],
                              bins=[0, 5, 12, 17, 35, 60, 100],
                              labels=["Baby\n0-5", "Child\n6-12", "Teen\n13-17",
                                      "Young Adult\n18-35", "Adult\n36-60", "Senior\n60+"])
    grp = tmp.groupby("AgeGroup", observed=False)["Survived"].agg(["mean", "count"]).reset_index()

    fig = go.Figure(go.Bar(
        x=[str(g) for g in grp["AgeGroup"]],
        y=grp["mean"],
        text=[f"{r:.1%}" for r in grp["mean"]],
        textposition="outside",
        marker=dict(
            color=grp["mean"],
            colorscale=[[0, PALETTE["died"]], [1, PALETTE["survived"]]],
            showscale=False,
        ),
        customdata=grp["count"],
        hovertemplate="Age Group: %{x}<br>Survival Rate: %{y:.1%}<br>Count: %{customdata}<extra></extra>",
    ))
    _apply_layout(fig, "Survival Rate by Age Group", height=400)
    fig.update_yaxes(title_text="Survival Rate", tickformat=".0%")
    return fig


# ── 6. Fare Distribution ─────────────────────────────────────────────────────
def plot_fare_distribution(df: pd.DataFrame) -> go.Figure:
    survived = np.log1p(df[df["Survived"] == 1]["Fare"].dropna())
    died = np.log1p(df[df["Survived"] == 0]["Fare"].dropna())

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=died, name="Died", nbinsx=40,
                               marker_color=PALETTE["died"], opacity=0.7))
    fig.add_trace(go.Histogram(x=survived, name="Survived", nbinsx=40,
                               marker_color=PALETTE["survived"], opacity=0.7))
    fig.update_layout(barmode="overlay", **LAYOUT_BASE, height=420,
                      title=dict(text="Fare Distribution (log scale) by Survival",
                                 font=dict(size=16, color="#38bdf8")))
    fig.update_xaxes(title_text="log(Fare + 1)", **AXIS_BASE)
    fig.update_yaxes(title_text="Count", **AXIS_BASE)
    return fig


# ── 7. Family Size ───────────────────────────────────────────────────────────
def plot_family_size(df: pd.DataFrame) -> go.Figure:
    tmp = df.copy()
    tmp["FamilySize"] = tmp["SibSp"] + tmp["Parch"] + 1
    grp = tmp.groupby("FamilySize")["Survived"].agg(["mean", "count"]).reset_index()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Survival Rate by Family Size",
                                        "Passenger Count by Family Size"],
                        horizontal_spacing=0.12)

    fig.add_trace(go.Bar(x=grp["FamilySize"], y=grp["mean"],
                         marker_color=PALETTE["survived"],
                         text=[f"{r:.0%}" for r in grp["mean"]],
                         textposition="outside",
                         hovertemplate="Family Size %{x}<br>Survival: %{y:.1%}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=grp["FamilySize"], y=grp["count"],
                         marker_color=PALETTE["neutral"],
                         hovertemplate="Family Size %{x}<br>Count: %{y}<extra></extra>"),
                  row=1, col=2)

    fig.update_layout(**LAYOUT_BASE, height=400,
                      title=dict(text="Family Size Analysis",
                                 font=dict(size=16, color="#38bdf8")),
                      showlegend=False)
    fig.update_xaxes(**AXIS_BASE, dtick=1)
    fig.update_yaxes(**AXIS_BASE)
    return fig


# ── 8. Embarked Analysis ─────────────────────────────────────────────────────
def plot_embarked(df: pd.DataFrame) -> go.Figure:
    tmp = df.copy()
    tmp["Embarked"] = tmp["Embarked"].fillna("S")
    grp = tmp.groupby("Embarked")["Survived"].agg(["mean", "sum", "count"]).reset_index()
    grp["Port"] = grp["Embarked"].map({"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"})

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Survival Rate by Port", "Count by Port"],
                        horizontal_spacing=0.12)

    colors = ["#34d399", "#60a5fa", "#f59e0b"]
    fig.add_trace(go.Bar(x=grp["Port"], y=grp["mean"],
                         marker_color=colors,
                         text=[f"{r:.1%}" for r in grp["mean"]],
                         textposition="outside",
                         hovertemplate="%{x}<br>Survival: %{y:.1%}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=grp["Port"], y=grp["sum"],
                         marker_color=PALETTE["survived"], name="Survived",
                         hovertemplate="%{y} survived<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Bar(x=grp["Port"], y=grp["count"] - grp["sum"],
                         marker_color=PALETTE["died"], name="Died",
                         hovertemplate="%{y} died<extra></extra>"), row=1, col=2)

    fig.update_layout(barmode="stack", **LAYOUT_BASE, height=400,
                      title=dict(text="Embarkation Port Analysis",
                                 font=dict(size=16, color="#38bdf8")))
    fig.update_xaxes(**AXIS_BASE)
    fig.update_yaxes(**AXIS_BASE)
    return fig


# ── 9. Missing Values ────────────────────────────────────────────────────────
def plot_missing_values(df: pd.DataFrame) -> go.Figure:
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
    missing = missing[missing > 0]

    fig = go.Figure(go.Bar(
        x=missing.values, y=missing.index, orientation="h",
        marker=dict(
            color=missing.values,
            colorscale=[[0, "#fbbf24"], [1, "#ef4444"]],
            showscale=True,
            colorbar=dict(title="Missing %", ticksuffix="%"),
        ),
        text=[f"{v:.1f}%" for v in missing.values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}% missing<extra></extra>",
    ))
    _apply_layout(fig, "Missing Value Analysis", height=350)
    fig.update_xaxes(title_text="Missing Percentage (%)", **AXIS_BASE)
    return fig


# ── 10. Correlation Heatmap ──────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    tmp = df.copy()
    tmp["Sex_enc"] = (tmp["Sex"] == "male").astype(int)
    tmp["Embarked_enc"] = tmp["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    tmp["FamilySize"] = tmp["SibSp"] + tmp["Parch"] + 1
    tmp["Has_Cabin"] = tmp["Cabin"].notna().astype(int)

    cols = ["Survived", "Pclass", "Sex_enc", "Age", "Fare",
            "SibSp", "Parch", "FamilySize", "Has_Cabin", "Embarked_enc"]
    corr = tmp[cols].corr().round(2)
    col_labels = ["Survived", "Pclass", "Sex\n(male=1)", "Age", "Fare",
                  "SibSp", "Parch", "FamilySize", "Has\nCabin", "Embarked"]

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=col_labels, y=col_labels,
        colorscale=[[0, PALETTE["died"]], [0.5, "#1e293b"], [1, PALETTE["survived"]]],
        zmid=0, text=corr.values,
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="x=%{x}<br>y=%{y}<br>r=%{z:.2f}<extra></extra>",
    ))
    _apply_layout(fig, "Correlation Heatmap", height=480)
    fig.update_xaxes(tickfont=dict(size=11))
    fig.update_yaxes(tickfont=dict(size=11))
    return fig


# ── 11. Title Analysis ───────────────────────────────────────────────────────
def plot_title_analysis(df: pd.DataFrame) -> go.Figure:
    tmp = df.copy()
    tmp["Title"] = tmp["Name"].apply(_extract_title)
    grp = tmp.groupby("Title")["Survived"].agg(["mean", "count"]).reset_index()
    grp = grp.sort_values("count", ascending=False)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Survival Rate by Title", "Count by Title"],
                        horizontal_spacing=0.12)

    fig.add_trace(go.Bar(x=grp["Title"], y=grp["mean"],
                         marker_color=PALETTE["survived"],
                         text=[f"{r:.0%}" for r in grp["mean"]],
                         textposition="outside",
                         hovertemplate="%{x}<br>Survival: %{y:.1%}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=grp["Title"], y=grp["count"],
                         marker_color=PALETTE["neutral"],
                         hovertemplate="%{x}: %{y} passengers<extra></extra>"),
                  row=1, col=2)

    fig.update_layout(**LAYOUT_BASE, height=420,
                      title=dict(text="Name Title Analysis (Extracted from Name Field)",
                                 font=dict(size=16, color="#38bdf8")),
                      showlegend=False)
    fig.update_xaxes(**AXIS_BASE)
    fig.update_yaxes(**AXIS_BASE)
    return fig


# ── 12. Pclass x Sex Heatmap ─────────────────────────────────────────────────
def plot_pclass_sex_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.pivot_table("Survived", index="Sex", columns="Pclass", aggfunc="mean")
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=["1st Class", "2nd Class", "3rd Class"],
        y=[s.capitalize() for s in pivot.index],
        colorscale=[[0, PALETTE["died"]], [1, PALETTE["survived"]]],
        text=[[f"{v:.1%}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=16, color="white"),
        hovertemplate="Sex: %{y}<br>Class: %{x}<br>Survival: %{z:.1%}<extra></extra>",
    ))
    _apply_layout(fig, "Survival Rate — Pclass × Sex Interaction", height=320)
    return fig


# ── 13. Cabin Deck Analysis ──────────────────────────────────────────────────
def plot_cabin_analysis(df: pd.DataFrame) -> go.Figure:
    tmp = df.copy()
    tmp["Cabin_Deck"] = tmp["Cabin"].str[0].fillna("Unknown")
    grp = tmp.groupby("Cabin_Deck")["Survived"].agg(["mean", "count"]).reset_index()
    grp = grp.sort_values("count", ascending=False)

    fig = go.Figure(go.Bar(
        x=grp["Cabin_Deck"], y=grp["mean"],
        marker=dict(
            color=grp["mean"],
            colorscale=[[0, PALETTE["died"]], [1, PALETTE["survived"]]],
            showscale=False,
        ),
        text=[f"{r:.0%}" for r in grp["mean"]],
        textposition="outside",
        customdata=grp["count"],
        hovertemplate="Deck %{x}<br>Survival: %{y:.1%}<br>Count: %{customdata}<extra></extra>",
    ))
    _apply_layout(fig, "Survival Rate by Cabin Deck (77% Missing = 'Unknown')", height=420)
    fig.update_xaxes(title_text="Cabin Deck")
    fig.update_yaxes(title_text="Survival Rate", tickformat=".0%")
    return fig


# ── 14. Pclass + Fare Scatter ────────────────────────────────────────────────
def plot_age_fare_scatter(df: pd.DataFrame) -> go.Figure:
    tmp = df.dropna(subset=["Age", "Fare"]).copy()
    tmp["Survived_Label"] = tmp["Survived"].map({0: "Died", 1: "Survived"})

    fig = px.scatter(tmp, x="Age", y="Fare", color="Survived_Label",
                     color_discrete_map={"Survived": PALETTE["survived"], "Died": PALETTE["died"]},
                     facet_col="Pclass", opacity=0.7,
                     labels={"Age": "Age", "Fare": "Fare (£)", "Survived_Label": ""},
                     hover_data=["Name", "Sex", "Embarked"])

    fig.update_layout(**LAYOUT_BASE, height=420,
                      title=dict(text="Age vs Fare scatter by Class and Survival",
                                 font=dict(size=16, color="#38bdf8")))
    fig.for_each_xaxis(lambda ax: ax.update(**AXIS_BASE))
    fig.for_each_yaxis(lambda ax: ax.update(**AXIS_BASE))
    return fig


# ── Master generator ─────────────────────────────────────────────────────────
CHART_META = [
    ("survival_overview",     "Overall Survival",          plot_survival_overview),
    ("survival_by_pclass",    "Survival by Class",         plot_survival_by_pclass),
    ("survival_by_sex",       "Survival by Sex",           plot_survival_by_sex),
    ("age_distribution",      "Age Distribution",          plot_age_distribution),
    ("survival_by_age_group", "Survival by Age Group",     plot_survival_by_age_group),
    ("fare_distribution",     "Fare Distribution",         plot_fare_distribution),
    ("family_size",           "Family Size",               plot_family_size),
    ("embarked",              "Embarkation Port",          plot_embarked),
    ("missing_values",        "Missing Values",            plot_missing_values),
    ("correlation_heatmap",   "Correlation Heatmap",       plot_correlation_heatmap),
    ("title_analysis",        "Name Title Analysis",       plot_title_analysis),
    ("pclass_sex_heatmap",    "Pclass × Sex Heatmap",      plot_pclass_sex_heatmap),
    ("cabin_analysis",        "Cabin Deck Analysis",       plot_cabin_analysis),
    ("age_fare_scatter",      "Age vs Fare Scatter",       plot_age_fare_scatter),
]

INSIGHTS = {
    "survival_overview":
        "38.4% of the 891 training passengers survived. This class imbalance is mild but means "
        "a naive 'all died' baseline achieves 61.6% accuracy — any model must beat this.",
    "survival_by_pclass":
        "1st class passengers survived at 63% vs 24% for 3rd class. Pclass is highly correlated "
        "with survival, reflecting both proximity to lifeboats (upper decks) and social priority.",
    "survival_by_sex":
        "Women survived at 74% vs 19% for men — the single strongest signal in the dataset. "
        "'Women and children first' was enforced rigorously, making Sex the most important feature.",
    "age_distribution":
        "Age has 19.9% missing values. The distribution is right-skewed with a peak around 20-30. "
        "Children under 16 show notably higher survival — impute Age by (Pclass, Title) group medians.",
    "survival_by_age_group":
        "Babies (0-5) survived at the highest rate. Teen and young adult males drag group rates down. "
        "Seniors had low survival — elderly passengers were deprioritized during the evacuation.",
    "fare_distribution":
        "Fare is extremely right-skewed. Log-transforming it (log1p) is essential before feeding it "
        "to distance-based models. Higher fare strongly correlates with 1st class and survival.",
    "family_size":
        "Solo travelers (FamilySize=1) and very large families (5+) had the lowest survival. "
        "Medium families of 2-4 fared best — they could coordinate escape without being too large.",
    "embarked":
        "Cherbourg passengers survived at 55% vs 34% for Southampton. This largely reflects "
        "that many Cherbourg boarders were wealthy 1st-class passengers (Pclass confound).",
    "missing_values":
        "Cabin is 77% missing — do NOT drop it. Create 'Has_Cabin' binary and 'Cabin_Deck' from the "
        "first letter. The presence of a cabin record is itself a survival signal.",
    "correlation_heatmap":
        "Sex (encoded male=1) and Pclass are the most negatively correlated with survival. "
        "Fare is the strongest positive predictor. Has_Cabin is also informative.",
    "title_analysis":
        "'Master' (young boys) survived at ~57% despite being male — confirming 'children first.' "
        "'Mrs' and 'Miss' have very high rates. Mr (adult males) is the lowest. Use Title as a feature.",
    "pclass_sex_heatmap":
        "The Sex×Pclass interaction is striking: even 3rd class women (49%) survived more than "
        "1st class men (37%). This interaction should be an explicit feature in your model.",
    "cabin_analysis":
        "Decks A-E (upper decks, 1st class) have high survival. 3rd class decks (F, G) are lower. "
        "The 77% 'Unknown' cabin group still has 30% survival — meaningful signal in missingness.",
    "age_fare_scatter":
        "High fare + young age = highest survival probability in all classes. 3rd class clusters "
        "at low fare and shows the most deaths. Note the 'bulge' of surviving women across all ages.",
}


def generate_all_figures(df: pd.DataFrame) -> list[dict]:
    """Returns list of {id, label, fig, insight} dicts."""
    results = []
    for chart_id, label, fn in CHART_META:
        try:
            fig = fn(df)
            results.append(dict(id=chart_id, label=label, fig=fig,
                                insight=INSIGHTS.get(chart_id, "")))
        except Exception as e:
            print(f"Warning: {chart_id} failed — {e}")
    return results
