"""
Evaluation utilities: confusion matrix, ROC, feature importance, SHAP.
All functions return plotly Figures.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import StratifiedKFold, cross_val_predict, learning_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc
import shap

from src.config import RANDOM_SEED, CV_FOLDS
from src.eda import LAYOUT_BASE, AXIS_BASE, PALETTE, DARK_BG, CARD_BG, TEXT_COLOR, GRID_COLOR


def plot_cv_comparison(cv_results: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cv_results["Model"],
        y=cv_results["CV_Mean"],
        error_y=dict(type="data", array=cv_results["CV_Std"], visible=True),
        marker=dict(
            color=cv_results["CV_Mean"],
            colorscale=[[0, PALETTE["died"]], [1, PALETTE["survived"]]],
            showscale=False,
        ),
        text=[f"{v:.4f}" for v in cv_results["CV_Mean"]],
        textposition="outside",
        hovertemplate="%{x}<br>CV Accuracy: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_BASE, height=420,
                      title=dict(text="5-Fold CV Accuracy Comparison",
                                 font=dict(size=16, color="#38bdf8")))
    fig.update_xaxes(**AXIS_BASE, title_text="Model")
    fig.update_yaxes(**AXIS_BASE, title_text="CV Accuracy", range=[0.7, 0.9])
    return fig


def plot_confusion_matrix(model, X, y, model_name="Model") -> go.Figure:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    y_pred = cross_val_predict(model, X, y, cv=skf)
    cm = confusion_matrix(y, y_pred)

    labels = ["Died", "Survived"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, CARD_BG], [1, PALETTE["survived"]]],
        text=cm, texttemplate="%{text}",
        textfont=dict(size=18, color="white"),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
    fig.update_layout(**LAYOUT_BASE, height=380,
                      title=dict(text=f"{model_name} Confusion Matrix (CV) — Acc: {acc:.3f}",
                                 font=dict(size=16, color="#38bdf8")))
    fig.update_xaxes(title_text="Predicted", **AXIS_BASE)
    fig.update_yaxes(title_text="Actual", **AXIS_BASE)
    return fig


def plot_roc_curves(models: dict, X, y) -> go.Figure:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, (name, model) in enumerate(models.items()):
        y_prob = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={roc_auc:.3f})",
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate=f"{name}<br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(dash="dash", color=GRID_COLOR)))
    fig.update_layout(**LAYOUT_BASE, height=480,
                      title=dict(text="ROC Curves — All Models (5-Fold CV)",
                                 font=dict(size=16, color="#38bdf8")))
    fig.update_xaxes(title_text="False Positive Rate", **AXIS_BASE)
    fig.update_yaxes(title_text="True Positive Rate", **AXIS_BASE)
    return fig


def plot_feature_importance(model, feature_names, model_name="Random Forest") -> go.Figure:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return go.Figure()

    idx = np.argsort(importances)[-20:]  # top 20
    fig = go.Figure(go.Bar(
        x=importances[idx],
        y=[feature_names[i] for i in idx],
        orientation="h",
        marker=dict(
            color=importances[idx],
            colorscale=[[0, PALETTE["neutral"]], [1, PALETTE["survived"]]],
            showscale=False,
        ),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_BASE, height=520,
                      title=dict(text=f"Feature Importance — {model_name} (Top 20)",
                                 font=dict(size=16, color="#38bdf8")))
    fig.update_xaxes(title_text="Importance", **AXIS_BASE)
    fig.update_yaxes(**AXIS_BASE)
    return fig


def plot_shap_summary(model, X, feature_names, model_name="XGBoost", max_display=15) -> go.Figure:
    """SHAP beeswarm-style bar chart."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs = np.abs(shap_values).mean(axis=0)
        idx = np.argsort(mean_abs)[-max_display:]

        fig = go.Figure(go.Bar(
            x=mean_abs[idx],
            y=[feature_names[i] for i in idx],
            orientation="h",
            marker=dict(
                color=mean_abs[idx],
                colorscale=[[0, PALETTE["neutral"]], [1, PALETTE["survived"]]],
                showscale=False,
            ),
            hovertemplate="%{y}: mean|SHAP|=%{x:.4f}<extra></extra>",
        ))
        fig.update_layout(**LAYOUT_BASE, height=480,
                          title=dict(text=f"SHAP Feature Importance — {model_name}",
                                     font=dict(size=16, color="#38bdf8")))
        fig.update_xaxes(title_text="Mean |SHAP value|", **AXIS_BASE)
        fig.update_yaxes(**AXIS_BASE)
        return fig
    except Exception as e:
        print(f"SHAP failed: {e}")
        return go.Figure()


def plot_learning_curve(model, X, y, model_name="Model") -> go.Figure:
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring="accuracy", n_jobs=-1
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_scores.mean(axis=1), name="Train",
        mode="lines+markers", line=dict(color=PALETTE["survived"], width=2),
        error_y=dict(type="data", array=train_scores.std(axis=1),
                     visible=True, color=PALETTE["survived"]),
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_scores.mean(axis=1), name="Validation",
        mode="lines+markers", line=dict(color=PALETTE["died"], width=2),
        error_y=dict(type="data", array=val_scores.std(axis=1),
                     visible=True, color=PALETTE["died"]),
    ))
    fig.update_layout(**LAYOUT_BASE, height=420,
                      title=dict(text=f"Learning Curve — {model_name}",
                                 font=dict(size=16, color="#38bdf8")))
    fig.update_xaxes(title_text="Training Set Size", **AXIS_BASE)
    fig.update_yaxes(title_text="Accuracy", **AXIS_BASE)
    return fig
