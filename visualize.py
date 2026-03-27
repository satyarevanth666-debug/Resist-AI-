from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def class_distribution_chart(df: pd.DataFrame):
    counts = df["Result"].value_counts().reset_index()
    counts.columns = ["Result", "Count"]
    return px.bar(
        counts,
        x="Result",
        y="Count",
        color="Result",
        title="Resistance Class Distribution",
        color_discrete_map={
            "Resistant": "#ef4444",
            "Susceptible": "#22c55e",
            "Intermediate": "#eab308",
        },
    )


def resistance_heatmap(df: pd.DataFrame):
    heatmap_df = (
        df.assign(ResistantBinary=(df["Result"] == "Resistant").astype(int))
        .pivot_table(
            index="Bacteria",
            columns="Antibiotic",
            values="ResistantBinary",
            aggfunc="mean",
            fill_value=0,
        )
        .sort_index()
    )
    return px.imshow(
        heatmap_df,
        color_continuous_scale="RdYlGn_r",
        title="Resistance Probability Heatmap (Bacteria vs Antibiotic)",
        labels=dict(x="Antibiotic", y="Bacteria", color="Resistance Rate"),
        aspect="auto",
    )


def confusion_matrix_chart(cm: list[list[int]]):
    labels = ["Susceptible", "Resistant", "Intermediate"]
    z = np.array(cm)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels[: z.shape[1]],
            y=labels[: z.shape[0]],
            colorscale="Blues",
            showscale=True,
        )
    )
    fig.update_layout(
        title="Confusion Matrix (Best Model)",
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    return fig


def feature_importance_chart(feature_importance: dict):
    fi_df = pd.DataFrame(
        {
            "Feature": list(feature_importance.keys()),
            "Importance": list(feature_importance.values()),
        }
    ).sort_values("Importance", ascending=False)
    return px.bar(
        fi_df,
        x="Feature",
        y="Importance",
        color="Importance",
        color_continuous_scale="Viridis",
        title="Feature Importance",
    )
