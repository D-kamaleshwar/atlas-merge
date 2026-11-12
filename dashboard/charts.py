"""Interactive Plotly Charts for ATLAS-MERGE Research Dashboard."""
from typing import Dict, List, Tuple, Any
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from core.geometry import GeometryProfile
from core.pareto import ParetoCandidate
from core.evaluation import EvaluationResult

# Professional Dark Palette
BG_COLOR = "#0b0f19"
PAPER_COLOR = "#131c2e"
GRID_COLOR = "rgba(51, 65, 85, 0.4)"
TEXT_COLOR = "#e2e8f0"
CYAN = "#38bdf8"
PURPLE = "#a855f7"
EMERALD = "#10b981"
AMBER = "#f59e0b"
ROSE = "#f43f5e"


def _apply_dark_layout(fig: go.Figure, title: str = "", height: int = 450) -> go.Figure:
    """Standardize dark scientific chart layout."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color=TEXT_COLOR, size=15, family="Segoe UI, sans-serif"),
            x=0.02,
            y=0.96,
        ),
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, size=12),
        margin=dict(l=50, r=30, t=50, b=45),
        height=height,
        xaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(color="#94a3b8"),
        ),
        legend=dict(
            bgcolor="rgba(19, 28, 46, 0.8)",
            bordercolor="rgba(51, 65, 85, 0.5)",
            borderwidth=1,
            font=dict(color=TEXT_COLOR),
        ),
    )
    return fig


def create_geometry_heatmap(profiles: Dict[str, GeometryProfile]) -> go.Figure:
    """Create interactive 8x4 Heatmap of Block Task-Vector Geometry."""
    block_names = list(profiles.keys())
    metrics = ["Cosine Similarity", "Sign-Conflict Density", "Relative L2 Norm", "Sparsity (Mean)"]

    z_data = []
    text_data = []

    for name in block_names:
        p = profiles[name]
        row = [p.cosine_similarity, p.sign_conflict_density, p.relative_norm, p.sparsity_mean]
        z_data.append(row)
        txt_row = [
            f"Cos Sim: {p.cosine_similarity:+.3f}",
            f"Conflict: {p.sign_conflict_density:.1%}",
            f"Rel Norm: {p.relative_norm:.2f}",
            f"Sparsity: {p.sparsity_mean:.1%}",
        ]
        text_data.append(txt_row)

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=metrics,
        y=block_names,
        text=text_data,
        hoverinfo="text+x+y",
        colorscale="Spectral_r",
        colorbar=dict(
            title="Value",
            titlefont=dict(color=TEXT_COLOR),
            tickfont=dict(color="#94a3b8"),
        ),
        texttemplate="%{z:.2f}",
        textfont=dict(color="#ffffff", size=11),
    ))

    _apply_dark_layout(fig, title="8-Block Task-Vector Geometry Profile Matrix", height=420)
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def create_pareto_scatter(
    all_candidates: List[ParetoCandidate],
    pareto_archive: List[ParetoCandidate],
    reference_point: Tuple[float, float] = (0.0, 0.0),
) -> go.Figure:
    """2D Pareto Frontier scatter plot (Math Proxy vs Code Proxy)."""
    fig = go.Figure()

    # 1. Dominated points (all candidates not in current archive)
    pareto_ids = {c.candidate_id for c in pareto_archive}
    dominated = [c for c in all_candidates if c.candidate_id not in pareto_ids]

    if dominated:
        fig.add_trace(go.Scatter(
            x=[c.math_score for c in dominated],
            y=[c.code_score for c in dominated],
            mode="markers",
            name="Dominated Candidates",
            marker=dict(size=7, color="#64748b", opacity=0.6, symbol="circle"),
            customdata=[[c.candidate_id, c.episode, c.retention_score, c.reward] for c in dominated],
            hovertemplate=(
                "<b>%{customdata[0]}</b> (Ep %{customdata[1]})<br>"
                "Math Proxy: %{x:.3f}<br>"
                "Code Proxy: %{y:.3f}<br>"
                "Retention: %{customdata[2]:.3f}<br>"
                "Reward: %{customdata[3]:.3f}<extra></extra>"
            ),
        ))

    # 2. Non-dominated Pareto frontier
    if pareto_archive:
        sorted_pareto = sorted(pareto_archive, key=lambda c: c.math_score)
        px = [c.math_score for c in sorted_pareto]
        py = [c.code_score for c in sorted_pareto]

        # Draw step frontier
        fig.add_trace(go.Scatter(
            x=px,
            y=py,
            mode="lines",
            name="Pareto Frontier",
            line=dict(color=CYAN, width=2, dash="dash"),
            hoverinfo="skip",
        ))

        # Highlight Pareto candidate points
        fig.add_trace(go.Scatter(
            x=px,
            y=py,
            mode="markers",
            name="Non-Dominated Optimal",
            marker=dict(
                size=12,
                color=CYAN,
                symbol="star",
                line=dict(width=1.5, color="#ffffff"),
            ),
            customdata=[[c.candidate_id, c.episode, c.retention_score, c.reward] for c in sorted_pareto],
            hovertemplate=(
                "⭐ <b>%{customdata[0]}</b> (Ep %{customdata[1]})<br>"
                "Math Proxy: %{x:.3f}<br>"
                "Code Proxy: %{y:.3f}<br>"
                "Retention: %{customdata[2]:.3f}<br>"
                "Reward: %{customdata[3]:.3f}<extra></extra>"
            ),
        ))

    # Reference point
    fig.add_trace(go.Scatter(
        x=[reference_point[0]],
        y=[reference_point[1]],
        mode="markers",
        name="Reference Point (0,0)",
        marker=dict(size=9, color=ROSE, symbol="x"),
        hoverinfo="name",
    ))

    _apply_dark_layout(fig, title="Pareto Trade-Off Space: Mathematical vs Code Proxy", height=480)
    fig.update_xaxes(title="Mathematical Reasoning Proxy (Synthetic GSM8K)", range=[0.0, 1.05])
    fig.update_yaxes(title="Code Generation Proxy (Synthetic MBPP)", range=[0.0, 1.05])
    return fig


def create_hypervolume_chart(history: List[Tuple[int, float]]) -> go.Figure:
    """Line chart tracking Hypervolume improvement over search episodes."""
    fig = go.Figure()
    if history:
        eps = [h[0] for h in history]
        hvs = [h[1] for h in history]

        fig.add_trace(go.Scatter(
            x=eps,
            y=hvs,
            mode="lines+markers",
            name="Hypervolume",
            line=dict(color=PURPLE, width=2.5),
            marker=dict(size=6, color=CYAN),
            hovertemplate="Episode %{x}: Hypervolume = %{y:.4f}<extra></extra>",
        ))

    _apply_dark_layout(fig, title="Pareto Hypervolume Evolution", height=350)
    fig.update_xaxes(title="Search Episode")
    fig.update_yaxes(title="Hypervolume Indicator")
    return fig


def create_ppo_training_charts(records: List[Any]) -> go.Figure:
    """Multi-panel chart displaying PPO learning dynamics (Reward, Losses, Entropy)."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Chebyshev Total Reward", "Policy & Value Losses", "Policy Entropy"),
        horizontal_spacing=0.08,
    )

    if records:
        eps = [r.episode for r in records]
        rewards = [r.reward for r in records]
        cheb = [r.chebyshev_reward for r in records]
        p_loss = [r.policy_loss for r in records]
        v_loss = [r.value_loss for r in records]
        entropy = [r.entropy for r in records]

        # Subplot 1: Rewards
        fig.add_trace(go.Scatter(x=eps, y=rewards, mode="lines+markers", name="Total Reward", line=dict(color=CYAN, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=eps, y=cheb, mode="lines", name="Chebyshev Term", line=dict(color=PURPLE, dash="dot")), row=1, col=1)

        # Subplot 2: Losses
        fig.add_trace(go.Scatter(x=eps, y=p_loss, mode="lines", name="Policy Loss", line=dict(color=AMBER)), row=1, col=2)
        fig.add_trace(go.Scatter(x=eps, y=v_loss, mode="lines", name="Value Loss", line=dict(color=ROSE)), row=1, col=2)

        # Subplot 3: Entropy
        fig.add_trace(go.Scatter(x=eps, y=entropy, mode="lines", name="Entropy", line=dict(color=EMERALD)), row=1, col=3)

    _apply_dark_layout(fig, title="PPO Reinforcement Learning Training Dynamics", height=380)
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig


def create_action_distribution_chart(records: List[Any]) -> go.Figure:
    """Bar chart summarizing operator choices across episodes and blocks."""
    op_counts = {"BASE_PRESERVE": 0, "TASK_ARITHMETIC": 0, "TIES": 0, "DARE_TIES": 0}
    for r in records:
        recipe = getattr(r, "recipe", {})
        for blk, cfg in recipe.items():
            op = cfg.get("operator", "TASK_ARITHMETIC")
            if op in op_counts:
                op_counts[op] += 1

    fig = go.Figure(data=[
        go.Bar(
            x=list(op_counts.keys()),
            y=list(op_counts.values()),
            marker=dict(color=[ROSE, AMBER, CYAN, PURPLE]),
            text=list(op_counts.values()),
            textposition="auto",
        )
    ])

    _apply_dark_layout(fig, title="Merge Operator Selection Frequency (PPO Policy)", height=350)
    fig.update_xaxes(title="Merge Operator")
    fig.update_yaxes(title="Total Block Selections")
    return fig


def create_tensor_distribution_histogram(
    base: torch.Tensor,
    t_math: torch.Tensor,
    t_code: torch.Tensor,
    merged: torch.Tensor,
) -> go.Figure:
    """Histogram comparison for interactive merge playground."""
    b_vals = base.detach().reshape(-1).cpu().numpy()[:2000]
    m_vals = (base + t_math).detach().reshape(-1).cpu().numpy()[:2000]
    c_vals = (base + t_code).detach().reshape(-1).cpu().numpy()[:2000]
    merg_vals = merged.detach().reshape(-1).cpu().numpy()[:2000]

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=b_vals, name="Base Model", opacity=0.4, marker_color="#94a3b8", nbinsx=40))
    fig.add_trace(go.Histogram(x=m_vals, name="Base + Math", opacity=0.4, marker_color=CYAN, nbinsx=40))
    fig.add_trace(go.Histogram(x=c_vals, name="Base + Code", opacity=0.4, marker_color=PURPLE, nbinsx=40))
    fig.add_trace(go.Histogram(x=merg_vals, name="Merged Result", opacity=0.6, marker_color=EMERALD, nbinsx=40))

    fig.update_layout(barmode="overlay")
    _apply_dark_layout(fig, title="Parameter Distribution Comparison (Sample of 2,000 Coordinates)", height=380)
    fig.update_xaxes(title="Weight Value")
    fig.update_yaxes(title="Count")
    return fig


def create_block_proxy_radar(eval_result: EvaluationResult) -> go.Figure:
    """Bar comparison of proxy scores across the 8 functional blocks."""
    blocks = list(eval_result.block_math_scores.keys())
    m_scores = [eval_result.block_math_scores[b] for b in blocks]
    c_scores = [eval_result.block_code_scores[b] for b in blocks]
    r_scores = [eval_result.block_retention_scores[b] for b in blocks]

    fig = go.Figure(data=[
        go.Bar(name="Math Alignment", x=blocks, y=m_scores, marker_color=CYAN),
        go.Bar(name="Code Alignment", x=blocks, y=c_scores, marker_color=PURPLE),
        go.Bar(name="Retention Score", x=blocks, y=r_scores, marker_color=EMERALD),
    ])

    _apply_dark_layout(fig, title="Block-Wise Alignment & Retention Scores", height=380)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Functional Transformer Block")
    fig.update_yaxes(title="Proxy Score [0.0 - 1.0]", range=[0.0, 1.1])
    return fig

