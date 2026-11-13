"""Interactive Plotly Charts for ATLAS-MERGE Research Dashboard."""
from typing import Dict, List, Tuple, Any
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from core.geometry import GeometryProfile
from core.pareto import ParetoCandidate
from core.evaluation import EvaluationResult

# Professional Light Theme Palette
BG_COLOR = "#f8fafc"
PAPER_COLOR = "#ffffff"
GRID_COLOR = "rgba(226, 232, 240, 0.9)"
BORDER_COLOR = "#cbd5e1"
TEXT_COLOR = "#0f172a"
TEXT_MUTED = "#64748b"
PRIMARY_BLUE = "#0284c7"
PURPLE = "#4f46e5"
VIOLET = "#7c3aed"
EMERALD = "#059669"
AMBER = "#d97706"
ROSE = "#e11d48"

# Backwards compatibility alias
CYAN = PRIMARY_BLUE


def _apply_dark_layout(fig: go.Figure, title: str = "", height: int = 450) -> go.Figure:
    """Standardize modern light scientific chart layout (named for backwards compatibility)."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(color=TEXT_COLOR, size=15, family="Inter, Segoe UI, sans-serif"),
            x=0.02,
            y=0.96,
        ),
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, size=12, family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=55, r=35, t=55, b=50),
        height=height,
        xaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=BORDER_COLOR,
            tickfont=dict(color=TEXT_MUTED, size=11),
            title_font=dict(color=TEXT_COLOR, size=12),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=BORDER_COLOR,
            tickfont=dict(color=TEXT_MUTED, size=11),
            title_font=dict(color=TEXT_COLOR, size=12),
        ),
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(color=TEXT_COLOR, size=11),
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            font=dict(color="#0f172a", family="Inter, Segoe UI, sans-serif", size=12),
            bordercolor=PRIMARY_BLUE,
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
            f"<b>{name}</b><br>Cosine Alignment: <b>{p.cosine_similarity:+.3f}</b><br>Alignment Category: {'Strong Positive' if p.cosine_similarity > 0.4 else ('Severe Opposing' if p.cosine_similarity < -0.3 else 'Orthogonal/Weak')}",
            f"<b>{name}</b><br>Sign-Conflict: <b>{p.sign_conflict_density:.1%}</b><br>Active Coordinates: {p.active_coordinate_ratio:.1%}",
            f"<b>{name}</b><br>Relative Norm (||Code|| / ||Math||): <b>{p.relative_norm:.2f}</b><br>Math Norm: {p.norm_math:.2f} | Code Norm: {p.norm_code:.2f}",
            f"<b>{name}</b><br>Mean Sparsity: <b>{p.sparsity_mean:.1%}</b><br>Math Zero-Ratio: {p.sparsity_math:.1%}",
        ]
        text_data.append(txt_row)

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=metrics,
        y=block_names,
        text=text_data,
        hoverinfo="text",
        colorscale="Blues",
        colorbar=dict(
            title=dict(
                text="<b>Metric Value</b>",
                font=dict(color=TEXT_COLOR, size=11),
            ),
            tickfont=dict(color=TEXT_MUTED, size=10),
            thickness=14,
            len=0.9,
        ),
        texttemplate="%{z:.2f}",
        textfont=dict(color="#0f172a", size=11, family="Inter, Segoe UI, sans-serif"),
    ))

    _apply_dark_layout(fig, title="8-Block Task-Vector Geometry Profile Matrix", height=430)
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def create_pareto_scatter(
    all_candidates: List[ParetoCandidate],
    pareto_archive: List[ParetoCandidate],
    reference_point: Tuple[float, float] = (0.0, 0.0),
) -> go.Figure:
    """2D Pareto Frontier scatter plot (Math Proxy vs Code Proxy)."""
    fig = go.Figure()

    pareto_ids = {c.candidate_id for c in pareto_archive}
    dominated = [c for c in all_candidates if c.candidate_id not in pareto_ids]

    # Dominated candidates
    if dominated:
        fig.add_trace(go.Scatter(
            x=[c.math_score for c in dominated],
            y=[c.code_score for c in dominated],
            mode="markers",
            name="Dominated Candidates",
            marker=dict(size=8, color="#94a3b8", opacity=0.6, symbol="circle"),
            customdata=[[c.candidate_id, c.episode, c.retention_score, c.reward] for c in dominated],
            hovertemplate=(
                "<b>Candidate:</b> %{customdata[0]} (Episode %{customdata[1]})<br>"
                "• Math Reasoning Proxy: <b>%{x:.4f}</b><br>"
                "• Code Generation Proxy: <b>%{y:.4f}</b><br>"
                "• Base Retention: <b>%{customdata[2]:.4f}</b><br>"
                "• Total Reward: <b>%{customdata[3]:.4f}</b><extra></extra>"
            ),
        ))

    # Pareto optimal candidates
    if pareto_archive:
        sorted_pareto = sorted(pareto_archive, key=lambda c: c.math_score)
        px = [c.math_score for c in sorted_pareto]
        py = [c.code_score for c in sorted_pareto]

        # Step frontier line
        fig.add_trace(go.Scatter(
            x=px,
            y=py,
            mode="lines",
            name="Pareto Frontier (Boundary)",
            line=dict(color=PRIMARY_BLUE, width=2.5, dash="dash"),
            hoverinfo="skip",
        ))

        # Star markers for optimal candidates
        fig.add_trace(go.Scatter(
            x=px,
            y=py,
            mode="markers",
            name="Non-Dominated Optimal",
            marker=dict(
                size=13,
                color="#0284c7",
                symbol="star",
                line=dict(width=1.5, color="#0f172a"),
            ),
            customdata=[[c.candidate_id, c.episode, c.retention_score, c.reward] for c in sorted_pareto],
            hovertemplate=(
                "⭐ <b>NON-DOMINATED: %{customdata[0]}</b> (Episode %{customdata[1]})<br>"
                "• Math Reasoning Proxy: <b>%{x:.4f}</b><br>"
                "• Code Generation Proxy: <b>%{y:.4f}</b><br>"
                "• Base Retention: <b>%{customdata[2]:.4f}</b><br>"
                "• Scalar Reward: <b>%{customdata[3]:.4f}</b><extra></extra>"
            ),
        ))

    # Reference point
    fig.add_trace(go.Scatter(
        x=[reference_point[0]],
        y=[reference_point[1]],
        mode="markers",
        name="Reference Point (0.0, 0.0)",
        marker=dict(size=10, color=ROSE, symbol="x", line=dict(width=2)),
        hoverinfo="name",
    ))

    _apply_dark_layout(fig, title="Pareto Multi-Objective Trade-Off Space (Math vs Code)", height=480)
    fig.update_xaxes(title="Mathematical Reasoning Proxy (Synthetic GSM8K)", range=[0.0, 1.05])
    fig.update_yaxes(title="Code Generation Proxy (Synthetic MBPP)", range=[0.0, 1.05])
    return fig


def create_pareto_3d_scatter(
    all_candidates: List[ParetoCandidate],
    pareto_archive: List[ParetoCandidate],
) -> go.Figure:
    """3D Interactive Scatter Plot across Math, Code, and Retention Proxies."""
    fig = go.Figure()

    pareto_ids = {c.candidate_id for c in pareto_archive}
    dominated = [c for c in all_candidates if c.candidate_id not in pareto_ids]

    if dominated:
        fig.add_trace(go.Scatter3d(
            x=[c.math_score for c in dominated],
            y=[c.code_score for c in dominated],
            z=[c.retention_score for c in dominated],
            mode="markers",
            name="Dominated Candidates",
            marker=dict(size=4, color="#94a3b8", opacity=0.5),
            customdata=[[c.candidate_id, c.episode, c.reward] for c in dominated],
            hovertemplate=(
                "<b>%{customdata[0]}</b> (Ep %{customdata[1]})<br>"
                "Math: %{x:.3f} | Code: %{y:.3f} | Ret: %{z:.3f}<br>"
                "Reward: %{customdata[2]:.3f}<extra></extra>"
            ),
        ))

    if pareto_archive:
        fig.add_trace(go.Scatter3d(
            x=[c.math_score for c in pareto_archive],
            y=[c.code_score for c in pareto_archive],
            z=[c.retention_score for c in pareto_archive],
            mode="markers",
            name="Pareto Optimal Frontier",
            marker=dict(
                size=8,
                color=PRIMARY_BLUE,
                symbol="diamond",
                opacity=0.95,
                line=dict(color="#0f172a", width=1),
            ),
            customdata=[[c.candidate_id, c.episode, c.reward] for c in pareto_archive],
            hovertemplate=(
                "⭐ <b>%{customdata[0]}</b> (Ep %{customdata[1]})<br>"
                "Math: %{x:.3f} | Code: %{y:.3f} | Ret: %{z:.3f}<br>"
                "Reward: %{customdata[2]:.3f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Math Proxy", backgroundcolor=BG_COLOR, gridcolor=GRID_COLOR),
            yaxis=dict(title="Code Proxy", backgroundcolor=BG_COLOR, gridcolor=GRID_COLOR),
            zaxis=dict(title="Retention Proxy", backgroundcolor=BG_COLOR, gridcolor=GRID_COLOR),
        ),
        paper_bgcolor=PAPER_COLOR,
        font=dict(color=TEXT_COLOR, size=11),
        margin=dict(l=10, r=10, t=40, b=10),
        height=500,
        title=dict(text="<b>3D Multi-Objective Trade-Off Space (Math × Code × Retention)</b>", font=dict(color=TEXT_COLOR, size=14)),
    )
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
            name="Hypervolume Indicator",
            line=dict(color=PURPLE, width=2.5),
            marker=dict(size=6, color=PRIMARY_BLUE, line=dict(color="#ffffff", width=1)),
            fill="tozeroy",
            fillcolor="rgba(79, 70, 229, 0.08)",
            hovertemplate="Episode %{x}: Hypervolume = <b>%{y:.4f}</b><extra></extra>",
        ))

    _apply_dark_layout(fig, title="Pareto Hypervolume Evolution Across Episodes", height=350)
    fig.update_xaxes(title="Search Episode")
    fig.update_yaxes(title="Hypervolume Indicator (Ref: 0.0, 0.0)")
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
        fig.add_trace(go.Scatter(x=eps, y=rewards, mode="lines+markers", name="Total Reward", line=dict(color=PRIMARY_BLUE, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=eps, y=cheb, mode="lines", name="Chebyshev Term", line=dict(color=PURPLE, dash="dot")), row=1, col=1)

        # Subplot 2: Losses
        fig.add_trace(go.Scatter(x=eps, y=p_loss, mode="lines", name="Policy Loss", line=dict(color=AMBER, width=1.8)), row=1, col=2)
        fig.add_trace(go.Scatter(x=eps, y=v_loss, mode="lines", name="Value Loss", line=dict(color=ROSE, width=1.8)), row=1, col=2)

        # Subplot 3: Entropy
        fig.add_trace(go.Scatter(x=eps, y=entropy, mode="lines", name="Entropy", line=dict(color=EMERALD, width=2)), row=1, col=3)

    _apply_dark_layout(fig, title="PPO Reinforcement Learning Training Dynamics", height=380)
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=BORDER_COLOR, title="Episode")
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=BORDER_COLOR)
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
            x=["Base-Preserve", "Task Arithmetic", "TIES", "DARE-TIES"],
            y=[op_counts["BASE_PRESERVE"], op_counts["TASK_ARITHMETIC"], op_counts["TIES"], op_counts["DARE_TIES"]],
            marker=dict(color=[ROSE, AMBER, PRIMARY_BLUE, PURPLE], line=dict(color="rgba(0,0,0,0.1)", width=1)),
            text=[op_counts["BASE_PRESERVE"], op_counts["TASK_ARITHMETIC"], op_counts["TIES"], op_counts["DARE_TIES"]],
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
    fig.add_trace(go.Histogram(x=b_vals, name="Base Model", opacity=0.45, marker_color="#94a3b8", nbinsx=40))
    fig.add_trace(go.Histogram(x=m_vals, name="Base + Math", opacity=0.45, marker_color=PRIMARY_BLUE, nbinsx=40))
    fig.add_trace(go.Histogram(x=c_vals, name="Base + Code", opacity=0.45, marker_color=PURPLE, nbinsx=40))
    fig.add_trace(go.Histogram(x=merg_vals, name="Merged Result", opacity=0.6, marker_color=EMERALD, nbinsx=40))

    fig.update_layout(barmode="overlay")
    _apply_dark_layout(fig, title="Parameter Distribution Comparison (2,000 Coordinates Sample)", height=380)
    fig.update_xaxes(title="Tensor Weight Value")
    fig.update_yaxes(title="Frequency Count")
    return fig


def create_block_proxy_radar(eval_result: EvaluationResult, retention_floor: float = 0.90) -> go.Figure:
    """Bar comparison of proxy scores across the 8 functional blocks with floor line."""
    blocks = list(eval_result.block_math_scores.keys())
    m_scores = [eval_result.block_math_scores[b] for b in blocks]
    c_scores = [eval_result.block_code_scores[b] for b in blocks]
    r_scores = [eval_result.block_retention_scores[b] for b in blocks]

    fig = go.Figure(data=[
        go.Bar(name="Math Reasoning Proxy", x=blocks, y=m_scores, marker_color=PRIMARY_BLUE),
        go.Bar(name="Code Generation Proxy", x=blocks, y=c_scores, marker_color=PURPLE),
        go.Bar(name="Retention Score", x=blocks, y=r_scores, marker_color=EMERALD),
    ])

    # Add threshold reference line
    fig.add_hline(
        y=retention_floor,
        line_dash="dot",
        line_color=AMBER,
        annotation_text=f"Retention Constraint Floor ({retention_floor:.2f})",
        annotation_position="bottom right",
        annotation_font=dict(color=AMBER, size=10),
    )

    _apply_dark_layout(fig, title="Block-Wise Alignment & Retention Scores", height=390)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Functional Transformer Block")
    fig.update_yaxes(title="Proxy Score [0.0 - 1.0]", range=[0.0, 1.1])
    return fig
