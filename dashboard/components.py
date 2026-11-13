"""UI Components, Design System, and Styling for ATLAS-MERGE Research Dashboard."""
from typing import Dict, Any, Optional
import platform
try:
    import psutil
except ImportError:
    psutil = None
import torch
import streamlit as st


def apply_custom_theme() -> None:
    """Inject custom CSS for modern high-contrast Light Theme research platform aesthetic."""
    st.markdown("""
        <style>
        /* =====================================================================
           ATLAS-MERGE DESIGN SYSTEM (Professional Light Research Theme)
           ===================================================================== */

        :root {
            --bg-base: #f8fafc;
            --bg-surface: #ffffff;
            --bg-card: #ffffff;
            --bg-card-hover: #f1f5f9;
            --border-subtle: #e2e8f0;
            --border-accent: #0284c7;
            --cyan-primary: #0284c7;
            --cyan-glow: rgba(2, 132, 199, 0.10);
            --indigo-accent: #4f46e5;
            --purple-accent: #7c3aed;
            --emerald-success: #059669;
            --amber-warning: #d97706;
            --rose-danger: #e11d48;
            --text-main: #0f172a;
            --text-muted: #475569;
            --text-dim: #64748b;
        }

        /* Base Application Canvas */
        .stApp {
            background-color: var(--bg-base);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            letter-spacing: -0.15px;
        }

        /* Top Header Styling */
        .atlas-header {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 22px 28px;
            margin-bottom: 22px;
            box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.05);
            position: relative;
            overflow: hidden;
        }
        .atlas-header::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #0284c7 0%, #4f46e5 50%, #7c3aed 100%);
        }
        .atlas-title-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }
        .atlas-title {
            font-size: 2.1rem;
            font-weight: 800;
            background: linear-gradient(90deg, #0284c7 0%, #4f46e5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            letter-spacing: -0.6px;
            line-height: 1.15;
        }
        .atlas-subtitle {
            font-size: 0.98rem;
            color: var(--text-muted);
            margin-top: 4px;
            font-weight: 400;
        }

        /* Badge Pills */
        .sim-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #e0f2fe;
            color: #0284c7;
            border: 1px solid #bae6fd;
            padding: 5px 14px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.3px;
            text-transform: uppercase;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.2px;
        }
        .status-badge.success {
            background: #dcfce7;
            color: #15803d;
            border: 1px solid #bbf7d0;
        }
        .status-badge.warning {
            background: #fef3c7;
            color: #b45309;
            border: 1px solid #fde68a;
        }
        .status-badge.info {
            background: #e0f2fe;
            color: #0369a1;
            border: 1px solid #bae6fd;
        }
        .status-badge.purple {
            background: #f3e8ff;
            color: #6b21a8;
            border: 1px solid #e9d5ff;
        }

        /* Metric Cards */
        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 18px 20px;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            height: 100%;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: var(--border-accent);
            box-shadow: 0 8px 18px -4px rgba(2, 132, 199, 0.12);
        }
        .metric-card-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        .metric-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: var(--text-dim);
            font-weight: 700;
        }
        .metric-icon {
            font-size: 1.15rem;
            opacity: 0.9;
        }
        .metric-value {
            font-size: 1.95rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.15;
            letter-spacing: -0.5px;
        }
        .metric-sub {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 6px;
            line-height: 1.35;
        }
        .metric-delta {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            font-size: 0.78rem;
            font-weight: 700;
            margin-top: 4px;
            padding: 2px 7px;
            border-radius: 4px;
        }
        .metric-delta.positive {
            color: #15803d;
            background: #dcfce7;
        }
        .metric-delta.negative {
            color: #b91c1c;
            background: #fee2e2;
        }

        /* Section Containers */
        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 28px 0 14px 0;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-subtle);
        }
        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #0f172a;
            display: flex;
            align-items: center;
            gap: 8px;
            letter-spacing: -0.3px;
        }
        .section-subtitle {
            font-size: 0.82rem;
            color: var(--text-dim);
        }

        /* Research Alert Boxes */
        .research-alert {
            background: #f0f9ff;
            border-left: 4px solid var(--cyan-primary);
            border-top: 1px solid #e0f2fe;
            border-right: 1px solid #e0f2fe;
            border-bottom: 1px solid #e0f2fe;
            border-radius: 8px;
            padding: 14px 20px;
            margin: 16px 0;
            color: #1e293b;
            font-size: 0.88rem;
            line-height: 1.5;
        }
        .research-alert-warning {
            background: #fffbeb;
            border-left: 4px solid var(--amber-warning);
            border-top: 1px solid #fef3c7;
            border-right: 1px solid #fef3c7;
            border-bottom: 1px solid #fef3c7;
            border-radius: 8px;
            padding: 14px 20px;
            margin: 16px 0;
            color: #92400e;
            font-size: 0.88rem;
            line-height: 1.5;
        }

        /* Block Architecture Cards */
        .block-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 12px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            transition: all 0.2s ease;
        }
        .block-card:hover {
            border-color: var(--border-accent);
            box-shadow: 0 6px 14px rgba(2, 132, 199, 0.10);
        }
        .block-card-title {
            font-weight: 700;
            color: #0f172a;
            font-size: 0.92rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .block-card-desc {
            font-size: 0.76rem;
            color: var(--text-muted);
            margin: 4px 0 8px 0;
        }
        .progress-track {
            background: #e2e8f0;
            height: 6px;
            border-radius: 3px;
            overflow: hidden;
            margin: 4px 0;
        }
        .progress-bar-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s ease;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid var(--border-subtle);
        }
        section[data-testid="stSidebar"] div.stRadio > div {
            gap: 6px;
        }
        section[data-testid="stSidebar"] div.stRadio label {
            background: #ffffff;
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 0.85rem;
            font-weight: 600;
            color: #475569;
            transition: all 0.2s ease;
        }
        section[data-testid="stSidebar"] div.stRadio label:hover {
            border-color: var(--border-accent);
            color: #0284c7;
            background: #f0f9ff;
        }

        /* Buttons & Controls */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            letter-spacing: 0.2px;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
            border: 1px solid #0284c7;
            color: #ffffff;
            box-shadow: 0 3px 10px 0 rgba(2, 132, 199, 0.25);
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 100%);
            box-shadow: 0 5px 15px 0 rgba(2, 132, 199, 0.35);
            transform: translateY(-1px);
        }

        /* Sliders and Inputs */
        div[data-testid="stSlider"] label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #334155;
        }

        /* Dataframes & Tables */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid var(--border-subtle);
            background: #ffffff;
        }

        /* Quick Navigation Link Card */
        .nav-quick-card {
            background: #ffffff;
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 14px 16px;
            transition: all 0.2s ease;
            text-decoration: none;
            color: inherit;
            display: block;
            margin-bottom: 8px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        }
        .nav-quick-card:hover {
            border-color: var(--border-accent);
            background: #f8fafc;
            transform: translateY(-2px);
            box-shadow: 0 6px 14px rgba(2, 132, 199, 0.08);
        }
        .nav-quick-title {
            font-weight: 700;
            color: #0284c7;
            font-size: 0.92rem;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .nav-quick-desc {
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-top: 3px;
        }
        </style>
    """, unsafe_allow_html=True)


def render_header(page_title: str, subtitle: str = "", active_metric_text: str = "") -> None:
    """Render top application header with branding and status badges.

    Completely removes redundant sub-meta bars to maintain a clean, crisp header without markdown leaks.
    """
    sub_text = subtitle or "Automated Multi-Objective Model Merging via Task Vector Geometry"
    status_pill = (
        f'<span class="status-badge info">● {active_metric_text}</span>'
        if active_metric_text
        else '<span class="status-badge success">● Engine Online</span>'
    )

    html = (
        '<div class="atlas-header">'
        '<div class="atlas-title-row">'
        '<div>'
        f'<h1 class="atlas-title">ATLAS-MERGE</h1>'
        f'<div class="atlas-subtitle">{sub_text}</div>'
        '</div>'
        '<div style="display: flex; align-items: center; gap: 8px;">'
        '<span class="sim-badge">🧬 Simulation-First Testbed</span>'
        f'{status_pill}'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_simulation_banner() -> None:
    """Explicitly clarify the simulation-first research setup with professional transparency."""
    html = (
        '<div class="research-alert">'
        'ℹ️ <strong>Academic Research Platform Notice:</strong> This interactive environment executes authentic '
        '<strong>PyTorch FP32 tensor algorithms</strong> (TIES, DARE-TIES, Task Arithmetic, Cosine Geometry Profiling, '
        'PPO Reinforcement Learning, Augmented Chebyshev Reward, and Pareto Hypervolume tracking) across 8 simulated transformer blocks. '
        'Checkpoints are generated as deterministic mathematical tensors without requiring 15+ GB HuggingFace model weight downloads on host hardware.'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(
    title: str,
    value: str,
    sub: str = "",
    delta: str = "",
    is_positive: bool = True,
    icon: str = "📊",
    badge: str = "",
) -> None:
    """Render high-impact styled metric tile with iconography, trend indicators, and badges."""
    delta_html = ""
    if delta:
        cls = "positive" if is_positive else "negative"
        symbol = "▲" if is_positive else "▼"
        delta_html = f'<div class="metric-delta {cls}">{symbol} {delta}</div>'

    badge_html = f'<span class="status-badge info">{badge}</span>' if badge else ""

    html = (
        '<div class="metric-card">'
        '<div class="metric-card-top">'
        f'<div class="metric-label">{title}</div>'
        f'<div>{badge_html} <span class="metric-icon">{icon}</span></div>'
        '</div>'
        f'<div class="metric-value">{value}</div>'
        f'{delta_html}'
        f'<div class="metric-sub">{sub}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_section_header(title: str, subtitle: str = "", icon: str = "📌", badge_text: str = "") -> None:
    """Render standard section header with divider."""
    badge_html = f'<span class="status-badge purple">{badge_text}</span>' if badge_text else ""
    sub_html = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""

    html = (
        '<div class="section-header">'
        '<div>'
        f'<div class="section-title"><span>{icon}</span> {title} {badge_html}</div>'
        f'{sub_html}'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_block_card(name: str, desc: str, cosine: float, conflict: float, norm_ratio: float, sparsity: float) -> None:
    """Render styled interactive card for an 8-block transformer layer."""
    cos_color = "#059669" if cosine > 0.3 else ("#e11d48" if cosine < -0.2 else "#d97706")
    conflict_pct = conflict * 100.0
    conflict_color = "#e11d48" if conflict > 0.6 else ("#d97706" if conflict > 0.35 else "#059669")

    html = (
        '<div class="block-card">'
        '<div class="block-card-title">'
        f'<span>{name}</span>'
        f'<span class="status-badge" style="background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1;">Norm: {norm_ratio:.2f}</span>'
        '</div>'
        f'<div class="block-card-desc">{desc}</div>'
        '<div style="font-size: 0.75rem; margin-top: 6px;">'
        '<div style="display: flex; justify-content: space-between; margin-bottom: 2px;">'
        '<span style="color: #64748b;">Cosine Alignment:</span>'
        f'<strong style="color: {cos_color};">{cosine:+.3f}</strong>'
        '</div>'
        '<div class="progress-track">'
        f'<div class="progress-bar-fill" style="width: {max(0, min(100, (cosine + 1) * 50)):.1f}%; background: {cos_color};"></div>'
        '</div>'
        '<div style="display: flex; justify-content: space-between; margin-top: 6px; margin-bottom: 2px;">'
        '<span style="color: #64748b;">Sign Conflict:</span>'
        f'<strong style="color: {conflict_color};">{conflict_pct:.1f}%</strong>'
        '</div>'
        '<div class="progress-track">'
        f'<div class="progress-bar-fill" style="width: {min(100, conflict_pct):.1f}%; background: {conflict_color};"></div>'
        '</div>'
        '<div style="display: flex; justify-content: space-between; margin-top: 6px; color: #64748b; font-size: 0.72rem;">'
        f'<span>Sparsity: {sparsity:.1%}</span>'
        '<span>Status: Active</span>'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def get_hardware_info() -> dict:
    """Detect local host hardware specifications."""
    cpu_name = platform.processor() or "Host Processor"
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1) if psutil else 16.0
    os_name = f"{platform.system()} {platform.release()}"
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "GPU Acceleration Optional (CPU FP32 Active)"
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1) if gpu_available else 0.0

    return {
        "os": os_name,
        "cpu": cpu_name,
        "ram_gb": ram_gb,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "cuda_active": gpu_available,
        "torch_version": torch.__version__,
        "device": "cuda" if gpu_available else "cpu (PyTorch FP32 Simulation)",
    }
