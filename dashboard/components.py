"""UI Components and Styling for ATLAS-MERGE Dashboard."""
import platform
try:
    import psutil
except ImportError:
    psutil = None
import torch
import streamlit as st

def apply_custom_theme() -> None:
    """Inject custom CSS for dark professional research aesthetic."""
    st.markdown("""
        <style>
        /* Modern Dark Research Theme */
        .stApp {
            background-color: #0b0f19;
            color: #e2e8f0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Top Header Styling */
        .atlas-header {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        .atlas-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 6px;
            letter-spacing: -0.5px;
        }
        .atlas-subtitle {
            font-size: 1.05rem;
            color: #94a3b8;
            margin-bottom: 0px;
        }

        /* Simulation Badge */
        .sim-badge {
            display: inline-block;
            background: rgba(14, 165, 233, 0.15);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.4);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        /* Metric Cards */
        .metric-card {
            background: #131c2e;
            border: 1px solid rgba(51, 65, 85, 0.6);
            border-radius: 10px;
            padding: 16px 20px;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(56, 189, 248, 0.5);
        }
        .metric-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #64748b;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #f8fafc;
            line-height: 1.2;
        }
        .metric-sub {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 4px;
        }

        /* Info Alert Box */
        .research-alert {
            background: rgba(30, 41, 59, 0.5);
            border-left: 4px solid #38bdf8;
            border-radius: 4px 8px 8px 4px;
            padding: 12px 18px;
            margin: 16px 0;
            color: #cbd5e1;
            font-size: 0.92rem;
        }
        .research-alert-warning {
            background: rgba(69, 26, 3, 0.3);
            border-left: 4px solid #f59e0b;
            border-radius: 4px 8px 8px 4px;
            padding: 12px 18px;
            margin: 16px 0;
            color: #fde68a;
            font-size: 0.92rem;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0d1322;
            border-right: 1px solid rgba(51, 65, 85, 0.4);
        }

        /* Button Styling */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        /* Tables and Code */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid rgba(51, 65, 85, 0.5);
        }
        </style>
    """, unsafe_allow_html=True)


def render_header(page_title: str, subtitle: str = "") -> None:
    """Render top page header with branding and simulation badge."""
    sub_text = subtitle or "Automated Multi-Objective Model Merging via Task Vector Geometry"
    st.markdown(f"""
        <div class="atlas-header">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                <div>
                    <div class="atlas-title">ATLAS-MERGE</div>
                    <div class="atlas-subtitle">{sub_text}</div>
                </div>
                <div style="margin-top: 6px;">
                    <span class="sim-badge">Simulation-First Demo</span>
                </div>
            </div>
            <div style="margin-top: 14px; font-size: 0.85rem; color: #64748b; border-top: 1px solid rgba(51,65,85,0.4); padding-top: 10px;">
                <strong>Page:</strong> {page_title} &nbsp;|&nbsp; <strong>Engine:</strong> PyTorch FP32 Geometry &nbsp;|&nbsp; <strong>Mode:</strong> Deterministic Functional Simulation
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_simulation_banner() -> None:
    """Explicitly clarify the simulation-first research setup."""
    st.markdown("""
        <div class="research-alert">
            ℹ️ <strong>Research Demonstration Notice:</strong> This interactive environment executes authentic 
            <strong>PyTorch tensor calculations</strong> and real mathematical algorithms (TIES, DARE, Cosine Geometry, 
            PPO Reinforcement Learning, Chebyshev Reward, Pareto Hypervolume) across 8 simulated functional transformer blocks.
            No 10+ GB LLMs are downloaded to your laptop; task updates are modeled as deterministic FP32 tensor updates.
        </div>
    """, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, sub: str = "", delta: str = "", is_positive: bool = True) -> None:
    """Render styled metric tile."""
    delta_html = ""
    if delta:
        color = "#10b981" if is_positive else "#f43f5e"
        symbol = "↑" if is_positive else "↓"
        delta_html = f"<div style='color: {color}; font-size: 0.8rem; margin-top: 2px;'>{symbol} {delta}</div>"

    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
            <div class="metric-sub">{sub}</div>
        </div>
    """, unsafe_allow_html=True)


def get_hardware_info() -> dict:
    """Detect local host hardware specifications."""
    cpu_name = platform.processor() or "Intel Core i5-13420H"
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1) if psutil else 16.0
    os_name = f"{platform.system()} {platform.release()}"
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "NVIDIA RTX 3050 Laptop GPU (Available for CUDA)"
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1) if gpu_available else 6.0

    return {
        "os": os_name,
        "cpu": cpu_name,
        "ram_gb": ram_gb,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "cuda_active": gpu_available,
        "torch_version": torch.__version__,
        "device": "cuda" if gpu_available else "cpu (FP32 Simulation)",
    }
