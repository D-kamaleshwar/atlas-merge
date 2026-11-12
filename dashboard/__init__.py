"""Dashboard package for ATLAS-MERGE."""
from .components import apply_custom_theme, render_header, render_simulation_banner, render_metric_card, get_hardware_info
from .charts import (
    create_geometry_heatmap,
    create_pareto_scatter,
    create_hypervolume_chart,
    create_ppo_training_charts,
    create_action_distribution_chart,
    create_tensor_distribution_histogram,
    create_block_proxy_radar,
)

__all__ = [
    "apply_custom_theme",
    "render_header",
    "render_simulation_banner",
    "render_metric_card",
    "get_hardware_info",
    "create_geometry_heatmap",
    "create_pareto_scatter",
    "create_hypervolume_chart",
    "create_ppo_training_charts",
    "create_action_distribution_chart",
    "create_tensor_distribution_histogram",
    "create_block_proxy_radar",
]

