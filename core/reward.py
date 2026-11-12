"""Multi-Objective Reward Function for ATLAS-MERGE.

Implements Preference-Conditioned Augmented Chebyshev Scalarization with
HellaSwag-style Retention Floor Constraints.
"""
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
from utils.validation import check_numerical_validity


@dataclass
class RewardBreakdown:
    """Detailed components of the computed multi-objective scalar reward."""
    total_reward: float
    chebyshev_reward: float
    chebyshev_distance: float
    augmentation_term: float
    retention_penalty: float
    math_preference: float
    code_preference: float
    retention_floor: float
    retention_achieved: float
    is_retention_violated: bool
    is_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MultiObjectiveReward:
    """Calculates preference-conditioned augmented Chebyshev scalarization rewards."""

    def __init__(
        self,
        rho: float = 0.05,
        ideal_math: float = 1.0,
        ideal_code: float = 1.0,
        retention_floor: float = 0.90,
        retention_penalty_weight: float = 5.0,
        invalid_penalty: float = -10.0,
    ):
        self.rho = rho
        self.ideal_math = ideal_math
        self.ideal_code = ideal_code
        self.retention_floor = retention_floor
        self.retention_penalty_weight = retention_penalty_weight
        self.invalid_penalty = invalid_penalty

    def calculate_reward(
        self,
        math_proxy: float,
        code_proxy: float,
        retention_proxy: float,
        w_math: float = 0.5,
        w_code: float = 0.5,
        is_valid: bool = True,
    ) -> RewardBreakdown:
        """Calculate the scalar reward according to preference-conditioned Chebyshev scalarization.
        
        Formula:
        d_inf = max( w_math * (ideal_math - math_proxy), w_code * (ideal_code - code_proxy) )
        R_cheb = - d_inf + rho * (w_math * math_proxy + w_code * code_proxy)
        P_ret = retention_penalty_weight * max(0.0, retention_floor - retention_proxy)
        Total Reward = R_cheb - P_ret
        """
        # Validity and NaN/Inf guard
        if (
            not is_valid
            or not check_numerical_validity(math_proxy, code_proxy, retention_proxy, w_math, w_code)
        ):
            return RewardBreakdown(
                total_reward=self.invalid_penalty,
                chebyshev_reward=self.invalid_penalty,
                chebyshev_distance=1.0,
                augmentation_term=0.0,
                retention_penalty=0.0,
                math_preference=w_math,
                code_preference=w_code,
                retention_floor=self.retention_floor,
                retention_achieved=0.0,
                is_retention_violated=True,
                is_valid=False,
            )

        # Normalize preference weights
        total_w = w_math + w_code
        if total_w > 0:
            w_m = w_math / total_w
            w_c = w_code / total_w
        else:
            w_m = 0.5
            w_c = 0.5

        # Distance to ideal point
        dist_math = w_m * (self.ideal_math - math_proxy)
        dist_code = w_c * (self.ideal_code - code_proxy)
        d_inf = max(dist_math, dist_code)

        # Augmented Chebyshev term to eliminate weak Pareto optima
        aug_term = self.rho * (w_m * math_proxy + w_c * code_proxy)
        r_cheb = -d_inf + aug_term

        # Retention constraint penalty
        ret_violation = max(0.0, self.retention_floor - retention_proxy)
        is_violated = ret_violation > 0.0
        ret_penalty = self.retention_penalty_weight * ret_violation

        total_reward = r_cheb - ret_penalty

        return RewardBreakdown(
            total_reward=float(total_reward),
            chebyshev_reward=float(r_cheb),
            chebyshev_distance=float(d_inf),
            augmentation_term=float(aug_term),
            retention_penalty=float(ret_penalty),
            math_preference=float(w_m),
            code_preference=float(w_c),
            retention_floor=float(self.retention_floor),
            retention_achieved=float(retention_proxy),
            is_retention_violated=is_violated,
            is_valid=True,
        )

