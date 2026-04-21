"""
PPA-HLS configuration: optimization objectives and weights.

Based on the optimization objectives table:
- Power-Only, Performance-Only, Area-Only
- Power+Performance (PP), Power+Area (PowerA), Performance+Area (PerfA)
- PPA (all three)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple


class PPAObjective(Enum):
    """Seven PPA optimization objectives."""
    POWER = "power"           # Power-Only: w_P=1, w_P'=0, w_A=0
    PERF = "perf"             # Performance-Only: w_P=0, w_P'=1, w_A=0
    AREA = "area"             # Area-Only: w_P=0, w_P'=0, w_A=1
    PP = "pp"                 # Power + Performance: w_P=1/2, w_P'=1/2, w_A=0
    POWERA = "powera"         # Power + Area: w_P=1/2, w_P'=0, w_A=1/2
    PERFA = "perfa"           # Performance + Area: w_P=0, w_P'=1/2, w_A=1/2
    PPA = "ppa"               # All three: w_P=1/3, w_P'=1/3, w_A=1/3


# (w_P: Power weight, w_P': Performance weight, w_A: Area weight)
OBJECTIVE_WEIGHTS: Dict[PPAObjective, Tuple[float, float, float]] = {
    PPAObjective.POWER:  (1.0, 0.0, 0.0),
    PPAObjective.PERF:   (0.0, 1.0, 0.0),
    PPAObjective.AREA:   (0.0, 0.0, 1.0),
    PPAObjective.PP:     (0.5, 0.5, 0.0),
    PPAObjective.POWERA: (0.5, 0.0, 0.5),
    PPAObjective.PERFA:  (0.0, 0.5, 0.5),
    PPAObjective.PPA:    (1/3, 1/3, 1/3),
}


@dataclass
class PPAWeights:
    """PPA weight configuration."""
    w_power: float
    w_perf: float
    w_area: float

    @classmethod
    def from_objective(cls, objective: PPAObjective) -> "PPAWeights":
        wp, wp_prime, wa = OBJECTIVE_WEIGHTS[objective]
        return cls(w_power=wp, w_perf=wp_prime, w_area=wa)

    def __iter__(self):
        return iter((self.w_power, self.w_perf, self.w_area))


# Default preference dataset sizes (reference from table)
PREFERENCE_DATASET_SIZES = {
    PPAObjective.POWER: 1966,
    PPAObjective.PERF: 1827,
    PPAObjective.AREA: 1936,
    PPAObjective.PP: 1808,
    PPAObjective.POWERA: 1927,
    PPAObjective.PERFA: 1813,
    PPAObjective.PPA: 1892,
}


# Training config (SFT-RL)
@dataclass
class TrainingConfig:
    """SFT-RL training configuration."""
    # SFT phase
    sft_epochs: int = 3
    sft_batch_size: int = 4
    sft_lr: float = 2e-5
    sft_max_length: int = 2048

    # DPO phase
    dpo_epochs: int = 1
    dpo_batch_size: int = 2
    dpo_lr: float = 5e-7
    dpo_beta: float = 0.1  # DPO beta parameter
    dpo_max_length: int = 4096

    # Model
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    use_peft: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
