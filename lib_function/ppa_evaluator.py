"""
PPA (Power, Performance, Area) evaluation for HLS designs.

Supports:
1. Parsing ForgeHLS JSON design data (BRAM, LUT, DSP, FF, clock period, latency)
2. Simulated PPA estimation when synthesis is unavailable
3. Integration with Vitis HLS / Design Compiler reports (extensible)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import json


@dataclass
class PPAMetrics:
    """PPA metrics for an HLS design."""
    power_mw: float = 0.0      # Power in mW
    perf_ns: float = 0.0       # Critical path / clock period in ns
    area_um2: float = 0.0      # Area in µm² (or use LUT+FF+BRAM as proxy)
    # HLS-specific (ForgeHLS format)
    bram_18k: int = 0
    lut: int = 0
    dsp: int = 0
    ff: int = 0
    estimated_clock_period_ns: float = 0.0
    best_case_latency: int = 0
    worst_case_latency: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "power_mw": self.power_mw,
            "perf_ns": self.perf_ns,
            "area_um2": self.area_um2,
            "BRAM_18K": self.bram_18k,
            "LUT": self.lut,
            "DSP": self.dsp,
            "FF": self.ff,
            "EstimatedClockPeriod": self.estimated_clock_period_ns,
            "Best-caseLatency": self.best_case_latency,
            "Worst-caseLatency": self.worst_case_latency,
        }


def _estimate_area_from_resources(lut: int, ff: int, bram: int, dsp: int) -> float:
    """
    Rough area proxy in µm² using typical FPGA resource costs.
    Uses LUT-equivalent: 1 BRAM ~ 1.2 LUT, 1 DSP ~ 0.5 LUT (heuristic).
    """
    lut_eq = lut + ff // 2 + bram * 1200 + dsp * 500
    # ~1 LUT ≈ 1 µm² at 7nm (very rough)
    return float(lut_eq)


def _estimate_power_from_resources(lut: int, ff: int, bram: int, dsp: int, clock_ns: float) -> float:
    """
    Rough power proxy in mW. Higher resources + higher frequency => more power.
    """
    if clock_ns <= 0:
        clock_ns = 10.0
    freq_mhz = 1000.0 / clock_ns
    # Heuristic: dynamic power scales with resources and frequency
    return (lut * 0.001 + ff * 0.0005 + bram * 0.5 + dsp * 2.0) * (freq_mhz / 100.0)


class PPAEvaluator:
    """
    Evaluates PPA metrics for HLS designs.

    Can load from ForgeHLS JSON or use simulated values for designs without synthesis.
    """

    def __init__(self, use_simulated_fallback: bool = True):
        self.use_simulated_fallback = use_simulated_fallback

    def from_forgehls_json(self, design: Dict[str, Any]) -> PPAMetrics:
        """Parse PPA from ForgeHLS design JSON."""
        lut = int(design.get("LUT", 0))
        ff = int(design.get("FF", 0))
        bram = int(design.get("BRAM_18K", 0))
        dsp = int(design.get("DSP", 0))
        clock_ns = float(design.get("EstimatedClockPeriod", design.get("TargetClockPeriod", 10.0)))
        best_lat = int(design.get("Best-caseLatency", 0))
        worst_lat = int(design.get("Worst-caseLatency", 0))

        area = _estimate_area_from_resources(lut, ff, bram, dsp)
        power = _estimate_power_from_resources(lut, ff, bram, dsp, clock_ns)

        return PPAMetrics(
            power_mw=power,
            perf_ns=clock_ns,
            area_um2=area,
            bram_18k=bram,
            lut=lut,
            dsp=dsp,
            ff=ff,
            estimated_clock_period_ns=clock_ns,
            best_case_latency=best_lat,
            worst_case_latency=worst_lat,
        )

    def from_synthesis_report(self, report_path: Union[str, Path]) -> Optional[PPAMetrics]:
        """Parse PPA from synthesis tool report (extensible for Vitis HLS / DC)."""
        path = Path(report_path)
        if not path.exists():
            return None
        # Placeholder: add real parser for csynth.rpt, utilization.rpt, etc.
        return None

    def estimate_from_code(self, source_code: str) -> PPAMetrics:
        """
        Simulated PPA estimation from C/C++ source (heuristic).
        Used when synthesis is not available.
        """
        lines = len(source_code.splitlines())
        chars = len(source_code)
        # Use both lines and chars for differentiation; rough heuristics based on code size
        complexity = lines * 20 + chars // 10
        lut = min(500000, 500 + complexity)
        ff = min(500000, 300 + complexity // 2)
        bram = min(2000, complexity // 100)
        dsp = min(2000, complexity // 200)
        clock_ns = 5.0 + complexity * 0.0001  # Larger designs tend to have longer critical path
        area = _estimate_area_from_resources(lut, ff, bram, dsp)
        power = _estimate_power_from_resources(lut, ff, bram, dsp, clock_ns)
        return PPAMetrics(
            power_mw=power,
            perf_ns=clock_ns,
            area_um2=area,
            bram_18k=bram,
            lut=lut,
            dsp=dsp,
            ff=ff,
            estimated_clock_period_ns=clock_ns,
            best_case_latency=lines * 10,
            worst_case_latency=lines * 100,
        )

    def evaluate(self, design: Union[Dict, str]) -> PPAMetrics:
        """
        Evaluate PPA from design (ForgeHLS dict or source code string).
        """
        if isinstance(design, dict):
            return self.from_forgehls_json(design)
        if isinstance(design, str):
            if self.use_simulated_fallback:
                return self.estimate_from_code(design)
        raise TypeError("design must be dict (ForgeHLS) or str (source code)")


def load_forgehls_designs(json_path: Union[str, Path]) -> List[Dict]:
    """Load ForgeHLS designs from JSON file."""
    path = Path(json_path)
    if not path.exists():
        return []
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "designs" in data:
        return data["designs"]
    return [data]
