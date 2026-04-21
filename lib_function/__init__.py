"""
NL2HLS: Natural Language to High-Level Synthesis with PPA Optimization

Implements PPA-HLS training following the PPA-RTL flow:
1. Base model generates HLS designs from functional descriptions
2. Synthesis → PPA metrics (Power, Performance, Area)
3. Weighted scoring → Preference data (winner > loser)
4. DPO (Direct Preference Optimization) for model refinement
"""

__version__ = "0.1.0"
