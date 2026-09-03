from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class FixedMIMOConfig:
    """Configuration for one fixed-antenna ideal-CSI MIMO wiretap setting."""

    Nt: int = 4
    Nr: int = 4
    Ne: int = 2
    snr_db: float = 15.0
    seed: int = 0
    channel_power: float = 1.0

    def __post_init__(self) -> None:
        for name in ("Nt", "Nr", "Ne"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError("{} must be positive".format(name))
        if self.channel_power <= 0:
            raise ValueError("channel_power must be positive")

    @property
    def Pt(self) -> float:
        return 10.0 ** (float(self.snr_db) / 10.0)

    @property
    def dimensions(self) -> tuple:
        return int(self.Nt), int(self.Nr), int(self.Ne)

    def metadata(self) -> Dict[str, Any]:
        return {
            "Nt": int(self.Nt),
            "Nr": int(self.Nr),
            "Ne": int(self.Ne),
            "snr_db": float(self.snr_db),
            "Pt": float(self.Pt),
            "seed": int(self.seed),
            "channel_law": "iid_complex_gaussian_unit_power",
            "channel_power": float(self.channel_power),
            "ideal_csi": True,
            "port_selection": False,
            "fluid_antennas": False,
        }

