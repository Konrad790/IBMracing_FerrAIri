from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SectorSpec:
    name: str
    start: float
    end: float


DOCUMENTED_CORKSCREW_LAYOUT_NAME = "corkscrew_documented_turns"

# Distances are derived from `tracks/road/corkscrew/corkscrew.xml`.
# T1 starts at the lap line, so the launch straight before the first corner is
# intentionally folded into the T1 sector.
DOCUMENTED_CORKSCREW_SECTORS: tuple[SectorSpec, ...] = (
    SectorSpec("T1", 0.0, 406.134),
    SectorSpec("T2", 406.134, 647.632),
    SectorSpec("T3", 647.632, 908.914),
    SectorSpec("T4", 908.914, 1400.075),
    SectorSpec("T5", 1400.075, 1822.799),
    SectorSpec("T6", 1822.799, 2255.351),
    SectorSpec("T7", 2255.351, 2326.695),
    SectorSpec("T8", 2326.695, 2389.034),
    SectorSpec("T8A", 2389.034, 2571.890),
    SectorSpec("T9", 2571.890, 2854.263),
    SectorSpec("T10", 2854.263, 3162.162),
    SectorSpec("T11", 3162.162, 3522.798),
)

SECTOR_LAYOUTS: dict[str, tuple[SectorSpec, ...]] = {
    DOCUMENTED_CORKSCREW_LAYOUT_NAME: DOCUMENTED_CORKSCREW_SECTORS,
}


def get_sector_layout(layout_name: str | None) -> tuple[SectorSpec, ...]:
    if layout_name is None:
        return ()
    return SECTOR_LAYOUTS.get(layout_name, ())


def get_track_length(layout_name: str | None) -> float:
    sectors = get_sector_layout(layout_name)
    if not sectors:
        return 0.0
    return sectors[-1].end


def normalize_dist_from_start(layout_name: str | None, dist_from_start: float) -> float:
    track_length = get_track_length(layout_name)
    if track_length <= 0.0:
        return max(0.0, dist_from_start)
    if dist_from_start < 0.0:
        return 0.0
    if dist_from_start >= track_length:
        return dist_from_start % track_length
    return dist_from_start


def get_sector_name(layout_name: str | None, dist_from_start: float) -> str | None:
    sectors = get_sector_layout(layout_name)
    if not sectors:
        return None

    normalized = normalize_dist_from_start(layout_name, dist_from_start)
    for sector in sectors:
        if normalized < sector.end:
            return sector.name
    return sectors[-1].name
