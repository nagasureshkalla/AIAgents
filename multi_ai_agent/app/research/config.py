"""Research depth presets, bounded by the configured limits."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import ResearchConfig


@dataclass(frozen=True)
class DepthProfile:
    name: str
    iterations: int
    sources: int  # total pages to read
    sources_per_iteration: int
    queries_per_iteration: int
    chunks_per_source: int
    evidence_chars: int  # evidence budget handed to the reasoning model


_PRESETS = {
    "quick": DepthProfile("quick", 1, 4, 4, 2, 3, 6000),
    "normal": DepthProfile("normal", 3, 8, 4, 3, 4, 9000),
    "deep": DepthProfile("deep", 5, 14, 5, 4, 5, 12000),
}


def depth_profile(depth: str | None, cfg: ResearchConfig) -> DepthProfile:
    base = _PRESETS.get((depth or cfg.default_depth).lower(), _PRESETS["normal"])
    return DepthProfile(
        name=base.name,
        iterations=min(base.iterations, cfg.max_iterations),
        sources=min(base.sources, cfg.max_sources),
        sources_per_iteration=min(base.sources_per_iteration, cfg.max_sources),
        queries_per_iteration=base.queries_per_iteration,
        chunks_per_source=base.chunks_per_source,
        evidence_chars=base.evidence_chars,
    )
