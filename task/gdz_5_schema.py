from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GZD5Question:
    name: str
    catalog_name: str
    answers: tuple[str, ...]


GZD5_QUESTIONS = (
    GZD5Question(
        "smooth_or_featured",
        "smooth-or-featured",
        ("smooth", "featured-or-disk", "artifact"),
    ),

    GZD5Question(
        "disk_edge_on",
        "disk-edge-on",
        ("yes", "no"),
    ),

    GZD5Question(
        "has_spiral_arms",
        "has-spiral-arms",
        ("yes", "no"),
    ),

    GZD5Question(
        "bar",
        "bar",
        ("strong", "weak", "no"),
    ),

    GZD5Question(
        "bulge_size",
        "bulge-size",
        ("dominant", "large", "moderate", "small", "none"),
    ),

    GZD5Question(
        "how_rounded",
        "how-rounded",
        ("round", "in-between", "cigar-shaped"),
    ),

    GZD5Question(
        "edge_on_bulge",
        "edge-on-bulge",
        ("boxy", "none", "rounded"),
    ),

    GZD5Question(
        "spiral_winding",
        "spiral-winding",
        ("tight", "medium", "loose"),
    ),

    GZD5Question(
        "spiral_arm_count",
        "spiral-arm-count",
        ("1", "2", "3", "4", "more-than-4", "cant-tell"),
    ),

    GZD5Question(
        "merging",
        "merging",
        ("none", "minor-disturbance", "major-disturbance", "merger"),
    ),
)


assert sum(len(q.answers) for q in GZD5_QUESTIONS) == 34
