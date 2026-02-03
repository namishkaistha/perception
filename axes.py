# axes.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class AnchorAxis:
    name: str
    left_anchors: List[str]
    right_anchors: List[str]


AXES: Dict[str, AnchorAxis] = {
    "spontaneity": AnchorAxis(
        name="spontaneity",
        left_anchors=["Rigid", "Structured", "Planned", "Methodical", "Scheduled", "Consistent", "Routine"],
        right_anchors=["Spontaneous", "Flexible", "Impulsive", "Adventurous", "Chaotic", "Free-spirited", "Easy-going", "Playful"],
    ),
    "risk": AnchorAxis(
        name="risk",
        left_anchors=[
            "Risk-averse",
            "Cautious",
            "Conservative",
            "Careful",
            "Calculative",
            "Prudent",
            "Safety-first",
            "Measured"
        ],
        right_anchors=[
            "Risk-taking",
            "Bold",
            "Adventurous",
            "Daring",
            "Fearless",
            "Willing to take risks",
            "Experimental",
            "Aggressive"
        ],
    ),
    # risk-averse -> risk-taking (or reverse)
   "ambition": AnchorAxis(
        name="ambition",
        left_anchors=[
            "Laid back",
            "Easygoing",
            "Content",
            "Unmotivated",
            "Passive",
            "Low-key",
            "Relaxed",
        ],
        right_anchors=[
            "Ambitious",
            "Driven",
            "Goal-oriented",
            "Motivated",
            "Hardworking",
            "Competitive",
            "High-achieving",
        ],
    ),

    "creativity":AnchorAxis(
        name="creativity",
        left_anchors=[
            "Unoriginal",
            "Conventional",
            "By-the-book",
            "Predictable",
            "Derivative",
            "Traditional",
        ],
        right_anchors=[
            "Creative",
            "Imaginative",
            "Original",
            "Innovative",
            "Artistic",
            "Visionary",
            "Inventive",
        ],
    ),

    "reliability": AnchorAxis(
        name="reliability",
        left_anchors=[
            "Unreliable",
            "Inconsistent",
            "Flaky",
            "Unpredictable",
            "Disorganized",
            "Forgetful",
        ],
        right_anchors=[
            "Reliable",
            "Consistent",
            "Dependable",
            "Trustworthy",
            "Steady",
            "Responsible",
        ],
    )
}
