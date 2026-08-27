"""
DermaSense CV-1 context-aware capture guidance.

Guidance is deterministic. The quality system identifies the problem;
this module maps the problem to a concrete corrective action.
"""

from __future__ import annotations


GUIDANCE = {
    "resolution": (
        "Use a higher-resolution image and avoid excessive cropping."
    ),
    "low_brightness": (
        "Move to a well-lit area and retake the image without glare."
    ),
    "high_brightness": (
        "Avoid direct glare or harsh light and retake the image."
    ),
    "low_contrast": (
        "Improve the lighting and retake the image with the lesion clearly visible."
    ),
    "motion_blur": (
        "Keep the camera steady and retake the image."
    ),
}


def guidance_for_issue(issue_type: str) -> str:
    """
    Return deterministic user guidance for a quality issue.

    Unknown issue types deliberately receive generic guidance rather
    than allowing free-form text generation.
    """

    if not isinstance(issue_type, str):
        raise TypeError(
            "issue_type must be a string"
        )

    return GUIDANCE.get(
        issue_type,
        "Retake the image under clearer conditions.",
    )
