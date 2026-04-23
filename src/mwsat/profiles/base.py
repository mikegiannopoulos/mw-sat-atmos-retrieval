from __future__ import annotations


REQUIRED_PROFILE_FIELDS = ("pressure", "temperature")


def validate_profile_data(profile: dict) -> dict:
    if not isinstance(profile, dict):
        raise TypeError("Profile data must be provided as a dictionary")

    missing_fields = [
        field for field in REQUIRED_PROFILE_FIELDS if field not in profile
    ]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"Profile data is missing required field(s): {missing}")

    return profile
