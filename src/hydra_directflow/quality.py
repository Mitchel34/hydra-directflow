"""Explicit USGS qualifier admission, separate from missing-data decisions.

The historical engineering default is exact ("A",). The confirmed primary
policy requires A and permits only additional R flags. It classifies without
rewriting, sorting or deduplicating source qualifiers. Choosing either policy
never fills a target, changes a replay clock, or enables scientific training.
"""
from hashlib import sha256
from pathlib import Path

STRICT_A = "STRICT_EXACT_A_ONLY_DEVELOPMENT"
PRIMARY_APPROVED = "PRIMARY_APPROVED_A_OPTIONAL_R_V1"


def _strict_exact_a(flags):
    # Preserve the original engineering predicate, including its tuple contract.
    return type(flags) is tuple and flags == ("A",)


def _primary_approved(flags):
    if type(flags) not in (tuple, list) or not flags:
        return False
    if not all(type(flag) is str and flag in ("A", "R") for flag in flags):
        return False
    return "A" in flags


def quality_admitter(policy_id=STRICT_A):
    """Resolve a supported policy; reject unknown IDs instead of falling back."""
    if type(policy_id) is not str or policy_id not in (STRICT_A, PRIMARY_APPROVED):
        raise ValueError("Explicit supported USGS quality policy ID required")
    return _strict_exact_a if policy_id == STRICT_A else _primary_approved


def admit_usgs_quality(flags, *, policy_id=STRICT_A):
    """Classify an original qualifier sequence without changing its contents."""
    return quality_admitter(policy_id)(flags)


def quality_policy_binding(*, input_quality_policy=STRICT_A, target_quality_policy=STRICT_A):
    """Portable receipt fields for independently selected history/label policies."""
    quality_admitter(input_quality_policy)
    quality_admitter(target_quality_policy)
    return {
        "input_quality_policy": input_quality_policy,
        "target_quality_policy": target_quality_policy,
        "quality_policy": input_quality_policy if input_quality_policy == target_quality_policy
                          else "SEPARATE_INPUT_AND_TARGET_QUALITY_POLICIES",
        "quality_implementation_sha256": {"quality.py": sha256(Path(__file__).read_bytes()).hexdigest()},
    }
