"""Label-free binary64 confidence gates and frozen-prefix threshold derivation.

The runtime gate sees one score and one fixed threshold. Derivation uses the
existing label-blind rank order only to compare feasible whole-tie acceptance
sets with a previously specified prefix; it does not optimize prediction quality.
"""
import math

from .selective import confidence_order


MAX_THRESHOLD = math.nextafter(1.0, math.inf)


def _binary64(value, name, upper):
    # Python float and numpy.float64 are binary64 floats. Reject implicit decimal,
    # integer, boolean, float32 and string conversion instead of changing precision.
    if not isinstance(value, float) or not math.isfinite(value) or not 0.0 <= value <= upper:
        raise ValueError(f'{name} must be a finite binary64 float in [0, {upper!r}]')
    return float(value)


def use_specialist(confidence, threshold):
    """Accept this request iff its exact unrounded confidence is >= threshold."""
    score = _binary64(confidence, 'Confidence', 1.0)
    cutoff = _binary64(threshold, 'Threshold', MAX_THRESHOLD)
    return score >= cutoff


def route(confidence, threshold):
    """Return a destination without labels, IDs, batch ranks or model execution."""
    return 'specialist' if use_specialist(confidence, threshold) else 'gpt-6-luna'


def derive_threshold(ids, confidence, accepted_count):
    """Choose the nearest realizable fixed >= gate to an existing ranked prefix.

All boundary ties must be included or excluded together. Choose the option with
minimum symmetric difference from the target prefix; equally good options include
all boundary ties. For k=0 the next binary64 value above the observed maximum
accepts none of these examples; for k=n the minimum observed confidence accepts all.
    """
    if len(ids) == 0 or len(ids) != len(confidence):
        raise ValueError('Nonempty aligned IDs and confidence required')
    if not all(isinstance(row_id, str) and row_id for row_id in ids):
        raise ValueError('Nonempty string IDs required')
    if type(accepted_count) is not int or not 0 <= accepted_count <= len(ids):
        raise ValueError('Accepted count must be an integer between zero and the sample size')
    scores = [_binary64(value, 'Confidence', 1.0) for value in confidence]
    order = confidence_order(ids, scores)
    original_ids = [ids[index] for index in order[:accepted_count]]
    original = set(original_ids)
    # Derive q from deterministic rank order, including signed-zero ties, so even
    # the exact floating-point representation is invariant to input permutations.
    boundary_index = order[accepted_count - 1] if accepted_count else order[0]
    boundary = scores[boundary_index]
    include_count = sum(score >= boundary for score in scores)
    exclude_count = sum(score > boundary for score in scores)
    include_difference = abs(include_count - accepted_count)
    exclude_difference = abs(exclude_count - accepted_count)
    include = include_difference <= exclude_difference
    threshold = boundary if include else math.nextafter(boundary, math.inf)
    fixed = {row_id for row_id, score in zip(ids, scores) if use_specialist(score, threshold)}
    added, removed = sorted(fixed - original), sorted(original - fixed)
    return {
        'threshold': threshold, 'threshold_decimal': repr(threshold), 'threshold_hex': threshold.hex(),
        'comparison': 'confidence >= threshold', 'n_examples': len(ids),
        'target_accepted_count': accepted_count, 'accepted_count': len(fixed),
        'actual_coverage': len(fixed) / len(ids),
        'boundary_confidence': boundary, 'boundary_ties_count': include_count - exclude_count,
        'boundary_decision': 'include_boundary_ties' if include else 'exclude_boundary_ties',
        'boundary_include_count': include_count, 'boundary_exclude_count': exclude_count,
        'original_accepted_ids': original_ids, 'fixed_accepted_ids': sorted(fixed),
        'added_ids': added, 'removed_ids': removed,
        'minimum_symmetric_difference': len(added) + len(removed),
    }
