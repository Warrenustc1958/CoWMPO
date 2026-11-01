# Copyright 2026 The RLinf Authors.
"""Conservative paired ranking of literal and category prompts in one shared image crop."""
import numpy as np


def preference(rows, prompts, margin=.05):
    """Require two distinct, uncensored same-prompt comparisons to agree by a fixed gap."""
    prompts = set(prompts)
    if len(prompts) < 2:
        return None
    decisions = []
    for prompt in sorted(prompts):
        entries = [r for r in rows if r['prompt'] == prompt]
        if len(entries) != 1:
            return None
        scores = entries[0]['scores']
        a, b = scores.get('chosen'), scores.get('alt')
        if a is None or b is None or not np.isfinite([a, b]).all():
            return None
        decisions.append('chosen' if a-b >= margin else 'alt' if b-a >= margin else None)
    return decisions[0] if decisions[0] is not None and len(set(decisions)) == 1 else None
