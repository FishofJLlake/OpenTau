#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from opentau.scripts.get_advantage_and_percentiles import (
    SOURCE_HUMAN_INTERVENTION_OVERRIDE,
    SOURCE_TD,
    apply_human_intervention_override,
    compute_cached_advantage_records,
)


def test_human_intervention_overrides_effective_advantage():
    effective, source = apply_human_intervention_override(-3.0, 1)
    assert effective == 1.0
    assert source == SOURCE_HUMAN_INTERVENTION_OVERRIDE


def test_non_intervention_preserves_raw_td_advantage():
    effective, source = apply_human_intervention_override(-3.0, 0)
    assert effective == -3.0
    assert source == SOURCE_TD


def test_cached_raw_effective_and_source_records_stay_aligned():
    values = {
        (0, 0): {"v0": 2.0, "reward": 0.5},
        (0, 1): {"v0": 3.0, "reward": 1.0},
    }
    records = [
        (0, 0, 0.0, 1),
        (0, 1, 0.1, 0),
    ]

    effective, raw, sources = compute_cached_advantage_records(
        values,
        records,
        n_steps_look_ahead=1,
    )

    expected_keys = {(0, 0.0), (0, 0.1)}
    assert set(effective) == expected_keys
    assert set(raw) == expected_keys
    assert set(sources) == expected_keys
    assert raw[(0, 0.0)] == 1.5
    assert effective[(0, 0.0)] == 1.0
    assert sources[(0, 0.0)] == SOURCE_HUMAN_INTERVENTION_OVERRIDE
    assert raw[(0, 0.1)] == -2.0
    assert effective[(0, 0.1)] == -2.0
    assert sources[(0, 0.1)] == SOURCE_TD
