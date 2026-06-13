#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import numpy as np

from opentau.scripts.compute_quantiles import compute_feature_quantiles


class _FakeArrowDataset:
    def __init__(self, values: np.ndarray, feature_key: str = "state"):
        self.values = values
        self.feature_key = feature_key
        self.column_names = [feature_key]

    def with_transform(self, transform):
        del transform
        return self

    def with_format(self, format_name):
        del format_name
        return self

    def select_columns(self, columns):
        assert columns == [self.feature_key]
        return self

    def iter(self, batch_size: int):
        for start in range(0, len(self.values), batch_size):
            yield {self.feature_key: self.values[start : start + batch_size]}


def test_histogram_quantiles_are_per_dimension_and_deterministic():
    values = np.stack(
        [
            np.arange(10_000, dtype=np.float32),
            np.arange(10_000, dtype=np.float32) * 2,
        ],
        axis=1,
    )
    dataset = _FakeArrowDataset(values)
    first = compute_feature_quantiles(dataset, "state")
    second = compute_feature_quantiles(dataset, "state")

    np.testing.assert_array_equal(first["q01"], second["q01"])
    np.testing.assert_array_equal(first["q99"], second["q99"])
    np.testing.assert_allclose(first["q01"], [100.0, 200.0], atol=4.0)
    np.testing.assert_allclose(first["q99"], [9900.0, 19800.0], atol=8.0)
