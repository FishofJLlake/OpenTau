#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import pytest
import torch

from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.policies.normalize import Normalize, Unnormalize


def test_quantile_normalize_and_unnormalize_per_dataset():
    features = {"state": PolicyFeature(type=FeatureType.STATE, shape=(2,))}
    stats = [
        {"state": {"q01": torch.tensor([0.0, 10.0]), "q99": torch.tensor([10.0, 30.0])}},
        {"state": {"q01": torch.tensor([-5.0, 0.0]), "q99": torch.tensor([5.0, 100.0])}},
    ]
    norm_map = {"STATE": NormalizationMode.QUANTILE}
    normalize = Normalize(features, norm_map, per_dataset_stats=stats)
    unnormalize = Unnormalize(features, norm_map, per_dataset_stats=stats)
    dataset_index = torch.tensor([0, 1], dtype=torch.long)
    values = torch.tensor([[0.0, 30.0], [5.0, 50.0]])

    normalized = normalize({"state": values}, dataset_index)["state"]
    torch.testing.assert_close(
        normalized,
        torch.tensor([[-1.0, 1.0], [1.0, 0.0]]),
        atol=1e-6,
        rtol=0.0,
    )
    recovered = unnormalize({"state": normalized}, dataset_index)["state"]
    torch.testing.assert_close(recovered, values, atol=1e-5, rtol=1e-5)


def test_quantile_does_not_clip_outliers():
    features = {"actions": PolicyFeature(type=FeatureType.ACTION, shape=(1,))}
    stats = [{"actions": {"q01": torch.tensor([0.0]), "q99": torch.tensor([10.0])}}]
    normalize = Normalize(
        features,
        {"ACTION": NormalizationMode.QUANTILE},
        per_dataset_stats=stats,
    )
    value = normalize({"actions": torch.tensor([[20.0]])}, torch.zeros(1, dtype=torch.long))["actions"]
    assert value.item() > 1.0


def test_quantile_constant_dimension_round_trips():
    features = {"state": PolicyFeature(type=FeatureType.STATE, shape=(1,))}
    stats = [{"state": {"q01": torch.tensor([3.0]), "q99": torch.tensor([3.0])}}]
    normalize = Normalize(features, {"STATE": NormalizationMode.QUANTILE}, per_dataset_stats=stats)
    unnormalize = Unnormalize(features, {"STATE": NormalizationMode.QUANTILE}, per_dataset_stats=stats)
    dataset_index = torch.zeros(1, dtype=torch.long)

    normalized = normalize({"state": torch.tensor([[3.0]])}, dataset_index)["state"]
    recovered = unnormalize({"state": normalized}, dataset_index)["state"]

    torch.testing.assert_close(normalized, torch.tensor([[-1.0]]))
    torch.testing.assert_close(recovered, torch.tensor([[3.0]]))


def test_quantile_rejects_visual_features():
    features = {"camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8))}
    stats = [
        {
            "camera0": {
                "q01": torch.zeros(3, 1, 1),
                "q99": torch.ones(3, 1, 1),
            }
        }
    ]
    with pytest.raises(ValueError, match="only supports STATE and ACTION"):
        Normalize(
            features,
            {"VISUAL": NormalizationMode.QUANTILE},
            per_dataset_stats=stats,
        )


def test_quantile_missing_stats_points_to_compute_command():
    features = {"state": PolicyFeature(type=FeatureType.STATE, shape=(1,))}
    with pytest.raises(KeyError, match="opentau.scripts.compute_quantiles"):
        Normalize(
            features,
            {"STATE": NormalizationMode.QUANTILE},
            per_dataset_stats=[{"state": {}}],
        )
