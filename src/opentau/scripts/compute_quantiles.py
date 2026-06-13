#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compute per-dimension q01/q99 statistics for state and action columns."""

import logging

import numpy as np

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.factory import make_dataset
from opentau.datasets.standard_data_format_mapping import resolve_feature_mapping
from opentau.datasets.utils import write_quantile_stats
from opentau.utils.utils import init_logging

NUM_QUANTILE_BINS = 5000
ARROW_BATCH_SIZE = 4096


def _as_matrix(values) -> np.ndarray:
    if hasattr(values, "to_pylist"):
        values = values.to_pylist()
    array = np.asarray(values)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array.reshape(-1, array.shape[-1])


def _iter_feature_batches(hf_dataset, feature_key: str):
    raw_dataset = hf_dataset.with_transform(None).with_format("arrow").select_columns([feature_key])
    for batch in raw_dataset.iter(batch_size=ARROW_BATCH_SIZE):
        matrix = _as_matrix(batch[feature_key])
        if matrix.shape[0] > 0:
            yield matrix


def compute_feature_quantiles(hf_dataset, feature_key: str) -> dict[str, np.ndarray]:
    """Compute deterministic histogram-based q01/q99 for one vector column."""
    column_names = set(hf_dataset.column_names)
    if feature_key not in column_names:
        raise KeyError(f"Feature {feature_key!r} is missing. Available columns: {sorted(column_names)}")

    minimum = None
    maximum = None
    count = 0
    for matrix in _iter_feature_batches(hf_dataset, feature_key):
        batch_min = np.min(matrix, axis=0)
        batch_max = np.max(matrix, axis=0)
        minimum = batch_min if minimum is None else np.minimum(minimum, batch_min)
        maximum = batch_max if maximum is None else np.maximum(maximum, batch_max)
        count += matrix.shape[0]

    if count < 2 or minimum is None or maximum is None:
        raise ValueError(f"Feature {feature_key!r} needs at least two rows, got {count}.")

    edges = []
    histograms = []
    for min_value, max_value in zip(minimum, maximum, strict=True):
        if min_value == max_value:
            min_value -= 1e-10
            max_value += 1e-10
        edges.append(np.linspace(min_value, max_value, NUM_QUANTILE_BINS + 1, dtype=np.float64))
        histograms.append(np.zeros(NUM_QUANTILE_BINS, dtype=np.int64))

    for matrix in _iter_feature_batches(hf_dataset, feature_key):
        for dim, dim_edges in enumerate(edges):
            histograms[dim] += np.histogram(matrix[:, dim], bins=dim_edges)[0]

    quantiles = []
    for quantile in (0.01, 0.99):
        target = quantile * count
        values = []
        for histogram, dim_edges in zip(histograms, edges, strict=True):
            cumulative = np.cumsum(histogram)
            index = int(np.searchsorted(cumulative, target, side="left"))
            index = min(index, NUM_QUANTILE_BINS - 1)
            count_before = cumulative[index - 1] if index > 0 else 0
            bin_count = histogram[index]
            fraction = 0.0 if bin_count == 0 else (target - count_before) / bin_count
            fraction = float(np.clip(fraction, 0.0, 1.0))
            values.append(dim_edges[index] + fraction * (dim_edges[index + 1] - dim_edges[index]))
        quantiles.append(np.asarray(values, dtype=np.float32))

    return {"q01": quantiles[0], "q99": quantiles[1]}


@parser.wrap()
def main(cfg: TrainPipelineConfig) -> None:
    for dataset_cfg in cfg.dataset_mixture.datasets:
        if dataset_cfg.repo_id is None:
            logging.info("Skipping VQA dataset %s: QUANTILE only applies to state/action.", dataset_cfg.vqa)
            continue

        dataset_result = make_dataset(dataset_cfg, cfg)
        dataset = dataset_result[0] if isinstance(dataset_result, tuple) else dataset_result
        mapping = resolve_feature_mapping(dataset_cfg.repo_id, dataset.control_mode)
        quantile_stats = {}
        for standard_key in ("state", "actions"):
            raw_key = mapping[standard_key]
            logging.info("Computing %s quantiles for %s (%s)", standard_key, dataset_cfg.repo_id, raw_key)
            quantile_stats[raw_key] = compute_feature_quantiles(dataset.hf_dataset, raw_key)

        write_quantile_stats(quantile_stats, dataset.root)
        logging.info("Wrote quantile stats for %s to %s", dataset_cfg.repo_id, dataset.root)


if __name__ == "__main__":
    init_logging()
    main()
