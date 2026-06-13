r"python src/opentau/scripts/get_advantage_and_percentiles.py  \
--config_path=outputs/train/2025-11-29/00-38-59_value/checkpoints/00520000 \
--batch_size=20 \
--dataloader_batch_size=20 \
--dataset_mixture=examples/advantage_config.json"

#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import draccus
import numpy as np
import torch
from torch.utils.data import DataLoader

from opentau.configs import parser
from opentau.configs.default import DatasetMixtureConfig
from opentau.configs.refs import resolve_refs_to_tempfile
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.factory import make_dataset
from opentau.datasets.utils import ADVANTAGE_SOURCES_PATH, ADVANTAGES_PATH, RAW_ADVANTAGES_PATH
from opentau.policies.factory import get_policy_class
from opentau.policies.value.reward import calculate_n_step_return
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import (
    auto_torch_device,
    init_logging,
)


def ensure_primitive(maybe_tensor):
    """Convert single-element tensors/arrays to Python scalars so they can be used as stable dict keys."""
    if isinstance(maybe_tensor, np.ndarray):
        return ensure_primitive(torch.from_numpy(maybe_tensor))
    if isinstance(maybe_tensor, torch.Tensor):
        assert maybe_tensor.numel() == 1, f"Tensor must be a single value, got shape={maybe_tensor.numel()}"
        return maybe_tensor.item()
    return maybe_tensor


_default0 = defaultdict(int)
POSITIVE_ADVANTAGE_OVERRIDE = 1.0
SOURCE_TD = "td"
SOURCE_HUMAN_INTERVENTION_OVERRIDE = "human_intervention_override"


def apply_human_intervention_override(
    raw_advantage: float, human_intervention: int | float
) -> tuple[float, str]:
    """Return the effective conditioning advantage and its audit source."""
    if human_intervention > 0:
        return POSITIVE_ADVANTAGE_OVERRIDE, SOURCE_HUMAN_INTERVENTION_OVERRIDE
    return raw_advantage, SOURCE_TD


def compute_cached_advantage_records(values, records, n_steps_look_ahead: int):
    """Compute aligned raw/effective/source maps from cached scalar records."""
    effective_advantages = {}
    raw_advantages = {}
    advantage_sources = {}
    for episode_index, current_idx, timestamp, human_intervention in records:
        look_ahead_idx = current_idx + n_steps_look_ahead
        vn = values.get((episode_index, look_ahead_idx), _default0)["v0"]
        reward = values.get((episode_index, current_idx), _default0)["reward"]
        v0 = values.get((episode_index, current_idx), _default0)["v0"]
        raw_advantage = ensure_primitive(reward + vn - v0)
        effective_advantage, source = apply_human_intervention_override(raw_advantage, human_intervention)
        key = (episode_index, timestamp)
        raw_advantages[key] = raw_advantage
        effective_advantages[key] = effective_advantage
        advantage_sources[key] = source
    return effective_advantages, raw_advantages, advantage_sources


# Store dataset_mixture_path before filtering (needed for parsing inside main)
# Handle both --dataset_mixture_path=<path> and --dataset_mixture=<path> (without nested fields)
_dataset_mixture_path_value = None
for arg in sys.argv:
    if arg.startswith("--dataset_mixture_path="):
        _dataset_mixture_path_value = arg.split("=", 1)[1]
        break
    elif arg.startswith("--dataset_mixture=") and "." not in arg.split("=", 1)[0]:
        # --dataset_mixture=<path> without nested fields (e.g., not --dataset_mixture.datasets.0.repo_id=...)
        _dataset_mixture_path_value = arg.split("=", 1)[1]
        break

# Create a wrapper that filters dataset_mixture path arguments before draccus parsing
_original_wrap = parser.wrap()


def _filter_dataset_mixture_path(fn):
    """Wrapper that filters --dataset_mixture_path and --dataset_mixture=<path> from sys.argv before draccus sees it."""
    wrapped_fn = _original_wrap(fn)

    def filtered_wrapper(*args, **kwargs):
        # If config is already provided, just call the function
        if len(args) > 0:
            return wrapped_fn(*args, **kwargs)

        # Otherwise, filter dataset_mixture path arguments from sys.argv before draccus parses
        original_argv = sys.argv.copy()
        try:
            filtered_args = []
            for arg in sys.argv:
                # Filter --dataset_mixture_path=<path>
                if (
                    arg.startswith("--dataset_mixture_path=")
                    or arg.startswith("--dataset_mixture=")
                    and "." not in arg.split("=", 1)[0]
                ):
                    continue
                else:
                    filtered_args.append(arg)
            sys.argv = filtered_args
            return wrapped_fn(*args, **kwargs)
        finally:
            sys.argv = original_argv

    return filtered_wrapper


@_filter_dataset_mixture_path
def main(cfg: TrainPipelineConfig):
    script_start = time.perf_counter()
    dataset_mixture_path = _dataset_mixture_path_value

    if dataset_mixture_path:
        logging.info(f"Loading dataset config from separate file: {dataset_mixture_path}")
        tmp_mixture = resolve_refs_to_tempfile(dataset_mixture_path)
        try:
            mixture_cfg = draccus.parse(
                config_class=DatasetMixtureConfig, config_path=str(tmp_mixture), args=[]
            )
        finally:
            tmp_mixture.unlink(missing_ok=True)
    else:
        logging.info("Using the dataset mixture config from the TrainPipelineConfig")
        mixture_cfg = cfg.dataset_mixture

    device = auto_torch_device()
    # torch.autograd.set_detect_anomaly(True)

    # TODO(shuheng): Do we need the random seed here?
    if cfg.seed is not None:
        set_seed(cfg.seed)

    logging.info("Creating policy")
    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
    policy.to(device=device, dtype=torch.bfloat16)
    policy.eval()

    # Effective advantages are consumed by policy conditioning; raw advantages
    # retain the TD residual before any intervention override.
    advantages = []
    raw_advantages = []

    for dataset_idx, dataset_cfg in enumerate(mixture_cfg.datasets):
        dataset_start = time.perf_counter()
        logging.info("Creating dataset %s", dataset_idx)
        ds_res = make_dataset(dataset_cfg, cfg, return_advantage_input=True)
        dataset = ds_res[0] if isinstance(ds_res, tuple) else ds_res
        dataloader_kwargs = {
            "batch_size": cfg.batch_size,
            "shuffle": False,
            "drop_last": False,
            "num_workers": cfg.num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        if cfg.num_workers > 0:
            dataloader_kwargs["persistent_workers"] = True
            if cfg.prefetch_factor is not None:
                dataloader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        dataloader = DataLoader(
            dataset,
            **dataloader_kwargs,
        )

        values = {}
        advantage_records = []
        ds_advantage = {}  # per-dataset advantages
        ds_raw_advantage = {}
        ds_advantage_source = {}
        first_pass_start = time.perf_counter()
        with torch.inference_mode():
            # The only dataloader pass: decode inputs, predict values, and retain
            # the compact scalar records needed for the TD look-ahead pass.
            for batch in dataloader:
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)

                reward_normalizers = batch.get("reward_normalizer")
                if reward_normalizers is None:
                    reward_normalizers = [cfg.policy.reward_config.reward_normalizer] * len(
                        batch["current_idx"]
                    )
                human_interventions = batch.get("human_intervention")
                if human_interventions is None:
                    human_interventions = [0] * len(batch["current_idx"])

                predicted_values = policy.predict_value(batch)
                for (
                    success,
                    episode_index,
                    episode_end_idx,
                    current_idx,
                    reward_normalizer,
                    human_intervention,
                    timestamp,
                    v0,
                ) in zip(
                    batch["success"],
                    batch["episode_index"],
                    batch["episode_end_idx"],
                    batch["current_idx"],
                    reward_normalizers,
                    human_interventions,
                    batch["timestamp"],
                    predicted_values,
                    strict=True,
                ):
                    (
                        success,
                        episode_index,
                        episode_end_idx,
                        current_idx,
                        reward_normalizer,
                        human_intervention,
                        timestamp,
                        v0,
                    ) = map(
                        ensure_primitive,
                        (
                            success,
                            episode_index,
                            episode_end_idx,
                            current_idx,
                            reward_normalizer,
                            human_intervention,
                            timestamp,
                            v0,
                        ),
                    )
                    reward = calculate_n_step_return(
                        success=success,
                        n_steps_look_ahead=cfg.policy.reward_config.N_steps_look_ahead,
                        episode_end_idx=episode_end_idx,
                        reward_normalizer=reward_normalizer,
                        current_idx=current_idx,
                        c_neg=cfg.policy.reward_config.C_neg,
                    )

                    values[(episode_index, current_idx)] = {"v0": v0, "reward": reward}
                    advantage_records.append((episode_index, current_idx, timestamp, human_intervention))

            logging.info(
                "Value pass complete: dataset=%s samples=%s elapsed=%.3fs",
                dataset_idx,
                len(advantage_records),
                time.perf_counter() - first_pass_start,
            )

            # Scalar-only pass over cached records; no image/video decode.
            ds_advantage, ds_raw_advantage, ds_advantage_source = compute_cached_advantage_records(
                values,
                advantage_records,
                cfg.policy.reward_config.N_steps_look_ahead,
            )
            advantages.extend(ds_advantage.values())
            raw_advantages.extend(ds_raw_advantage.values())

        # Convert tuple keys to strings for JSON serialization
        advantage_data_json = {f"{ep_idx},{ts}": f"{val:.6f}" for (ep_idx, ts), val in ds_advantage.items()}
        raw_advantage_data_json = {
            f"{ep_idx},{ts}": f"{val:.6f}" for (ep_idx, ts), val in ds_raw_advantage.items()
        }
        source_data_json = {f"{ep_idx},{ts}": source for (ep_idx, ts), source in ds_advantage_source.items()}

        dataset_root = Path(dataset.root)
        advantages_path = dataset_root / ADVANTAGES_PATH
        raw_advantages_path = dataset_root / RAW_ADVANTAGES_PATH
        advantage_sources_path = dataset_root / ADVANTAGE_SOURCES_PATH
        advantages_path.parent.mkdir(parents=True, exist_ok=True)
        with open(advantages_path, "w") as f:
            json.dump(advantage_data_json, f, indent=4)
        with open(raw_advantages_path, "w") as f:
            json.dump(raw_advantage_data_json, f, indent=4)
        with open(advantage_sources_path, "w") as f:
            json.dump(source_data_json, f, indent=4)
        logging.info(
            "Advantage files written: dataset=%s records=%s elapsed=%.3fs",
            dataset_idx,
            len(advantage_data_json),
            time.perf_counter() - dataset_start,
        )

    # Calculate percentiles of advantages: 0th, 5th, 10th, ..., 100th
    if not advantages:
        raise ValueError("No advantage records were produced.")
    percentiles = list(range(0, 101, 5))  # [0, 5, 10, 15, ..., 100]
    advantage_percentiles = np.percentile(np.array(advantages), percentiles)
    raw_advantage_percentiles = np.percentile(np.array(raw_advantages), percentiles)

    print("Effective advantage percentiles for policy conditioning:")
    for p, val in zip(percentiles, advantage_percentiles, strict=False):
        print(f"  {p:03d}th percentile: {val:.6f}")
    print("Raw TD advantage percentiles for deciding epsilon threshold:")
    for p, val in zip(percentiles, raw_advantage_percentiles, strict=False):
        print(f"  {p:03d}th percentile: {val:.6f}")
    logging.info("Advantage computation finished in %.3fs", time.perf_counter() - script_start)


if __name__ == "__main__":
    init_logging()
    main()
