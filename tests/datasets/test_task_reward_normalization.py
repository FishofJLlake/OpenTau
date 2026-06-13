#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

from opentau.datasets.lerobot_dataset import (
    LeRobotDataset,
    _compute_task_reward_normalizers,
)


def test_task_reward_normalizer_uses_maximum_selected_episode_length():
    normalizers = _compute_task_reward_normalizers(
        episodes=[0, 1, 2, 3],
        episode_lengths={0: 10, 1: 25, 2: 7, 3: 100},
        episode_to_task_index={0: 4, 1: 4, 2: 9},
    )

    assert normalizers == {4: 25, 9: 7}


def test_task_reward_normalizer_falls_back_to_config():
    dataset = object.__new__(LeRobotDataset)
    dataset.cfg = SimpleNamespace(
        policy=SimpleNamespace(
            reward_config=SimpleNamespace(reward_normalizer=400),
        )
    )
    dataset.task_reward_normalizers = {4: 25}

    assert dataset._get_reward_normalizer_for_task(4) == 25
    assert dataset._get_reward_normalizer_for_task(9) == 400
