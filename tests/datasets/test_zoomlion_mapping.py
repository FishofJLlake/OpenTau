#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from opentau.datasets.standard_data_format_mapping import resolve_feature_mapping


def test_zoomlion_realman_dualarm_mapping():
    mapping = resolve_feature_mapping("zoomlion/realman_dualarm")
    assert mapping == {
        "camera0": "observation.images.head",
        "camera1": "observation.images.left",
        "camera2": "observation.images.right",
        "state": "observation.state",
        "actions": "action",
        "prompt": "task",
        "response": "response",
    }
