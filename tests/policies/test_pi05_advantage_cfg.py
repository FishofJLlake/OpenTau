#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from opentau.policies.pi05.configuration_pi05 import PI05Config
from opentau.policies.pi05.modeling_pi05 import (
    PI05FlowMatching,
    apply_classifier_free_guidance,
)


class _FakePrefixModel(nn.Module):
    def forward(self, *, inputs_embeds, **_kwargs):
        prefix_embs = inputs_embeds[0]
        batch_size, sequence_length, hidden_dim = prefix_embs.shape
        prefix_out = torch.zeros(batch_size, sequence_length, hidden_dim, device=prefix_embs.device)
        cache = {
            0: {
                "key_states": torch.zeros(batch_size, 1, 1, hidden_dim, device=prefix_embs.device),
                "value_states": torch.zeros(batch_size, 1, 1, hidden_dim, device=prefix_embs.device),
            }
        }
        return (prefix_out, None), cache


def _make_advantage_model(
    mode: str,
    *,
    threshold: float = 0.0,
    dropout: float = 0.3,
    guidance_scale: float = 1.0,
    predict_response: bool = False,
):
    model = object.__new__(PI05FlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        advantage=mode,
        advantage_threshold=threshold,
        cfg_dropout=dropout,
        guidance_scale=guidance_scale,
        chunk_size=2,
        max_action_dim=1,
        num_steps=1,
        predict_response=predict_response,
        response_max_length=2,
    )
    model.register_buffer("_advantage_positive_tokens", torch.tensor([10, 11, 0]), persistent=False)
    model.register_buffer("_advantage_positive_masks", torch.tensor([True, True, False]), persistent=False)
    model.register_buffer("_advantage_negative_tokens", torch.tensor([20, 21, 22]), persistent=False)
    model.register_buffer("_advantage_negative_masks", torch.tensor([True, True, True]), persistent=False)
    model._missing_inference_advantage_logged = False
    return model


def _prepare_sampling_model(mode: str, *, guidance_scale: float, predict_response: bool = False):
    model = _make_advantage_model(
        mode,
        guidance_scale=guidance_scale,
        predict_response=predict_response,
    )
    model.paligemma_with_expert = _FakePrefixModel()
    model.test_events = []

    def embed_prefix(self, _images, _img_masks, lang_tokens, _lang_masks, **_kwargs):
        batch_size = lang_tokens.shape[0]
        return (
            torch.zeros(batch_size, 1, 2),
            torch.ones(batch_size, 1, dtype=torch.bool),
            torch.ones(batch_size, 1, dtype=torch.long),
        )

    def append_advantage(
        self,
        prefix_pad_masks,
        past_key_values,
        _prefix_offsets,
        _advantage_tokens,
        advantage_masks,
    ):
        self.test_events.append(("append", advantage_masks.clone()))
        return torch.cat([prefix_pad_masks, advantage_masks], dim=1), past_key_values

    def denoise_step(self, _prefix_pad_masks, _past_key_values, x_t, _time):
        self.test_events.append(("denoise", x_t.shape[0]))
        if x_t.shape[0] == 2:
            conditional = torch.full_like(x_t[:1], 3.0)
            unconditional = torch.full_like(x_t[:1], 1.0)
            return torch.cat([conditional, unconditional], dim=0)
        return torch.ones_like(x_t)

    model.embed_prefix = MethodType(embed_prefix, model)
    model._append_advantage_to_cache = MethodType(append_advantage, model)
    model.denoise_step = MethodType(denoise_step, model)
    return model


def _sample(model, advantage=None):
    return model.sample_actions(
        images=[],
        img_masks=[],
        lang_tokens=torch.zeros(1, 1, dtype=torch.long),
        lang_masks=torch.ones(1, 1, dtype=torch.bool),
        action_prefix=torch.zeros(1, 2, 1),
        delay=torch.zeros(1, 1, dtype=torch.long),
        noise=torch.zeros(1, 2, 1),
        advantage=advantage,
    )


def test_use_mode_missing_inference_advantage_falls_back_to_ignore():
    model = _make_advantage_model("use")
    tokens, masks = model.prepare_advantage_tokens(
        None,
        batch_size=2,
        device=torch.device("cpu"),
        apply_cfg_dropout=False,
        inference=True,
    )
    assert tokens is None
    assert masks is None


def test_use_mode_missing_training_advantage_raises():
    model = _make_advantage_model("use")
    with pytest.raises(ValueError, match="requires an advantage tensor"):
        model.prepare_advantage_tokens(
            None,
            batch_size=1,
            device=torch.device("cpu"),
            apply_cfg_dropout=False,
            inference=False,
        )


def test_advantage_threshold_selects_positive_and_negative_rows():
    model = _make_advantage_model("use", threshold=0.5)
    tokens, masks = model.prepare_advantage_tokens(
        torch.tensor([0.5, 0.49]),
        batch_size=2,
        device=torch.device("cpu"),
        apply_cfg_dropout=False,
        inference=False,
    )
    assert tokens.tolist() == [[10, 11, 0], [20, 21, 22]]
    assert masks.tolist() == [[True, True, False], [True, True, True]]


def test_cfg_dropout_masks_whole_indicator_per_sample(monkeypatch):
    model = _make_advantage_model("on", dropout=0.3)
    monkeypatch.setattr(
        torch,
        "rand",
        lambda size, device=None: torch.tensor([0.0, 1.0], device=device)[:size],
    )
    _, masks = model.prepare_advantage_tokens(
        None,
        batch_size=2,
        device=torch.device("cpu"),
        apply_cfg_dropout=True,
        inference=False,
    )
    assert masks[0].logical_not().all()
    assert masks[1].tolist() == [True, True, False]


def test_classifier_free_guidance_formula():
    conditional = torch.tensor([[[3.0, 5.0]]])
    unconditional = torch.tensor([[[1.0, 2.0]]])
    result = apply_classifier_free_guidance(conditional, unconditional, 2.0)
    torch.testing.assert_close(result, torch.tensor([[[5.0, 8.0]]]))


def test_use_mode_missing_inference_advantage_uses_single_unconditioned_branch():
    model = _prepare_sampling_model("use", guidance_scale=2.0)
    result = _sample(model)

    assert [event[0] for event in model.test_events] == ["denoise"]
    assert model.test_events[0][1] == 1
    torch.testing.assert_close(result, torch.full((1, 2, 1), -1.0))


def test_on_mode_without_advantage_still_appends_positive_condition():
    model = _prepare_sampling_model("on", guidance_scale=1.0)
    _sample(model)

    append_event, denoise_event = model.test_events
    assert append_event[0] == "append"
    assert append_event[1].tolist() == [[True, True, False]]
    assert denoise_event == ("denoise", 1)


def test_cfg_uses_one_double_batch_denoise_call_and_combines_velocities():
    model = _prepare_sampling_model("use", guidance_scale=2.0)
    result = _sample(model, advantage=torch.tensor([1.0]))

    append_event, denoise_event = model.test_events
    assert append_event[1].tolist() == [
        [True, True, False],
        [False, False, False],
    ]
    assert denoise_event == ("denoise", 2)
    torch.testing.assert_close(result, torch.full((1, 2, 1), -5.0))


def test_response_generation_precedes_advantage_conditioning():
    model = _prepare_sampling_model("on", guidance_scale=1.0, predict_response=True)
    original_prepare = model.prepare_advantage_tokens

    def infer_response(
        self,
        prefix_out,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        past_key_values,
        prefix_offsets,
        response_tokens,
        _auto_step,
        _batch_size,
        _device,
    ):
        self.test_events.append(("response", response_tokens.shape[1]))
        return (
            prefix_out,
            torch.cat([prefix_embs, torch.zeros(1, 1, prefix_embs.shape[-1])], dim=1),
            torch.cat([prefix_pad_masks, torch.ones(1, 1, dtype=torch.bool)], dim=1),
            torch.cat([prefix_att_masks, torch.ones(1, 1, dtype=torch.long)], dim=1),
            prefix_offsets + 1,
            torch.cat([response_tokens, torch.ones(1, 1, dtype=torch.long)], dim=1),
            past_key_values,
        )

    def prepare_advantage(self, *args, **kwargs):
        self.test_events.append(("prepare_advantage", None))
        return original_prepare(*args, **kwargs)

    model.infer_response = MethodType(infer_response, model)
    model.prepare_advantage_tokens = MethodType(prepare_advantage, model)
    _sample(model)

    assert [event[0] for event in model.test_events] == [
        "response",
        "response",
        "prepare_advantage",
        "append",
        "denoise",
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cfg_dropout": 1.0}, "cfg_dropout must be in"),
        ({"cfg_dropout": -0.1}, "cfg_dropout must be in"),
        ({"guidance_scale": 0.9}, "guidance_scale must be"),
    ],
)
def test_invalid_cfg_settings_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PI05Config(**kwargs)
