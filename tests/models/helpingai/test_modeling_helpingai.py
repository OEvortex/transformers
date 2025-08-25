# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch HelpingAI model."""

import unittest

from pytest import mark

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import (
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    slow,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        HelpingaiConfig,
        HelpingaiForCausalLM,
        HelpingaiForQuestionAnswering,
        HelpingaiForSequenceClassification,
        HelpingaiForTokenClassification,
        HelpingaiModel,
    )


class HelpingaiModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = HelpingaiConfig
        base_model_class = HelpingaiModel
        causal_lm_class = HelpingaiForCausalLM
        sequence_class = HelpingaiForSequenceClassification
        question_answering_class = HelpingaiForQuestionAnswering
        token_classification_class = HelpingaiForTokenClassification

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
    ):
        super().__init__(
            parent,
            batch_size,
            seq_length,
            is_training,
            use_input_mask,
            use_token_type_ids,
            use_labels,
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            type_sequence_label_size,
            initializer_range,
            num_labels,
            num_choices,
            pad_token_id,
            scope,
        )
        self.num_key_value_heads = num_key_value_heads


@require_torch
class HelpingaiModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = HelpingaiModelTester
    test_missing_keys = False
    test_torchscript = False
    test_generate_without_input_ids = False

    def setUp(self):
        super().setUp()


@require_torch_accelerator
class HelpingaiIntegrationTest(unittest.TestCase):
    def tearDown(self):
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    @slow
    def test_model_from_pretrained(self):
        # This test would be enabled once a pretrained model is available
        # For now, we just test that the model can be instantiated
        config = HelpingaiConfig()
        model = HelpingaiForCausalLM(config)
        self.assertIsInstance(model, HelpingaiForCausalLM)

    @mark.skip(reason="Model is not currently public - will update test post release")
    @slow
    def test_model_generation(self):
        EXPECTED_TEXT_COMPLETION = (
            """Once upon a time,In a village there was a farmer who had three sons. The farmer was very old and he"""
        )
        prompt = "Once upon a time"
        tokenizer = AutoTokenizer.from_pretrained("HelpingAI/hai3.1-checkpoint-0002")
        model = HelpingaiForCausalLM.from_pretrained("HelpingAI/hai3.1-checkpoint-0002", device_map="auto")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        generated_ids = model.generate(input_ids, max_new_tokens=20)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @mark.skip(reason="Model is not currently public - will update test post release")
    @slow
    @require_flash_attn
    @mark.flash_attn_test
    def test_model_generation_flash_attn(self):
        EXPECTED_TEXT_COMPLETION = (
            """In a village there was a farmer who had three sons. The farmer was very old and he"""
        )
        prompt = "Once upon a time"
        tokenizer = AutoTokenizer.from_pretrained("HelpingAI/hai3.1-checkpoint-0002")
        model = HelpingaiForCausalLM.from_pretrained(
            "HelpingAI/hai3.1-checkpoint-0002", device_map="auto", attn_implementation="flash_attention_2"
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        generated_ids = model.generate(input_ids, max_new_tokens=20)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text[len(prompt) :])