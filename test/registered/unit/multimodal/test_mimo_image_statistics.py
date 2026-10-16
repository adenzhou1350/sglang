"""Checkpoint image statistics must reach both ordinary and EPD MiMo paths."""

import asyncio
import unittest
from contextlib import ExitStack
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch

from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessor
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.models.mimo_v2 import MiMoV2ForCausalLM
from sglang.srt.multimodal.encoder_preprocessing import invoke_encoder_preprocessor
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.mimo_v2 import (
    ImageInput,
    MiMoProcessor,
    MiMoV2Processor,
    MultimodalSpecialTokens,
    VideoInput,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CLIP = SimpleNamespace(
    image_mean=[0.48145466, 0.4578275, 0.40821073],
    image_std=[0.26862954, 0.26130258, 0.27577711],
)
IMAGENET = SimpleNamespace(
    image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]
)


class TestMiMoImageStatistics(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            patch(
                "sglang.srt.multimodal.processors.mimo_v2.MiMoAudioPipeline",
                return_value=SimpleNamespace(),
            )
        )
        self.stack.enter_context(
            patch.object(
                BaseMultimodalProcessor,
                "__init__",
                lambda obj, hf, args, processor, *a, **kw: setattr(
                    obj, "_processor", processor
                ),
            )
        )
        self.stack.enter_context(
            patch.object(MultimodalSpecialTokens, "build", return_value=None)
        )
        self.stack.enter_context(
            patch(
                "sglang.srt.multimodal.processors.mimo_v2.envs."
                "SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU.get",
                return_value=False,
            )
        )
        self.config = SimpleNamespace(
            vision_config={"patch_size": 16, "spatial_merge_size": 2},
            processor_config={
                "audio_sampling_rate": 24000,
                "use_video_timestamps": True,
                "image_min_pixels": 1024,
                "image_max_pixels": 1024,
                "video_min_pixels": 1024,
                "video_max_pixels": 1024,
                "video_total_max_pixels": 2048,
                **{
                    name: index
                    for index, name in enumerate(
                        (
                            "vision_start_token_id",
                            "vision_end_token_id",
                            "image_token_id",
                            "video_token_id",
                            "audio_token_id",
                            "audio_start_token_id",
                            "audio_end_token_id",
                            "video_start_token_id",
                            "video_end_token_id",
                        ),
                        start=1,
                    )
                },
            },
        )
        self.image = torch.stack(
            [torch.full((32, 32), float(value)) for value in (32, 128, 224)]
        )
        self.frames = self.image.repeat(2, 1, 1, 1)
        self.timestamps = torch.tensor([0.0, 1.0])

    def processor(self, statistics):
        hf_processor = SimpleNamespace(
            tokenizer=SimpleNamespace(pad_token_id=0), image_processor=statistics
        )
        processor = MiMoV2Processor(self.config, SimpleNamespace(), hf_processor)
        return processor.mimo_processor

    def normalized(self, statistics):
        mean = torch.as_tensor(statistics.image_mean).reshape(-1, 1, 1)
        std = torch.as_tensor(statistics.image_std).reshape(-1, 1, 1)
        return (self.image / 255.0 - mean) / std

    def test_checkpoint_image(self):
        actual = self.processor(IMAGENET).process_image(ImageInput(self.image))
        torch.testing.assert_close(actual, self.normalized(IMAGENET))

    def test_checkpoint_video(self):
        processor = self.processor(IMAGENET)
        actual, _, _, _ = processor.process_video(
            VideoInput(video=(self.frames, self.timestamps))
        )
        expected, _ = processor._flatten_visual_inputs(
            self.normalized(IMAGENET).repeat(2, 1, 1, 1), "video"
        )
        torch.testing.assert_close(actual, expected)

    def test_missing_statistics_preserves_clip(self):
        actual = self.processor(None).process_image(ImageInput(self.image))
        torch.testing.assert_close(actual, self.normalized(CLIP))

    def test_cache_isolated_between_checkpoints(self):
        other = SimpleNamespace(image_mean=[0.3, 0.4, 0.5], image_std=[0.2, 0.3, 0.4])
        first, second = self.processor(IMAGENET), self.processor(other)
        cases = ((first, IMAGENET), (second, other), (first, IMAGENET))
        for processor, statistics in cases:
            actual = processor.process_image(ImageInput(self.image))
            torch.testing.assert_close(actual, self.normalized(statistics))

    def test_encoder_image_after_audio_first_initialization(self):
        model = SimpleNamespace(config=self.config, _encoder_processor=None)
        hook = partial(MiMoV2ForCausalLM.preprocess_mm_for_encoder, model)
        invoke_encoder_preprocessor(hook, [], Modality.AUDIO, {})
        actual = invoke_encoder_preprocessor(
            hook, [self.image], Modality.IMAGE, {}, image_processor=IMAGENET
        )
        expected, _ = model._encoder_processor._flatten_visual_inputs(
            self.normalized(IMAGENET), "image"
        )
        torch.testing.assert_close(actual["pixel_values"], expected)

    def test_encoder_video_uses_checkpoint_statistics(self):
        model = SimpleNamespace(config=self.config, _encoder_processor=None)
        hook = partial(MiMoV2ForCausalLM.preprocess_mm_for_encoder, model)
        owner = SimpleNamespace(image_processor=IMAGENET, vision_config={})
        with (
            patch.object(
                MiMoProcessor,
                "_load_video_for_encoder",
                return_value=(b"video", (self.frames, self.timestamps)),
            ),
            patch.object(MiMoProcessor, "has_audio_track", return_value=False),
        ):
            actual = asyncio.run(
                EncoderPreprocessor._process_video_items(owner, [b"video"], hook)
            )
        expected, _ = model._encoder_processor._flatten_visual_inputs(
            self.normalized(IMAGENET).repeat(2, 1, 1, 1), "video"
        )
        torch.testing.assert_close(actual["pixel_values_videos"], expected)

    def test_encoder_video_preserves_legacy_three_argument_hook(self):
        def legacy(items, modality, config):
            return {"count": len(items), "modality": modality, "config": config}

        owner = SimpleNamespace(image_processor=IMAGENET, vision_config={"video": {}})
        result = asyncio.run(
            EncoderPreprocessor._process_video_items(owner, [b"video"], legacy)
        )
        self.assertEqual(
            result,
            {"count": 1, "modality": Modality.VIDEO, "config": owner.vision_config},
        )


if __name__ == "__main__":
    unittest.main()
