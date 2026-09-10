from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _UnreadableDeviceTensor:
    def max(self):
        raise AssertionError("has_encoder_tokens must not reduce the device tensor")

    def item(self):
        raise AssertionError(
            "has_encoder_tokens must not materialize the device tensor"
        )

    def __bool__(self):
        raise AssertionError("has_encoder_tokens must not branch on the device tensor")


@pytest.mark.parametrize(
    ("encoder_lens_cpu", "expected"),
    [
        ([0], False),
        ([0, 0, 0], False),
        ([1], True),
        ([0, 128, 0], True),
    ],
)
def test_has_encoder_tokens_uses_host_mirror(encoder_lens_cpu, expected):
    batch = SimpleNamespace(
        encoder_lens=_UnreadableDeviceTensor(),
        encoder_lens_cpu=encoder_lens_cpu,
    )

    assert ForwardBatch.has_encoder_tokens(batch) is expected


def test_has_encoder_tokens_handles_non_encoder_batch():
    batch = SimpleNamespace(encoder_lens=None, encoder_lens_cpu=None)

    assert ForwardBatch.has_encoder_tokens(batch) is False


def test_has_encoder_tokens_requires_host_identity():
    batch = SimpleNamespace(
        encoder_lens=_UnreadableDeviceTensor(), encoder_lens_cpu=None
    )

    with pytest.raises(AssertionError):
        ForwardBatch.has_encoder_tokens(batch)
