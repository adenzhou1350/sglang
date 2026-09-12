import random
import unittest
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import get_batch_request_metadata
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestScheduleBatchMetadata(unittest.TestCase):
    @staticmethod
    def _reference(reqs):
        mode = CaptureHiddenMode.NULL
        for req in reqs:
            mode = max(mode, req.return_hidden_states_mode)
        return (
            any(req.return_logprob for req in reqs),
            any(req.grammar for req in reqs),
            mode,
        )

    def test_empty_batch(self):
        self.assertEqual(
            get_batch_request_metadata([]),
            (False, False, CaptureHiddenMode.NULL),
        )

    def test_randomized_equivalence(self):
        rng = random.Random(0x5A17)
        for _ in range(10_000):
            reqs = [
                SimpleNamespace(
                    return_logprob=rng.random() < 0.17,
                    grammar=object() if rng.random() < 0.13 else None,
                    return_hidden_states_mode=CaptureHiddenMode(rng.randrange(3)),
                )
                for _ in range(rng.choice((0, 1, 2, 8, 32, 128)))
            ]
            self.assertEqual(get_batch_request_metadata(reqs), self._reference(reqs))


if __name__ == "__main__":
    unittest.main()
