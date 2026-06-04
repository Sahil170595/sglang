"""page_size > 1 variants (flashinfer).

EAGLE3 page64 topk1 + topk8 tree (spec v2) + EAGLE/Llama-2 page4 (topk1 and
topk8, spec v1). Runs on the cheap (5090) runner.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecFeatureKit,
    SpecLogprobKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base, EagleLlama2Base

register_cuda_ci(est_time=720, stage="base-b", runner_config="1-gpu-small")


class TestEagle3Page64(Eagle3Base, SpecAccuracyKit, SpecLogprobKit, SpecFeatureKit):
    """EAGLE3 spec v2, page_size=64 (flashinfer): + logprob losslessness."""

    page_size = 64
    disable_overlap = False
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagle3Page64Topk8(Eagle3Base, SpecAccuracyKit, SpecFeatureKit):
    """EAGLE3 spec v2 tree (topk=8) + page_size=64: the holey per-branch draft path
    (topk>1 + page>1 stays on v2 via partial-page duplication)."""

    page_size = 64
    spec_topk = 8
    spec_tokens = 32
    disable_overlap = False
    cuda_graph_max_bs = 5
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama2Page4Topk1(EagleLlama2Base, SpecAccuracyKit, SpecFeatureKit):
    """Llama-2 topk=1 + page_size=4."""

    spec_topk = 1
    spec_tokens = 6
    page_size = 4
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama2Page4Topk8(EagleLlama2Base, SpecAccuracyKit, SpecFeatureKit):
    """Llama-2 topk>1 tree + page_size=4 (spec v1)."""

    page_size = 4
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


if __name__ == "__main__":
    unittest.main()
