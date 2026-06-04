"""topk > 1 tree drafting (EAGLE3 topk16 + EAGLE/Llama-2 topk8), page1 and page>1.

topk > 1 runs on spec v1 (no overlap) and on spec v2 (overlap) for supported
backends -- including page_size > 1, where each branch's draft region is laid out
holey per page and the prefix partial-tail page is duplicated into it. flashinfer
is pinned because this runs on the cheap (5090) runner, where fa3 (Hopper-only)
isn't available -- functional sanity only, no perf/stress. (topk > 1 on fa3,
including page>1, is covered on the Hopper runner in test_spec_eagle_fa3.py.)
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecCorrectnessKit,
    SpecFeatureKit,
    SpecLogprobKit,
    SpecPenaltyKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base, EagleLlama2Base

register_cuda_ci(est_time=1500, stage="base-b", runner_config="1-gpu-small")


class TestEagle3Topk16(Eagle3Base, SpecCorrectnessKit, SpecAccuracyKit, SpecLogprobKit):
    """EAGLE3 topk=16 tree (spec v1): correctness + gsm8k + logprob losslessness."""

    spec_topk = 16
    spec_tokens = 64
    disable_overlap = True  # topk>1 -> spec v1
    cuda_graph_max_bs = 5
    acc_length_thres = 3.1
    batch_accept_len_thres = 1.75
    gsm8k_accept_len_thres = 2.4  # EAGLE3 topk16 gsm8k accept ~2.48
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagle3Topk16SpecV2(TestEagle3Topk16, SpecFeatureKit):
    """EAGLE3 topk=16 tree on spec v2 (overlap, page1): guards the v2 tree path's
    accepted-path compaction, validated by logprob_spec_v2_match."""

    disable_overlap = False


class TestEagle3Page64Topk8(Eagle3Base, SpecAccuracyKit, SpecFeatureKit):
    """EAGLE3 spec v2 tree (topk=8) + page_size=64: the holey per-branch draft path
    (topk>1 + page>1 stays on v2 via partial-page duplication)."""

    page_size = 64
    spec_topk = 8
    spec_tokens = 32
    disable_overlap = False
    cuda_graph_max_bs = 5
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama2Suite(
    EagleLlama2Base,
    SpecCorrectnessKit,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
):
    """EAGLE/Llama-2 topk=8 full coverage (kits listed in bases)."""

    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama2Chunked4(EagleLlama2Base, SpecCorrectnessKit):
    """Correctness under tiny chunked prefill."""

    chunked_prefill_size = 4
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama3TokenMap(EagleLlama2Base, SpecAccuracyKit):
    """EAGLE on Llama-3-8B with a FR-Spec token map (topk=4)."""

    model = "meta-llama/Meta-Llama-3-8B-Instruct"
    draft_model = "lmsys/sglang-EAGLE-LLaMA3-Instruct-8B"
    spec_topk = 4
    spec_tokens = 8
    cuda_graph_max_bs = 5
    gsm8k_accept_len_thres = 2.5  # FR-Spec token map lowers accept (~2.57)
    extra_args = (
        "--speculative-token-map",
        "thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt",
    )
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama2Page4Topk8(EagleLlama2Base, SpecAccuracyKit, SpecFeatureKit):
    """Llama-2 topk>1 tree + page_size=4 (spec v1)."""

    page_size = 4
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


if __name__ == "__main__":
    unittest.main()
