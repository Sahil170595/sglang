"""Repro: spec v2 tree drafting (topk>1) + page_size>1 KV-pool leak under multi-turn.

Observed 2026-06-03 on branch lsyin/spec-v2-topk-page (page>1 work, ~commit 04f6f3b4ea),
EAGLE3 Llama-3.1-8B, fa3, eager. The idle memory invariant check
(scheduler_components/invariant_checker._report_leak via on_idle) trips:

    ValueError: pool memory leak detected! [full] total=731136, available=722944,
      evictable=0, protected=0, session_held=0, uncached=0   (8192 = topk*pages*page slots)

Key conditions to expose it (NOT a normal config -- this is a stress probe):
  - spec v2 tree drafting: speculative_eagle_topk > 1 AND page_size > 1 (the holey
    per-branch draft layout this branch adds).
  - LARGE page (512) so the whole short multi-turn conversation stays in page 0
    (prefix_base == 0 every step -> every draft step exercises the page>1 holey path).
  - MULTI-TURN, few tokens per turn: the scheduler goes idle after every turn, so the
    idle leak check (_check_all_pools) runs each turn -- a per-request leak that a
    concurrent gsm8k run only hits at the very end shows up immediately here.
  - --disable-cuda-graph --disable-piecewise-cuda-graph: exercise the eager
    draft_decode_set_expand_metadata path directly.
  - Strict idle mem check ON so the leak is a hard failure, not a warning.

ROOT CAUSE / FIX: the page>1 over-allocation (prepare_for_decode reserves
kv_allocated_len = committed + 2*get_alloc_len_per_decode, the holey footprint
topk*num_new_pages*page) exceeded the req_to_token row width, which only reserved
4 + num_draft_tokens of spec headroom. release_kv_cache then silently clamped the
free range to the row width and the over-allocated slots were never returned ->
leak (and the holey gather could index OOB at larger page). Fixed in
model_runner_kv_cache_mixin._init_pools by widening extra_max_context_len to
max(4 + num_draft_tokens, 2 * get_alloc_len_per_decode) for spec v2 page>1 topk>1.
This test now passes; it guards against regression. (Note: the busy invariant
check self_check_during_busy is still disabled for topk>1, so only the idle path
catches a leak today.)

Run manually (needs a GPU + the two HF models cached):
    SGLANG_ENABLE_SPEC_V2=1 \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1 \
    python -m pytest test/manual/spec/test_spec_v2_page_kv_leak_repro.py -s
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT_MODEL = "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"

# page_size must exceed the full multi-turn conversation length so prefix_base stays 0
# (every draft step then runs the page>1 holey branch with last_page == whole context).
PAGE_SIZE = 512
SPEC_TOPK = 8
TURNS = 30
TOKENS_PER_TURN = 8


class TestSpecV2PageKvLeakRepro(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            TARGET_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft-model-path",
                DRAFT_MODEL,
                "--speculative-num-steps",
                "5",
                "--speculative-eagle-topk",
                str(SPEC_TOPK),
                "--speculative-num-draft-tokens",
                "32",
                "--page-size",
                str(PAGE_SIZE),
                "--attention-backend",
                "fa3",
                "--mem-fraction-static",
                "0.75",
                "--max-running-requests",
                "8",
                "--chunked-prefill-size",
                str(PAGE_SIZE),  # must be divisible by page_size
                "--dtype",
                "float16",
                "--context-length",
                "8192",
                "--trust-remote-code",
                "--disable-cuda-graph",
                "--disable-piecewise-cuda-graph",
            ],
            env={
                "SGLANG_ENABLE_SPEC_V2": "1",
                "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
                # Make the idle leak invariant a hard error instead of a warning.
                "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "1",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_multi_turn_no_kv_leak(self):
        """Grow one conversation a few tokens per turn (staying under page_size) and
        assert the server survives. With the leak present, the scheduler aborts mid-loop
        with 'pool memory leak detected!' and the next request gets connection-refused.
        """
        requests.get(self.base_url + "/flush_cache")
        ctx = "Let's have a long conversation about science and the universe."
        for t in range(TURNS):
            r = requests.post(
                self.base_url + "/generate",
                json={
                    "text": ctx,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": TOKENS_PER_TURN,
                    },
                },
                timeout=600,
            )
            self.assertEqual(
                r.status_code, 200, f"turn {t} failed (server likely crashed): {r.text}"
            )
            ctx = ctx + r.json()["text"] + f" Tell me more about detail {t}:"

        # Server must still be healthy (a leak crash kills the scheduler).
        info = requests.get(self.base_url + "/server_info").json()
        print(
            "avg_spec_accept_length=",
            info["internal_states"][0].get("avg_spec_accept_length"),
        )


if __name__ == "__main__":
    unittest.main()
