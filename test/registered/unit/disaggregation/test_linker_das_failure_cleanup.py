"""DAS-specific consumers obey the external-linker output discard decision."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.managers.schedule_batch import FINISH_ABORT, ReqKvInfo
from sglang.srt.mem_cache.common import (
    _release_donated_swa_slots,
    checkpoint_kv_cache,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("failed,donate", [(False, False), (False, True), (True, True)])
def test_failed_load_cannot_use_one_time_prompt_donation(failed, donate):
    req = SimpleNamespace(
        rid="prompt",
        finished=lambda: False,
        discard_output_reason=FINISH_ABORT(status_code=503) if failed else None,
        skip_radix_cache_insert=True,
        allow_radix_cache_insert_once=donate,
        extend_range=SimpleNamespace(end=8),
        prefix_len=0,
    )
    cache = SimpleNamespace(checkpoint=Mock())
    checkpoint_kv_cache(req, cache)
    if donate and not failed:
        cache.checkpoint.assert_called_once_with(req, up_to=8)
    else:
        cache.checkpoint.assert_not_called()
    assert req.prefix_len == 8


def test_donated_swa_cleanup_uses_kv_record_and_runs_once():
    req = SimpleNamespace(dsv4_donated_swa_len=4, kv=ReqKvInfo(req_pool_idx=1))
    rows = torch.arange(16).reshape(2, 8)
    free = Mock()
    cache = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(free_swa=free),
        req_to_token_pool=SimpleNamespace(req_to_token=rows),
    )
    _release_donated_swa_slots(req, cache)
    _release_donated_swa_slots(req, cache)
    free.assert_called_once()
    assert torch.equal(free.call_args.args[0], rows[1, :4])
