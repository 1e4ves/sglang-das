"""DAS-specific consumers obey the external-linker output discard decision."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.disaggregation.hidden_state import (
    get_pd_hidden_req_state as pd_hidden_state,
)
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
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


@pytest.mark.parametrize("send_owner_direct", [False, True])
def test_failed_hidden_rows_are_not_written_or_sent(send_owner_direct):
    class Request:
        pass

    failed, healthy = Request(), Request()
    for i, req in enumerate((failed, healthy)):
        req.rid = str(i)
        req.discard_output_reason = FINISH_ABORT(status_code=503) if i == 0 else None
        req.pending_bootstrap = False
        state = pd_hidden_state(req)
        state.src_indices = [2 * i + 1, 2 * i + 2]
        state.meta = {"hidden_start": 0, "hidden_len": 2}
    pool = SimpleNamespace(hidden_size=1, write=Mock())
    hidden = torch.tensor([[11.0], [12.0], [21.0], [22.0]])
    scheduler = SimpleNamespace(
        disagg_metadata_buffers=SimpleNamespace(pd_hidden_pool=pool),
        _extract_pd_hidden_states_from_result=lambda result: hidden,
        ps=SimpleNamespace(pp_rank=0),
        _send_pd_hidden_only_chunk=Mock(),
    )
    batch = SimpleNamespace(
        reqs=[failed, healthy], extend_lens=[2, 2], prefix_lens=[0, 0]
    )
    SchedulerDisaggregationPrefillMixin._write_pd_hidden_rows_for_batch(
        scheduler, batch, object(), send_owner_direct=send_owner_direct
    )
    pool.write.assert_called_once()
    indices, written = pool.write.call_args.args
    assert indices == [3, 4]
    assert torch.equal(written, hidden[2:])
    if send_owner_direct:
        scheduler._send_pd_hidden_only_chunk.assert_called_once_with(healthy)
    else:
        scheduler._send_pd_hidden_only_chunk.assert_not_called()
    assert pd_hidden_state(failed).current_src_indices is None


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
