"""DSpark transfers prebuilt draft KV through the normal PD lifecycle."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from sglang.srt.disaggregation import decode, prefill
from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.utils import KVClassType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("mla", [True, False])
@pytest.mark.parametrize("pp_rank,pp_size", [(0, 1), (0, 2), (1, 2)])
def test_register_target_and_draft_kv_on_owning_pp_stage(mla, pp_rank, pp_size):
    # Each call returns fresh lists: registration appends draft buffers for MLA.
    def make_pool(ptrs, layers):
        return SimpleNamespace(
            start_layer=0,
            end_layer=layers,
            layer_num=layers,
            page_size=2,
            head_num=1,
            get_kv_layer_ids=lambda: list(range(layers)),
            get_contiguous_buf_infos=lambda: (
                list(ptrs),
                [1024] * len(ptrs),
                [64] * len(ptrs),
            ),
        )

    target = make_pool([10, 20] if mla else [10, 20, 30, 40], 2)
    draft = make_pool([50] if mla else [50, 60], 1)
    queue = prefill.PrefillBootstrapQueue.__new__(prefill.PrefillBootstrapQueue)
    queue.tp_rank = 0
    queue.pp_rank = pp_rank
    queue.pp_size = pp_size
    queue.transfer_backend = "mooncake"
    queue.is_mla_backend = mla
    queue.token_to_kv_pool = target
    queue.draft_token_to_kv_pool = draft
    queue.metadata_buffers = SimpleNamespace(get_buf_infos=lambda: ([], [], []))
    queue.scheduler = SimpleNamespace(
        ps=SimpleNamespace(dp_rank=0, gpu_id=0),
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(kv_cache_dtype_str="bfloat16")
        ),
        model_config=SimpleNamespace(
            num_hidden_layers=2, get_total_num_kv_heads=lambda: 1
        ),
        server_args=SimpleNamespace(),
    )
    manager_factory = Mock()

    def get_kv_class(backend, kind):
        return SimpleNamespace if kind == KVClassType.KVARGS else manager_factory

    with (
        patch.object(prefill, "get_kv_class", get_kv_class),
        patch.object(prefill, "setup_state_kv_args") as setup_state,
        patch.object(
            prefill,
            "get_disagg",
            return_value=SimpleNamespace(disaggregation_ib_device=None),
        ),
        patch.object(
            prefill,
            "envs",
            SimpleNamespace(
                SGLANG_DISAGG_STAGING_BUFFER=SimpleNamespace(get=lambda: False)
            ),
        ),
    ):
        queue._init_kv_manager()

    args = manager_factory.call_args.args[0]
    owns_draft = pp_rank == pp_size - 1
    expected = (
        ([10, 20, 50] if mla else [10, 20, 50, 30, 40, 60])
        if owns_draft
        else ([10, 20] if mla else [10, 20, 30, 40])
    )
    assert args.kv_data_ptrs == expected
    assert args.kv_layer_ids == (
        ([0, 1, 2] if mla else [0, 1, 2, 0, 1, 2])
        if owns_draft
        else ([0, 1] if mla else [0, 1, 0, 1])
    )
    assert setup_state.call_args.args[2] is (draft if owns_draft else None)


def test_chunked_kv_send_finishes_with_state_and_metadata():
    sender = Mock(_source_event=None)
    sender.should_send_kv_chunk.return_value = True
    req = SimpleNamespace(
        rid="chunked",
        kv=SimpleNamespace(req_pool_idx=0),
        start_send_idx=0,
        origin_input_ids=list(range(8)),
        extend_range=SimpleNamespace(end=8),
        disagg_decode_prefix_len=0,
        disagg_kv_sender=sender,
    )
    event = Mock()
    scheduler = SimpleNamespace(
        enable_staging=False,
        forward_stream=object(),
        device_module=SimpleNamespace(Event=lambda: event),
        token_to_kv_pool_allocator=SimpleNamespace(
            page_size=2, translate_kv_indices_for_transfer=lambda indices: indices
        ),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.arange(8).reshape(1, 8)),
        disagg_prefill_bootstrap_queue=SimpleNamespace(
            kv_manager=SimpleNamespace(
                kv_args=SimpleNamespace(state_types=[StateType.DSA])
            )
        ),
        disagg_metadata_buffers=Mock(),
        disagg_prefill_pending_chunk_rids=set(),
    )
    send = prefill.SchedulerDisaggregationPrefillMixin.send_kv_chunk
    with patch.object(prefill, "_is_npu", False):
        assert send(scheduler, req, end_idx=5)
        assert req.start_send_idx == 4
        assert scheduler.disagg_prefill_pending_chunk_rids == {req.rid}
        scheduler.disagg_metadata_buffers.set_buf.assert_not_called()
        np.testing.assert_array_equal(sender.send.call_args.args[0], [0, 1])
        assert sender.send.call_args.args[1] is None

        assert send(scheduler, req, last_chunk=True)

    assert req.start_send_idx == 8
    assert not scheduler.disagg_prefill_pending_chunk_rids
    assert sender.send.call_count == 2
    np.testing.assert_array_equal(sender.send.call_args.args[0], [2, 3])
    np.testing.assert_array_equal(sender.send.call_args.args[1][0], [0, 1, 2, 3])
    assert sender.send.call_args.kwargs["num_kv_tokens"] == 4
    scheduler.disagg_metadata_buffers.set_buf.assert_called_once_with(req)
    assert event.record.call_count == 2
    event.record.assert_called_with(scheduler.forward_stream)


@pytest.mark.parametrize("restore_pending", [False, True])
def test_decode_success_commits_once_local_restore_finishes(restore_pending):
    queue = decode.DecodeTransferQueue.__new__(decode.DecodeTransferQueue)
    queue.enable_staging = False
    queue.scheduler = SimpleNamespace(enable_decode_hicache=restore_pending)
    req = SimpleNamespace(rid="received", bootstrap_room=123, finished_reason=None)
    entry = SimpleNamespace(
        req=req,
        metadata_buffer_index=0,
        hicache_restore_status=decode.HiCacheRestoreResult.PENDING,
    )
    queue.queue = [entry]
    queue._poll_with_metadata_gate = Mock(return_value=[KVPoll.Success])
    queue._process_hicache_local_restores = Mock()
    queue._commit_transfer_to_req = Mock()
    queue.metadata_buffers = SimpleNamespace(bootstrap_room=[123])
    queue.req_to_metadata_buffer_idx_allocator = Mock()

    if restore_pending:
        assert queue.pop_transferred() == []
        queue._commit_transfer_to_req.assert_not_called()
        queue.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
        # This test models the scheduler finishing its local restore before the
        # next poll. Network KV success alone must not bypass that dependency.
        entry.hicache_restore_status = decode.HiCacheRestoreResult.READY

    assert queue.pop_transferred() == [req]
    assert queue.queue == []
    assert queue.metadata_buffers.bootstrap_room == [0]
    queue._commit_transfer_to_req.assert_called_once_with(entry)
    queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(0)
