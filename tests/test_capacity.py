"""CPU-only falsification checks for the capacity experiment contracts.

These tests deliberately use synthetic receipts.  They do not load torch/vLLM and
do not establish that a physical GPU trial is feasible or representative.
"""
import copy
import unittest

from capacity_analysis import (merge_group_records, native_accounting,
                               pre_boundary, request_metrics, summarize)
from capacity_contract import (check_devices, engine_arguments, freeze,
                               membership_hash, physical_gpu_binding, route_plan,
                               topology, validate_membership, validate_residency,
                               validate_slo)


SLO = {"ttft_seconds": 1.0, "mean_tpot_seconds": 1.0,
       "end_to_end_seconds": 2.0}


def device(i, memory=32 * 1024 ** 3):
    return {"uuid": "GPU-%04d" % i, "total_memory": memory,
            "free_memory": memory}


def ep1_row():
    layers = []
    for i in range(16):
        layers.append({"name": "layer-%d" % i, "is_cuda": True,
                       "dtype": "torch.float16", "w13_shape": [64],
                       "w2_shape": [64], "global_to_local_map": list(range(64))})
    return {"pid": 10, "gpu_uuid": "GPU-ONE", "dp_rank": 0,
            "ep_rank": 0, "tp_world_size": 1, "pp_world_size": 1,
            "dp_world_size": 1, "ep_world_size": 1, "model_type": "olmoe",
            "model_dtype": "torch.float16", "global_num_experts": 64,
            "kv_capacity_blocks": 1, "kv_tensors": [{"is_cuda": True}],
            "expert_layers": layers}


def offered_plan():
    return [{"request_id": "main-r0", "group": "D1", "owner": 0,
             "offset_ns": 0, "family": "narrative"},
            {"request_id": "main-r1", "group": "D1", "owner": 1,
             "offset_ns": 10, "family": "identifiers"}]


def normal_events(coalesced=False):
    events = []
    for i, row in enumerate(offered_plan()):
        rid = row["request_id"]
        events.append({"event": "submitted", "request_id": rid,
                       "monotonic_ns": 100 + i * 100, "group": row["group"],
                       "owner": row["owner"], "family": row["family"]})
        chunks = [[1], [1, 2]] if not coalesced else [[1, 2]]
        for n, tokens in enumerate(chunks):
            events.append({"event": "output_chunk", "request_id": rid,
                           "monotonic_ns": 110 + i * 100 + n * 10,
                           "token_ids": tokens})
        events.append({"event": "terminal", "request_id": rid,
                       "monotonic_ns": 130 + i * 100, "token_ids": [1, 2],
                       "finish_reason": "length"})
    return events


class FrozenOfferTests(unittest.TestCase):
    def test_fixed_85_and_smoke_calendar_hashes_are_stable(self):
        main = freeze()
        smoke = freeze(smoke=True)
        self.assertEqual((len(main["arrivals"]), main["sha256"]),
                         (85, "76f6d9c84e379d648e4ff2b9418061f71f84a1f1c55092cf09a9f332e143c712"))
        self.assertEqual((len(smoke["arrivals"]), smoke["sha256"]),
                         (8, "3c8427d54cbb81a69c1d04ff8485646ad435458cdd32f9a2964f8947cad6ad8e"))

    def test_c0_c1_keep_the_same_85_global_offers_and_boundary_is_one_way(self):
        work = freeze()
        c0, c1 = route_plan("C0", work), route_plan("C1", work)
        fields = ("request_id", "offset_ns", "family")
        self.assertEqual([{k: r[k] for k in fields} for r in c0["arrivals"]],
                         [{k: r[k] for k in fields} for r in c1["arrivals"]])
        self.assertTrue(all(r["group"] == "D1" for r in c0["arrivals"]))
        boundary = c1["boundary_offset_ns"]
        self.assertTrue(all(r["group"] == "D1" for r in c1["arrivals"]
                            if r["offset_ns"] < boundary))
        self.assertTrue(all(r["offset_ns"] >= boundary for r in c1["arrivals"]
                            if r["group"] == "D2"))

    def test_c_groups_require_four_distinct_full_gpus_not_two_gpu_or_mig_aliases(self):
        for arm in ("C0", "C1"):
            groups = topology(arm)
            self.assertEqual(sum(g["dp"] for g in groups), 4)
            self.assertEqual(len({d for g in groups for d in g["devices"]}), 4)
            with self.assertRaisesRegex(ValueError, "requires 4 distinct"):
                check_devices(arm, [device(0), device(1)])
        with self.assertRaisesRegex(ValueError, "MIG"):
            physical_gpu_binding([device(0)], "GPU-0000, 0000:01:00.0, Enabled")
        with self.assertRaisesRegex(ValueError, "PCI"):
            physical_gpu_binding([device(0), device(1)],
                                 "GPU-0000, 0000:01:00.0, Disabled\nGPU-0001, 0000:01:00.0, Disabled")

    def test_ep1_t4_floor_rejects_point_eight_utilization(self):
        t4 = 16 * 1024 ** 3
        with self.assertRaisesRegex(ValueError, "INFEASIBLE_STATIC"):
            check_devices("A", [device(0, t4)], utilization=.8)


class TopologyAndIdentityTests(unittest.TestCase):
    def test_engine_arguments_keep_dp_ep_separate(self):
        args = engine_arguments(2, "reference", "/model", 30101)
        self.assertEqual((args["data_parallel_size"], args["data_parallel_size_local"]), (2, 2))
        self.assertEqual((args["tensor_parallel_size"], args["pipeline_parallel_size"]), (1, 1))
        self.assertTrue(args["enable_expert_parallel"])
        with self.assertRaises(ValueError):
            engine_arguments(3, "reference", "/model", 30101)

    def test_ep1_residency_is_not_ep2_and_membership_binds_nonce_gpu_rank(self):
        row = ep1_row()
        self.assertTrue(validate_residency([row], ["GPU-ONE"], 1))
        wrong = copy.deepcopy(row); wrong["ep_world_size"] = 2
        with self.assertRaisesRegex(ValueError, "DP1/EP1"):
            validate_residency([wrong], ["GPU-ONE"], 1)
        incarnation = "nonce"
        receipt = {"pid": 10, "gpu_uuid": "GPU-ONE", "rank": 0,
                   "group_incarnation": incarnation,
                   "members": [membership_hash(incarnation, "GPU-ONE", 0)]}
        self.assertTrue(validate_membership([receipt], [row], incarnation))
        for key, value in (("group_incarnation", "wrong"), ("gpu_uuid", "GPU-OTHER")):
            broken = copy.deepcopy(receipt); broken[key] = value
            with self.assertRaises(ValueError):
                validate_membership([broken], [row], incarnation)


class AccountingTests(unittest.TestCase):
    def test_slo_validation_and_all_offer_denominator_includes_unsent_and_censored(self):
        self.assertFalse(validate_slo({k: None for k in SLO}))
        for bad in ({"ttft_seconds": 0, "mean_tpot_seconds": None, "end_to_end_seconds": None},
                    {"ttft_seconds": float("nan"), "mean_tpot_seconds": None, "end_to_end_seconds": None}):
            with self.assertRaises(ValueError): validate_slo(bad)
        records = request_metrics(offered_plan(), normal_events()[:4], 100, 500, SLO)
        self.assertEqual({r["status"] for r in records}, {"NORMAL", "UNSENT"})
        summary = summarize(records, 1_000_000_000, 100, SLO)
        self.assertEqual(summary["planned"], 2)
        self.assertEqual(summary["slo_goodput"]["verified_passes"], 1)

    def test_coalesced_chunks_make_tpot_unknown_and_never_false_goodput(self):
        records = request_metrics(offered_plan(), normal_events(coalesced=True), 100, 500, SLO)
        self.assertTrue(all(r["token_timing_coverage"].startswith("COALESCED") for r in records))
        self.assertTrue(all(r["slo"] == "UNKNOWN" for r in records))
        summary = summarize(records, 1_000_000_000, 100, SLO)
        self.assertEqual(summary["slo_goodput"]["verified_passes"], 0)
        self.assertEqual(summary["slo_goodput"]["unknown"], 2)

    def test_known_slo_failure_dominates_an_unknown_other_metric(self):
        strict = {**SLO, "ttft_seconds": 1e-9}
        records = request_metrics(offered_plan(), normal_events(coalesced=True), 100, 500, strict)
        self.assertTrue(all(r["ttft_seconds"] is not None and r["mean_tpot_seconds"] is None
                            for r in records))
        self.assertTrue(all(r["slo"] == "FAIL" for r in records))

    def test_duplicate_cumulative_chunk_cannot_extend_last_token_timing(self):
        events = normal_events()
        # Valid duplicate transport data did not reveal a token, so it must not
        # become the last-token timestamp.
        first = events[:4]
        first.insert(2, {"event": "output_chunk", "request_id": "main-r0",
                         "monotonic_ns": 120, "token_ids": [1]})
        records = request_metrics(offered_plan()[:1], first, 100, 500, SLO)
        self.assertEqual(records[0]["mean_tpot_seconds"], 10 / 1e9)

    def test_clock_order_and_followup_deadline_are_enforced(self):
        early = normal_events(); early[0]["monotonic_ns"] = 99
        with self.assertRaisesRegex(ValueError, "Early offer"):
            request_metrics(offered_plan(), early, 100, 500, SLO)
        reversed_terminal = normal_events(); reversed_terminal[-1]["monotonic_ns"] = 1
        with self.assertRaisesRegex(ValueError, "precedes"):
            request_metrics(offered_plan(), reversed_terminal, 100, 500, SLO)
        records = request_metrics(offered_plan(), normal_events(), 100, 500, SLO,
                                  followup_deadline_ns=150)
        self.assertEqual(records[1]["status"], "CENSORED_LATE_TERMINAL")

    def test_cross_group_duplicate_orphan_binding_is_rejected(self):
        plan = {"arrivals": offered_plan()}
        records = request_metrics(offered_plan(), normal_events(), 100, 500, SLO)
        self.assertEqual(len(merge_group_records(plan, [{"records": records}])), 2)
        duplicate = records + [copy.deepcopy(records[0])]
        with self.assertRaisesRegex(ValueError, "duplicate/orphan/missing"):
            merge_group_records(plan, [{"records": duplicate}])
        moved = copy.deepcopy(records); moved[0]["group"] = "D2"
        with self.assertRaisesRegex(ValueError, "binding"):
            merge_group_records(plan, [{"records": moved}])

    def test_generic_kv_ownership_and_release_conservation(self):
        rows = offered_plan()
        inputs = {"narrative": [1], "identifiers": [2]}
        joins = {"main-r0": {"internal_request_id": "i0"},
                 "main-r1": {"internal_request_id": "i1"}}
        core = []
        for rank, rid, family in ((0, "i0", "narrative"), (1, "i1", "identifiers")):
            core += [{"event": "core_initialized", "dp_rank": rank, "pid": 10 + rank, "free_blocks": 8},
                     {"event": "request_accepted", "dp_rank": rank, "pid": 10 + rank, "request_id": rid, "prompt_token_ids": inputs[family]},
                     {"event": "allocation", "dp_rank": rank, "monotonic_ns": 1, "seq": 1, "request_id": rid, "accepted": True, "before_blocks": [[]], "after_blocks": [[{"block_id": rank, "ref_cnt": 1}]]},
                     {"event": "kv_free", "dp_rank": rank, "monotonic_ns": 2, "seq": 2, "request_id": rid, "remaining_blocks": [[]], "released_blocks": [[{"block_id": rank, "ref_cnt": 0}]], "free_blocks": 8},
                     {"event": "output_applied", "dp_rank": rank, "grants": {rid: 1}, "after": {}, "deferred_frees": 0}]
        report = native_accounting("main", rows, core, joins, 2, inputs, diagnostic=True)
        self.assertEqual(report["scheduler_KV"], "PASS_REQUEST_OWNERSHIP")
        broken = copy.deepcopy(core); broken[3]["released_blocks"][0][0]["ref_cnt"] = 1
        with self.assertRaisesRegex(ValueError, "reference"):
            native_accounting("main", rows, broken, joins, 2, inputs, diagnostic=True)

    def test_pre_boundary_uses_only_published_past_snapshot_and_no_future_fields(self):
        core = [{"dp_rank": 0, "monotonic_ns": 20, "load_state": {"waiting": []}},
                {"dp_rank": 0, "monotonic_ns": 101, "load_after": {"waiting": ["future"]}}]
        report = pre_boundary(core, 100, 2)
        self.assertEqual(report["ranks"][0]["state"], {"waiting": []})
        self.assertEqual(report["ranks"][1]["status"], "UNKNOWN_NO_PUBLISHED_SNAPSHOT")
        self.assertEqual(report["future_output_lengths"], "NOT_USED")


if __name__ == "__main__":
    unittest.main()
