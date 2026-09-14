"""Engine detection: never a guess, and never silent."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from conftest import coll_line, write

from collprof import engines


def test_a_serving_run_is_recognised(sglang_run: Path):
    spec, reason = engines.detect(sglang_run)
    assert spec.name == "sglang-disagg"
    assert "4 log(s)" in reason


def test_a_training_run_is_recognised(primus_run: Path):
    spec, reason = engines.detect(primus_run)
    assert spec.name == "primus"
    assert "2 log(s)" in reason


def test_an_unrecognised_run_lists_what_was_looked_for(tmp_path: Path):
    with pytest.raises(SystemExit) as exc:
        engines.detect(tmp_path)
    message = str(exc.value)
    assert "no known engine" in message
    for name in engines.REGISTRY:
        assert name in message
    assert "--engine" in message


def test_an_ambiguous_run_refuses_to_pick(tmp_path: Path):
    """Two layouts in one directory means the answer is unknowable, not that the first wins."""
    write(tmp_path / "prefill_NODE0.log", [coll_line()])
    write(tmp_path / "node_0" / "stdout.out", [coll_line()])
    with pytest.raises(SystemExit, match="more than one engine"):
        engines.detect(tmp_path)


def test_an_unknown_engine_name_lists_the_known_ones():
    with pytest.raises(SystemExit, match="unknown engine"):
        engines.get("vllm-disagg")


def test_every_registered_engine_declares_what_a_report_needs():
    """The registry is the contract; an engine missing a piece produces a misleading report."""
    for name, spec in engines.REGISTRY.items():
        assert spec.name == name
        assert spec.summary, f"{name} has no summary for the report header"
        assert spec.logs.globs, f"{name} declares no log globs"
        assert spec.limits.max_msg_bytes > 0
        if spec.iteration_metric:
            keys = {m.key for m in spec.metrics}
            assert spec.iteration_metric in keys, f"{name} counts iterations with an absent metric"


def test_the_moe_parallelism_knobs_the_kimi_entries_set_are_classified():
    """The MoE parallelism knobs the Kimi-K2 entries set are perf-relevant, not noise."""
    from collprof.engines.sglang_disagg import SPEC

    perf = SPEC.run_config.perf_relevant
    for setting in ("moe_dense_tp_size", "enable_dp_lm_head",
                    "enable_dp_attention_local_control_broadcast"):
        assert setting in perf, setting
        assert setting not in SPEC.run_config.noise, setting


def test_the_ab_catalog_entries_pin_the_kernel_variant():
    """Both A/B catalog entries pin MoRI to its throughput kernel, so the pair isolates the
    backend rather than the dispatch mode."""
    import json
    from pathlib import Path

    catalog = json.loads((Path(__file__).resolve().parents[5] / "scripts" / "sglang_disagg"
                          / "models.json").read_text())
    ab = {m["name"]: m["env_vars"] for m in catalog if m["name"].endswith("-ab")}

    assert len(ab) == 2, "the A/B pair"
    for name, env in ab.items():
        assert env["SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD"] == "0", name


def _catalog_entries(text: str) -> dict:
    """The two-level `models.yaml` shape, without PyYAML.

    This tooling is standard library only apart from an optional `openpyxl`, and a test that
    imports `yaml` fails at collection on a fresh environment rather than reporting anything.
    """
    entries: dict = {}
    name = block = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        hit = re.match(r"^(\S+):\s*$", line)
        if hit:
            name, block = hit.group(1), None
            entries.setdefault(name, {})
            continue
        if name is None:
            continue
        hit = re.match(r"^  (\w+):\s*\"(.*)\"\s*$", line)
        if hit:
            entries[name][hit.group(1)] = hit.group(2)
            block = None
            continue
        hit = re.match(r"^  (\w+):\s*$", line)
        if hit:
            block = hit.group(1)
            entries[name].setdefault(block, {})
            continue
        hit = re.match(r"^    (\w+):\s*\"(.*)\"\s*$", line)
        if hit and block:
            entries[name][block][hit.group(1)] = hit.group(2)
    return entries


def test_the_ab_pair_differs_only_by_the_backend_flag():
    """One factor is the whole design of the pair, across every field and not just `dp_flags`.

    `--deepep-mode normal` was set on the DeepEP arm alone; the effective `ServerArgs` matched
    only because `normal` is sglang's default, so the pair was one-factor by accident.
    """
    catalog = _catalog_entries((Path(__file__).resolve().parents[5] / "scripts" / "sglang_disagg"
                                / "models.yaml").read_text())
    pair = {name: entry for name, entry in catalog.items() if name.endswith("-AB")}
    assert len(pair) == 2, f"the A/B pair, got {sorted(pair)}"

    (left_name, left), (right_name, right) = sorted(pair.items())
    assert set(left) == set(right), "the same fields are configured on both arms"
    assert "base_flags" in left and "prefill" in left, "the parser found the fields it checks"

    for field in sorted(set(left) | set(right)):
        lhs, rhs = left.get(field), right.get(field)
        if field != "dp_flags":
            # Everything else -- graph capture, memory fraction, request limits, the per-role
            # blocks -- must be identical, or the pair measures more than the backend.
            assert lhs == rhs, f"{field} differs: {left_name}={lhs!r} {right_name}={rhs!r}"
            continue
        assert len(lhs.split()) == len(rhs.split()), f"dp_flags differ in length: {lhs} / {rhs}"
        differ = [(a, b) for a, b in zip(lhs.split(), rhs.split()) if a != b]
        assert differ in ([("deepep", "mori")], [("mori", "deepep")]), \
            f"only the backend may differ, got {differ}"
