"""The setup registry: every name in the config and the database means one thing."""

from __future__ import annotations

import pytest

from fraud_detect.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from fraud_detect.dashboard import data
from fraud_detect.dashboard import registry as reg


def test_every_config_feature_set_maps_to_exactly_one_setup():
    names = load_config(DEFAULT_CONFIG)["feature_sets"]
    assert names
    for name in names:
        fs = reg.parse_feature_set(name)
        assert fs.setup in reg.SETUPS and fs.variant in reg.VARIANTS
        assert fs.budget in reg.BUDGETS


def test_every_committed_pipeline_entry_maps_to_a_setup():
    entries = data.load_entries(REPO_ROOT / "results" / "runs")
    assert data.unmapped_feature_sets(entries) == []
    assert sum(e.meta["kind"] == "pipeline" for e in entries) > 0


@pytest.mark.parametrize(
    ("name", "setup", "variant", "size", "budget", "display"),
    [
        ("pilot_m1_own", "B", reg.ALL, None, reg.PILOT, "B"),
        ("m1_own", "B", reg.ALL, None, reg.FULL, "B"),
        ("m1_own_no36", "B", reg.NO36, None, reg.FULL, "B"),
        ("pilot_m2_own_nbr_no19", "E1", reg.NO19, None, reg.PILOT, "E1"),
        ("pilot_m5_own_gnnssl", "E2", reg.ALL, "8", reg.PILOT, "E2"),
        ("pilot_m5_own_gnnssl_no19d16", "E2", reg.NO19, "16", reg.PILOT, "E2 (16)"),
        (
            "pilot_m5r_own_gnnrand_no19raw",
            "E3",
            reg.NO19,
            "raw 64",
            reg.PILOT,
            "E3 (raw 64)",
        ),
        ("pilot_m7r_own_nbr_gnnrand", "E7r", reg.ALL, None, reg.PILOT, "E7r"),
        ("pilot_m8s_own_gnnsup", "E8s", reg.ALL, None, reg.PILOT, "E8s"),
        ("pilot_g5g_rgcn", "E5g", reg.ALL, None, reg.PILOT, "E5g"),
        ("pilot_g5n3_sage_rw", "N3", reg.ALL, None, reg.PILOT, "N3"),
        ("pilot_g5w_sage_big", "E5w", reg.ALL, None, reg.PILOT, "E5w"),
        ("pilot_g6w_mlp_big", "E6w", reg.ALL, None, reg.PILOT, "E6w"),
        ("pilot_g9_sign", "E9", reg.ALL, None, reg.PILOT, "E9"),
        ("pilot_g9n3_sign_rw", "N9", reg.ALL, None, reg.PILOT, "N9"),
        ("pilot_n1_own_deg_no19", "N1", reg.NO19, None, reg.PILOT, "N1"),
        ("pilot_m1_own_top4no19", "B", reg.TOP4NO19, None, reg.PILOT, "B"),
        ("pilot_m1_own_only36", "B", reg.ONLY36, None, reg.PILOT, "B"),
    ],
)
def test_parsing_a_name(name, setup, variant, size, budget, display):
    fs = reg.parse_feature_set(name)
    assert fs.setup.setup_id == setup
    assert (fs.variant, fs.embed_size, fs.budget) == (variant, size, budget)
    assert fs.display_id == display


@pytest.mark.parametrize(
    "name",
    [
        "",
        "nonsense",
        "pilot_",
        "t100_m1_own",
        "m1_own_extra",
        "pilot_m1_own_no19d16",
        "m9_own",
        "pilot_g5_sage_nope",
    ],
)
def test_an_unknown_name_is_refused(name):
    with pytest.raises(reg.UnknownFeatureSetError):
        reg.parse_feature_set(name)
    assert not reg.is_known(name)


def test_setup_ids_and_bases_are_unique_and_families_known():
    ids = [s.setup_id for s in reg.SETUPS]
    bases = [s.base for s in reg.SETUPS]
    assert len(set(ids)) == len(ids) and len(set(bases)) == len(bases)
    assert {s.family for s in reg.SETUPS} == set(reg.FAMILIES)


def test_only_the_model_6_setups_carry_a_label_derived_column():
    marked = {s.setup_id for s in reg.SETUPS if s.label_derived_column}
    assert marked == {"E8", "E8s"}


def test_only_b_and_the_edge_free_networks_use_no_graph():
    assert {s.setup_id for s in reg.SETUPS if not s.uses_graph} == {"B", "E6", "E6w"}


def test_variant_labels_name_the_feature_count():
    assert reg.variant_label(reg.ALL, "amazon") == "all 25 features"
    assert reg.variant_label(reg.ALL, "yelpchi") == "all 32 features"
    assert reg.variant_label(reg.ALL, "other") == "all features"
    assert "f19" in reg.variant_label(reg.NO19, "amazon")


def test_the_report_scripts_read_the_registry():
    import importlib.util  # noqa: PLC0415
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location(
            "gnn_report", REPO_ROOT / "scripts" / "gnn_report.py"
        )
        gnn = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gnn)
        spec = importlib.util.spec_from_file_location(
            "rq_report", REPO_ROOT / "scripts" / "rq_report.py"
        )
        rq = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rq)
    finally:
        sys.path.pop(0)
    for base, (setup_id, label) in gnn.SETUPS.items():
        s = reg.parse_feature_set(reg.PILOT_PREFIX + base).setup
        assert (s.setup_id, s.label) == (setup_id, label)
    assert rq.parse("pilot_m5_own_gnnssl_no19d16") == ("E2 (16)", "without f19", "16")
    assert rq.parse("pilot_m1_own") == ("B", "all 25", "")
    assert rq.parse("pilot_n2_own_nbrrw_no19") == ("N2", "without f19", "")
    assert rq.SETUP_LABELS["N1"] == "N1: + degree only"
