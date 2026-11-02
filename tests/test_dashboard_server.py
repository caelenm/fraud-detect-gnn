"""The dashboard's server: real HTTP on an ephemeral loopback port, invented data."""

from __future__ import annotations

import http.client
import json
import threading

import numpy as np
import pandas as pd
import pytest
from helpers import fake_dataset, fake_live_inputs, finish_fake_run, table_worker
from synthetic import fake_ladder_db, fake_run_entry

from fraud_detect import columns as C
from fraud_detect import results_db
from fraud_detect.config import DEFAULT_CONFIG, get_paths, load_config
from fraud_detect.dashboard import classify as cl
from fraud_detect.dashboard import data
from fraud_detect.dashboard import server as srv
from fraud_detect.dashboard.worker import WorkerError

ROW_LEVEL_KEYS = {
    "node_id",
    "prob",
    "probs_test",
    "preds_test",
    "y_true",
    "y_pred",
    "group_id",
    "actual",
    "saved_prob",
}


class Client:
    def __init__(self, server) -> None:
        self.port = server.server_address[1]

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            payload = response.read()
            return response.status, dict(response.getheaders()), payload
        finally:
            conn.close()

    def get(self, path, headers=None):
        status, _, payload = self.request("GET", path, headers=headers)
        return status, json.loads(payload)

    def classify(self, body, headers=None):
        raw = json.dumps(body) if not isinstance(body, bytes) else body
        status, _, payload = self.request(
            "POST",
            "/api/classify",
            raw,
            {"Content-Type": "application/json", **(headers or {})},
        )
        return status, json.loads(payload)


def serve(app):
    server = srv.make_server(app, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@pytest.fixture
def aggregate(tmp_path):
    """The server on an invented database, with no run outputs at all."""
    db = tmp_path / "db"
    db.mkdir()
    fake_ladder_db(db)
    config = load_config(DEFAULT_CONFIG)
    for key in ("raw_dir", "processed_dir", "outputs_dir"):
        config["paths"][key] = str(tmp_path / key)
    app = srv.App(config, db, use_model=False)
    server = serve(app)
    yield app, Client(server)
    server.shutdown()
    server.server_close()


# ---- Aggregate sections ----------------------------------------------------
def test_datasets_describe_the_page(aggregate):
    _, client = aggregate
    status, body = client.get("/api/datasets")
    assert status == 200
    assert [d["name"] for d in body["datasets"]] == ["amazon", "yelpchi"]
    amazon = body["datasets"][0]
    assert {v["name"] for v in amazon["variants"]} == {"all", "no19"}
    assert body["datasets"][1]["variants"] == [
        {"name": "all", "label": "all 32 features"}
    ]
    assert body["default_variant"] == "all"  # decided 2026-11-02
    assert body["failed_checks"] == [] and body["n_checks"] == 12
    assert body["general_caveats"] and body["not_run"] and body["demo_notes"]
    assert any("pilot budget" in note for note in body["general_caveats"])


def test_methods_follow_every_filter(aggregate):
    _, client = aggregate
    status, amazon = client.get("/api/methods?ds=amazon&budget=pilot&variant=all")
    assert status == 200
    assert [r["display_id"] for r in amazon["rows"]] == ["B", "E1", "E2", "E5", "E6"]
    _, yelp = client.get("/api/methods?ds=yelpchi&budget=pilot&variant=all")
    assert amazon["rows"][0]["test_pr_auc"] != yelp["rows"][0]["test_pr_auc"]
    assert amazon["rows"][0]["no_skill"] == pytest.approx(0.1)
    assert yelp["rows"][0]["no_skill"] == pytest.approx(0.15)
    _, no19 = client.get("/api/methods?ds=amazon&budget=pilot&variant=no19")
    assert [r["display_id"] for r in no19["rows"]] == ["B", "E1"]
    _, full = client.get("/api/methods?ds=amazon&budget=full&variant=all")
    assert [r["display_id"] for r in full["rows"]] == ["B"]
    _, none = client.get("/api/methods?ds=yelpchi&budget=full&variant=all")
    assert none["rows"] == []


def test_the_best_setup_and_the_f19_warning_are_in_the_view(aggregate):
    _, client = aggregate
    _, amazon = client.get("/api/methods?ds=amazon&budget=pilot&variant=all")
    assert amazon["best"]["setup_id"] == "E1"  # the highest CV, not the highest test
    assert any("likely leaks" in c for c in amazon["caveats"])
    _, no19 = client.get("/api/methods?ds=amazon&budget=pilot&variant=no19")
    assert not any("likely leaks" in c for c in no19["caveats"])
    _, yelp = client.get("/api/methods?ds=yelpchi&budget=pilot&variant=all")
    assert not any("f19" in c for c in yelp["caveats"])
    _, best = client.get("/api/best")
    assert set(best) == {"amazon-all", "amazon-no19", "yelpchi-all"}
    assert best["yelpchi-all"]["setup_id"] == "E2"


def test_contrasts_and_cross_dataset(aggregate):
    _, client = aggregate
    _, view = client.get("/api/contrasts?ds=amazon&variant=all")
    assert [q["b_id"] for q in view["questions"]] == ["E1", "E5"]
    assert {r["a_id"] for r in view["families"]["primary"]} == {"B", "E6"}
    assert view["default_check"] is None
    _, cross = client.get("/api/cross")
    assert [s["setup"] for s in cross["setups"]] == ["B", "E1", "E2", "E5", "E6"]
    agreement = {r["contrast"]: r for r in cross["agreement"]}
    assert agreement["E1 − B"]["agrees"] is True
    assert agreement["E5 − E6"]["agrees"] is False


def test_checks_endpoint_lists_every_check(aggregate):
    _, client = aggregate
    _, body = client.get("/api/checks")
    assert len(body["checks"]) == 12 and all(c["ok"] for c in body["checks"])


@pytest.mark.parametrize(
    "path",
    [
        "/api/methods?ds=nope",
        "/api/methods?ds=amazon&variant=nope",
        "/api/methods?ds=amazon&budget=t100",
        "/api/methods",
        "/api/contrasts?ds=amazon&variant=top4",
        "/api/contrasts",
    ],
)
def test_bad_parameters_are_a_400_with_a_reason(aggregate, path):
    _, client = aggregate
    status, body = client.get(path)
    assert status == 400 and body["error"]


def _keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _keys(v)}
    return set()


@pytest.mark.parametrize(
    "path",
    [
        "/api/datasets",
        "/api/methods?ds=amazon&variant=all",
        "/api/contrasts?ds=amazon",
        "/api/cross",
        "/api/best",
        "/api/checks",
    ],
)
def test_the_aggregate_endpoints_hold_no_row_level_fields(aggregate, path):
    _, client = aggregate
    status, body = client.get(path)
    assert status == 200
    assert not _keys(body) & ROW_LEVEL_KEYS


def test_a_fresh_clone_has_every_aggregate_section_and_says_what_classify_needs(
    aggregate,
):
    app, client = aggregate
    app.load_panels()  # no outputs/ under this config at all
    _, status = client.get("/api/status")
    assert {p["state"] for p in status["panels"]} == {"unavailable"}
    assert all(p["message"] for p in status["panels"])
    code, body = client.classify({"target": "amazon-all", "draw": "any"})
    assert code == 400 and body["error"]
    assert client.get("/api/cross")[0] == 200


def test_clean_turns_numpy_and_nan_into_plain_json():
    out = srv.clean(
        {
            "a": np.float64(1.5),
            "b": float("nan"),
            "c": [np.int64(2), float("inf")],
            "d": (np.bool_(True),),
            3: None,
        }
    )
    assert out == {"a": 1.5, "b": None, "c": [2, None], "d": [True], "3": None}
    json.dumps(out, allow_nan=False)


def test_a_run_counted_twice_stops_the_server_at_start(tmp_path):
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", tag_id="a")
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", tag_id="b")
    with pytest.raises(data.DashboardDataError, match="more than once"):
        srv.App(load_config(DEFAULT_CONFIG), tmp_path, use_model=False)


def test_a_failed_check_reaches_the_page(tmp_path):
    fake_run_entry(tmp_path, "amazon", "pilot_m1_own", test=1.5)
    app = srv.App(load_config(DEFAULT_CONFIG), tmp_path, use_model=False)
    assert app.api_datasets()["failed_checks"] == ["Metrics are in range"]


# ---- Staying local ---------------------------------------------------------
def test_the_server_binds_the_loopback_address_only(aggregate):
    app, _ = aggregate
    server = srv.make_server(app, 0)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()
    assert srv.HOST == "127.0.0.1"


@pytest.mark.parametrize(
    "host", ["example.com", "example.com:8555", "evil.localhost:1", ""]
)
def test_a_request_addressed_to_another_host_is_refused(aggregate, host):
    _, client = aggregate
    status, body = client.get("/api/datasets", {"Host": host})
    assert status == 403 and "localhost" in body["error"]
    status, _, _ = client.request("GET", "/", headers={"Host": host})
    assert status == 403


def test_localhost_names_are_accepted(aggregate):
    _, client = aggregate
    for host in (f"127.0.0.1:{client.port}", f"localhost:{client.port}"):
        assert client.get("/api/datasets", {"Host": host})[0] == 200


def test_a_post_from_another_origin_is_refused(aggregate):
    _, client = aggregate
    status, _ = client.classify(
        {"target": "amazon-all"}, {"Origin": "https://example.com"}
    )
    assert status == 403
    ok = {"Origin": f"http://localhost:{client.port}"}
    assert client.classify({"target": "amazon-all"}, ok)[0] != 403


@pytest.mark.parametrize(
    "path",
    [
        "/../configs/default.yaml",
        "/..%2f..%2fconfigs/default.yaml",
        "/report.css/../run.py",
        "/static/app.js",
        "/results/runs",
        "/index.html",
        "/api/nope",
    ],
)
def test_nothing_outside_the_fixed_list_is_served(aggregate, path):
    _, client = aggregate
    status, _, _ = client.request("GET", path)
    assert status == 404


def test_every_response_carries_the_security_headers(aggregate):
    _, client = aggregate
    for path in (
        "/",
        "/app.js",
        "/dashboard.css",
        "/report.css",
        "/api/datasets",
        "/nope",
    ):
        _, headers, _ = client.request("GET", path)
        policy = headers["Content-Security-Policy"]
        assert "default-src 'none'" in policy and "connect-src 'self'" in policy
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["Referrer-Policy"] == "no-referrer"


def test_static_files_have_their_types_and_head_works(aggregate):
    _, client = aggregate
    wanted = {
        "/": "text/html",
        "/app.js": "text/javascript",
        "/dashboard.css": "text/css",
        "/report.css": "text/css",
    }
    for path, kind in wanted.items():
        status, headers, body = client.request("GET", path)
        assert status == 200 and headers["Content-Type"].startswith(kind) and body
        status, headers, body = client.request("HEAD", path)
        assert status == 200 and body == b"" and int(headers["Content-Length"]) > 0


def test_posts_must_be_small_json(aggregate):
    _, client = aggregate
    status, _ = client.classify(b"{not json")
    assert status == 400
    status, _, _ = client.request(
        "POST", "/api/classify", "{}", {"Content-Type": "text/plain"}
    )
    assert status == 415
    status, _, _ = client.request(
        "POST",
        "/api/classify",
        b"x" * (srv.MAX_BODY + 1),
        {"Content-Type": "application/json"},
    )
    assert status == 400
    status, _, _ = client.request(
        "POST", "/api/other", "{}", {"Content-Type": "application/json"}
    )
    assert status == 404
    assert client.classify([1, 2])[0] == 400


@pytest.mark.parametrize(
    "body",
    [{"target": "nope"}, {"target": "amazon-all", "draw": "nope"}, {"draw": "any"}, {}],
)
def test_classify_refuses_bad_requests(aggregate, body):
    _, client = aggregate
    status, reply = client.classify(body)
    assert status == 400 and reply["error"]


# ---- The classify panel, end to end ----------------------------------------
E1, BASE = "pilot_m2_own_nbr", "pilot_m1_own"
CV_B = {("dummy", "none"): 0.2, ("lgbm", "none"): 0.6}
CV_E1 = {("dummy", "none"): 0.2, ("lgbm", "none"): 0.5, ("catboost", "none"): 0.7}
KEY = "fake-all"


def record(db, paths) -> None:
    """The database entry of a faked run, through the real recording code."""
    fields, table = results_db.from_pipeline_run(paths.feature_set_dir)
    meta = results_db.build_meta(
        description="Invented run",
        device="fake-laptop",
        started_utc="2026-01-01T00:00:00Z",
        **fields,
    )
    results_db.add_entry(db, meta, table)


@pytest.fixture
def live(tmp_path, monkeypatch):
    path, config = fake_dataset(tmp_path, n_nodes=200)
    db = tmp_path / "db"
    db.mkdir()
    for name, cv in ((BASE, CV_B), (E1, CV_E1)):
        finish_fake_run(path, config, name, cv)
        record(db, get_paths({**config, "feature_set": name}))
    p = get_paths({**config, "feature_set": E1})
    fake_live_inputs(p)
    monkeypatch.setattr(cl, "TARGET_VARIANTS", {"fake": ("all",)})
    return config, db, p


def probs_of(config) -> np.ndarray:
    side = cl.compare.load_side(cl.split_config(config, "fake"), E1)
    return side.prob


def started(config, db, p, offset=0.0, use_model=True, factory=None):
    probs = probs_of(config)
    factory = factory or (lambda pairs: table_worker(pairs[0][0], p, probs, offset))
    app = srv.App(config, db, use_model=use_model, worker_factory=factory)
    server = serve(app)
    app.load_panels()
    return app, server, Client(server)


def test_live_classification_end_to_end(live):
    config, db, p = live
    app, server, client = started(config, db, p)
    try:
        _, status = client.get("/api/status")
        (panel,) = status["panels"]
        assert panel["state"] == "ready" and panel["target"]["mode"] == "live"
        assert panel["target"]["live_verified"] and panel["target"]["checks_ok"]
        assert panel["target"]["best"]["setup_id"] == "E1"
        code, result = client.classify({"target": KEY, "draw": "fraud", "seed": 11})
        assert code == 200 and result["mode"] == "live" and result["verified"]
        assert result["actual"] == 1 and result["baseline"]["setup"] == "B"
        assert result["node_id"] in set(pd.read_csv(p.test_ids)[C.NODE_ID])
        assert result["node_id"] not in set(pd.read_csv(p.train_ids)[C.NODE_ID])
        again = client.classify({"target": KEY, "draw": "fraud", "seed": 11})[1]
        assert again["node_id"] == result["node_id"]
        assert client.classify({"target": KEY, "draw": "any", "seed": -1})[0] == 400
    finally:
        server.shutdown()
        server.server_close()
        app.close()
    assert app.worker.closed


def test_no_model_mode_shows_saved_predictions_and_says_so(live):
    config, db, p = live
    app, server, client = started(config, db, p, use_model=False)
    try:
        _, status = client.get("/api/status")
        (panel,) = status["panels"]
        assert panel["state"] == "ready" and panel["target"]["mode"] == "saved"
        assert "--no-model" in panel["target"]["note"]
        assert app.worker is None
        code, result = client.classify({"target": KEY, "draw": "any", "seed": 1})
        assert code == 200 and result["mode"] == "saved" and result["verified"] is False
    finally:
        server.shutdown()
        server.server_close()


def test_a_model_that_disagrees_at_start_up_is_not_shown_as_live(live):
    config, db, p = live
    app, server, client = started(config, db, p, offset=1e-3)
    try:
        _, status = client.get("/api/status")
        (panel,) = status["panels"]
        assert panel["target"]["mode"] == "saved" and not panel["target"]["live_verified"]
        assert not panel["target"]["checks_ok"]
        code, result = client.classify({"target": KEY, "draw": "any", "seed": 1})
        assert code == 200 and result["mode"] == "saved"
    finally:
        server.shutdown()
        server.server_close()


def test_a_worker_that_cannot_start_leaves_saved_predictions_and_the_reason(live):
    config, db, p = live

    class Broken:
        def start(self):
            raise WorkerError("uv is not on PATH")

        def close(self):
            pass

    app, server, client = started(config, db, p, factory=lambda _: Broken())
    try:
        _, status = client.get("/api/status")
        (panel,) = status["panels"]
        assert status["worker_error"] == "uv is not on PATH"
        assert panel["state"] == "ready" and panel["target"]["mode"] == "saved"
        assert "uv is not on PATH" in panel["target"]["note"]
        assert client.classify({"target": KEY, "draw": "any", "seed": 2})[0] == 200
    finally:
        server.shutdown()
        server.server_close()


def test_a_panel_with_a_failed_check_is_not_offered(live):
    config, db, p = live
    export = next(p.df_analyze_output_dir.rglob("results/test00"))
    path = export / "X_test_00.csv"
    pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
    app, server, client = started(config, db, p)
    try:
        _, status = client.get("/api/status")
        (panel,) = status["panels"]
        assert panel["state"] == "unavailable" and "lines up" in panel["message"]
        assert client.classify({"target": KEY, "draw": "any"})[0] == 400
    finally:
        server.shutdown()
        server.server_close()


def test_a_classify_response_is_answered_quickly(live):
    import time  # noqa: PLC0415

    config, db, p = live
    app, server, client = started(config, db, p)
    try:
        start = time.monotonic()
        for seed in range(5):
            assert client.classify({"target": KEY, "draw": "any", "seed": seed})[0] == 200
        assert (time.monotonic() - start) / 5 < 1.0  # the plan's bound, per click
    finally:
        server.shutdown()
        server.server_close()
