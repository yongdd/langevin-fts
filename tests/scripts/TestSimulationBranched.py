"""Export branched trees and reject invalid connections without running a solver."""

import copy
import json
from pathlib import Path
import runpy
import threading

import pytest


MODEL = runpy.run_path(str(Path(__file__).with_name("TestSimulationModel.py")))
CHECKS = MODEL["CHECKS"]
BUILDER = MODEL["BUILDER"]


def values(shape="star"):
    form = MODEL["values"]()
    if shape == "star":
        edges = [("A", 0, 1), ("A", 0, 2), ("A", 0, 3),
                 ("B", 1, 4), ("B", 2, 5), ("B", 3, 6)]
    elif shape == "comb":
        edges = [("A", 0, 1), ("A", 1, 2), ("A", 2, 3),
                 ("B", 1, 4), ("B", 2, 5)]
    else:
        edges = [("A", 5, 19), ("B", 19, 2147483647)]
    form["model"]["polymers"][0]["blocks"] = [
        {"type": name, "length": "0.2", "v": v, "u": u} for name, v, u in edges
    ]
    return form


@pytest.mark.parametrize("mode", ("SCFT", "L-FTS", "CL-FTS"))
@pytest.mark.parametrize("shape", ("star", "comb", "sparse_linear"))
def test_branched_export(mode, shape, tmp_path, monkeypatch):
    form = values(shape)
    form["mode"] = mode
    expected = [{**b, "length": .2} for b in form["model"]["polymers"][0]["blocks"]]
    calls = CHECKS["execute_export"](form, tmp_path, monkeypatch)
    params = calls["params"]
    if mode == "SCFT":
        CHECKS["VALIDATION"]["validate_system_params"](params)
    else:
        CHECKS["VALIDATION"]["validate_lfts_params"](params, complex_fields=mode == "CL-FTS")
    assert params["polymers"][0]["blocks"] == expected
    assert params["polymers"][1]["blocks"] == [{"type": "C", "length": 1}]
    assert set(calls["fields"]) == {"A", "B", "C"}


def test_branched_row_order_and_node_labels_are_preserved():
    form = values()
    blocks = form["model"]["polymers"][0]["blocks"]
    blocks.reverse()
    for b in blocks:
        b["v"], b["u"] = str(b["u"] + 10), str(b["v"] + 10)
    params = BUILDER["model_parameters"](form["model"])
    assert params["polymers"][0]["blocks"] == [
        {**b, "length": .2, "v": int(b["v"]), "u": int(b["u"])} for b in blocks
    ]


@pytest.mark.parametrize("problem, message", (
    ("self", "different nodes"), ("duplicate", "duplicate connection"),
    ("cycle", "cycle"), ("disconnected", "one connected polymer"),
    ("negative", "node IDs"), ("overflow", "node IDs"),
    ("fractional", "integer"), ("boolean", "node IDs"),
    ("null", "node IDs"), ("float", "node IDs"),
    ("missing_u", "missing or unexpected fields"),
    ("mixed", "missing or unexpected fields"),
    ("mixed_first", "missing or unexpected fields"),
))
def test_invalid_branched_connections(problem, message):
    form = values()
    blocks = form["model"]["polymers"][0]["blocks"]
    if problem == "self": blocks[0]["u"] = blocks[0]["v"]
    elif problem == "duplicate": blocks[1].update(v=blocks[0]["u"], u=blocks[0]["v"])
    elif problem == "cycle": blocks.append({"type": "A", "length": "0.2", "v": 1, "u": 2})
    elif problem == "disconnected": blocks[0].update(v=100, u=101)
    elif problem == "negative": blocks[0]["u"] = -1
    elif problem == "overflow": blocks[0]["u"] = 2147483648
    elif problem == "fractional": blocks[0]["u"] = "1.5"
    elif problem == "boolean": blocks[0]["u"] = True
    elif problem == "null": blocks[0]["u"] = None
    elif problem == "float": blocks[0]["u"] = 1.0
    elif problem == "missing_u": del blocks[0]["u"]
    elif problem == "mixed": del blocks[1]["v"]; del blocks[1]["u"]
    elif problem == "mixed_first": del blocks[0]["v"]; del blocks[0]["u"]
    with pytest.raises(ValueError, match=message):
        BUILDER["build_script"](form)


def test_branched_server_save_and_invalid_graph_rejection(tmp_path):
    web = runpy.run_path(str(Path(__file__).with_name("TestSimulationWeb.py")))
    server = BUILDER["create_server"](0, tmp_path / "exports")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        form = values("comb")
        invalid = copy.deepcopy(form)
        invalid["model"]["polymers"][0]["blocks"].append({"type": "A", "length": "0.2", "v": 0, "u": 3})
        payload = {"values": invalid, "filename": "comb.py", "directory": "branched"}
        status, _, body = web["request"](server, "POST", "/api/save", payload)
        assert status == 400
        assert "cycle" in json.loads(body)["error"]
        assert not server.save_dir.exists()
        payload["values"] = form
        status, _, body = web["request"](server, "POST", "/api/save", payload)
        assert status == 201
        result = json.loads(body)
        assert result["path"] == str(server.save_dir / "branched/comb.py")
        assert Path(result["path"]).read_text() == result["script"]
        compile(result["script"], result["path"], "exec")
        assert not (Path(result["path"]).parent / "simulation").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
