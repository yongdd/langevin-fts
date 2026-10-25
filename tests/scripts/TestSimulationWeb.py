"""Headless HTTP checks for the browser script editor; no simulation is executed."""

import http.client
import json
from pathlib import Path
import runpy
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest


BUILDER = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "gui" / "simulation_gui.py"))


@pytest.fixture(scope="module")
def web_server():
    server = BUILDER["create_server"](0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(server, method="GET", path="/", body=None, headers=None):
    headers = dict(headers or {})
    if isinstance(body, dict):
        body = json.dumps(body).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")
    connection = http.client.HTTPConnection(*server.server_address, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, response.headers, response.read()
    finally:
        connection.close()


def test_editor_page_and_assets(web_server):
    status, headers, page = request(web_server)
    assert status == 200
    assert "text/html" in headers["Content-Type"]
    assert 'lang="en"' in page.decode("utf-8")
    assert 'id="download"' in page.decode("utf-8")
    assert web_server.server_address[0] == "127.0.0.1"
    for path, content_type in (("/app.js", "text/javascript"), ("/style.css", "text/css")):
        status, headers, body = request(web_server, path=path)
        assert status == 200
        assert content_type in headers["Content-Type"]
        assert body


def test_editor_assets_are_not_git_ignored():
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        pytest.skip("Source archive has no Git ignore metadata")
    assets = ["index.html", "app.js", "model-editor.js", "style.css", "icon.svg", "space-groups.json"]
    result = subprocess.run(["git", "check-ignore", "--no-index", "--",
                             *["scripts/gui/assets/" + name for name in assets]],
                            cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5)
    assert result.returncode == 1, result.stdout + result.stderr


def test_presets_and_script_generation(web_server):
    status, _, body = request(web_server, path="/api/presets")
    assert status == 200
    presets = json.loads(body)
    assert list(presets) == list(BUILDER["PRESETS"])
    for name, values in presets.items():
        status, _, body = request(web_server, "POST", "/api/generate", values)
        assert status == 200
        script = json.loads(body)["script"]
        compile(script, name, "exec")
        assert "def main():" in script


@pytest.mark.parametrize("update", (
    {"nx": "32, 32", "lx": "3"}, {"initial": "gyroid", "nx": "64", "lx": "4.0"},
    {"reduce_memory": "false"}, {"ds": "1e1000000000"}, {"threads": 2},
))
def test_invalid_settings_return_useful_errors(web_server, update):
    values = BUILDER["preset_settings"](BUILDER["PRESETS"][0])
    values.update(update)
    status, _, body = request(web_server, "POST", "/api/generate", values)
    assert status == 400
    assert json.loads(body)["error"]


@pytest.mark.parametrize("body", (b"{broken", b"[]", b"null", b"{}", b"\xff"))
def test_malformed_requests(web_server, body):
    status, _, response = request(web_server, "POST", "/api/generate", body,
                                  {"Content-Type": "application/json"})
    assert status == 400
    assert json.loads(response)["error"]


@pytest.mark.parametrize("length, expected", (("-1", 400), ("invalid", 400), ("65537", 413)))
def test_body_size_headers(web_server, length, expected):
    status, _, body = request(web_server, "POST", "/api/generate", b"",
                              {"Content-Type": "application/json", "Content-Length": length})
    assert status == expected
    assert json.loads(body)["error"]


def test_unsupported_request_type(web_server):
    status, _, _ = request(web_server, "POST", "/api/generate", b"mode=SCFT",
                           {"Content-Type": "application/x-www-form-urlencoded"})
    assert status == 415


def test_unrecognized_form_fields(web_server):
    values = BUILDER["preset_settings"](BUILDER["PRESETS"][0])
    values["unexpected"] = "field"
    status, _, _ = request(web_server, "POST", "/api/generate", values)
    assert status == 400


def test_browser_origin_and_tunnel_ports(web_server):
    values = BUILDER["preset_settings"](BUILDER["PRESETS"][0])
    status, _, _ = request(web_server, "POST", "/api/generate", values,
                           {"Origin": "https://unrelated.example"})
    assert status == 403
    status, _, _ = request(web_server, headers={"Host": "unrelated.example"})
    assert status == 403
    # The local end of an SSH tunnel may use a different port from the server.
    status, _, _ = request(web_server, "POST", "/api/generate", values,
                           {"Host": "localhost:9999", "Origin": "http://localhost:9999"})
    assert status == 200


@pytest.mark.parametrize("path", ("/CLAUDE.md", "/../CLAUDE.md", "/.git/config", "/api/run"))
def test_no_filesystem_or_execution_routes(web_server, path):
    status, _, _ = request(web_server, path=path)
    assert status == 404


@pytest.fixture
def save_server(tmp_path):
    server = BUILDER["create_server"](0, save_dir=tmp_path / "scripts")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def save_payload(filename="simulation.py", directory=""):
    return {"values": BUILDER["preset_settings"](BUILDER["PRESETS"][0]),
            "filename": filename, "directory": directory}


def test_server_save_config_and_new_script(save_server):
    status, _, body = request(save_server, path="/api/config")
    assert status == 200
    assert json.loads(body)["save_dir"] == str(save_server.save_dir)
    assert not save_server.save_dir.exists()
    status, _, body = request(save_server, "POST", "/api/save", save_payload(directory="gyroid/run01"))
    assert status == 201
    saved = json.loads(body)
    path = save_server.save_dir / "gyroid" / "run01" / "simulation.py"
    assert saved["path"] == str(path)
    assert path.read_text() == saved["script"]
    compile(path.read_text(), str(path), "exec")
    assert not (path.parent / "simulation").exists()  # Saving does not run the script.


def test_server_save_preserves_existing_file(save_server):
    payload = save_payload()
    status, _, body = request(save_server, "POST", "/api/save", payload)
    assert status == 201
    path = Path(json.loads(body)["path"])
    before = path.read_bytes()
    payload["values"]["chi_n"] = "21"
    status, _, _ = request(save_server, "POST", "/api/save", payload)
    assert status == 409
    assert path.read_bytes() == before


@pytest.mark.parametrize("filename, directory", (
    ("../escape.py", ""), ("/tmp/escape.py", ""), ("simulation.sh", ""),
    ("simulation.py", "../escape"), ("simulation.py", "/tmp/escape"),
    ("simulation.py", "nested/../escape"), ("simulation.py", "nested\\escape"),
    ("simulation.py", "\0"), ("simulation.py", False), (None, ""),
))
def test_server_save_rejects_invalid_paths(save_server, filename, directory):
    status, _, body = request(save_server, "POST", "/api/save", save_payload(filename, directory))
    assert status == 400
    assert json.loads(body)["error"]
    assert not save_server.save_dir.exists()


def test_server_save_rejects_symlink_directory(save_server, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    save_server.save_dir.mkdir()
    (save_server.save_dir / "linked").symlink_to(outside, target_is_directory=True)
    status, _, body = request(save_server, "POST", "/api/save", save_payload(directory="linked"))
    assert status == 500
    assert json.loads(body)["error"]
    assert not (outside / "simulation.py").exists()


def test_server_save_preserves_symlink_file(save_server, tmp_path):
    outside = tmp_path / "original.py"
    outside.write_text("original content")
    save_server.save_dir.mkdir()
    (save_server.save_dir / "simulation.py").symlink_to(outside)
    status, _, _ = request(save_server, "POST", "/api/save", save_payload())
    assert status == 409
    assert outside.read_text() == "original content"


def test_server_save_validates_before_creating_files(save_server):
    payload = save_payload()
    payload["values"]["ds"] = "0"
    status, _, _ = request(save_server, "POST", "/api/save", payload)
    assert status == 400
    assert not save_server.save_dir.exists()


def test_concurrent_server_saves_preserve_first_result(save_server):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: request(save_server, "POST", "/api/save", save_payload()), range(2)))
    assert sorted(result[0] for result in results) == [201, 409]
    winner = next(json.loads(body) for status, _, body in results if status == 201)
    assert Path(winner["path"]).read_text() == winner["script"]


def test_server_save_checks_origin(save_server):
    status, _, _ = request(save_server, "POST", "/api/save", save_payload(),
                           {"Origin": "https://unrelated.example"})
    assert status == 403
    assert not save_server.save_dir.exists()


@pytest.mark.parametrize("directory", ("x" * 513, "/".join(["layer"] * 17)))
def test_server_save_limits_directory_creation(save_server, directory):
    status, _, _ = request(save_server, "POST", "/api/save", save_payload(directory=directory))
    assert status == 400
    assert not save_server.save_dir.exists()
