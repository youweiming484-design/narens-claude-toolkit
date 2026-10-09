import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import support
from mcp_fixer import cli

FIXTURES = support.TESTS / "fixtures"
CLEAN = str(FIXTURES / "clean_tools.json")
MESSY = str(FIXTURES / "messy_tools.json")


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(list(argv))
        except SystemExit as exc:  # argparse's own usage errors
            code = exc.code
    return code, out.getvalue(), err.getvalue()


class CliCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def write(self, name, text):
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    def assert_usage_error(self, *argv, fragment=None):
        code, out, err = run(*argv)
        self.assertEqual(code, 2, (out, err))
        self.assertEqual(out, "")
        self.assertTrue(err.startswith("error: "), err)
        self.assertEqual(err.count("\n"), 1, err)
        self.assertNotIn("Traceback", err)
        if fragment:
            self.assertIn(fragment, err)


class FileModeTests(CliCase):
    def test_empty_tool_list_file_is_valid(self):
        for text in ("[]", '{"tools": []}'):
            for command in ("score", "patch"):
                with self.subTest(text=text, command=command):
                    path = self.write("empty-tools.json", text)
                    with patch.object(cli, "list_tools_stdio") as server:
                        code, out, err = run(
                            command, "--tools-json", path
                        )
                        self.assertEqual((code, err), (0, ""))
                        self.assertTrue(out)
                        server.assert_not_called()

    def test_a_clean_file_scores_100_as_text(self):
        code, out, err = run("score", "--tools-json", CLEAN)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("mcp-fixer score: 100/100", out)

    def test_json_format(self):
        code, out, _ = run("score", "--tools-json", MESSY, "--format", "json")
        data = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(data["schemaVersion"], 1)
        self.assertEqual(data["source"]["kind"], "file")
        rules = {f["rule"] for f in data["findings"]}
        self.assertTrue({"D001", "D002", "N001", "P001", "P002", "P003", "N002"} <= rules, rules)

    def test_the_messy_file_scores_70_by_hand(self):
        # run: D002 + N001 + P003 + P001 + P002 (10 each) + P005 (4) = -54 -> 46
        # searchItems: D001 (-25) -> 75; list_items: clean -> 100
        # mean 73.67, minus 4 for mixed naming styles (N002) = 69.67 -> 70
        _, out, _ = run("score", "--tools-json", MESSY, "--format", "json")
        self.assertEqual(json.loads(out)["score"], 70)

    def test_a_bare_array_and_an_object_with_tools_both_work(self):
        tools = json.loads(Path(CLEAN).read_text(encoding="utf-8"))["tools"]
        bare = self.write("bare.json", json.dumps(tools))
        self.assertEqual(run("score", "--tools-json", bare)[0], 0)

    def test_a_bom_is_tolerated(self):
        path = self.dir / "bom.json"
        path.write_bytes(b"\xef\xbb\xbf" + Path(CLEAN).read_bytes())
        self.assertEqual(run("score", "--tools-json", str(path))[0], 0)

    def test_the_same_file_gives_byte_identical_json(self):
        first = run("score", "--tools-json", MESSY, "--format", "json")[1]
        second = run("score", "--tools-json", MESSY, "--format", "json")[1]
        self.assertEqual(first, second)

    def test_non_ascii_text_report_does_not_crash(self):
        path = self.write("na.json", json.dumps([{"name": "recuperer", "description": "", "inputSchema": {"type": "object", "properties": {}}, "title": "日本語"}], ensure_ascii=False))
        code, out, err = run("score", "--tools-json", path)
        self.assertEqual((code, err), (0, ""))


class ConsoleEncodingTests(CliCase):
    def test_a_legacy_console_encoding_does_not_crash_the_report(self):
        name = "日本語_tool"  # cannot be encoded in cp1252
        path = self.write("jp.json", json.dumps([{"name": name, "description": "", "inputSchema": {"type": "object", "properties": {}}}], ensure_ascii=False))
        env = dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONPATH=str(support.SRC))
        done = subprocess.run(
            [sys.executable, "-m", "mcp_fixer", "score", "--tools-json", path],
            capture_output=True, env=env, timeout=60,
        )
        self.assertEqual(done.returncode, 0, done.stderr.decode("utf-8", "replace"))
        self.assertIn(name, done.stdout.decode("utf-8"))


class DeepJsonTests(CliCase):
    def test_a_file_nested_too_deeply_to_parse_is_a_usage_error(self):
        self.assert_usage_error("score", "--tools-json", self.write("deep.json", "[" * 5000), fragment="not valid JSON")

    def test_a_schema_nested_very_deeply_never_gives_a_traceback(self):
        depth = 900
        schema = '{"type":"object","properties":{"x":' * depth + '{"type":"string"}' + "}}" * depth
        text = '[{"name":"deep_tool","description":"A deeply nested tool schema here.","inputSchema":' + schema + "}]"
        code, out, err = run("score", "--tools-json", self.write("deepschema.json", text))
        self.assertIn(code, (0, 2))
        self.assertNotIn("Traceback", err)
        if code == 2:
            self.assertEqual(err.count("\n"), 1)


class MinScoreAndOutTests(CliCase):
    def test_min_score_passes_at_the_boundary_and_fails_below(self):
        _, out, _ = run("score", "--tools-json", MESSY, "--format", "json")
        score = json.loads(out)["score"]
        self.assertEqual(run("score", "--tools-json", MESSY, "--min-score", str(score))[0], 0)
        code, text, err = run("score", "--tools-json", MESSY, "--min-score", str(score + 1))
        self.assertEqual((code, err), (1, ""))
        self.assertIn("mcp-fixer score:", text)  # the report is still printed

    def test_out_writes_the_report_to_the_file_instead_of_stdout(self):
        target = self.dir / "report.json"
        code, out, err = run("score", "--tools-json", MESSY, "--format", "json", "--out", str(target))
        self.assertEqual((code, out, err), (0, "", ""))
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["schemaVersion"], 1)
        self.assertNotIn("\r", target.read_bytes().decode("utf-8"))

    def test_an_unwritable_out_is_a_usage_error(self):
        self.assert_usage_error(
            "score", "--tools-json", CLEAN, "--out", str(self.dir / "no-such-dir" / "r.json"),
            fragment="cannot write",
        )


class UsageErrorTests(CliCase):
    def test_neither_input_mode(self):
        self.assert_usage_error("score", fragment="--tools-json")

    def test_both_input_modes(self):
        self.assert_usage_error("score", "--tools-json", CLEAN, "--", sys.executable, "-V", fragment="not both")

    def test_a_missing_file(self):
        self.assert_usage_error("score", "--tools-json", str(self.dir / "nope.json"), fragment="cannot read")

    def test_a_file_that_is_not_json(self):
        self.assert_usage_error("score", "--tools-json", self.write("bad.json", "{not json"), fragment="not valid JSON")

    def test_a_file_that_is_not_utf8(self):
        path = self.dir / "latin.json"
        path.write_bytes(b"\x80\x81\x82")
        self.assert_usage_error("score", "--tools-json", str(path), fragment="UTF-8")

    def test_a_file_with_the_wrong_shape(self):
        for text in ('{"foo": []}', '"just a string"', "5", '{"tools": "x"}'):
            with self.subTest(text):
                self.assert_usage_error("score", "--tools-json", self.write("shape.json", text), fragment="must be a JSON array")

    def test_a_bad_env_pair(self):
        for pair in ("NOEQUALS", "=value"):
            with self.subTest(pair):
                self.assert_usage_error("score", "--env", pair, "--", sys.executable, "-V", fragment="KEY=VALUE")

    def test_bad_numbers(self):
        for flag, value in (("--timeout", "0"), ("--timeout", "-3"), ("--timeout", "nan"), ("--timeout", "inf"),
                            ("--min-score", "nan"), ("--min-score", "inf")):
            with self.subTest((flag, value)):
                self.assert_usage_error("score", "--tools-json", CLEAN, flag, value)

    def test_empty_tools_json_without_server(self):
        for command in ("score", "patch"):
            with self.subTest(command=command):
                self.assert_usage_error(
                    command, "--tools-json", "", fragment="empty"
                )

    def test_empty_tools_json_with_server(self):
        for command in ("score", "patch"):
            with self.subTest(command=command):
                with patch.object(
                    cli,
                    "list_tools_stdio",
                    return_value=([], {
                        "serverName": "demo-server",
                        "serverVersion": "1.0",
                        "protocolVersion": "2025-06-18",
                    }),
                ) as server:
                    self.assert_usage_error(
                        command, "--tools-json", "",
                        "--", "demo-server",
                        fragment="not both",
                    )
                    server.assert_not_called()


class ServerModeTests(CliCase):
    def server_args(self, mode="normal", *extra):
        return ["--", sys.executable, str(support.FAKE_SERVER), "--mode", mode, "--tools", CLEAN_TOOLS, *extra]

    def test_a_stdio_server_is_scored(self):
        code, out, err = run("score", "--format", "json", "--timeout", "10", *self.server_args("normal"))
        self.assertEqual((code, err), (0, ""))
        data = json.loads(out)
        self.assertEqual(data["source"], {"kind": "stdio", "protocolVersion": "2025-06-18", "serverName": "fake-server", "serverVersion": "1.2.3"})
        self.assertEqual(data["score"], 100)
        self.assertEqual(data["metrics"]["toolCount"], 2)

    def test_the_server_command_keeps_its_own_dashes(self):
        code, out, _ = run("score", "--timeout", "10", *self.server_args("normal", "--page-size", "1"))
        self.assertEqual(code, 0)
        self.assertIn("server: fake-server 1.2.3", out)

    def test_a_server_that_hangs_is_a_clean_error(self):
        self.assert_usage_error("score", "--timeout", "0.8", *self.server_args("hang"), fragment="timed out")

    def test_a_server_that_exits_is_a_clean_error(self):
        self.assert_usage_error("score", "--timeout", "5", *self.server_args("exit"), fragment="exited with code 4")

    def test_a_missing_server_command_is_a_clean_error(self):
        self.assert_usage_error("score", "--", "definitely-not-a-real-command-xyz", fragment="cannot find the server command")

    def test_env_values_reach_the_server(self):
        script = self.write("echo_env.py", (
            "import json, os, sys\n"
            "for line in sys.stdin:\n"
            "    m = json.loads(line)\n"
            "    if m.get('method') == 'initialize':\n"
            "        print(json.dumps({'jsonrpc': '2.0', 'id': m['id'], 'result': {'protocolVersion': '2025-06-18', 'serverInfo': {'name': os.environ.get('GREETING', 'unset'), 'version': '1'}}}), flush=True)\n"
            "    elif m.get('method') == 'tools/list':\n"
            "        print(json.dumps({'jsonrpc': '2.0', 'id': m['id'], 'result': {'tools': []}}), flush=True)\n"
        ))
        code, out, _ = run("score", "--format", "json", "--env", "GREETING=hello", "--timeout", "10", "--", sys.executable, script)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["source"]["serverName"], "hello")


class PatchCommandTests(CliCase):
    def test_a_patch_is_written_to_stdout_and_validates(self):
        code, out, err = run("patch", "--tools-json", MESSY)
        self.assertEqual((code, err), (0, ""))
        from mcp_fixer import patch_format
        patch = patch_format.validate_patch(json.loads(out))
        self.assertEqual(list(patch["tools"]), ["run", "searchItems"])

    def test_the_same_input_gives_byte_identical_output(self):
        self.assertEqual(run("patch", "--tools-json", MESSY)[1], run("patch", "--tools-json", MESSY)[1])

    def test_out_writes_a_new_file_and_prints_nothing(self):
        target = self.dir / "p.json"
        code, out, err = run("patch", "--tools-json", MESSY, "--out", str(target))
        self.assertEqual((code, out, err), (0, "", ""))
        self.assertEqual(target.read_text(encoding="utf-8"), run("patch", "--tools-json", MESSY)[1])
        self.assertNotIn(b"\r", target.read_bytes())

    def test_an_existing_out_file_is_never_overwritten_without_force(self):
        target = self.dir / "p.json"
        target.write_text("my edits", encoding="utf-8")
        self.assert_usage_error("patch", "--tools-json", MESSY, "--out", str(target), fragment="exists; use --force")
        self.assertEqual(target.read_text(encoding="utf-8"), "my edits")

    def test_force_overwrites(self):
        target = self.dir / "p.json"
        target.write_text("my edits", encoding="utf-8")
        code, _, _ = run("patch", "--tools-json", MESSY, "--out", str(target), "--force")
        self.assertEqual(code, 0)
        self.assertIn('"patchVersion"', target.read_text(encoding="utf-8"))

    def test_an_unwritable_out_is_a_usage_error(self):
        self.assert_usage_error("patch", "--tools-json", CLEAN, "--out", str(self.dir / "no-dir" / "p.json"), fragment="cannot write")

    def test_the_same_input_errors_as_score(self):
        self.assert_usage_error("patch", fragment="--tools-json")
        self.assert_usage_error("patch", "--tools-json", CLEAN, "--", sys.executable, "-V", fragment="not both")
        self.assert_usage_error("patch", "--tools-json", str(self.dir / "nope.json"), fragment="cannot read")
        self.assert_usage_error("patch", "--tools-json", self.write("bad.json", "{nope"), fragment="not valid JSON")
        self.assert_usage_error("patch", "--tools-json", CLEAN, "--timeout", "nan")

    def test_a_stdio_server_gives_a_patch_with_its_source(self):
        args = ["--", sys.executable, str(support.FAKE_SERVER), "--mode", "normal", "--tools", CLEAN_TOOLS]
        code, out, err = run("patch", "--timeout", "10", *args)
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(json.loads(out)["source"], {"serverName": "fake-server", "serverVersion": "1.2.3"})

    def test_a_schema_nested_very_deeply_never_gives_a_traceback(self):
        schema = '{"type":"object","properties":{"x":' * 900 + '{"type":"string"}' + "}}" * 900
        text = '[{"name":"deep_tool","description":"A deeply nested tool schema here.","inputSchema":' + schema + "}]"
        code, out, err = run("patch", "--tools-json", self.write("deep.json", text))
        self.assertIn(code, (0, 2))
        self.assertNotIn("Traceback", err)


class WrapCommandTests(CliCase):
    def test_a_missing_command_is_a_usage_error(self):
        patch = self.write("p.json", json.dumps({"patchVersion": 1, "tools": {}}))
        self.assert_usage_error("wrap", "--patch", patch, fragment="server command")

    def test_a_missing_patch_file_is_a_usage_error(self):
        self.assert_usage_error("wrap", "--patch", str(self.dir / "nope.json"), "--", sys.executable, "-V", fragment="cannot read")

    def test_an_invalid_patch_is_a_usage_error_naming_the_tool(self):
        patch = self.write("p.json", json.dumps({"patchVersion": 1, "tools": {"run": {"rename": ""}}}))
        self.assert_usage_error("wrap", "--patch", patch, "--", sys.executable, "-V", fragment="tool 'run': rename must be")

    def test_a_server_command_that_does_not_exist_is_a_usage_error(self):
        patch = self.write("p.json", json.dumps({"patchVersion": 1, "tools": {}}))
        self.assert_usage_error("wrap", "--patch", patch, "--", "definitely-not-a-real-command-xyz", fragment="cannot find the server command")

    def test_the_patch_option_is_required(self):
        code, out, err = run("wrap", "--", sys.executable, "-V")
        self.assertEqual(code, 2)
        self.assertIn("--patch", err)


# A tools file for the fake server (a bare array, as the fake server expects).
_TOOLS_DIR = tempfile.TemporaryDirectory()
CLEAN_TOOLS = str(Path(_TOOLS_DIR.name) / "tools.json")
Path(CLEAN_TOOLS).write_text(json.dumps(json.loads(Path(CLEAN).read_text(encoding="utf-8"))["tools"]), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
