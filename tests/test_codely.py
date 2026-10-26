"""Codely CLI: its models as `codely --cmd "/model list"` lists them, a generation called in its -p mode with
--output-format json (the answer in its reply's `response`; the schema asked for in the prompt, as it has no
--json-schema), every tool it lists excluded so that it answers rather than acts (its own web tools alone when a
question asks for them), its error object reported as it reports it, a reply whose JSON does not parse asked for once
more. The commands are looked at, not run: no model, no network.

Run from the repo root:
    python3 -m unittest discover tests
"""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from textfold import agents, llm, models

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
LISTING = """Fetching models from Codely OAuth...
Provider Models:

Provider: Codely OAuth
Endpoint: https://codely-litellm.tuanjie.cn/v1/models

Available Models (3):

\x1b[32m● \x1b[0m\x1b[32mDeepSeek-V4.1-Flash\x1b[0m \x1b[32m(current)\x1b[0m
\x1b[90m○ \x1b[0mGLM-5.3
\x1b[90m○ \x1b[0mKIMI-K3.1-Pro

Use "/model use <model_name>" to switch to a different model
"""
TOOLS = "Available built-in tools:\n- read_file: ReadFile\n- web_search: WebSearch\n- web_fetch: WebFetch\n- write_file: WriteFile\n"


def reply(out, **kw):
    """A `codely -p --output-format json` reply, as the CLI writes it."""
    return SimpleNamespace(returncode=kw.pop("code", 0), stderr=kw.pop("stderr", ""),
                           stdout=json.dumps({"response": out if isinstance(out, str) else json.dumps(out),
                                              "stats": {}}, **kw))


class Catalog(unittest.TestCase):
    def test_its_models_as_model_list_lists_them(self):
        listing = SimpleNamespace(returncode=0, stdout=LISTING, stderr="")
        with mock.patch.object(agents, "argv", lambda p: [p]), mock.patch.object(agents, "_run", return_value=listing):
            got = agents._codely_models("/bin/codely")
        self.assertEqual([m["id"] for m in got], ["codely/codely/DeepSeek-V4.1-Flash", "codely/codely/GLM-5.3",
                                                  "codely/codely/KIMI-K3.1-Pro"])
        self.assertEqual(got[0]["name"], "Codely · DeepSeek-V4.1-Flash")
        self.assertEqual(got[0]["runtime"], "codely")
        self.assertIn("minimal", got[0]["efforts"])   # (its least: no level off)

    def test_a_listing_it_cannot_read_says_so(self):
        listing = SimpleNamespace(returncode=0, stdout="Please sign in\n", stderr="")
        with mock.patch.object(agents, "argv", lambda p: [p]), mock.patch.object(agents, "_run", return_value=listing):
            with self.assertRaises(RuntimeError) as e:
                agents._codely_models("/bin/codely")
        self.assertIn("sign in", str(e.exception))

    def test_a_model_id(self):
        spec = models.resolve("codely/codely/GLM-5.3", {"model": None})
        self.assertEqual((spec["kind"], spec["runtime"], spec["model"]), ("agent", "codely", "GLM-5.3"))
        with mock.patch.object(agents, "run_codely", return_value=({"ok": True}, None)) as run:
            self.assertEqual(llm._once("p", SCHEMA, spec, 30, "s"), ({"ok": True}, None))
        self.assertEqual(run.call_args.args[:2], ("GLM-5.3", "p"))


class Tools(unittest.TestCase):
    def test_the_names_it_lists(self):
        listing = SimpleNamespace(returncode=0, stdout=TOOLS, stderr="")
        with mock.patch.dict(agents._CODELY_TOOLS, {}, clear=True), \
                mock.patch.object(agents, "argv", lambda p: [p]), mock.patch.object(agents, "child_env", lambda: {}), \
                mock.patch("subprocess.run", lambda cmd, **kw: listing):
            self.assertEqual(agents.codely_tools("/bin/codely"), ["read_file", "web_search", "web_fetch", "write_file"])
            agents.codely_tools("/bin/codely")   # (asked once per CLI: the cache answers)

    def test_no_list_no_generation(self):
        gone = SimpleNamespace(returncode=0, stdout="", stderr="not signed in")
        with mock.patch.dict(agents._CODELY_TOOLS, {}, clear=True), \
                mock.patch.object(agents, "argv", lambda p: [p]), mock.patch.object(agents, "child_env", lambda: {}), \
                mock.patch("subprocess.run", lambda cmd, **kw: gone):
            with self.assertRaises(RuntimeError) as e:
                agents.codely_tools("/bin/codely")   # (a tool left in would act on its own)
        self.assertIn("not signed in", str(e.exception))


class Generation(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(agents._CODELY_TOOLS, {}, clear=True)   # (each generation lists the tools itself)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_it(self, tools=False, effort=None, replies=None):
        calls = []

        def fake(cmd, **kw):
            if "--list-tools" in cmd:
                return SimpleNamespace(returncode=0, stdout=TOOLS, stderr="")
            calls.append((cmd, kw))
            return replies[len(calls) - 1] if replies else reply({"ok": True})

        with mock.patch.object(agents, "path_of", return_value="/bin/codely"), \
                mock.patch.object(agents, "argv", lambda p: [p]), \
                mock.patch.object(agents, "child_env", lambda: {}), \
                mock.patch("subprocess.run", fake):
            out, cost = agents.run_codely("GLM-5.3", "Fold this.", SCHEMA, "The system.", effort, 30, tools)
        return out, cost, calls

    def test_a_generation(self):
        out, cost, calls = self.run_it(effort="off")
        self.assertEqual((out, cost), ({"ok": True}, None))
        cmd, kw = calls[0]
        self.assertEqual(cmd[1], "-p")
        self.assertEqual(cmd[2], "")   # (the prompt goes on stdin, as claude -p's)
        self.assertEqual(cmd[cmd.index("--model") + 1], "GLM-5.3")
        self.assertEqual(cmd[cmd.index("--output-format") + 1], "json")
        self.assertEqual(cmd[cmd.index("--path-policy") + 1], "strict")   # (the workspace alone)
        self.assertEqual(cmd[cmd.index("--reasoning_effort") + 1], "minimal")   # (its off)
        self.assertNotIn("--json-schema", cmd)   # (it has none)
        self.assertIn("--no-agents", cmd)
        excluded = {cmd[i + 1] for i, x in enumerate(cmd) if x == "--exclude-tools"}
        self.assertEqual(excluded, {"read_file", "web_search", "web_fetch", "write_file"})
        self.assertTrue(kw["input"].startswith("The system.\n\nFold this."))   # (no flag for its system prompt)
        self.assertIn(json.dumps(SCHEMA), kw["input"])   # (the schema asked for in the prompt)
        self.assertNotEqual(Path(kw["cwd"]).resolve(), Path.cwd().resolve())   # a scratch folder

    def test_its_web_tools_alone_when_a_question_asks_for_them(self):
        out, _, calls = self.run_it(tools=True)
        self.assertEqual(out, {"ok": True})
        cmd = calls[0][0]
        excluded = {cmd[i + 1] for i, x in enumerate(cmd) if x == "--exclude-tools"}
        self.assertEqual(excluded, {"read_file", "write_file"})   # (web_search and web_fetch stay)

    def test_a_reply_whose_json_does_not_parse_is_asked_again(self):
        out, _, calls = self.run_it(replies=[reply("not json at all"), reply({"ok": True})])
        self.assertEqual(out, {"ok": True})
        self.assertEqual(len(calls), 2)

    def test_its_error_as_it_reports_it(self):
        refused = SimpleNamespace(returncode=1, stderr="", stdout=json.dumps(
            {"error": {"type": "MODEL_ERROR", "message": "401 team not allowed to access model", "status": 401}}))
        with self.assertRaises(RuntimeError) as e:
            self.run_it(replies=[refused])
        self.assertIn("team not allowed", str(e.exception))

    def test_an_exit_without_a_reply_is_reported(self):
        gone = SimpleNamespace(returncode=2, stdout="", stderr="boom")
        with self.assertRaises(RuntimeError) as e:
            self.run_it(replies=[gone])
        self.assertIn("boom", str(e.exception))


if __name__ == "__main__":
    unittest.main()
