"""Qoder CLI: its models as `--list-models` lists them, a generation called as claude -p is (its --reasoning-effort,
off being "none", and the schema asked for in the prompt: Qoder CLI has no --json-schema), its web tools when a
question asks for them, its error as it reports it, a reply whose JSON does not parse asked for once more. The
commands are looked at, not run: no model, no network.

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


def reply(out, **kw):
    """A `qodercli -p --output-format json` result, as the CLI writes it."""
    return SimpleNamespace(returncode=kw.pop("code", 0), stderr=kw.pop("stderr", ""),
                           stdout=json.dumps({"type": "result", "subtype": "success", "is_error": False,
                                              "result": out if isinstance(out, str) else json.dumps(out),
                                              "total_cost_usd": 0, **kw}))


class Catalog(unittest.TestCase):
    def test_its_models_as_list_models_lists_them(self):
        listing = SimpleNamespace(returncode=0, stdout="MODEL\nAuto\nDeepSeek-Flash\n", stderr="")
        with mock.patch.object(agents, "argv", lambda p: [p]), mock.patch.object(agents, "_run", return_value=listing):
            got = agents._qodercli_models("/bin/qodercli")
        self.assertEqual([m["id"] for m in got], ["qodercli/qodercli/Auto", "qodercli/qodercli/DeepSeek-Flash"])
        self.assertEqual(got[0]["name"], "Qoder CLI · Auto")
        self.assertIn("none", got[0]["efforts"])

    def test_a_model_id(self):
        spec = models.resolve("qodercli/qodercli/DeepSeek-Flash", {"model": None})
        self.assertEqual((spec["kind"], spec["runtime"], spec["model"]), ("agent", "qodercli", "DeepSeek-Flash"))
        with mock.patch.object(agents, "run_qodercli", return_value=({"ok": True}, 0)) as run:
            self.assertEqual(llm._once("p", SCHEMA, spec, 30, "s"), ({"ok": True}, 0))
        self.assertEqual(run.call_args.args[:2], ("DeepSeek-Flash", "p"))


class Generation(unittest.TestCase):
    def run_it(self, tools=False, effort=None, replies=None):
        calls = []

        def fake(cmd, **kw):
            calls.append((cmd, kw))
            return (replies[len(calls) - 1] if replies else reply({"ok": True}))

        with mock.patch.object(agents, "path_of", return_value="/bin/qodercli"), \
                mock.patch.object(agents, "argv", lambda p: [p]), \
                mock.patch.object(agents, "env_for", lambda p: {}), \
                mock.patch("subprocess.run", fake):
            out, cost = agents.run_qodercli("DeepSeek-Flash", "Fold this.", SCHEMA, "The system.", effort, 30, tools)
        return out, cost, calls

    def test_a_generation(self):
        out, cost, calls = self.run_it(effort="off")
        self.assertEqual((out, cost), ({"ok": True}, 0))
        cmd, kw = calls[0]
        self.assertEqual(cmd[1:4], ["-p", "--model", "DeepSeek-Flash"])
        self.assertEqual(cmd[cmd.index("--tools") + 1], "")
        self.assertEqual(cmd[cmd.index("--system-prompt") + 1], "The system.")
        self.assertEqual(cmd[cmd.index("--reasoning-effort") + 1], "none")   # (its off)
        self.assertIn("--setting-sources", cmd)   # (the reader's own settings stay out)
        self.assertEqual(cmd[cmd.index("--output-format") + 1], "json")
        self.assertNotIn("--json-schema", cmd)   # (it has none)
        self.assertNotIn("--setting-sources", self.run_it(tools=True)[2][0][0])
        self.assertIn('"ok"', kw["input"])   # (the schema asked for in the prompt)
        self.assertNotEqual(Path(kw["cwd"]).resolve(), Path.cwd().resolve())   # a scratch folder
        self.assertEqual(kw["input"], "Fold this.\n\nReply with one JSON object that matches this JSON Schema, and nothing else:\n"
                                      + json.dumps(SCHEMA))

    def test_its_web_tools_when_a_question_asks_for_them(self):
        out, _, calls = self.run_it(tools=True)
        self.assertEqual(out, {"ok": True})
        cmd = calls[0][0]
        self.assertEqual(cmd[cmd.index("--tools") + 1], "WebSearch,WebFetch")
        self.assertEqual(cmd[cmd.index("--allowed-tools") + 1], "WebSearch,WebFetch")
        self.assertIn("--strict-mcp-config", cmd)
        self.assertEqual(cmd[cmd.index("--append-system-prompt") + 1], "The system.")   # (its own system prompt stays)
        self.assertNotIn("--system-prompt", cmd)

    def test_a_reply_whose_json_does_not_parse_is_asked_again(self):
        out, _, calls = self.run_it(replies=[reply("not json at all"), reply({"ok": True})])
        self.assertEqual(out, {"ok": True})
        self.assertEqual(len(calls), 2)

    def test_its_error_as_it_reports_it(self):
        with self.assertRaises(RuntimeError) as e:
            self.run_it(replies=[reply("quota exceeded", is_error=True)])
        self.assertIn("quota exceeded", str(e.exception))

    def test_asked_with_the_user_settings_when_they_carry_the_sign_in(self):
        refused = SimpleNamespace(returncode=1, stdout="", stderr="Please sign in")
        out, _, calls = self.run_it(replies=[refused, reply({"ok": True})])
        self.assertEqual(out, {"ok": True})
        self.assertIn("--setting-sources", calls[0][0])
        self.assertNotIn("--setting-sources", calls[1][0])


if __name__ == "__main__":
    unittest.main()
