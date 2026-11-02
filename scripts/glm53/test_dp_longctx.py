import json
from pathlib import Path
import tempfile
import unittest

import dp_longctx


def chars_by_four(text):
    return max(1, len(text) // 4)


def measured(messages, tools=None):
    # the stand-in for the server's count: the rendered text's characters by four
    return chars_by_four(dp_longctx.session_text(messages, tools or []))


class SessionShapeTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        self.files = []
        for i in range(12):
            path = root / "src" / f"file{i}.cpp"
            path.parent.mkdir(exist_ok=True)
            path.write_text(f"// file {i}\n" + "".join(f"int f{i}_{j}(int x) {{ return x * {j}; }}\n" for j in range(400)))
            self.files.append(path)

    def tearDown(self):
        self.folder.cleanup()

    def test_session_reaches_the_target_within_two_percent(self):
        # A session is a Claude Code history: a system prompt, the tools, a user request, then Read calls and their
        # results (the corpus's files) until the whole rendered prompt measures the target, no more and no more than
        # two percent less.
        target = 20000
        messages = dp_longctx.build_session(self.files, target, measured, dp_longctx.TOOLS, seed=3)
        size = measured(messages, dp_longctx.TOOLS)
        self.assertLessEqual(size, target, (size, target))
        self.assertGreaterEqual(size, int(target * 0.98), (size, target))
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        steps = messages[2:]
        self.assertEqual(len(steps) % 2, 0)
        self.assertGreater(len(steps), 2)
        for call, result in zip(steps[::2], steps[1::2]):
            self.assertEqual(call["role"], "assistant")
            self.assertEqual(len(call["tool_calls"]), 1)
            tool_call = call["tool_calls"][0]
            self.assertEqual(tool_call["function"]["name"], "Read")
            self.assertEqual(result["role"], "tool")
            self.assertEqual(result["tool_call_id"], tool_call["id"])
            path = json.loads(tool_call["function"]["arguments"])["file_path"]
            self.assertTrue(Path(path).read_text().startswith(result["content"]), path)

    def test_session_never_passes_the_target_when_a_file_tokenizes_denser(self):
        # between exact counts the builder estimates the next file at the rate seen so far; a file that tokenizes
        # four times denser than the ones before it must still leave the history at or under the target, since the
        # server refuses a prompt over the slot
        def dense_measure(messages, tools=None):
            size = measured(messages, tools)
            for m in messages:
                if m["role"] == "tool" and m["content"].startswith("// dense"):
                    size += 3 * len(m["content"]) // 4
            return size
        for i in (1, 4, 6, 9, 11):
            self.files[i].write_text("// dense\n" + self.files[i].read_text())
        for seed in range(8):
            messages = dp_longctx.build_session(self.files, 20000, dense_measure, dp_longctx.TOOLS, seed=seed)
            self.assertLessEqual(dense_measure(messages, dp_longctx.TOOLS), 20000, seed)

    def test_two_seeds_read_the_files_in_different_orders(self):
        a = dp_longctx.build_session(self.files, 12000, measured, dp_longctx.TOOLS, seed=1)
        b = dp_longctx.build_session(self.files, 12000, measured, dp_longctx.TOOLS, seed=2)
        self.assertNotEqual(a[2]["tool_calls"][0]["function"]["arguments"], b[2]["tool_calls"][0]["function"]["arguments"])

    def test_next_turn_keeps_the_history_as_its_prefix(self):
        # Every later turn appends the reply and one more step; the earlier messages stay byte for byte, so the
        # server's cache can keep the prefix.
        messages = dp_longctx.build_session(self.files, 8000, measured, dp_longctx.TOOLS, seed=5)
        before = json.dumps(messages)
        longer = dp_longctx.next_turn(messages, "I read it.", self.files[0], "now summarize")
        self.assertEqual(json.dumps(longer[:len(messages)]), before)
        self.assertEqual(longer[len(messages)]["role"], "assistant")
        self.assertEqual(longer[len(messages)]["content"], "I read it.")
        self.assertEqual(longer[len(messages) + 1]["role"], "user")
        self.assertIn("now summarize", longer[len(messages) + 1]["content"])
        self.assertEqual([m["role"] for m in longer[len(messages) + 2:]], ["assistant", "tool"])


class SummaryTest(unittest.TestCase):
    def timing(self, stream, turn, prompt_n, cache_n, predicted_n, prompt_ms, predicted_ms):
        return {"stream": stream, "turn": turn, "wall_ms": prompt_ms + predicted_ms,
                "timings": {"prompt_n": prompt_n, "cache_n": cache_n, "predicted_n": predicted_n,
                            "prompt_ms": prompt_ms, "predicted_ms": predicted_ms}}

    def test_summary_names_the_turn_that_lost_its_cache(self):
        results = [
            self.timing("a", 1, 400000, 0, 64, 120000.0, 1000.0),
            self.timing("b", 1, 400000, 0, 64, 125000.0, 1100.0),
            self.timing("a", 2, 900, 400050, 64, 400.0, 1000.0),
            self.timing("b", 2, 400900, 0, 64, 121000.0, 1000.0),  # the slot lost its cache
        ]
        summary = dp_longctx.summarize(results)
        self.assertEqual(summary["cache_lost"], [{"stream": "b", "turn": 2, "prompt_n": 400900, "cache_n": 0}])
        turn1, turn2 = summary["turns"][0], summary["turns"][1]
        self.assertEqual(turn1["turn"], 1)
        self.assertEqual(turn1["streams"], 2)
        self.assertEqual(turn1["prompt_n_max"], 400000)
        self.assertAlmostEqual(turn1["prompt_ms_median"], 122500.0)
        self.assertAlmostEqual(turn1["decode_tps_median"], (64 / 1000.0 * 1000 + 64 / 1100.0 * 1000) / 2, places=3)
        self.assertAlmostEqual(turn1["decode_tps_aggregate"], 64 / 1000.0 * 1000 + 64 / 1100.0 * 1000, places=3)
        self.assertEqual(turn2["prompt_n_max"], 400900)
        self.assertAlmostEqual(turn2["prompt_ms_median"], (400.0 + 121000.0) / 2)

    def test_peak_is_per_card(self):
        self.assertEqual(dp_longctx.peak([[1, 2, 3, 4], [5, 1, 1, 1], [2, 2, 9, 2]]), [5, 2, 9, 4])
        self.assertEqual(dp_longctx.peak([]), [])


if __name__ == "__main__":
    unittest.main()
