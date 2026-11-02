import os
import tempfile
import threading
import pytest
from utils import *

server = ServerPreset.tinyllama2()

class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0
    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content
    def wait_for(self, text):
        content = ""
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            content += self.drain()
            if text in content:
                return content
            time.sleep(0.01)
        return content

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 100
    server.kv_unified = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


LONG_PROMPT = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
    "He met many creatures along the way including dragons and fairies "
    "and wizards who helped him on his noble quest to save the kingdom."
)


# idle slot cleared on launch should restore from cache-ram
def test_clear_and_restore():
    global server
    server.start()
    log = LogReader(server.log_path)

    # verify feature is enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in log.wait_for("__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__")

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    original_prompt_n = res.body["timings"]["prompt_n"]

    # Slot 0 is the only slot with KV — should NOT be cleared
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()

    # Launching slot 1 clears idle slot 0
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in log.wait_for("__TEST_TAG_CACHE_IDLE_SLOT__")

    # Re-send same prompt — should restore from cache-ram
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "updating prompt cache" in log.wait_for("updating prompt cache")
    assert res.body["timings"]["cache_n"] > 0
    assert res.body["timings"]["prompt_n"] < original_prompt_n

    # Follow-up — slot 0 kept its KV, no clearing needed
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT + " The knight finally reached the castle gates.",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


# the same, with the returning request naming the slot it was on: the slot is empty, and its prompt comes from cache-ram
def test_clear_and_restore_named_slot():
    global server
    server.start()
    log = LogReader(server.log_path)

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    original_prompt_n = res.body["timings"]["prompt_n"]

    # Launching slot 1 clears idle slot 0
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in log.wait_for("__TEST_TAG_CACHE_IDLE_SLOT__")

    # Re-send the same prompt to slot 0 by id: it restores from cache-ram
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "updating prompt cache" in log.wait_for("updating prompt cache")
    assert res.body["timings"]["cache_n"] > 0
    assert res.body["timings"]["prompt_n"] < original_prompt_n


# a request for a busy slot waits for it: the prompt cache does not save or replace the slot's state under the running request
def test_named_busy_slot_left_alone():
    global server
    server.start()
    log = LogReader(server.log_path)

    other = {"prompt": "The little dog ran to the park with a red ball.", "id_slot": 0, "cache_prompt": True}
    res = server.make_request("POST", "/completion", data=other)
    assert res.status_code == 200

    # Launching slot 1 saves idle slot 0 to cache-ram and clears it
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    long_run = {"prompt": LONG_PROMPT, "id_slot": 0, "n_predict": 160, "ignore_eos": True, "cache_prompt": False}
    res = server.make_request("POST", "/completion", data=long_run)
    assert res.status_code == 200
    expected = res.body["content"]
    log.drain()

    # the same run streamed, and once slot 0 generates, the other prompt asks for slot 0: its cached state is the best match
    other_res = []
    thread = None
    content = ""
    for data in server.make_stream_request("POST", "/completion", data={**long_run, "stream": True}):
        if thread is None:
            thread = threading.Thread(target=lambda: other_res.append(server.make_request("POST", "/completion", data=other)))
            thread.start()
        if not data["stop"]:
            content += data["content"]
    assert thread is not None
    thread.join()
    assert other_res[0].status_code == 200

    if "requested slot is unavailable" not in log.drain():
        pytest.skip("the other request did not reach the server while slot 0 was generating")  # ty: ignore[too-many-positional-arguments]
    assert content == expected


def test_disabled_with_flag():
    global server
    server.no_cache_idle_slots = True
    server.start()
    log = LogReader(server.log_path)

    # Feature should not be enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" not in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # Request on different slot — should NOT trigger clearing
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


def test_ram_prefix_beats_short_resident_prefix():
    server.start()
    long_prompt = LONG_PROMPT * 2
    short_prompt = LONG_PROMPT[:90]
    first = server.make_request("POST", "/completion", data={
        "prompt": long_prompt, "id_slot": 0, "cache_prompt": True,
    })
    assert first.status_code == 200
    # Admission of slot 1 happens before idle slot 0 is saved. This leaves the
    # long prefix in RAM and the short exact prefix resident in slot 1.
    short = server.make_request("POST", "/completion", data={
        "prompt": short_prompt, "id_slot": 1, "cache_prompt": True, "n_predict": 1,
    })
    assert short.status_code == 200
    resumed = server.make_request("POST", "/completion", data={
        "prompt": long_prompt + " The tool returned the location of the sword.",
        "cache_prompt": True,
    })
    assert resumed.status_code == 200
    assert resumed.body["timings"]["cache_n"] >= first.body["timings"]["prompt_n"] - 1


def test_idle_slot_kept_when_snapshot_exceeds_ram_limit():
    server.cache_ram = 1
    server.n_ctx = 4096
    server.start()
    log = LogReader(server.log_path)
    first = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT * 10, "id_slot": 0, "cache_prompt": True, "n_predict": 1,
    })
    assert first.status_code == 200
    other = server.make_request("POST", "/completion", data={
        "prompt": "A different conversation", "id_slot": 1, "cache_prompt": True,
    })
    assert other.status_code == 200
    assert "exceeds cache size limit" in log.wait_for("exceeds cache size limit")
    resumed = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT * 10 + " The tool returned successfully.",
        "id_slot": 0, "cache_prompt": True,
    })
    assert resumed.status_code == 200
    assert resumed.body["timings"]["cache_n"] >= first.body["timings"]["prompt_n"] - 1


def test_kv_pressure_preserves_idle_tool_continuation():
    server.no_cache_idle_slots = True
    server.n_ctx = 512
    server.n_batch = 128
    server.n_ubatch = 128
    server.start()
    log = LogReader(server.log_path)
    tokenized = server.make_request("POST", "/tokenize", data={"content": LONG_PROMPT * 5})
    assert tokenized.status_code == 200
    tokens = tokenized.body["tokens"]
    assert len(tokens) > 400
    first = server.make_request("POST", "/completion", data={
        "prompt": tokens[:350], "id_slot": 0, "cache_prompt": True, "n_predict": 1,
    })
    assert first.status_code == 200
    other = server.make_request("POST", "/completion", data={
        "prompt": list(reversed(tokens[:350])), "id_slot": 1, "cache_prompt": True, "n_predict": 1,
    })
    assert other.status_code == 200
    assert "purging slot 0" in log.wait_for("purging slot 0")
    resumed = server.make_request("POST", "/completion", data={
        "prompt": tokens[:370], "id_slot": 0, "cache_prompt": True, "n_predict": 1,
    })
    assert resumed.status_code == 200
    assert resumed.body["timings"]["cache_n"] >= 349
