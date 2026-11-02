#!/usr/bin/env python3
"""Serve the documented GLM-5.3 Flash recipe; --dry-run prints JSON without starting a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "models/templates/GLM-5.3-Flash.jinja"
DRAFT_VOCAB = ROOT / "scripts/glm53/assets/mtp-draft-vocab-98304.i32"
DEFAULT_MTP_DEPTH = "2"
ASSET_SHA256 = {
    TEMPLATE: "0bc9a9e2801e8dfbea88403de8523d87cb26cc91d9308e68bce4b50b90708ddb",
    DRAFT_VOCAB: "8e83de6161b0f75c212f7079b00e10af2c03701ae17afe62ee48bca35717e3e6",
}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local first GGUF shard; other shards must be alongside it")
    parser.add_argument("--mmproj", type=Path, help="Local vision projector; omit for text-only serving")
    parser.add_argument("--server", type=Path, default=ROOT / "build/bin/llama-server",
                        help="llama-server from this fork, with matching libraries beside it")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--mtp-depth", choices=["1", "2", "off"], default=DEFAULT_MTP_DEPTH)
    parser.add_argument("--prompt-step", type=int, default=1024, help="Prompt tokens per step alongside generation; 0 removes the cap")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def make_plan(args):
    if not 1 <= args.port <= 65535 or args.prompt_step < 0:
        raise ValueError("port must be 1..65535 and prompt-step must be nonnegative")
    if not args.host or any(char in args.host for char in "\n\r\0"):
        raise ValueError("host must be a nonempty address without control characters")
    server = args.server.expanduser().resolve()
    argv = [str(server), "-m", str(args.model.expanduser().resolve())]
    if args.mmproj:
        argv += ["--mmproj", str(args.mmproj.expanduser().resolve())]
    argv += ["-ngl", "99", "-sm", "tensor", "-ts", "1,1", "--jinja", "-fa", "on",
             "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "-ctki", "q8_0",
             "-b", "2048", "-ub", "2048", "--prompt-step", str(args.prompt_step),
             "--alias", "GLM-5.3-Flash-Uncensored-IQ3_XXS", "--chat-template-file", str(TEMPLATE)]
    if args.mtp_depth != "off":
        argv += ["--spec-type", "draft-mtp", "--spec-draft-n-max", args.mtp_depth,
                 "--spec-draft-mtp-vocab", str(DRAFT_VOCAB), "--spec-draft-mtp-adaptive", "0.18"]
    argv += ["-c", "524288", "-np", "3", "--kv-unified", "--cache-ram", "32768",
             "--no-cache-idle-slots", "--metrics", "--host", args.host, "--port", str(args.port),
             "--log-timestamps"]
    env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0,1",
        "GGML_CUDA_GRAPH_MAX": "0", "GGML_CUDA_ALLREDUCE": "internal",
        "LLAMA_INDEXER_Q8_QUERY": "1", "LLAMA_MTP_DEVICE_H": "1", "LLAMA_TOK_EMBD_GPU": "1",
        "LLAMA_KPOOL_DIRECT_INDICES": "0", "GGML_CUDA_MMQ_MOE_NCOLS": "0",
        "GGML_CUDA_MMQ_L2_PF": "1", "GGML_SCHED_DEBUG": "0",
        "LD_LIBRARY_PATH": str(server.parent),
    }
    return argv, env


def execution_environment(settings, inherited=None):
    source = os.environ if inherited is None else inherited
    # Keep unrelated application settings; remove implicit engine/profile/debug overrides.
    env = {key: value for key, value in source.items()
           if not key.startswith(("GGML_", "LLAMA_", "CUDA_", "NVIDIA_"))}
    env.update(settings)
    return env


def verify_assets():
    for path, expected in ASSET_SHA256.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("serving asset SHA256 mismatch: " + str(path))


def main(args):
    argv, settings = make_plan(args)
    verify_assets()
    if args.dry_run:
        print(json.dumps({"argv": argv, "env": settings}, indent=2))
        return 0
    for label, path in [("server", Path(argv[0])), ("model", args.model.expanduser()),
                        ("mmproj", args.mmproj.expanduser() if args.mmproj else None)]:
        if path is not None and not path.is_file():
            raise ValueError("missing local " + label + ": " + str(path))
    if not os.access(argv[0], os.X_OK):
        raise ValueError("server is not executable: " + argv[0])
    os.execve(argv[0], argv, execution_environment(settings))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(arguments()))
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
