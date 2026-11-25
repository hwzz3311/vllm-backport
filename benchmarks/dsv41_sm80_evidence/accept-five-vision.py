import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import os
import re
import struct
import threading
import time
import urllib.error
import urllib.request
import zlib

BASE = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:18159").rstrip("/")
MODEL = os.environ.get("VLLM_MODEL", "")
API_KEY = os.environ.get("VLLM_API_KEY")
HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["Authorization"] = f"Bearer {API_KEY}"
IMAGE_TOKEN = 129264
COLORS = {"red": (232, 25, 24), "green": (24, 197, 57), "blue": (21, 57, 232), "yellow": (247, 218, 18), "purple": (132, 27, 192)}

def png_chunk(name, data):
    return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))

def image_url(color, variation):
    side = 1344
    rgb = bytes(COLORS[color])
    accent = bytes((variation, 255 - variation, 19))
    line = b"\0" + rgb * side
    stripe = b"\0" + accent * 24 + rgb * (side - 24)
    raw = stripe * 24 + line * (side - 24)
    png = b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
    png += png_chunk(b"IDAT", zlib.compress(raw, 5)) + png_chunk(b"IEND", b"")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}

def payload(index, case, prefix_tokens):
    color = list(COLORS)[index - 1] if case == "distinct" else "red"
    variant = 20 * index if case == "distinct" else 3
    images = [image_url(color if j == 0 else tuple(COLORS)[j % 3], variant + j) for j in range(4)]
    prefix = f"Test sequence {index}. Ignore the repeated text that follows." + " stone" * prefix_tokens if prefix_tokens else ""
    content = [{"type": "text", "text": prefix + "\nInspect these four images."}, *images,
               {"type": "text", "text": "First write the dominant color in the first picture as one English word. Then list integers 1 through 60."}]
    return color, {"model": MODEL, "messages": [{"role": "user", "content": content}]}

def post(path, body, timeout):
    request = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                     headers=HEADERS)
    return urllib.request.urlopen(request, timeout=timeout)

def gauges():
    keys = ("num_requests_running", "num_requests_waiting", "generation_tokens_total", "num_preemptions_total", "kv_cache_usage_perc")
    with urllib.request.urlopen(urllib.request.Request(BASE + "/metrics", headers=HEADERS), timeout=3) as response:
        stats = response.read().decode()
    result = {}
    for line in stats.splitlines():
        m = re.match(r"^vllm:(\w+)(?:\{[^}]*\})? ([\deE+.-]+)$", line)
        if m and m[1] in keys:
            result[m[1]] = float(m[2])
    return result

def run(index, color, body, start_gate, timeout, min_output):
    body = dict(body, stream=True, stream_options={"include_usage": True}, temperature=0,
                reasoning_effort="none", min_tokens=min_output, max_tokens=min_output + 36)
    start_gate.wait()
    started = time.monotonic()
    first = None
    out = []
    usage = None
    stop = None
    error = None
    done = False
    try:
        with post("/v1/chat/completions", body, timeout=timeout) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice["delta"].get("content")
                    if delta:
                        first = first or time.monotonic()
                        out.append(delta)
                    stop = choice.get("finish_reason") or stop
    except urllib.error.HTTPError as exc:
        error = f"HTTP {exc.code}: {exc.read(350).decode(errors='replace')}"
    except (OSError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    answer = "".join(out).strip()
    words = re.findall(r"[A-Za-z]+", answer.lower())
    return {"request": index, "expected_color": color, "first_word": words[0] if words else None,
            "color_match": bool(words and words[0] == color), "usage": usage,
            "ttft_s": round(first - started, 3) if first else None, "elapsed_s": round(time.monotonic() - started, 3),
            "finish_reason": stop, "done": done, "error": error}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=("shared", "distinct"))
    parser.add_argument("--prefix-tokens", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=50)
    parser.add_argument("--min-output", type=int, default=240)
    opts = parser.parse_args()
    if not MODEL:
        parser.error("Set VLLM_MODEL to the served model name")
    prepared = [payload(i, opts.case, opts.prefix_tokens) for i in range(1, 6)]
    before = gauges()
    if before.get("num_requests_running", 0) or before.get("num_requests_waiting", 0):
        print(json.dumps({"case": opts.case, "busy": before}), flush=True)
        return 2
    with post("/tokenize", prepared[0][1], timeout=35) as response:
        sample = json.load(response)
    runs = [len(list(group)) for match, group in itertools.groupby(sample.get("tokens", []),
                                                                    key=lambda token: token == IMAGE_TOKEN) if match]
    print(json.dumps({"case": opts.case, "input_tokens": sample.get("count"), "image_token_runs": runs,
                      "before": before}), flush=True)
    start_gate = threading.Barrier(5)
    peak = 0
    peak_kv = 0
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = [pool.submit(run, i, color, body, start_gate, opts.request_timeout, opts.min_output) for i, (color, body) in enumerate(prepared, start=1)]
        stamp = 0
        while not all(job.done() for job in jobs):
            frame = gauges()
            peak = max(peak, frame.get("num_requests_running", 0))
            peak_kv = max(peak_kv, frame.get("kv_cache_usage_perc", 0))
            now = time.monotonic()
            if now - stamp >= 4:
                print(json.dumps({"case": opts.case, "sample": frame}), flush=True)
                stamp = now
            time.sleep(0.3)
        rows = [job.result() for job in jobs]
    for row in rows:
        print(json.dumps(row), flush=True)
    completed = sum(row["done"] and row["usage"] is not None for row in rows)
    print(json.dumps({"case": opts.case, "completed": completed, "peak_running": peak, "peak_kv": peak_kv,
                      "correct_colors": sum(row["color_match"] for row in rows), "after": gauges()}), flush=True)
    return 0 if completed == 5 and all(row["color_match"] for row in rows) else 1

if __name__ == "__main__":
    raise SystemExit(main())
