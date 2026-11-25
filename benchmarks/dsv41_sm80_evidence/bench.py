"""Measure independent cold-prefill, decode, and visual requests over the API."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import re
from secrets import token_hex
from threading import Barrier, Event, Thread
import time
from urllib.request import Request, urlopen

BASE = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:18182").rstrip("/")
PHRASES = (
    "SILVER MAPLE ORBIT", "COPPER WILLOW PULSE", "GOLDEN CEDAR VECTOR",
    "AMBER QUARTZ ANCHOR", "VIOLET JUNIPER BRIDGE", "INDIGO BIRCH SIGNAL",
)
METRICS = {
    "running": "num_requests_running",
    "waiting": "num_requests_waiting",
    "kv_usage": "kv_cache_usage_perc",
    "preemptions": "num_preemptions_total",
    "prefix_hits": "prefix_cache_hits_total",
    "drafted": "spec_decode_num_draft_tokens_total",
    "accepted": "spec_decode_num_accepted_tokens_total",
}

def emit(event, **data):
    print(json.dumps(dict(event=event, unix_time=time.time(), **data), ensure_ascii=False),
          flush=True)

def call(path, payload=None, timeout=7200):
    data = json.dumps(payload).encode() if payload is not None else None
    return urlopen(Request(BASE + path, data=data, headers={"Content-Type": "application/json"}),
                   timeout=timeout)

def metrics():
    with call("/metrics", timeout=8) as response:
        body = response.read().decode()
    result = {}
    for key, name in METRICS.items():
        values = re.findall(rf"^vllm:{name}(?:\{{[^\n]*\}})?\s+(\S+)", body, re.M)
        if values:
            result[key] = sum(map(float, values))
    return result

def tokenize(model, text):
    with call("/tokenize", dict(model=model, prompt=text, add_special_tokens=False),
              timeout=30) as response:
        return json.load(response)["tokens"]

def visual(model):
    from PIL import Image
    with io.BytesIO() as file:
        Image.new("RGB", (640, 480), color=(0, 0, 255)).save(file, "PNG")
        data = base64.b64encode(file.getvalue()).decode()
    payload = {
        "model": model, "max_tokens": 64, "temperature": 0,
        "chat_template_kwargs": {"thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}},
            {"type": "text", "text": "What is the dominant color in the image? Reply with a single color word."},
        ]}],
    }
    start = time.monotonic()
    with call("/v1/chat/completions", payload, timeout=300) as response:
        result = json.load(response)
    message = result["choices"][0]["message"]
    answer = message.get("content") or ""
    passed = "blue" in answer.lower() or "蓝" in answer
    emit("vision", passed=passed, seconds=round(time.monotonic()-start, 3),
         answer=answer, usage=result["usage"])
    if not passed:
        raise RuntimeError("Vision answer did not identify the blue image")

def text(model, length, count, max_tokens, kv_pool):
    suffix = tokenize(model, "\nWhat was the exact calibration phrase in the first record? "
                      "Reply with the phrase.\nAnswer:")
    prompts = []
    for phrase in PHRASES[:count]:
        nonce = token_hex(16)
        prefix = tokenize(model, f"{nonce}. First record: the calibration phrase is {phrase}.\n")
        filler = tokenize(model, "".join(
            f"{nonce}, station {station}: routine inspection, sample numbering logged.\n"
            for station in range(256)
        ))
        remaining = length - len(prefix) - len(suffix)
        assert remaining > 0 and filler
        prompts.append(prefix + (filler * ((remaining + len(filler) - 1) // len(filler)))[:remaining] + suffix)
    assert len({tuple(prompt[:8]) for prompt in prompts}) == count
    before = metrics()
    peak = {"running": 0, "kv_with_all_running": 0}
    complete = Event()
    barrier = Barrier(count)

    def sample():
        next_report = time.monotonic()
        while not complete.is_set():
            try:
                row = metrics()
                active = row.get("running", 0)
                peak["running"] = max(peak["running"], active)
                if active == count:
                    peak["kv_with_all_running"] = max(peak["kv_with_all_running"],
                                                     row.get("kv_usage", 0))
                if time.monotonic() >= next_report:
                    emit("progress", **row)
                    next_report = time.monotonic() + 60
            except (OSError, ValueError) as error:
                if time.monotonic() >= next_report:
                    emit("metrics_error", error=str(error))
                    next_report = time.monotonic() + 15
            complete.wait(0.5)

    def generate(index):
        barrier.wait(timeout=10)
        start = time.monotonic()
        first = last = usage = None
        parts = []
        payload = dict(model=model, prompt=prompts[index], max_tokens=max_tokens,
                       temperature=0, stream=True, stream_options={"include_usage": True})
        with call("/v1/completions", payload) as response:
            for line in response:
                if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                    continue
                event = json.loads(line[6:])
                if "error" in event:
                    raise RuntimeError(event["error"])
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    if text := choice.get("text"):
                        last = time.monotonic()
                        if first is None:
                            first = last
                            emit("first_token", request=index, ttft=round(first - start, 3))
                        parts.append(text)
        if first is None or usage is None:
            raise RuntimeError(f"Request {index} completed without streamed text or usage")
        answer = "".join(parts)
        result = dict(request=index, prompt_tokens=usage["prompt_tokens"],
                      output_tokens=usage["completion_tokens"], ttft=round(first - start, 3),
                      total_seconds=round(time.monotonic() - start, 3),
                      decode_seconds=round(last - first, 3),
                      answer_ok=PHRASES[index] in answer.upper(), answer=answer)
        emit("request_done", **result)
        return result

    emit("started", requests=count, prompt_tokens=length, gpu_pool=kv_pool, before=before)
    sampler = Thread(target=sample, daemon=True)
    sampler.start()
    try:
        with ThreadPoolExecutor(max_workers=count) as workers:
            results = list(workers.map(generate, range(count)))
    finally:
        complete.set()
        sampler.join()

    after = metrics()
    delta = {name: after[name] - baseline for name, baseline in before.items()
             if name in after and name in ("preemptions", "prefix_hits", "drafted", "accepted")}
    peak_minimum = 0.9 * count * length / kv_pool if kv_pool and length >= 500_000 else 0
    passed = (all(row["answer_ok"] and row["prompt_tokens"] == length for row in results)
              and peak["running"] == count and peak["kv_with_all_running"] >= peak_minimum
              and delta.get("preemptions", 0) == 0)
    emit("verdict", passed=passed, peak=peak, minimum_kv=peak_minimum, delta=delta,
         effective_prefill_tok_s=round(count * length / max(row["ttft"] for row in results), 2))
    if not passed:
        raise RuntimeError("Text concurrency, token accounting or exact retrieval failed")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--count", type=int, default=6, choices=(1, 4, 6))
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--kv-pool", type=int, default=0)
    args = parser.parse_args()
    with call("/v1/models", timeout=10) as response:
        info = json.load(response)["data"][0]
    model = info["id"]
    if args.vision:
        visual(model)
        return
    assert args.length + args.max_tokens <= info["max_model_len"], info
    if args.kv_pool:
        assert args.count * (args.length + args.max_tokens) <= args.kv_pool
    text(model, args.length, args.count, args.max_tokens, args.kv_pool)

if __name__ == "__main__":
    main()
