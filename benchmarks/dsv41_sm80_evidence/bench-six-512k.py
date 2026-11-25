#!/usr/bin/env python3
"""Check six independent near-512K streams and measure overlapped generation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import re
from secrets import token_hex
from threading import Barrier, Event, Thread
import time

from bench import PHRASES, call, emit, metrics, tokenize


REQUESTS = 6
CONTEXT = 524_288
CONTEXT_MARGIN = 128
METRIC_DELTAS = ("preemptions", "prefix_hits", "drafted", "accepted")


def make_prompts(model, length):
    suffix = tokenize(
        model,
        "\nRepeat the exact calibration phrase from the first record, "
        "then explain how to audit the records in detail.\nAnswer:",
    )
    prompts = []
    for phrase in PHRASES[:REQUESTS]:
        nonce = token_hex(16)
        prefix = tokenize(
            model, f"{nonce}. First record: the calibration phrase is {phrase}.\n"
        )
        filler = tokenize(
            model,
            "".join(
                f"{nonce}, station {station}: routine inspection, sample logged.\n"
                for station in range(256)
            ),
        )
        remaining = length - len(prefix) - len(suffix)
        if remaining <= 0 or not filler:
            raise ValueError("Prompt length is too short or tokenization returned no filler")
        repetitions = (remaining + len(filler) - 1) // len(filler)
        prompts.append(prefix + (filler * repetitions)[:remaining] + suffix)
    if len({tuple(prompt[:8]) for prompt in prompts}) != REQUESTS:
        raise RuntimeError("The six prompts must have distinct first cache blocks")
    return prompts


def p95(values):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[math.ceil(0.95 * len(ordered)) - 1], 3)


def generate(model, prompt, max_tokens, index, start_barrier):
    payload = dict(
        model=model, prompt=prompt, max_tokens=max_tokens, temperature=0,
        ignore_eos=True, stream=True, return_token_ids=True,
        stream_options={"include_usage": True},
    )
    start_barrier.wait(timeout=10)
    started = time.monotonic()
    updates, parts = [], []
    token_updates = []
    usage = None
    done = False
    with call("/v1/completions", payload) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            message = line[6:].strip()
            if message == b"[DONE]":
                done = True
                break
            event = json.loads(message)
            if "error" in event:
                raise RuntimeError(f"Request {index} failed: {event['error']}")
            usage = event.get("usage") or usage
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            stamp = time.monotonic()
            if token_ids := choice.get("token_ids"):
                token_updates.append((stamp, len(token_ids)))
            if piece := choice.get("text"):
                updates.append(stamp)
                parts.append(piece)
                if len(updates) == 1:
                    emit("first_token", request=index, ttft_s=round(updates[0] - started, 3))
    if not done or not usage or len(updates) < 2:
        raise RuntimeError(f"Request {index} did not return a complete stream with usage")
    reply = re.sub(r"\s+", " ", "".join(parts)).upper()
    return dict(request=index, started=started, updates=updates, usage=usage,
                token_updates=token_updates, answer_ok=PHRASES[index] in reply)


def run(model, options):
    prompts = make_prompts(model, options.length)
    before = metrics()
    if not {"running", "preemptions"} <= before.keys():
        raise RuntimeError("Running-request or preemption metrics are unavailable")
    if before["running"] or before.get("waiting", 0):
        raise RuntimeError("The benchmark requires an idle service on port 18182")

    finished = Event()
    peak = dict(running=0, kv_with_six_running=0)

    def sample():
        next_report = 0.0
        while not finished.is_set():
            try:
                snapshot = metrics()
                running = snapshot.get("running", 0)
                peak["running"] = max(peak["running"], running)
                if running == REQUESTS:
                    peak["kv_with_six_running"] = max(
                        peak["kv_with_six_running"], snapshot.get("kv_usage", 0)
                    )
                if time.monotonic() >= next_report:
                    emit("progress", **snapshot)
                    next_report = time.monotonic() + 60
            except (OSError, ValueError) as error:
                if time.monotonic() >= next_report:
                    emit("metrics_error", error=str(error))
                    next_report = time.monotonic() + 15
            finished.wait(0.5)

    emit("started", requests=REQUESTS, prompt_tokens=options.length,
         max_tokens=options.max_tokens, gpu_kv_pool=options.kv_pool, before=before)
    sampler = Thread(target=sample, daemon=True)
    sampler.start()
    barrier = Barrier(REQUESTS)
    try:
        with ThreadPoolExecutor(max_workers=REQUESTS) as workers:
            futures = [workers.submit(generate, model, prompt, options.max_tokens, index, barrier)
                       for index, prompt in enumerate(prompts)]
            rows = [future.result() for future in futures]
    finally:
        finished.set()
        sampler.join()

    after = metrics()
    delta = {name: after[name] - before[name] for name in METRIC_DELTAS
             if name in before and name in after}
    overlap_start = max(row["updates"][0] for row in rows)
    overlap_end = min(row["updates"][-1] for row in rows)
    all_updates_in_overlap = []
    tokens_in_overlap = []
    requests_passed = True
    for row in rows:
        updates = row["updates"]
        usage = row["usage"]
        output = usage["completion_tokens"]
        updates_in_overlap = sum(overlap_start <= stamp <= overlap_end for stamp in updates)
        all_updates_in_overlap.append(updates_in_overlap)
        token_updates = row["token_updates"]
        overlap_output = None
        if sum(count for _, count in token_updates) == output:
            overlap_output = sum(count for stamp, count in token_updates
                                 if overlap_start < stamp <= overlap_end)
        tokens_in_overlap.append(overlap_output)
        request_passed = (usage["prompt_tokens"] == options.length
                          and output == options.max_tokens and row["answer_ok"]
                          and updates_in_overlap >= 2)
        requests_passed &= request_passed
        elapsed = updates[-1] - updates[0]
        emit("request_done", request=row["request"], passed=request_passed,
             prompt_tokens=usage["prompt_tokens"], output_tokens=output,
             answer_ok=row["answer_ok"], ttft_s=round(updates[0] - row["started"], 3),
             tpot_ms=round(1_000 * elapsed / (output - 1), 3) if output > 1 else None,
             stream_update_gap_p95_ms=p95([1_000 * (b - a) for a, b in zip(updates, updates[1:])]),
             stream_updates_during_overlap=updates_in_overlap,
             output_tokens_during_overlap=overlap_output)

    outputs = sum(row["usage"]["completion_tokens"] for row in rows)
    earliest_start = min(row["started"] for row in rows)
    earliest_text = min(row["updates"][0] for row in rows)
    latest_text = max(row["updates"][-1] for row in rows)
    drafted = delta.get("drafted", 0)
    accepted = delta.get("accepted")
    overlap_seconds = overlap_end - overlap_start
    overlap_tokens = None
    if all(count is not None for count in tokens_in_overlap):
        overlap_tokens = sum(tokens_in_overlap)
    e2e_seconds = latest_text - earliest_start
    decode_seconds = latest_text - earliest_text
    passed = (requests_passed and overlap_seconds > 0
              and peak["running"] == REQUESTS and delta.get("preemptions") == 0)
    emit("verdict", passed=passed, peak=peak, delta=delta, output_tokens=outputs,
         six_way_decode_overlap_s=round(max(0, overlap_seconds), 3),
         min_stream_updates_during_overlap=min(all_updates_in_overlap),
         output_tokens_during_overlap=overlap_tokens,
         aggregate_overlap_output_tok_s=(round(overlap_tokens / overlap_seconds, 2)
                                         if overlap_tokens is not None and overlap_seconds > 0
                                         else None),
         aggregate_e2e_output_tok_s=(round(outputs / e2e_seconds, 2)
                                     if e2e_seconds > 0 else None),
         aggregate_decode_output_tok_s_estimate=(round((outputs - REQUESTS) / decode_seconds, 2)
                                                 if decode_seconds > 0 else None),
         draft_acceptance_rate=round(accepted / drafted, 4) if drafted and accepted is not None else None)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=523_136)
    parser.add_argument("--max-tokens", type=int, default=1_024)
    parser.add_argument("--kv-pool", type=int, required=True)
    options = parser.parse_args()
    if options.kv_pool < REQUESTS * CONTEXT:
        parser.error(f"GPU KV capacity must be at least {REQUESTS * CONTEXT:,} tokens")
    if (options.length < 128 or options.max_tokens < 2
            or options.length + options.max_tokens > CONTEXT - CONTEXT_MARGIN):
        parser.error("Prompt and output lengths must leave at least 128 tokens of context margin")
    try:
        with call("/v1/models", timeout=10) as response:
            model = json.load(response)["data"][0]
        if model["max_model_len"] < CONTEXT:
            raise ValueError(f"The service max model length is below {CONTEXT:,} tokens")
        return 0 if run(model["id"], options) else 1
    except (OSError, RuntimeError, ValueError, KeyError, IndexError) as error:
        emit("verdict", passed=False, error=str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
