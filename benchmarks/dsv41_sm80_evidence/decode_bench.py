"""Measure the aggregate throughput of six short, concurrently decoding requests."""

from concurrent.futures import ThreadPoolExecutor
import json
import sys
from threading import Barrier
import time

from bench import call, emit, metrics

PROMPTS = [
    "Provide an accessible explanation of how database indexes work.\n",
    "Explain why city trees reduce heat and describe two limitations.\n",
    "Describe the difference between a public key and a private key.\n",
    "Explain how continental drift shapes mountains and coastlines.\n",
    "Write a clear short essay on musical rhythm and syncopation.\n",
    "Describe how an engineer measures the latency of a computer network.\n",
]

def main():
    count = int(sys.argv[1])
    with call("/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]
    before = metrics()
    ready = Barrier(count)

    def generate(index):
        payload = dict(
            model=model, prompt=PROMPTS[index], max_tokens=128,
            ignore_eos=True, temperature=0, stream=True,
            stream_options={"include_usage": True},
        )
        ready.wait(timeout=10)
        start = time.monotonic()
        first = last = usage = None
        with call("/v1/completions", payload, timeout=180) as response:
            for line in response:
                if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                    continue
                event = json.loads(line[6:])
                assert "error" not in event, event
                usage = event.get("usage") or usage
                if any(choice.get("text") for choice in event.get("choices", [])):
                    last = time.monotonic()
                    if first is None:
                        first = last
        assert usage and usage["completion_tokens"] == 128 and first is not None, usage
        result = dict(request=index, start=start, first=first, last=last,
                      ttft=round(first-start, 4), decode_tok_s=round(127/(last-first), 2))
        emit("request_done", **result)
        return result

    with ThreadPoolExecutor(max_workers=count) as workers:
        results = list(workers.map(generate, range(count)))
    after = metrics()
    drafted = after.get("drafted", 0) - before.get("drafted", 0)
    accepted = after.get("accepted", 0) - before.get("accepted", 0)
    span = max(row["last"] for row in results) - min(row["start"] for row in results)
    emit("verdict", passed=True, requests=count, output_tokens=count*128,
         e2e_output_tok_s=round(count*128/span, 2), drafted=drafted, accepted=accepted,
         acceptance_rate=round(accepted/drafted, 4) if drafted else None)

if __name__ == "__main__":
    main()
