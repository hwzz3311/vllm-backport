# DeepSeek V4.1 Flash 1017 on 4× A800

These measurements were collected on four NVIDIA A800 PCIe 80 GB GPUs with TP4 and the server patches based on `0f5c1d6`. `measurements.json` contains sanitized request statistics, measured kernel error, and the sampled production counters behind this summary. TTFT includes queueing.

| Checkpoint and workload | Observed results |
| --- | --- |
| NVFP4, five independently submitted synthetic vision requests, 4,017 prompt tokens each | 5/5 correct, peak 5 running, TTFT 4.47–17.44 s, zero preemptions at the end |
| NVFP4, five independently submitted long synthetic vision requests, 94,031 prompt tokens each | 5/5 correct, peak 2 running, TTFT 50.48–248.42 s, zero preemptions at the end |
| W4A8, six short decode requests with 128 generated tokens each | 283.48 aggregate end-to-end output tok/s, 520/1,240 draft tokens accepted (41.94%). Six requests were submitted together. The run did not record a common six-way decode window. |
| W4A8, six cold 8,192 / 32,768-token prompts | All retrieval answers correct. Effective input throughput 3,108 / 3,923 tok/s. Peak running count 4 / 2, below the six-way acceptance threshold. Effective throughput divides total prompt tokens by the latest TTFT in each group. |
| W4A8, serving window, batch budget 3072 and DSpark K5 | 106.78 output tok/s across 125 positive-throughput samples with one request running. Aggregate log-window peak 294.2 output tok/s, maximum observed running count 2. The GPU KV pool reported 2,994,183 tokens. |

The NVFP4 vision run used batch budget 1280, `gpu_memory_utilization=0.978`, host Engram, selective ordinary weight offload, and a 512 GiB private-rank native KV prefix cache. The W4A8 short synthetic run used batch budget 1280, `gpu_memory_utilization=0.965`, and DSpark K5. The W4A8 serving observation used `gpu_memory_utilization=0.975`, host Engram, zero ordinary weight offload, FP8 MLA KV, and a 128 GiB native host prefix cache.

Real-weight checks cover W4A8 Marlin experts (relative L2 0.9585–1.0256%), the W8A8 output projection `wo_a` (maximum relative L2 0.0668%), and Engram lookup on two sampled layers. INT8 Engram and the NVFP4 checkpoint's MXFP4 Engram both matched the BF16 reference bit-for-bit in GPU and pinned-host lookup tests.

## Reproduce the synthetic workloads

Run these clients against an isolated server with the relevant checkpoint loaded. Each NVFP4 request uses four generated images. Set `VLLM_API_KEY` only if the test server uses authentication.

```bash
export VLLM_BASE_URL=http://127.0.0.1:18159
export VLLM_MODEL='your-served-nvfp4-model-name'
python accept-five-vision.py distinct
python accept-five-vision.py distinct --prefix-tokens 90000 --min-output 72 --request-timeout 450
```

```bash
export VLLM_BASE_URL=http://127.0.0.1:18182
python decode_bench.py 6
python bench.py --count 6 --length 8192 --max-tokens 64
python bench.py --count 6 --length 32768 --max-tokens 64
```

For the six-way near-512K test, use the GPU capacity reported at startup. The acceptance script requires a pool of at least 3,145,728 tokens, six concurrent generation streams, exact retrieval, and zero new preemptions.

```bash
python bench-six-512k.py --kv-pool "$GPU_KV_TOKENS"
```

The historical 2,994,183-token configuration cannot pass that capacity check. The performance effect of DSpark itself still requires a matched K0/K3/K5 run. The Marlin repack optimization needs before/after peak VRAM and load time, and the native KV patch needs cold versus host-prefix-hit TTFT with output agreement. The public backport revision was checked for patch application. Repeat model tests on the submitted branch before making upstream performance claims.
