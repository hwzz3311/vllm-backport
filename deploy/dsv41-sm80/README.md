# DeepSeek V4.1 Flash W4A8 on 4× A800

Run the W4A8 checkpoint using the existing SM80 backport image and the patched Python files in this release. The image supplies the compiled CUDA extensions. Use Linux, Docker with NVIDIA container support, and four available A800 80 GB GPUs. Engram uses about 185 GiB of host RAM and the native prefix cache uses another 128 GiB in total. Allow additional memory for startup and the OS.

## Download the checkpoint

```bash
modelscope download \
  --model baidu-aicapx/DeepSeek-V4.1-Flash-1017-W4A8-INT4-Dynamic \
  --local_dir ./DeepSeek-V4.1-Flash-1017-W4A8-INT4-Dynamic
```

The public checkpoint's `config.json` and `model.safetensors.index.json` matched the server copies. Their SHA-256 digests are `296ba735252dab5fbacb24523ac1f9ccc994f1dedd229532c2c0146e8dbb554a` and `7665d9259c273cd2909b2fa539d45cee43199b79124bdb9f21ab6836b33cbc25`.

## Start on Linux

From the extracted [RC 2 startup archive](https://github.com/hwzz3311/vllm-backport/releases/tag/dsv41-sm80-v0.1.0-rc.2), run the commands below. From a Git checkout, use `bash deploy/dsv41-sm80/launch.sh` in place of `bash launch.sh`. If upgrading, stop the old container before reusing its GPUs and port.

```bash
bash launch.sh --check
MODEL_DIR=./DeepSeek-V4.1-Flash-1017-W4A8-INT4-Dynamic \
  GPU_DEVICES=0,1,2,3 bash launch.sh --start
```

`--check` checks the Python overlay and the presence of the compiled extension without using a GPU and prints the selected vLLM arguments. `--start` runs the container in the background and writes `server.log`, `startup.log`, `argv.txt`, and `container.cid` to the printed run directory. The default endpoint is `http://127.0.0.1:18182`. Set `PORT` or `LOG_DIR` to override those defaults.

## Select a profile

| Profile | Batched tokens | GPU memory utilization | Sequence limit | Per-request context |
| --- | ---: | ---: | ---: | ---: |
| `default` | 3072 | 0.975 | 6 | 524,288 |
| `capacity` | 1280 | 0.965 | 6 | 524,288 |

Use `PREFILL_PROFILE=capacity` to select the fallback. Both profiles omit the per-request long-prefill threshold flag and retain the overall batched-token budget. DSpark K5 is the default. `DSPARK_TOKENS=3` and `DSPARK_TOKENS=0` select the existing comparison modes. Both profiles keep ordinary model weights on GPU, Engram in host memory, and a separate 128 GiB native host prefix cache.

The launcher uses `lazymio/vllm-backport:latest-sm80`. To use the same base image as the server tests, set `VLLM_IMAGE=lazymio/vllm-backport@sha256:2627d2a97e7f539b99ac98f511c589f5f520d3fc5630b195aca7f6ef865ead4e`.

## Verify the service

Once `/health` returns HTTP 200, send a small completion request.

```bash
curl -fsS http://127.0.0.1:18182/health
curl -fsS http://127.0.0.1:18182/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv41-w4a8-512k","prompt":"List three primary colors.","max_tokens":32,"temperature":0}'
```

The earlier server run reported 2,994,183 active GPU KV tokens. Six independent full 512K contexts need 3,145,728 tokens. RC 2 passed syntax and startup-argument checks. Multi-image concurrency with the updated profiles still needs a GPU run. [Server benchmark evidence](https://github.com/hwzz3311/vllm-backport/tree/a03addbc419a894de5f8991402af80ebb16f9b16/benchmarks/dsv41_sm80_evidence)
