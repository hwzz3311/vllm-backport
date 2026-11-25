# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 sparse MLA attention for SM8x (Ampere: A100/A800/A6000).

Reuses the ROCm Triton sparse-MLA implementation wholesale, exactly like the
V4 Ampere layer: its kernels, ragged metadata builders and bf16 o_proj
reference path are plain Triton/torch (the aiter-only preshuffle GEMMs and the
fused aiter norm+quant self-disable off ROCm), and
``vllm.v1.attention.ops.fp8_sm80`` supplies e4m3 encode/decode below SM89
where Triton refuses native fp8 converts.
"""

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassInt8ScaledMMLinearKernel,
)
from vllm.models.deepseek_v4_1.amd.rocm import (
    DeepseekV4ROCMAiterMLASparseBackend,
    DeepseekV41ROCMAiterMLAAttention,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fused_inverse_rope_gptj


class DeepseekV41AmpereMLASparseBackend(DeepseekV4ROCMAiterMLASparseBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV41"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class DeepseekV41AmpereMLAAttention(DeepseekV41ROCMAiterMLAAttention):
    """SM8x DeepSeek V4.1 attention: ROCm Triton path on CUDA Ampere."""

    backend_cls = DeepseekV41AmpereMLASparseBackend

    # Keep only one sequence's dequantized KV in the prefill workspace.
    # Both profiling and the actual prefill loop use this same tile size;
    # scheduler concurrency and each sequence's context limit are unchanged.
    PREFILL_CHUNK_SIZE = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The ROCm layer routes wo_a through the ordinary MXFP8 linear kernel
        # (is_bmm=False) because its bf16 einsum dequantizes the raw fp8 weight
        # itself. On CUDA that ordinary list is Marlin, whose repack would turn
        # the weight into packed garbage for the einsum. Keep the grouped-BMM
        # kernel list instead: on SM8x it resolves to the emulation kernel,
        # which dequantizes wo_a to a plain bf16 [g*r, d] weight at load time;
        # _get_cached_wo_a_bf16 then only views it (no second dequant).
        self.wo_a.is_bmm = True

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.wo_a.weight.dtype != torch.int8:
            return super()._o_proj(o, positions)

        scheme = getattr(self.wo_a, "scheme", None)
        kernel = getattr(scheme, "kernel", None)
        if not (
            isinstance(kernel, CutlassInt8ScaledMMLinearKernel)
            and kernel.config.is_channelwise
            and kernel.config.input_symmetric
            and not kernel.config.is_static_input_scheme
        ):
            raise NotImplementedError("SM80 INT8 WO_A requires dynamic CUTLASS W8A8")

        if o.shape[0] == 0:
            return self.wo_b(o.new_empty((0, self.n_local_groups * self.o_lora_rank)))
        rotated = _fused_inverse_rope_gptj(
            o, positions, self.rotary_emb.cos_sin_cache, self.rope_head_dim
        )
        grouped = rotated.view(o.shape[0], self.n_local_groups, -1)
        chunks = []
        for group, x in enumerate(grouped.unbind(dim=1)):
            start = group * self.o_lora_rank
            stop = start + self.o_lora_rank
            xq, scale, _ = ops.scaled_int8_quant(x.contiguous())
            chunks.append(
                ops.cutlass_scaled_mm(
                    xq,
                    self.wo_a.weight[:, start:stop],
                    scale_a=scale,
                    scale_b=self.wo_a.weight_scale[start:stop],
                    out_dtype=x.dtype,
                )
            )
        return self.wo_b(torch.cat(chunks, dim=-1))
