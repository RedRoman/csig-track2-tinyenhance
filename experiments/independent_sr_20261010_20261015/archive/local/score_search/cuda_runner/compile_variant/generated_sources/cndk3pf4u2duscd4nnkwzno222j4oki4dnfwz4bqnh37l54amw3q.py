# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/o5/co5bmtlggu7lrvnoujirjyeoxs3wazvwyyvhamwv4ktmiuczymub.py
# Topologically Sorted Source Nodes: [sample], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   sample => convolution
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg1_1, %arg2_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4, 'x': 262144}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1179648, 'x': 1572864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3
    xnumel = 262144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 262144*y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 3*x1), tmp0, ymask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/mc/cmcqijezxg63atamvknkxfm7cwsqn2zra4xafl5bqvicqv6nhjjn.py
# Topologically Sorted Source Nodes: [sample], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   sample => convolution
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg1_1, %arg2_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 13824, 'x': 6912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 3*x2 + 27*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/du/cduycxm3idpek7a4rv4t5wk77kk7xyq7alof4piq6awytfnx73we.py
# Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_2 = async_compile.triton('triton_per_fused_native_group_norm_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2322432, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
    xnumel = 96768
    XBLOCK: tl.constexpr = 1
    r0_numel = 347
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x1 = ((xindex // 32) % 216)
    x2 = xindex // 6912
    x0 = (xindex % 32)
    x4 = xindex
    tmp0 = r0_3 + 347*x1
    tmp1 = tl.full([1], 74899, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r0_3 + 347*x1 + 74899*x2
    tmp4 = tl.full([1], 1048576, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + (4*x0 + 128*(((r0_3 + 347*x1 + 74899*x2) % 262144)) + ((((r0_3 + 347*x1 + 74899*x2) // 262144) % 4))), r0_mask & tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (4*x0 + ((((r0_3 + 347*x1 + 74899*x2) // 262144) % 4))), r0_mask & tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
    tmp12 = tl.where(tmp6, tmp10, tmp11)
    tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
    tmp14 = tl.where(tmp2, tmp12, tmp13)
    tmp15 = 0.0
    tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
    tmp17 = tl.where(tmp6, tmp15, tmp16)
    tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
    tmp19 = tl.where(tmp2, tmp17, tmp18)
    tmp20 = 1.0
    tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
    tmp22 = tl.where(tmp6, tmp20, tmp21)
    tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
    tmp24 = tl.where(tmp2, tmp22, tmp23)
    tmp25 = tl.broadcast_to(tmp14, [R0_BLOCK])
    tmp26 = tl.broadcast_to(tmp19, [R0_BLOCK])
    tmp27 = tl.broadcast_to(tmp24, [R0_BLOCK])
    tmp29 = tl.where(r0_mask, tmp25, 0)
    tmp30 = tl.where(r0_mask, tmp26, 0)
    tmp31 = tl.where(r0_mask, tmp27, 0)
    tmp32, tmp33, tmp34 = triton_helpers.welford(tmp29, tmp30, tmp31, 0)
    tmp35 = triton_helpers.promote_to_tensor(tmp32)
    tmp36 = triton_helpers.promote_to_tensor(tmp33)
    tmp37 = triton_helpers.promote_to_tensor(tmp34)
    tl.store(out_ptr0 + (x4), tmp35, None)
    tl.store(out_ptr1 + (x4), tmp36, None)
    tl.store(out_ptr2 + (x4), tmp37, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/d4/cd4hdmqctvy2ahlvlumxx477z6hpxjnoicwhf4lnd32ct6bfin4w.py
# Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_3 = async_compile.triton('triton_per_fused_native_group_norm_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1182720, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 896
    r0_numel = 108
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 32)
    x1 = xindex // 32
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*r0_2 + 3456*x1), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 32*r0_2 + 3456*x1), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 32*r0_2 + 3456*x1), r0_mask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp8 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp9 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/kf/ckfoqyl34fs5snu5yhvmem7v6udlu4mlbolytcxlphzkq6mo363l.py
# Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_4 = async_compile.triton('triton_per_fused_native_group_norm_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 21504, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 448
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 32)
    x1 = xindex // 32
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*r0_2 + 64*x1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 32*r0_2 + 64*x1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 32*r0_2 + 64*x1), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/gx/cgxlgze6diql3cqpk6z7i6jsweap4ny6qmila2hygqk42mu4bfuz.py
# Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_5 = async_compile.triton('triton_per_fused_native_group_norm_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5888, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 14
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 32*r0_1), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 32*r0_1), r0_mask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp8 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp9 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/n5/cn5gomr4pkbm2n6dvatf5mhvo4p4rubxf5eiucemwivei3rwjz3a.py
# Topologically Sorted Source Nodes: [hidden_states, hidden_states_1], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states => add_1, mul_1
#   hidden_states_1 => convert_element_type_5, mul_2, sigmoid
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_2), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_1,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %sigmoid), kwargs = {})
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_6 = async_compile.triton('triton_poi_fused_native_group_norm_silu_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 201327360}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x0 // 4), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0 // 4), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 1048576.0
    tmp8 = (tmp6 / tmp7)
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = tl.sigmoid(tmp18)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/cq/ccqkd4tcnai62yunth4tlj6d2lmd3564kopedp23lbhmzaz2q4kl.py
# Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_1 => convert_element_type_5, mul_2, sigmoid
#   hidden_states_2 => convolution_1
# Graph fragment:
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_1,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %sigmoid), kwargs = {})
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_5, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_7 = async_compile.triton('triton_poi_fused_convolution_silu_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 589824, 'x': 294912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/ts/ctso4i57e4ofxtccguo3kswntj4hrm4fecvx47eiffd7llolcg4b.py
# Topologically Sorted Source Nodes: [hidden_states_7], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_7 => var_mean_2
# Graph fragment:
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_4, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_8 = async_compile.triton('triton_per_fused_native_group_norm_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2322432, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
    xnumel = 96768
    XBLOCK: tl.constexpr = 1
    r0_numel = 347
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x1 = ((xindex // 32) % 216)
    x2 = xindex // 6912
    x0 = (xindex % 32)
    x4 = xindex
    tmp0 = r0_3 + 347*x1
    tmp1 = tl.full([1], 74899, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r0_3 + 347*x1 + 74899*x2
    tmp4 = tl.full([1], 1048576, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + (4*x0 + 128*(((r0_3 + 347*x1 + 74899*x2) % 262144)) + ((((r0_3 + 347*x1 + 74899*x2) // 262144) % 4))), r0_mask & tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (4*x0 + ((((r0_3 + 347*x1 + 74899*x2) // 262144) % 4))), r0_mask & tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr2 + (4*x0 + 128*(((r0_3 + 347*x1 + 74899*x2) % 262144)) + ((((r0_3 + 347*x1 + 74899*x2) // 262144) % 4))), r0_mask & tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (4*x0 + ((((r0_3 + 347*x1 + 74899*x2) // 262144) % 4))), r0_mask & tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.full(tmp16.shape, 0, tmp16.dtype)
    tmp18 = tl.where(tmp6, tmp16, tmp17)
    tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
    tmp20 = tl.where(tmp2, tmp18, tmp19)
    tmp21 = 0.0
    tmp22 = tl.full(tmp21.shape, 0, tmp21.dtype)
    tmp23 = tl.where(tmp6, tmp21, tmp22)
    tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
    tmp25 = tl.where(tmp2, tmp23, tmp24)
    tmp26 = tl.full(tmp14.shape, 0, tmp14.dtype)
    tmp27 = tl.where(tmp6, tmp14, tmp26)
    tmp28 = tl.full(tmp27.shape, 0, tmp27.dtype)
    tmp29 = tl.where(tmp2, tmp27, tmp28)
    tmp30 = tl.broadcast_to(tmp20, [R0_BLOCK])
    tmp31 = tl.broadcast_to(tmp25, [R0_BLOCK])
    tmp32 = tl.broadcast_to(tmp29, [R0_BLOCK])
    tmp34 = tl.where(r0_mask, tmp30, 0)
    tmp35 = tl.where(r0_mask, tmp31, 0)
    tmp36 = tl.where(r0_mask, tmp32, 0)
    tmp37, tmp38, tmp39 = triton_helpers.welford(tmp34, tmp35, tmp36, 0)
    tmp40 = triton_helpers.promote_to_tensor(tmp37)
    tmp41 = triton_helpers.promote_to_tensor(tmp38)
    tmp42 = triton_helpers.promote_to_tensor(tmp39)
    tl.store(out_ptr0 + (x4), tmp40, None)
    tl.store(out_ptr1 + (x4), tmp41, None)
    tl.store(out_ptr2 + (x4), tmp42, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/hj/chji663ikrrzytiuii4qaf66wwnauosfjdqdubszw6cglirnjilh.py
# Topologically Sorted Source Nodes: [hidden_states_7, hidden_states_8], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_7 => add_6, mul_7
#   hidden_states_8 => convert_element_type_17, mul_8, sigmoid_2
# Graph fragment:
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %unsqueeze_14), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_17), kwargs = {})
#   %sigmoid_2 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_6,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, %sigmoid_2), kwargs = {})
#   %convert_element_type_17 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_8, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_9 = async_compile.triton('triton_poi_fused_native_group_norm_silu_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 268436480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0 // 4), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0 // 4), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 1048576.0
    tmp14 = (tmp12 / tmp13)
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp11 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 + tmp23
    tmp25 = tl.sigmoid(tmp24)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp27, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/kj/ckj4yvuoenyvfzhwvnsa3bxajupme6ja2ffcvy6jel6sn7krx7ym.py
# Topologically Sorted Source Nodes: [sample, hidden_states_4, hidden_states_6, add, output_tensor, hidden_states_11, hidden_states_13, add_1, output_tensor_1, hidden_states_14], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   add => add_4
#   add_1 => add_9
#   hidden_states_11 => convert_element_type_23, mul_11, sigmoid_3
#   hidden_states_13 => convolution_4
#   hidden_states_14 => constant_pad_nd
#   hidden_states_4 => convert_element_type_11, mul_5, sigmoid_1
#   hidden_states_6 => convolution_2
#   output_tensor => div
#   output_tensor_1 => div_1
#   sample => convolution
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg1_1, %arg2_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_1 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %sigmoid_1), kwargs = {})
#   %convert_element_type_11 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5, torch.bfloat16), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_11, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %convolution_2), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_4, 1.0), kwargs = {})
#   %sigmoid_3 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_8,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %sigmoid_3), kwargs = {})
#   %convert_element_type_23 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_11, torch.bfloat16), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_23, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, %convolution_4), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_9, 1.0), kwargs = {})
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%div_1, [0, 1, 0, 1], 0.0), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_div_silu_10 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_div_silu_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_div_silu_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 336857088}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_div_silu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33685632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 65664
    x1 = ((xindex // 128) % 513)
    x3 = (xindex % 65664)
    x0 = (xindex % 128)
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 512, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x3 + 65536*x2), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.load(in_ptr2 + (x3 + 65536*x2), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tl.load(in_ptr4 + (x3 + 65536*x2), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = tmp18 * tmp13
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp5, tmp19, tmp20)
    tl.store(out_ptr0 + (x4), tmp21, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/3y/c3yiflyctnlu3oyatn2s6ogc36ssdn4y5ysybqvekf5gpg4xuyi5.py
# Topologically Sorted Source Nodes: [sample, hidden_states_4, hidden_states_6, add, output_tensor, hidden_states_11, hidden_states_13, add_1, output_tensor_1, hidden_states_14, hidden_states_15], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   add => add_4
#   add_1 => add_9
#   hidden_states_11 => convert_element_type_23, mul_11, sigmoid_3
#   hidden_states_13 => convolution_4
#   hidden_states_14 => constant_pad_nd
#   hidden_states_15 => convolution_5
#   hidden_states_4 => convert_element_type_11, mul_5, sigmoid_1
#   hidden_states_6 => convolution_2
#   output_tensor => div
#   output_tensor_1 => div_1
#   sample => convolution
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg1_1, %arg2_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_1 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %sigmoid_1), kwargs = {})
#   %convert_element_type_11 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5, torch.bfloat16), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_11, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %convolution_2), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_4, 1.0), kwargs = {})
#   %sigmoid_3 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_8,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %sigmoid_3), kwargs = {})
#   %convert_element_type_23 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_11, torch.bfloat16), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_23, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, %convolution_4), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_9, 1.0), kwargs = {})
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%div_1, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_5 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd, %arg19_1, %arg20_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_div_silu_11 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_div_silu_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_div_silu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50331904}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_div_silu_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/bo/cboa442z2eumwhocqr32zfawkws66d6sdjv5zvosubbibjcay2lb.py
# Topologically Sorted Source Nodes: [hidden_states_16], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_16 => var_mean_4
# Graph fragment:
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_8, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_12 = async_compile.triton('triton_per_fused_native_group_norm_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 5, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1572864, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_12(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 128)
    x2 = xindex // 4096
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0 + 128*(((r0_3 + 128*x1 + 16384*x2) % 65536)) + ((r0_3 + 128*x1 + 16384*x2) // 65536)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None]
    tmp7 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = (tmp6 / tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tmp10 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.sum(tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp9, None)
    tl.store(out_ptr1 + (x4), tmp14, None)
    tl.store(out_ptr2 + (x4), tmp8, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/ub/cub6e2f5mwg5a7h2dbp2ghtpithwudkbnurhoqj7ssiutl7yc5aa.py
# Topologically Sorted Source Nodes: [hidden_states_16], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_16 => var_mean_4
# Graph fragment:
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_8, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_13 = async_compile.triton('triton_per_fused_native_group_norm_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 798720, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_13(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 32)
    x1 = xindex // 32
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*r0_2 + 4096*x1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 32*r0_2 + 4096*x1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 32*r0_2 + 4096*x1), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/j3/cj3vebtfw37edva7yttc7ca2i5snhi3k6otbikii3aqeqrerkdnd.py
# Topologically Sorted Source Nodes: [hidden_states_16], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_16 => var_mean_4
# Graph fragment:
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_8, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_14 = async_compile.triton('triton_per_fused_native_group_norm_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6656, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_14(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*r0_1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 32*r0_1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 32*r0_1), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/qp/cqpll2di3pfbpzfnifbroqhewntk3m5phopymytfh5kpwafiy5f4.py
# Topologically Sorted Source Nodes: [hidden_states_16, hidden_states_17], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_16 => add_11, mul_13
#   hidden_states_17 => convert_element_type_29, mul_14, sigmoid_4
# Graph fragment:
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_9, %unsqueeze_26), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %unsqueeze_29), kwargs = {})
#   %sigmoid_4 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_11,), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %sigmoid_4), kwargs = {})
#   %convert_element_type_29 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_14, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_15 = async_compile.triton('triton_poi_fused_native_group_norm_silu_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50332160}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0 // 4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0 // 4), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 262144.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/dq/cdq4os4qlvhashrbp7zy5y27pq5obrbmt75sjrylnu353pem5kgz.py
# Topologically Sorted Source Nodes: [hidden_states_17, hidden_states_18], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_17 => convert_element_type_29, mul_14, sigmoid_4
#   hidden_states_18 => convolution_6
# Graph fragment:
#   %sigmoid_4 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_11,), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %sigmoid_4), kwargs = {})
#   %convert_element_type_29 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_14, torch.bfloat16), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_29, %arg23_1, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_16 = async_compile.triton('triton_poi_fused_convolution_silu_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1179648, 'x': 589824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/ze/czeqqa2hhleszatib3nqhswx3zku7ha7qrodlkrr5kjgrtybcp4r.py
# Topologically Sorted Source Nodes: [hidden_states_19], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_19 => var_mean_5
# Graph fragment:
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_10, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_17 = async_compile.triton('triton_per_fused_native_group_norm_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 5, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_17(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 256)
    x2 = xindex // 8192
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (8*x0 + 256*(((r0_3 + 128*x1 + 32768*x2) % 65536)) + ((r0_3 + 128*x1 + 32768*x2) // 65536)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (8*x0 + ((r0_3 + 128*x1 + 32768*x2) // 65536)), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp8 = tl.sum(tmp6, 1)[:, None]
    tmp9 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = (tmp8 / tmp10)
    tmp12 = tmp4 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp11, None)
    tl.store(out_ptr1 + (x4), tmp16, None)
    tl.store(out_ptr2 + (x4), tmp10, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/ew/cewl4fz5v232xl4ql7gf3f24ilcatnagmjc4i6v7hebnjyh2x6f4.py
# Topologically Sorted Source Nodes: [hidden_states_19], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_19 => var_mean_5
# Graph fragment:
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_10, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_18 = async_compile.triton('triton_per_fused_native_group_norm_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1597440, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_18(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 32)
    x1 = xindex // 32
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*r0_2 + 4096*x1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 32*r0_2 + 4096*x1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 32*r0_2 + 4096*x1), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/5o/c5oj2mlcj7zbw6pvqqalmbohki4tshea74ormhi5pzzwo2gez5px.py
# Topologically Sorted Source Nodes: [hidden_states_19], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_19 => var_mean_5
# Graph fragment:
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_10, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_19 = async_compile.triton('triton_per_fused_native_group_norm_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24576, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_19(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 32)
    x1 = xindex // 32
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*r0_2 + 64*x1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 32*r0_2 + 64*x1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 32*r0_2 + 64*x1), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/5e/c5epbbera5bxt5uxuuo4c7msvd4hhy2g5aqavxqhpxxo263gm7y2.py
# Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_19 => add_13, mul_16
#   hidden_states_20 => convert_element_type_35, mul_17, sigmoid_5
# Graph fragment:
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_11, %unsqueeze_32), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %unsqueeze_35), kwargs = {})
#   %sigmoid_5 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_13,), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_13, %sigmoid_5), kwargs = {})
#   %convert_element_type_35 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_17, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_20 = async_compile.triton('triton_poi_fused_native_group_norm_silu_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 100664832}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x0 // 8), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0 // 8), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 524288.0
    tmp8 = (tmp6 / tmp7)
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = tl.sigmoid(tmp18)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/oj/cojh2k5ypdu45ezx55rmfz7pqzmgwmsefyvo4jikgfln4my2i3o2.py
# Topologically Sorted Source Nodes: [hidden_states_20, hidden_states_22], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_20 => convert_element_type_35, mul_17, sigmoid_5
#   hidden_states_22 => convolution_7
# Graph fragment:
#   %sigmoid_5 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_13,), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_13, %sigmoid_5), kwargs = {})
#   %convert_element_type_35 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_17, torch.bfloat16), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_35, %arg27_1, %arg28_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_21 = async_compile.triton('triton_poi_fused_convolution_silu_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 2359296, 'x': 1179648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_21(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/qp/cqpwccsytu5ytu6i2pcva4jdxjuyyxnzi5uddglofakjh5lz4347.py
# Topologically Sorted Source Nodes: [hidden_states_23], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_23 => var_mean_6
# Graph fragment:
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_12, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_22 = async_compile.triton('triton_per_fused_native_group_norm_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 5, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 256)
    x2 = xindex // 8192
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (8*x0 + 256*(((r0_3 + 128*x1 + 32768*x2) % 65536)) + ((r0_3 + 128*x1 + 32768*x2) // 65536)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (8*x0 + ((r0_3 + 128*x1 + 32768*x2) // 65536)), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (8*x0 + 256*(((r0_3 + 128*x1 + 32768*x2) % 65536)) + ((r0_3 + 128*x1 + 32768*x2) // 65536)), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (8*x0 + ((r0_3 + 128*x1 + 32768*x2) // 65536)), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp14 = tl.sum(tmp12, 1)[:, None]
    tmp15 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = (tmp14 / tmp16)
    tmp18 = tmp10 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.sum(tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp17, None)
    tl.store(out_ptr1 + (x4), tmp22, None)
    tl.store(out_ptr2 + (x4), tmp16, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/62/c62h5qhfp5qa2vtizjehtoxcvizj2y2vq23cibfbbmhvbmvbjwdd.py
# Topologically Sorted Source Nodes: [hidden_states_23, hidden_states_24], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_23 => add_16, mul_19
#   hidden_states_24 => convert_element_type_41, mul_20, sigmoid_6
# Graph fragment:
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, %unsqueeze_38), kwargs = {})
#   %add_16 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %unsqueeze_41), kwargs = {})
#   %sigmoid_6 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_16,), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_16, %sigmoid_6), kwargs = {})
#   %convert_element_type_41 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_20, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_23 = async_compile.triton('triton_poi_fused_native_group_norm_silu_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 134219776}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0 // 8), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0 // 8), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 524288.0
    tmp14 = (tmp12 / tmp13)
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp11 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 + tmp23
    tmp25 = tl.sigmoid(tmp24)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp27, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/pe/cpeaxlh46zj6jxjvabfafxevrbnapmlhlqqavexz3gsfu7c4evay.py
# Topologically Sorted Source Nodes: [input_tensor, hidden_states_20, hidden_states_22, add_2, output_tensor_2, hidden_states_27, hidden_states_29, add_3, output_tensor_3, hidden_states_30], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   add_2 => add_14
#   add_3 => add_19
#   hidden_states_20 => convert_element_type_35, mul_17, sigmoid_5
#   hidden_states_22 => convolution_7
#   hidden_states_27 => convert_element_type_47, mul_23, sigmoid_7
#   hidden_states_29 => convolution_10
#   hidden_states_30 => constant_pad_nd_1
#   input_tensor => convolution_8
#   output_tensor_2 => div_2
#   output_tensor_3 => div_3
# Graph fragment:
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_5, %arg29_1, %arg30_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_5 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_13,), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_13, %sigmoid_5), kwargs = {})
#   %convert_element_type_35 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_17, torch.bfloat16), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_35, %arg27_1, %arg28_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_8, %convolution_7), kwargs = {})
#   %div_2 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_14, 1.0), kwargs = {})
#   %sigmoid_7 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_18,), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_18, %sigmoid_7), kwargs = {})
#   %convert_element_type_47 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_23, torch.bfloat16), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_47, %arg37_1, %arg38_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_2, %convolution_10), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_19, 1.0), kwargs = {})
#   %constant_pad_nd_1 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%div_3, [0, 1, 0, 1], 0.0), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_div_silu_24 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_div_silu_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_div_silu_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 169086976}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_div_silu_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16908544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 65792
    x1 = ((xindex // 256) % 257)
    x3 = (xindex % 65792)
    x0 = (xindex % 256)
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 256, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x3 + 65536*x2), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.load(in_ptr2 + (x3 + 65536*x2), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tl.load(in_ptr4 + (x3 + 65536*x2), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = tmp18 * tmp13
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp5, tmp19, tmp20)
    tl.store(out_ptr0 + (x4), tmp21, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/ds/cdsx7xgjp7hysr24b5yllao5igptvnnuz6rcktoa6sqrabaaqrpk.py
# Topologically Sorted Source Nodes: [input_tensor, hidden_states_20, hidden_states_22, add_2, output_tensor_2, hidden_states_27, hidden_states_29, add_3, output_tensor_3, hidden_states_30, hidden_states_31], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   add_2 => add_14
#   add_3 => add_19
#   hidden_states_20 => convert_element_type_35, mul_17, sigmoid_5
#   hidden_states_22 => convolution_7
#   hidden_states_27 => convert_element_type_47, mul_23, sigmoid_7
#   hidden_states_29 => convolution_10
#   hidden_states_30 => constant_pad_nd_1
#   hidden_states_31 => convolution_11
#   input_tensor => convolution_8
#   output_tensor_2 => div_2
#   output_tensor_3 => div_3
# Graph fragment:
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_5, %arg29_1, %arg30_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_5 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_13,), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_13, %sigmoid_5), kwargs = {})
#   %convert_element_type_35 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_17, torch.bfloat16), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_35, %arg27_1, %arg28_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_8, %convolution_7), kwargs = {})
#   %div_2 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_14, 1.0), kwargs = {})
#   %sigmoid_7 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_18,), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_18, %sigmoid_7), kwargs = {})
#   %convert_element_type_47 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_23, torch.bfloat16), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_47, %arg37_1, %arg38_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_2, %convolution_10), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_19, 1.0), kwargs = {})
#   %constant_pad_nd_1 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%div_3, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_11 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_1, %arg39_1, %arg40_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_div_silu_25 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_div_silu_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_div_silu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25166336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_div_silu_25(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/fw/cfww7njem3zjhtmafph4s35h3tiwfa4brtd7vvb63h46o6hvhq4c.py
# Topologically Sorted Source Nodes: [hidden_states_32], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_32 => var_mean_8
# Graph fragment:
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_16, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_26 = async_compile.triton('triton_per_fused_native_group_norm_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 5, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 786432, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_26(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 64)
    x2 = xindex // 2048
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (8*x0 + 256*(((r0_3 + 128*x1 + 8192*x2) % 16384)) + ((r0_3 + 128*x1 + 8192*x2) // 16384)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None]
    tmp7 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = (tmp6 / tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tmp10 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.sum(tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp9, None)
    tl.store(out_ptr1 + (x4), tmp14, None)
    tl.store(out_ptr2 + (x4), tmp8, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/zh/czhur4u2q4opowfo4kpvvkxhjistlhp637gwgdpk6zwlyceanatp.py
# Topologically Sorted Source Nodes: [hidden_states_32], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_32 => var_mean_8
# Graph fragment:
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_16, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_27 = async_compile.triton('triton_per_fused_native_group_norm_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 405504, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_27(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 32)
    x1 = xindex // 32
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*r0_2 + 2048*x1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 32*r0_2 + 2048*x1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 32*r0_2 + 2048*x1), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/4n/c4nduril5ttbvlnvyhabyfgrsicj5r6wrkrezl5wpx24l7buqkpr.py
# Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_32 => add_21, mul_25
#   hidden_states_33 => convert_element_type_53, mul_26, sigmoid_8
# Graph fragment:
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, %unsqueeze_50), kwargs = {})
#   %add_21 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_25, %unsqueeze_53), kwargs = {})
#   %sigmoid_8 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_21,), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %sigmoid_8), kwargs = {})
#   %convert_element_type_53 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_26, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_28 = async_compile.triton('triton_poi_fused_native_group_norm_silu_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25166848}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0 // 8), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0 // 8), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 131072.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/h6/ch6miouo2psm2dnxv2rdbdsoae6ekfjy7citdkctk3ejr44v4ern.py
# Topologically Sorted Source Nodes: [hidden_states_33, hidden_states_34], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_33 => convert_element_type_53, mul_26, sigmoid_8
#   hidden_states_34 => convolution_12
# Graph fragment:
#   %sigmoid_8 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_21,), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %sigmoid_8), kwargs = {})
#   %convert_element_type_53 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_26, torch.bfloat16), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_53, %arg43_1, %arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_29 = async_compile.triton('triton_poi_fused_convolution_silu_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 4718592, 'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_29(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/rk/crk3tkeostqf57cotekousdzisq4c4o7yoyss33noe7dsknjdarv.py
# Topologically Sorted Source Nodes: [hidden_states_35], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_35 => var_mean_9
# Graph fragment:
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_18, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_30 = async_compile.triton('triton_per_fused_native_group_norm_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 5, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 35127296, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_30(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 512)
    x1 = xindex // 512
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 65536*x1 + ((r0_2 + 128*x1) // 16384)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + ((r0_2 + 128*x1) // 16384)), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp8 = tl.sum(tmp6, 1)[:, None]
    tmp9 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = (tmp8 / tmp10)
    tmp12 = tmp4 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
    tl.store(out_ptr2 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/qj/cqjdw2lh5ds4upaahv2xjyrqowpri6qv4c6bkkew2m7fghhd5db5.py
# Topologically Sorted Source Nodes: [hidden_states_35], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_35 => var_mean_9
# Graph fragment:
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_18, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_31 = async_compile.triton('triton_per_fused_native_group_norm_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 798720, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_31(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
    tl.store(out_ptr2 + (x0), tmp15, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/fe/cfe3yekt4v6gknt2dyc3ms5xqpnt3hlt2j27k6cmordt4wkq632c.py
# Topologically Sorted Source Nodes: [hidden_states_35], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_35 => var_mean_9
# Graph fragment:
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_18, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_32 = async_compile.triton('triton_per_fused_native_group_norm_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 512, 'r0_': 6144}}
)
@triton.jit
def triton_per_fused_native_group_norm_32(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 16*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 16*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/xw/cxwtof6mhmcnldmh7dnxqz7xsg3wz5edsjmiqxkum7od32vxjxjs.py
# Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_36], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_35 => add_23, mul_28
#   hidden_states_36 => convert_element_type_59, mul_29, sigmoid_9
# Graph fragment:
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, %unsqueeze_56), kwargs = {})
#   %add_23 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_28, %unsqueeze_59), kwargs = {})
#   %sigmoid_9 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_23,), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_23, %sigmoid_9), kwargs = {})
#   %convert_element_type_59 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_29, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_33 = async_compile.triton('triton_poi_fused_native_group_norm_silu_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50334720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x0 // 16), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0 // 16), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 262144.0
    tmp8 = (tmp6 / tmp7)
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = tl.sigmoid(tmp18)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/f3/cf3oclrbyjll3mplb3cxs2ux4qmllzqbpu4vq64vqdhdgb23pwf3.py
# Topologically Sorted Source Nodes: [hidden_states_36, hidden_states_38], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_36 => convert_element_type_59, mul_29, sigmoid_9
#   hidden_states_38 => convolution_13
# Graph fragment:
#   %sigmoid_9 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_23,), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_23, %sigmoid_9), kwargs = {})
#   %convert_element_type_59 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_29, torch.bfloat16), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_59, %arg47_1, %arg48_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_34 = async_compile.triton('triton_poi_fused_convolution_silu_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 9437184, 'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_34(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/om/compa2jo4h5gxx4qp3nrhfhwmidl75dgdyyoagvmr2fsx5xipjhu.py
# Topologically Sorted Source Nodes: [hidden_states_39], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_39 => var_mean_10
# Graph fragment:
#   %var_mean_10 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_20, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_35 = async_compile.triton('triton_per_fused_native_group_norm_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 5, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 68681728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 512)
    x1 = xindex // 512
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 65536*x1 + ((r0_2 + 128*x1) // 16384)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + ((r0_2 + 128*x1) // 16384)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + 512*r0_2 + 65536*x1 + ((r0_2 + 128*x1) // 16384)), None).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x0 + ((r0_2 + 128*x1) // 16384)), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp14 = tl.sum(tmp12, 1)[:, None]
    tmp15 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = (tmp14 / tmp16)
    tmp18 = tmp10 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.sum(tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, None)
    tl.store(out_ptr1 + (x3), tmp22, None)
    tl.store(out_ptr2 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/7u/c7uegj5kwidlh4qo5ecyfk3ajjxkmmnzum2ydr2lwjkjckgq7k2a.py
# Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_39 => add_26, mul_31
#   hidden_states_40 => convert_element_type_65, mul_32, sigmoid_10
# Graph fragment:
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_21, %unsqueeze_62), kwargs = {})
#   %add_26 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %unsqueeze_65), kwargs = {})
#   %sigmoid_10 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_26,), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_26, %sigmoid_10), kwargs = {})
#   %convert_element_type_65 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_32, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_36 = async_compile.triton('triton_poi_fused_native_group_norm_silu_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 67112960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0 // 16), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0 // 16), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 262144.0
    tmp14 = (tmp12 / tmp13)
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp11 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 + tmp23
    tmp25 = tl.sigmoid(tmp24)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp27, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/2q/c2qvqowplndfrq3vastzewv7mo742emx3sxfgyfbw4gkbfe6gick.py
# Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_36, hidden_states_38, add_4, output_tensor_4, hidden_states_43, hidden_states_45, add_5, output_tensor_5, hidden_states_46], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   add_4 => add_24
#   add_5 => add_29
#   hidden_states_36 => convert_element_type_59, mul_29, sigmoid_9
#   hidden_states_38 => convolution_13
#   hidden_states_43 => convert_element_type_71, mul_35, sigmoid_11
#   hidden_states_45 => convolution_16
#   hidden_states_46 => constant_pad_nd_2
#   input_tensor_1 => convolution_14
#   output_tensor_4 => div_4
#   output_tensor_5 => div_5
# Graph fragment:
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_11, %arg49_1, %arg50_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_9 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_23,), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_23, %sigmoid_9), kwargs = {})
#   %convert_element_type_59 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_29, torch.bfloat16), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_59, %arg47_1, %arg48_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %convolution_13), kwargs = {})
#   %div_4 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_24, 1.0), kwargs = {})
#   %sigmoid_11 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_28,), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_28, %sigmoid_11), kwargs = {})
#   %convert_element_type_71 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_35, torch.bfloat16), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_71, %arg57_1, %arg58_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_4, %convolution_16), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_29, 1.0), kwargs = {})
#   %constant_pad_nd_2 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%div_5, [0, 1, 0, 1], 0.0), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_div_silu_37 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_div_silu_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_div_silu_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 85204992}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_div_silu_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8520192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 66048
    x1 = ((xindex // 512) % 129)
    x3 = (xindex % 66048)
    x0 = (xindex % 512)
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 128, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x3 + 65536*x2), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.load(in_ptr2 + (x3 + 65536*x2), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tl.load(in_ptr4 + (x3 + 65536*x2), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = tmp18 * tmp13
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp5, tmp19, tmp20)
    tl.store(out_ptr0 + (x4), tmp21, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/yd/cyd2bcwlg4d2ilnmtcau5qb2q4vmipeq37pj3ckhjfbihweciejf.py
# Topologically Sorted Source Nodes: [hidden_states_48], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_48 => var_mean_12
# Graph fragment:
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_24, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_38 = async_compile.triton('triton_per_fused_native_group_norm_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 5, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 393216, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_38(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 256)
    x1 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 512*(((r0_2 + 128*x1) % 4096)) + ((r0_2 + 128*x1) // 4096)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (2*x0 + ((r0_2 + 128*x1) // 4096)), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp8 = tl.sum(tmp6, 1)[:, None]
    tmp9 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = (tmp8 / tmp10)
    tmp12 = tmp4 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
    tl.store(out_ptr2 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/d2/cd25awexafurwmooti7zouvrrhtaztc6w2m5crb34434ck2hh647.py
# Topologically Sorted Source Nodes: [hidden_states_48], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_48 => var_mean_12
# Graph fragment:
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_24, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_39 = async_compile.triton('triton_per_fused_native_group_norm_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 202752, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_39(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 256*r0_1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 256*r0_1), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
    tl.store(out_ptr2 + (x0), tmp15, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/nz/cnzdtrbi2hkbbfbgtoqsqrqea65rkx3uzdatfh4ges4veuqmowek.py
# Topologically Sorted Source Nodes: [hidden_states_48], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_48 => var_mean_12
# Graph fragment:
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_24, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_40 = async_compile.triton('triton_per_fused_native_group_norm_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 512, 'r0_': 3072}}
)
@triton.jit
def triton_per_fused_native_group_norm_40(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 8*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 8*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 8*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/3w/c3wvvzn77af2b4pgitzcehclb6jw6tb3orz5y3wmegqagxlxypf2.py
# Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_48 => add_31, mul_37
#   hidden_states_49 => convert_element_type_77, mul_38, sigmoid_12
# Graph fragment:
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_25, %unsqueeze_74), kwargs = {})
#   %add_31 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_37, %unsqueeze_77), kwargs = {})
#   %sigmoid_12 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_31,), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_31, %sigmoid_12), kwargs = {})
#   %convert_element_type_77 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_38, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_41 = async_compile.triton('triton_poi_fused_native_group_norm_silu_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12585984}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x0 // 16), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0 // 16), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 65536.0
    tmp8 = (tmp6 / tmp7)
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = tl.sigmoid(tmp18)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/ny/cny5znxpbpgg3shi3up7ojjqzg2gwchpz3oepwo34xwnvpptawkp.py
# Topologically Sorted Source Nodes: [hidden_states_55], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_55 => var_mean_14
# Graph fragment:
#   %var_mean_14 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_28, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_42 = async_compile.triton('triton_per_fused_native_group_norm_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 5, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 393216, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 256)
    x1 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 512*(((r0_2 + 128*x1) % 4096)) + ((r0_2 + 128*x1) // 4096)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (2*x0 + ((r0_2 + 128*x1) // 4096)), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (2*x0 + 512*(((r0_2 + 128*x1) % 4096)) + ((r0_2 + 128*x1) // 4096)), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (2*x0 + ((r0_2 + 128*x1) // 4096)), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp14 = tl.sum(tmp12, 1)[:, None]
    tmp15 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = (tmp14 / tmp16)
    tmp18 = tmp10 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.sum(tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, None)
    tl.store(out_ptr1 + (x3), tmp22, None)
    tl.store(out_ptr2 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/bd/cbdjyzhhzcqd6oqfvqutc66btmpdf3qins63zj6eknbytdkykkze.py
# Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_55 => add_36, mul_43
#   hidden_states_56 => convert_element_type_89, mul_44, sigmoid_14
# Graph fragment:
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_29, %unsqueeze_86), kwargs = {})
#   %add_36 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %unsqueeze_89), kwargs = {})
#   %sigmoid_14 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_36,), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_36, %sigmoid_14), kwargs = {})
#   %convert_element_type_89 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_44, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_43 = async_compile.triton('triton_poi_fused_native_group_norm_silu_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16781312}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0 // 16), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0 // 16), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 65536.0
    tmp14 = (tmp12 / tmp13)
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp11 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 + tmp23
    tmp25 = tl.sigmoid(tmp24)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp27, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/yp/cypeps7g4qghbon2a7wzpcemefpr3dmeaibl3ibdnowns67kfbwi.py
# Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_36, hidden_states_38, add_4, output_tensor_4, hidden_states_43, hidden_states_45, add_5, output_tensor_5, hidden_states_46, hidden_states_47, hidden_states_52, hidden_states_54, add_6, output_tensor_6, hidden_states_59, hidden_states_61, add_7, output_tensor_7], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   add_4 => add_24
#   add_5 => add_29
#   add_6 => add_34
#   add_7 => add_39
#   hidden_states_36 => convert_element_type_59, mul_29, sigmoid_9
#   hidden_states_38 => convolution_13
#   hidden_states_43 => convert_element_type_71, mul_35, sigmoid_11
#   hidden_states_45 => convolution_16
#   hidden_states_46 => constant_pad_nd_2
#   hidden_states_47 => convolution_17
#   hidden_states_52 => convert_element_type_83, mul_41, sigmoid_13
#   hidden_states_54 => convolution_19
#   hidden_states_59 => convert_element_type_95, mul_47, sigmoid_15
#   hidden_states_61 => convolution_21
#   input_tensor_1 => convolution_14
#   output_tensor_4 => div_4
#   output_tensor_5 => div_5
#   output_tensor_6 => div_6
#   output_tensor_7 => div_7
# Graph fragment:
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_11, %arg49_1, %arg50_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_9 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_23,), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_23, %sigmoid_9), kwargs = {})
#   %convert_element_type_59 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_29, torch.bfloat16), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_59, %arg47_1, %arg48_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %convolution_13), kwargs = {})
#   %div_4 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_24, 1.0), kwargs = {})
#   %sigmoid_11 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_28,), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_28, %sigmoid_11), kwargs = {})
#   %convert_element_type_71 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_35, torch.bfloat16), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_71, %arg57_1, %arg58_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_4, %convolution_16), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_29, 1.0), kwargs = {})
#   %constant_pad_nd_2 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%div_5, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_17 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_2, %arg59_1, %arg60_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_13 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_33,), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_33, %sigmoid_13), kwargs = {})
#   %convert_element_type_83 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_41, torch.bfloat16), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_83, %arg67_1, %arg68_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %convolution_19), kwargs = {})
#   %div_6 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_34, 1.0), kwargs = {})
#   %sigmoid_15 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_38,), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, %sigmoid_15), kwargs = {})
#   %convert_element_type_95 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_47, torch.bfloat16), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_95, %arg75_1, %arg76_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_6, %convolution_21), kwargs = {})
#   %div_7 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_39, 1.0), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_div_silu_44 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_div_silu_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_div_silu_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20974592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_div_silu_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp12 * tmp7
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/3r/c3rnurhq3posgfxret3x6ncxf3vxs7uhkdsh4u2vvh7ii53orarv.py
# Topologically Sorted Source Nodes: [hidden_states_62], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_62 => var_mean_16
# Graph fragment:
#   %var_mean_16 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_32, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_45 = async_compile.triton('triton_per_fused_native_group_norm_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 5, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 393216, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_45(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 256)
    x1 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 512*(((r0_2 + 128*x1) % 4096)) + ((r0_2 + 128*x1) // 4096)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None]
    tmp7 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = (tmp6 / tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tmp10 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.sum(tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, None)
    tl.store(out_ptr1 + (x3), tmp14, None)
    tl.store(out_ptr2 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/yg/cyg267hiyrzv4vku2q6la5yj7xtosvquod5l5r465556xck645yq.py
# Topologically Sorted Source Nodes: [hidden_states_62, hidden_states_63], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_62 => add_41, mul_49
#   hidden_states_63 => convert_element_type_101, mul_50, sigmoid_16
# Graph fragment:
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_33, %unsqueeze_98), kwargs = {})
#   %add_41 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, %unsqueeze_101), kwargs = {})
#   %sigmoid_16 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_41,), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_41, %sigmoid_16), kwargs = {})
#   %convert_element_type_101 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_50, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_46 = async_compile.triton('triton_poi_fused_native_group_norm_silu_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12584960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0 // 16), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0 // 16), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 65536.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/6v/c6vzpfyookc7wznj3o5op5kcqg2uqa2kwg3zyybh7v5qxrdfk26p.py
# Topologically Sorted Source Nodes: [group_norm_18], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_18 => var_mean_18
# Graph fragment:
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_37, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_47 = async_compile.triton('triton_per_fused_native_group_norm_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 5, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 393216, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_47(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 256)
    x1 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 512*(((r0_2 + 128*x1) % 4096)) + ((r0_2 + 128*x1) // 4096)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (2*x0 + 512*(((r0_2 + 128*x1) % 4096)) + ((r0_2 + 128*x1) // 4096)), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (2*x0 + ((r0_2 + 128*x1) // 4096)), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp12 = tl.sum(tmp10, 1)[:, None]
    tmp13 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = (tmp12 / tmp14)
    tmp16 = tmp8 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.sum(tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp20, None)
    tl.store(out_ptr2 + (x3), tmp14, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/xl/cxlxlq5e3iujntbbiw7ukzhgm3fr3zeg6zksoojtlzzbhozkgoix.py
# Topologically Sorted Source Nodes: [group_norm_18], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_18 => add_46, convert_element_type_109, mul_55
# Graph fragment:
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_38, %unsqueeze_109), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_55, %unsqueeze_111), kwargs = {})
#   %convert_element_type_109 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_46, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_48 = async_compile.triton('triton_poi_fused_native_group_norm_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 8388608, 'x': 8391680}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 512*y0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1 + 512*y0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x1 // 16), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1 // 16), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 65536.0
    tmp12 = (tmp10 / tmp11)
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp9 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 4096*x1), tmp23, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/tn/ctnwh7ftxghme4bnfspk5ypehubvsecva7rckedbcickvaj5s7s2.py
# Topologically Sorted Source Nodes: [hidden_states_71], Original ATen: [aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   hidden_states_71 => _scaled_dot_product_efficient_attention
# Graph fragment:
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_6, %permute_7, %permute_8, None, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_49 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12583936}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_49(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/mk/cmkxj7wwf3op7ni35ogdga53555nvpggicf2dqkk6jvi55rdcanb.py
# Topologically Sorted Source Nodes: [hidden_states_66, hidden_states_68, add_8, output_tensor_8, hidden_states_77, hidden_states_78], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div]
# Source node to ATen node mapping:
#   add_8 => add_44
#   hidden_states_66 => convert_element_type_107, mul_53, sigmoid_17
#   hidden_states_68 => convolution_23
#   hidden_states_77 => add_50
#   hidden_states_78 => div_9
#   output_tensor_8 => div_8
# Graph fragment:
#   %sigmoid_17 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_43,), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %sigmoid_17), kwargs = {})
#   %convert_element_type_107 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_53, torch.bfloat16), kwargs = {})
#   %convolution_23 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_107, %arg83_1, %arg84_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_7, %convolution_23), kwargs = {})
#   %div_8 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_44, 1), kwargs = {})
#   %add_50 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_54, %div_8), kwargs = {})
#   %div_9 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_50, 1), kwargs = {})
triton_poi_fused_add_convolution_div_silu_50 = async_compile.triton('triton_poi_fused_add_convolution_div_silu_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_silu_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20973568}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_silu_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 + tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 + tmp9
    tmp11 = tmp10 * tmp8
    tl.store(in_out_ptr0 + (x2), tmp11, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/w6/cw62ztxuweu5hsu7aebgesl4yxnge3xt5uqtajjhno5jkkkxkyco.py
# Topologically Sorted Source Nodes: [sample_1, sample_2], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   sample_1 => add_57, mul_63
#   sample_2 => convert_element_type_138, mul_64, sigmoid_20
# Graph fragment:
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_60, %unsqueeze_126), kwargs = {})
#   %add_57 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_63, %unsqueeze_129), kwargs = {})
#   %sigmoid_20 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_20), kwargs = {})
#   %convert_element_type_138 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_64, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_51 = async_compile.triton('triton_poi_fused_native_group_norm_silu_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16780288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x0 // 16), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0 // 16), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 65536.0
    tmp12 = (tmp10 / tmp11)
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp9 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = tl.sigmoid(tmp22)
    tmp24 = tmp22 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp25, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/zx/czxhn7r4nmfbidwf4xz7lf6wsvgrzbjhygxmbadijhfvwod6vbcq.py
# Topologically Sorted Source Nodes: [sample_2, sample_3], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   sample_2 => convert_element_type_138, mul_64, sigmoid_20
#   sample_3 => convolution_26
# Graph fragment:
#   %sigmoid_20 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_20), kwargs = {})
#   %convert_element_type_138 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_64, torch.bfloat16), kwargs = {})
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_138, %arg105_1, %arg106_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_52 = async_compile.triton('triton_poi_fused_convolution_silu_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 147456, 'x': 73728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_52(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/ju/cjumeq6htyrruyx7lmgtkso674gjdaawhgwk7gyfewyt7khvtjr4.py
# Topologically Sorted Source Nodes: [sample_2, sample_3], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   sample_2 => convert_element_type_138, mul_64, sigmoid_20
#   sample_3 => convolution_26
# Graph fragment:
#   %sigmoid_20 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_20), kwargs = {})
#   %convert_element_type_138 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_64, torch.bfloat16), kwargs = {})
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_138, %arg105_1, %arg106_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_53 = async_compile.triton('triton_poi_fused_convolution_silu_53', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_53', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 196624}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_53(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 8)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/mv/cmvwjwag37uemhky5jrf75cz6vzvhoatms2eqsqnnlvbbhtvke6r.py
# Topologically Sorted Source Nodes: [sample_2, sample_3, enc], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   enc => convolution_27
#   sample_2 => convert_element_type_138, mul_64, sigmoid_20
#   sample_3 => convolution_26
# Graph fragment:
#   %sigmoid_20 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_20), kwargs = {})
#   %convert_element_type_138 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_64, torch.bfloat16), kwargs = {})
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_138, %arg105_1, %arg106_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_27 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_26, %arg107_1, %arg108_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_54 = async_compile.triton('triton_poi_fused_convolution_silu_54', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 65552, 'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_54(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 8*x1), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x1 + 4096*y0), tmp2, ymask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/ds/cdsstxerqg7hh7spwxfcrag6aysw2ja2a2fc4t5mxvvrsbmqfjdq.py
# Topologically Sorted Source Nodes: [clamp, exp_1, mul, exp], Original ATen: [aten.clamp, aten.exp, aten.mul]
# Source node to ATen node mapping:
#   clamp => clamp_max, clamp_min, convert_element_type_139, convert_element_type_140
#   exp => exp
#   exp_1 => exp_1
#   mul => mul_65
# Graph fragment:
#   %convert_element_type_139 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_49, torch.float32), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_139, -30.0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 20.0), kwargs = {})
#   %convert_element_type_140 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max, torch.bfloat16), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%convert_element_type_140,), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_140, 0.5), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_65,), kwargs = {})
triton_poi_fused_clamp_exp_mul_55 = async_compile.triton('triton_poi_fused_clamp_exp_mul_55', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_exp_mul_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 229376}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_exp_mul_55(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16384 + x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -30.0
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = 20.0
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl_math.exp(tmp6)
    tmp8 = 0.5
    tmp9 = tmp6 * tmp8
    tmp10 = tl_math.exp(tmp9)
    tl.store(out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr1 + (x0), tmp7, None)
    tl.store(out_ptr2 + (x0), tmp10, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 3, 512, 512), (786432, 262144, 512, 1))
    assert_size_stride(arg1_1, (128, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg2_1, (128, ), (1, ))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (128, ), (1, ))
    assert_size_stride(arg17_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg18_1, (128, ), (1, ))
    assert_size_stride(arg19_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (128, ), (1, ))
    assert_size_stride(arg22_1, (128, ), (1, ))
    assert_size_stride(arg23_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg24_1, (256, ), (1, ))
    assert_size_stride(arg25_1, (256, ), (1, ))
    assert_size_stride(arg26_1, (256, ), (1, ))
    assert_size_stride(arg27_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg28_1, (256, ), (1, ))
    assert_size_stride(arg29_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg30_1, (256, ), (1, ))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (256, ), (1, ))
    assert_size_stride(arg36_1, (256, ), (1, ))
    assert_size_stride(arg37_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (256, ), (1, ))
    assert_size_stride(arg42_1, (256, ), (1, ))
    assert_size_stride(arg43_1, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg44_1, (512, ), (1, ))
    assert_size_stride(arg45_1, (512, ), (1, ))
    assert_size_stride(arg46_1, (512, ), (1, ))
    assert_size_stride(arg47_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg48_1, (512, ), (1, ))
    assert_size_stride(arg49_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (512, ), (1, ))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg54_1, (512, ), (1, ))
    assert_size_stride(arg55_1, (512, ), (1, ))
    assert_size_stride(arg56_1, (512, ), (1, ))
    assert_size_stride(arg57_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg58_1, (512, ), (1, ))
    assert_size_stride(arg59_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg60_1, (512, ), (1, ))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (512, ), (1, ))
    assert_size_stride(arg63_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg64_1, (512, ), (1, ))
    assert_size_stride(arg65_1, (512, ), (1, ))
    assert_size_stride(arg66_1, (512, ), (1, ))
    assert_size_stride(arg67_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (512, ), (1, ))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg72_1, (512, ), (1, ))
    assert_size_stride(arg73_1, (512, ), (1, ))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg76_1, (512, ), (1, ))
    assert_size_stride(arg77_1, (512, ), (1, ))
    assert_size_stride(arg78_1, (512, ), (1, ))
    assert_size_stride(arg79_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg80_1, (512, ), (1, ))
    assert_size_stride(arg81_1, (512, ), (1, ))
    assert_size_stride(arg82_1, (512, ), (1, ))
    assert_size_stride(arg83_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg84_1, (512, ), (1, ))
    assert_size_stride(arg85_1, (512, ), (1, ))
    assert_size_stride(arg86_1, (512, ), (1, ))
    assert_size_stride(arg87_1, (512, 512), (512, 1))
    assert_size_stride(arg88_1, (512, ), (1, ))
    assert_size_stride(arg89_1, (512, 512), (512, 1))
    assert_size_stride(arg90_1, (512, ), (1, ))
    assert_size_stride(arg91_1, (512, 512), (512, 1))
    assert_size_stride(arg92_1, (512, ), (1, ))
    assert_size_stride(arg93_1, (512, 512), (512, 1))
    assert_size_stride(arg94_1, (512, ), (1, ))
    assert_size_stride(arg95_1, (512, ), (1, ))
    assert_size_stride(arg96_1, (512, ), (1, ))
    assert_size_stride(arg97_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg98_1, (512, ), (1, ))
    assert_size_stride(arg99_1, (512, ), (1, ))
    assert_size_stride(arg100_1, (512, ), (1, ))
    assert_size_stride(arg101_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg102_1, (512, ), (1, ))
    assert_size_stride(arg103_1, (512, ), (1, ))
    assert_size_stride(arg104_1, (512, ), (1, ))
    assert_size_stride(arg105_1, (8, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg106_1, (8, ), (1, ))
    assert_size_stride(arg107_1, (8, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg108_1, (8, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 3, 512, 512), (786432, 1, 1536, 3), torch.bfloat16)
        # Topologically Sorted Source Nodes: [sample], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg0_1, buf0, 3, 262144, stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((128, 3, 3, 3), (27, 1, 9, 3), torch.bfloat16)
        # Topologically Sorted Source Nodes: [sample], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(arg1_1, buf1, 384, 9, stream=stream0)
        del arg1_1
        # Topologically Sorted Source Nodes: [sample], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (1, 128, 512, 512), (33554432, 1, 65536, 128), 'torch.ops.aten.convolution.default')
        del buf0
        del buf1
        buf3 = empty_strided_cuda((1, 32, 1, 1, 14, 216), (96768, 1, 96768, 96768, 6912, 32), torch.float32)
        buf4 = empty_strided_cuda((1, 32, 1, 1, 14, 216), (96768, 1, 96768, 96768, 6912, 32), torch.float32)
        buf5 = empty_strided_cuda((1, 32, 1, 1, 14, 216), (96768, 1, 96768, 96768, 6912, 32), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_2.run(buf2, arg2_1, buf3, buf4, buf5, 96768, 347, stream=stream0)
        buf6 = empty_strided_cuda((1, 32, 1, 1, 14, 2), (896, 1, 896, 896, 64, 32), torch.float32)
        buf7 = empty_strided_cuda((1, 32, 1, 1, 14, 2), (896, 1, 896, 896, 64, 32), torch.float32)
        buf8 = empty_strided_cuda((1, 32, 1, 1, 14, 2), (896, 1, 896, 896, 64, 32), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_3.run(buf3, buf4, buf5, buf6, buf7, buf8, 896, 108, stream=stream0)
        buf9 = empty_strided_cuda((1, 32, 1, 1, 14), (448, 1, 448, 448, 32), torch.float32)
        buf10 = empty_strided_cuda((1, 32, 1, 1, 14), (448, 1, 448, 448, 32), torch.float32)
        buf11 = empty_strided_cuda((1, 32, 1, 1, 14), (448, 1, 448, 448, 32), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf6, buf7, buf8, buf9, buf10, buf11, 448, 2, stream=stream0)
        buf12 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 32, 32), torch.float32)
        buf13 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf9, buf10, buf11, buf12, buf13, 32, 14, stream=stream0)
        buf16 = empty_strided_cuda((1, 128, 512, 512), (33554432, 1, 65536, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states, hidden_states_1], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_6.run(buf2, arg2_1, buf12, buf13, arg3_1, arg4_1, buf16, 33554432, stream=stream0)
        del arg3_1
        del arg4_1
        buf17 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg5_1, buf17, 16384, 9, stream=stream0)
        del arg5_1
        # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2], Original ATen: [aten.silu, aten.convolution]
        buf18 = extern_kernels.convolution(buf16, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (1, 128, 512, 512), (33554432, 1, 65536, 128), 'torch.ops.aten.convolution.default')
        buf19 = buf5; del buf5  # reuse
        buf20 = buf4; del buf4  # reuse
        buf21 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_3], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_2.run(buf18, arg6_1, buf19, buf20, buf21, 96768, 347, stream=stream0)
        buf22 = buf8; del buf8  # reuse
        buf23 = buf7; del buf7  # reuse
        buf24 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_3], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_3.run(buf19, buf20, buf21, buf22, buf23, buf24, 896, 108, stream=stream0)
        buf25 = buf9; del buf9  # reuse
        buf26 = buf11; del buf11  # reuse
        buf27 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_3], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf22, buf23, buf24, buf25, buf26, buf27, 448, 2, stream=stream0)
        buf28 = buf13; del buf13  # reuse
        buf29 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_3], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf25, buf26, buf27, buf28, buf29, 32, 14, stream=stream0)
        buf32 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_6.run(buf18, arg6_1, buf28, buf29, arg7_1, arg8_1, buf32, 33554432, stream=stream0)
        del arg6_1
        del arg7_1
        del arg8_1
        del buf18
        buf33 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_6], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg9_1, buf33, 16384, 9, stream=stream0)
        del arg9_1
        # Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_6], Original ATen: [aten.silu, aten.convolution]
        buf34 = extern_kernels.convolution(buf32, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (1, 128, 512, 512), (33554432, 1, 65536, 128), 'torch.ops.aten.convolution.default')
        buf35 = buf21; del buf21  # reuse
        buf36 = buf20; del buf20  # reuse
        buf37 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_7], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_8.run(buf2, arg2_1, buf34, arg10_1, buf35, buf36, buf37, 96768, 347, stream=stream0)
        buf38 = buf24; del buf24  # reuse
        buf39 = buf23; del buf23  # reuse
        buf40 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_7], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_3.run(buf35, buf36, buf37, buf38, buf39, buf40, 896, 108, stream=stream0)
        buf41 = buf27; del buf27  # reuse
        buf42 = buf26; del buf26  # reuse
        buf43 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_7], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf38, buf39, buf40, buf41, buf42, buf43, 448, 2, stream=stream0)
        buf44 = buf29; del buf29  # reuse
        buf45 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_7], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf41, buf42, buf43, buf44, buf45, 32, 14, stream=stream0)
        buf48 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_7, hidden_states_8], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_9.run(buf2, arg2_1, buf34, arg10_1, buf44, buf45, arg11_1, arg12_1, buf48, 33554432, stream=stream0)
        del arg11_1
        del arg12_1
        buf49 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_8, hidden_states_9], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg13_1, buf49, 16384, 9, stream=stream0)
        del arg13_1
        # Topologically Sorted Source Nodes: [hidden_states_8, hidden_states_9], Original ATen: [aten.silu, aten.convolution]
        buf50 = extern_kernels.convolution(buf48, buf49, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (1, 128, 512, 512), (33554432, 1, 65536, 128), 'torch.ops.aten.convolution.default')
        buf51 = buf37; del buf37  # reuse
        buf52 = buf36; del buf36  # reuse
        buf53 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_10], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_2.run(buf50, arg14_1, buf51, buf52, buf53, 96768, 347, stream=stream0)
        buf54 = buf40; del buf40  # reuse
        buf55 = buf39; del buf39  # reuse
        buf56 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_10], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_3.run(buf51, buf52, buf53, buf54, buf55, buf56, 896, 108, stream=stream0)
        del buf51
        del buf52
        del buf53
        buf57 = buf43; del buf43  # reuse
        buf58 = buf42; del buf42  # reuse
        buf59 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_10], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf54, buf55, buf56, buf57, buf58, buf59, 448, 2, stream=stream0)
        del buf54
        del buf55
        del buf56
        buf60 = buf45; del buf45  # reuse
        buf61 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_10], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf57, buf58, buf59, buf60, buf61, 32, 14, stream=stream0)
        del buf57
        del buf58
        del buf59
        buf64 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_6.run(buf50, arg14_1, buf60, buf61, arg15_1, arg16_1, buf64, 33554432, stream=stream0)
        del arg14_1
        del arg15_1
        del arg16_1
        del buf50
        buf65 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_11, hidden_states_13], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg17_1, buf65, 16384, 9, stream=stream0)
        del arg17_1
        # Topologically Sorted Source Nodes: [hidden_states_11, hidden_states_13], Original ATen: [aten.silu, aten.convolution]
        buf66 = extern_kernels.convolution(buf64, buf65, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (1, 128, 512, 512), (33554432, 1, 65536, 128), 'torch.ops.aten.convolution.default')
        del buf64
        buf67 = empty_strided_cuda((1, 128, 513, 513), (33685632, 1, 65664, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [sample, hidden_states_4, hidden_states_6, add, output_tensor, hidden_states_11, hidden_states_13, add_1, output_tensor_1, hidden_states_14], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_div_silu_10.run(buf2, arg2_1, buf34, arg10_1, buf66, arg18_1, buf67, 33685632, stream=stream0)
        del arg10_1
        del arg18_1
        del arg2_1
        del buf2
        del buf34
        del buf66
        buf68 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [sample, hidden_states_4, hidden_states_6, add, output_tensor, hidden_states_11, hidden_states_13, add_1, output_tensor_1, hidden_states_14, hidden_states_15], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg19_1, buf68, 16384, 9, stream=stream0)
        del arg19_1
        # Topologically Sorted Source Nodes: [sample, hidden_states_4, hidden_states_6, add, output_tensor, hidden_states_11, hidden_states_13, add_1, output_tensor_1, hidden_states_14, hidden_states_15], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.constant_pad_nd]
        buf69 = extern_kernels.convolution(buf67, buf68, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (1, 128, 256, 256), (8388608, 1, 32768, 128), 'torch.ops.aten.convolution.default')
        del buf67
        del buf68
        buf70 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [sample, hidden_states_4, hidden_states_6, add, output_tensor, hidden_states_11, hidden_states_13, add_1, output_tensor_1, hidden_states_14, hidden_states_15], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_div_silu_11.run(buf70, arg20_1, 8388608, stream=stream0)
        del arg20_1
        buf71 = empty_strided_cuda((1, 32, 1, 1, 16, 128), (65536, 1, 65536, 65536, 4096, 32), torch.float32)
        buf72 = empty_strided_cuda((1, 32, 1, 1, 16, 128), (65536, 1, 65536, 65536, 4096, 32), torch.float32)
        buf73 = empty_strided_cuda((1, 32, 1, 1, 16, 128), (65536, 1, 65536, 65536, 4096, 32), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_16], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_12.run(buf70, buf71, buf72, buf73, 65536, 128, stream=stream0)
        buf74 = empty_strided_cuda((1, 32, 1, 1, 16), (512, 1, 512, 512, 32), torch.float32)
        buf75 = empty_strided_cuda((1, 32, 1, 1, 16), (512, 1, 512, 512, 32), torch.float32)
        buf76 = empty_strided_cuda((1, 32, 1, 1, 16), (512, 1, 512, 512, 32), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_16], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_13.run(buf71, buf72, buf73, buf74, buf75, buf76, 512, 128, stream=stream0)
        buf77 = buf61; del buf61  # reuse
        buf78 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_16], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_14.run(buf74, buf75, buf76, buf77, buf78, 32, 16, stream=stream0)
        buf81 = empty_strided_cuda((1, 128, 256, 256), (8388608, 1, 32768, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_16, hidden_states_17], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_15.run(buf70, buf77, buf78, arg21_1, arg22_1, buf81, 8388608, stream=stream0)
        del arg21_1
        del arg22_1
        buf82 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_17, hidden_states_18], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_16.run(arg23_1, buf82, 32768, 9, stream=stream0)
        del arg23_1
        # Topologically Sorted Source Nodes: [hidden_states_17, hidden_states_18], Original ATen: [aten.silu, aten.convolution]
        buf83 = extern_kernels.convolution(buf81, buf82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (1, 256, 256, 256), (16777216, 1, 65536, 256), 'torch.ops.aten.convolution.default')
        del buf81
        del buf82
        buf84 = empty_strided_cuda((1, 32, 1, 1, 16, 256), (131072, 1, 131072, 131072, 8192, 32), torch.float32)
        buf85 = empty_strided_cuda((1, 32, 1, 1, 16, 256), (131072, 1, 131072, 131072, 8192, 32), torch.float32)
        buf86 = empty_strided_cuda((1, 32, 1, 1, 16, 256), (131072, 1, 131072, 131072, 8192, 32), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_19], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_17.run(buf83, arg24_1, buf84, buf85, buf86, 131072, 128, stream=stream0)
        buf87 = empty_strided_cuda((1, 32, 1, 1, 16, 2), (1024, 1, 1024, 1024, 64, 32), torch.float32)
        buf88 = empty_strided_cuda((1, 32, 1, 1, 16, 2), (1024, 1, 1024, 1024, 64, 32), torch.float32)
        buf89 = empty_strided_cuda((1, 32, 1, 1, 16, 2), (1024, 1, 1024, 1024, 64, 32), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_19], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_18.run(buf84, buf85, buf86, buf87, buf88, buf89, 1024, 128, stream=stream0)
        buf90 = buf76; del buf76  # reuse
        buf91 = buf75; del buf75  # reuse
        buf92 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_19], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf87, buf88, buf89, buf90, buf91, buf92, 512, 2, stream=stream0)
        buf93 = buf78; del buf78  # reuse
        buf94 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_19], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_14.run(buf90, buf91, buf92, buf93, buf94, 32, 16, stream=stream0)
        # Topologically Sorted Source Nodes: [input_tensor], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf70, arg29_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (1, 256, 256, 256), (16777216, 1, 65536, 256), 'torch.ops.aten.convolution.default')
        del arg29_1
        buf98 = empty_strided_cuda((1, 256, 256, 256), (16777216, 1, 65536, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_20.run(buf83, arg24_1, buf93, buf94, arg25_1, arg26_1, buf98, 16777216, stream=stream0)
        del arg24_1
        del arg25_1
        del arg26_1
        del buf83
        buf99 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_20, hidden_states_22], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_21.run(arg27_1, buf99, 65536, 9, stream=stream0)
        del arg27_1
        # Topologically Sorted Source Nodes: [hidden_states_20, hidden_states_22], Original ATen: [aten.silu, aten.convolution]
        buf100 = extern_kernels.convolution(buf98, buf99, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (1, 256, 256, 256), (16777216, 1, 65536, 256), 'torch.ops.aten.convolution.default')
        buf101 = buf86; del buf86  # reuse
        buf102 = buf85; del buf85  # reuse
        buf103 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_23], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_22.run(buf96, arg30_1, buf100, arg28_1, buf101, buf102, buf103, 131072, 128, stream=stream0)
        buf104 = buf89; del buf89  # reuse
        buf105 = buf88; del buf88  # reuse
        buf106 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_23], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_18.run(buf101, buf102, buf103, buf104, buf105, buf106, 1024, 128, stream=stream0)
        buf107 = buf92; del buf92  # reuse
        buf108 = buf91; del buf91  # reuse
        buf109 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_23], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf104, buf105, buf106, buf107, buf108, buf109, 512, 2, stream=stream0)
        buf110 = buf94; del buf94  # reuse
        buf111 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_23], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_14.run(buf107, buf108, buf109, buf110, buf111, 32, 16, stream=stream0)
        buf114 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_23, hidden_states_24], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_23.run(buf96, arg30_1, buf100, arg28_1, buf110, buf111, arg31_1, arg32_1, buf114, 16777216, stream=stream0)
        del arg31_1
        del arg32_1
        buf115 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_24, hidden_states_25], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_21.run(arg33_1, buf115, 65536, 9, stream=stream0)
        del arg33_1
        # Topologically Sorted Source Nodes: [hidden_states_24, hidden_states_25], Original ATen: [aten.silu, aten.convolution]
        buf116 = extern_kernels.convolution(buf114, buf115, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (1, 256, 256, 256), (16777216, 1, 65536, 256), 'torch.ops.aten.convolution.default')
        buf117 = buf103; del buf103  # reuse
        buf118 = buf102; del buf102  # reuse
        buf119 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_26], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_17.run(buf116, arg34_1, buf117, buf118, buf119, 131072, 128, stream=stream0)
        buf120 = buf106; del buf106  # reuse
        buf121 = buf105; del buf105  # reuse
        buf122 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_26], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_18.run(buf117, buf118, buf119, buf120, buf121, buf122, 1024, 128, stream=stream0)
        del buf117
        del buf118
        del buf119
        buf123 = buf109; del buf109  # reuse
        buf124 = buf108; del buf108  # reuse
        buf125 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_26], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf120, buf121, buf122, buf123, buf124, buf125, 512, 2, stream=stream0)
        del buf120
        del buf121
        del buf122
        buf126 = buf111; del buf111  # reuse
        buf127 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_26], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_14.run(buf123, buf124, buf125, buf126, buf127, 32, 16, stream=stream0)
        buf130 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_26, hidden_states_27], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_20.run(buf116, arg34_1, buf126, buf127, arg35_1, arg36_1, buf130, 16777216, stream=stream0)
        del arg34_1
        del arg35_1
        del arg36_1
        del buf116
        buf131 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_27, hidden_states_29], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_21.run(arg37_1, buf131, 65536, 9, stream=stream0)
        del arg37_1
        # Topologically Sorted Source Nodes: [hidden_states_27, hidden_states_29], Original ATen: [aten.silu, aten.convolution]
        buf132 = extern_kernels.convolution(buf130, buf131, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (1, 256, 256, 256), (16777216, 1, 65536, 256), 'torch.ops.aten.convolution.default')
        del buf130
        buf133 = empty_strided_cuda((1, 256, 257, 257), (16908544, 1, 65792, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_tensor, hidden_states_20, hidden_states_22, add_2, output_tensor_2, hidden_states_27, hidden_states_29, add_3, output_tensor_3, hidden_states_30], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_div_silu_24.run(buf96, arg30_1, buf100, arg28_1, buf132, arg38_1, buf133, 16908544, stream=stream0)
        del arg28_1
        del arg30_1
        del arg38_1
        del buf100
        del buf132
        del buf96
        buf134 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [input_tensor, hidden_states_20, hidden_states_22, add_2, output_tensor_2, hidden_states_27, hidden_states_29, add_3, output_tensor_3, hidden_states_30, hidden_states_31], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_21.run(arg39_1, buf134, 65536, 9, stream=stream0)
        del arg39_1
        # Topologically Sorted Source Nodes: [input_tensor, hidden_states_20, hidden_states_22, add_2, output_tensor_2, hidden_states_27, hidden_states_29, add_3, output_tensor_3, hidden_states_30, hidden_states_31], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.constant_pad_nd]
        buf135 = extern_kernels.convolution(buf133, buf134, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (1, 256, 128, 128), (4194304, 1, 32768, 256), 'torch.ops.aten.convolution.default')
        del buf133
        del buf134
        buf136 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [input_tensor, hidden_states_20, hidden_states_22, add_2, output_tensor_2, hidden_states_27, hidden_states_29, add_3, output_tensor_3, hidden_states_30, hidden_states_31], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_div_silu_25.run(buf136, arg40_1, 4194304, stream=stream0)
        del arg40_1
        buf137 = empty_strided_cuda((1, 32, 1, 1, 16, 64), (32768, 1, 32768, 32768, 2048, 32), torch.float32)
        buf138 = empty_strided_cuda((1, 32, 1, 1, 16, 64), (32768, 1, 32768, 32768, 2048, 32), torch.float32)
        buf139 = empty_strided_cuda((1, 32, 1, 1, 16, 64), (32768, 1, 32768, 32768, 2048, 32), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_32], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_26.run(buf136, buf137, buf138, buf139, 32768, 128, stream=stream0)
        buf140 = buf125; del buf125  # reuse
        buf141 = buf124; del buf124  # reuse
        buf142 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_32], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_27.run(buf137, buf138, buf139, buf140, buf141, buf142, 512, 64, stream=stream0)
        del buf137
        del buf138
        del buf139
        buf143 = buf127; del buf127  # reuse
        buf144 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_32], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_14.run(buf140, buf141, buf142, buf143, buf144, 32, 16, stream=stream0)
        buf147 = empty_strided_cuda((1, 256, 128, 128), (4194304, 1, 32768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_28.run(buf136, buf143, buf144, arg41_1, arg42_1, buf147, 4194304, stream=stream0)
        del arg41_1
        del arg42_1
        buf148 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_33, hidden_states_34], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_29.run(arg43_1, buf148, 131072, 9, stream=stream0)
        del arg43_1
        # Topologically Sorted Source Nodes: [hidden_states_33, hidden_states_34], Original ATen: [aten.silu, aten.convolution]
        buf149 = extern_kernels.convolution(buf147, buf148, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (1, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
        del buf147
        del buf148
        buf150 = reinterpret_tensor(buf73, (1, 32, 1, 1, 16, 128), (65536, 16, 65536, 65536, 1, 512), 0); del buf73  # reuse
        buf151 = reinterpret_tensor(buf72, (1, 32, 1, 1, 16, 128), (65536, 16, 65536, 65536, 1, 512), 0); del buf72  # reuse
        buf152 = reinterpret_tensor(buf71, (1, 32, 1, 1, 16, 128), (65536, 16, 65536, 65536, 1, 512), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_30.run(buf149, arg44_1, buf150, buf151, buf152, 65536, 128, stream=stream0)
        buf153 = reinterpret_tensor(buf142, (1, 32, 1, 1, 16), (512, 16, 512, 512, 1), 0); del buf142  # reuse
        buf154 = reinterpret_tensor(buf141, (1, 32, 1, 1, 16), (512, 16, 512, 512, 1), 0); del buf141  # reuse
        buf155 = reinterpret_tensor(buf140, (1, 32, 1, 1, 16), (512, 16, 512, 512, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_31.run(buf150, buf151, buf152, buf153, buf154, buf155, 512, 128, stream=stream0)
        buf156 = buf144; del buf144  # reuse
        buf157 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_32.run(buf153, buf154, buf155, buf156, buf157, 32, 16, stream=stream0)
        # Topologically Sorted Source Nodes: [input_tensor_1], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf136, arg49_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (1, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
        del arg49_1
        del buf136
        buf161 = reinterpret_tensor(buf70, (1, 512, 128, 128), (8388608, 1, 65536, 512), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_36], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_33.run(buf149, arg44_1, buf156, buf157, arg45_1, arg46_1, buf161, 8388608, stream=stream0)
        del arg44_1
        del arg45_1
        del arg46_1
        del buf149
        buf162 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_36, hidden_states_38], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_34.run(arg47_1, buf162, 262144, 9, stream=stream0)
        del arg47_1
        # Topologically Sorted Source Nodes: [hidden_states_36, hidden_states_38], Original ATen: [aten.silu, aten.convolution]
        buf163 = extern_kernels.convolution(buf161, buf162, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (1, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
        buf164 = buf152; del buf152  # reuse
        buf165 = buf151; del buf151  # reuse
        buf166 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_39], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_35.run(buf159, arg50_1, buf163, arg48_1, buf164, buf165, buf166, 65536, 128, stream=stream0)
        buf167 = buf155; del buf155  # reuse
        buf168 = buf154; del buf154  # reuse
        buf169 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_39], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_31.run(buf164, buf165, buf166, buf167, buf168, buf169, 512, 128, stream=stream0)
        buf170 = buf157; del buf157  # reuse
        buf171 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_39], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_32.run(buf167, buf168, buf169, buf170, buf171, 32, 16, stream=stream0)
        buf174 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_36.run(buf159, arg50_1, buf163, arg48_1, buf170, buf171, arg51_1, arg52_1, buf174, 8388608, stream=stream0)
        del arg51_1
        del arg52_1
        buf175 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_41], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_34.run(arg53_1, buf175, 262144, 9, stream=stream0)
        del arg53_1
        # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_41], Original ATen: [aten.silu, aten.convolution]
        buf176 = extern_kernels.convolution(buf174, buf175, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (1, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
        buf177 = buf166; del buf166  # reuse
        buf178 = buf165; del buf165  # reuse
        buf179 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_42], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_30.run(buf176, arg54_1, buf177, buf178, buf179, 65536, 128, stream=stream0)
        buf180 = buf169; del buf169  # reuse
        buf181 = buf168; del buf168  # reuse
        buf182 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_42], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_31.run(buf177, buf178, buf179, buf180, buf181, buf182, 512, 128, stream=stream0)
        del buf177
        del buf178
        del buf179
        buf183 = buf171; del buf171  # reuse
        buf184 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_42], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_32.run(buf180, buf181, buf182, buf183, buf184, 32, 16, stream=stream0)
        del buf180
        del buf181
        del buf182
        buf187 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_42, hidden_states_43], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_33.run(buf176, arg54_1, buf183, buf184, arg55_1, arg56_1, buf187, 8388608, stream=stream0)
        del arg54_1
        del arg55_1
        del arg56_1
        del buf176
        buf188 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_43, hidden_states_45], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_34.run(arg57_1, buf188, 262144, 9, stream=stream0)
        del arg57_1
        # Topologically Sorted Source Nodes: [hidden_states_43, hidden_states_45], Original ATen: [aten.silu, aten.convolution]
        buf189 = extern_kernels.convolution(buf187, buf188, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (1, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
        del buf187
        buf190 = empty_strided_cuda((1, 512, 129, 129), (8520192, 1, 66048, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_36, hidden_states_38, add_4, output_tensor_4, hidden_states_43, hidden_states_45, add_5, output_tensor_5, hidden_states_46], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_div_silu_37.run(buf159, arg50_1, buf163, arg48_1, buf189, arg58_1, buf190, 8520192, stream=stream0)
        del arg48_1
        del arg50_1
        del arg58_1
        del buf159
        del buf163
        del buf189
        buf191 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_36, hidden_states_38, add_4, output_tensor_4, hidden_states_43, hidden_states_45, add_5, output_tensor_5, hidden_states_46, hidden_states_47], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_34.run(arg59_1, buf191, 262144, 9, stream=stream0)
        del arg59_1
        # Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_36, hidden_states_38, add_4, output_tensor_4, hidden_states_43, hidden_states_45, add_5, output_tensor_5, hidden_states_46, hidden_states_47], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.constant_pad_nd]
        buf192 = extern_kernels.convolution(buf190, buf191, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (1, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        del buf190
        buf193 = empty_strided_cuda((1, 32, 1, 1, 8, 64), (16384, 8, 16384, 16384, 1, 256), torch.float32)
        buf194 = empty_strided_cuda((1, 32, 1, 1, 8, 64), (16384, 8, 16384, 16384, 1, 256), torch.float32)
        buf195 = empty_strided_cuda((1, 32, 1, 1, 8, 64), (16384, 8, 16384, 16384, 1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_48], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_38.run(buf192, arg60_1, buf193, buf194, buf195, 16384, 128, stream=stream0)
        buf196 = empty_strided_cuda((1, 32, 1, 1, 8), (256, 8, 256, 256, 1), torch.float32)
        buf197 = empty_strided_cuda((1, 32, 1, 1, 8), (256, 8, 256, 256, 1), torch.float32)
        buf198 = empty_strided_cuda((1, 32, 1, 1, 8), (256, 8, 256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_48], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_39.run(buf193, buf194, buf195, buf196, buf197, buf198, 256, 64, stream=stream0)
        buf199 = buf184; del buf184  # reuse
        buf200 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_48], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_40.run(buf196, buf197, buf198, buf199, buf200, 32, 8, stream=stream0)
        buf203 = empty_strided_cuda((1, 512, 64, 64), (2097152, 1, 32768, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_41.run(buf192, arg60_1, buf199, buf200, arg61_1, arg62_1, buf203, 2097152, stream=stream0)
        del arg61_1
        del arg62_1
        buf204 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_49, hidden_states_50], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_34.run(arg63_1, buf204, 262144, 9, stream=stream0)
        del arg63_1
        # Topologically Sorted Source Nodes: [hidden_states_49, hidden_states_50], Original ATen: [aten.silu, aten.convolution]
        buf205 = extern_kernels.convolution(buf203, buf204, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (1, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf206 = buf195; del buf195  # reuse
        buf207 = buf194; del buf194  # reuse
        buf208 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_51], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_38.run(buf205, arg64_1, buf206, buf207, buf208, 16384, 128, stream=stream0)
        buf209 = buf198; del buf198  # reuse
        buf210 = buf197; del buf197  # reuse
        buf211 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_51], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_39.run(buf206, buf207, buf208, buf209, buf210, buf211, 256, 64, stream=stream0)
        buf212 = buf200; del buf200  # reuse
        buf213 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_51], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_40.run(buf209, buf210, buf211, buf212, buf213, 32, 8, stream=stream0)
        buf216 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_51, hidden_states_52], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_41.run(buf205, arg64_1, buf212, buf213, arg65_1, arg66_1, buf216, 2097152, stream=stream0)
        del arg64_1
        del arg65_1
        del arg66_1
        del buf205
        buf217 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_52, hidden_states_54], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_34.run(arg67_1, buf217, 262144, 9, stream=stream0)
        del arg67_1
        # Topologically Sorted Source Nodes: [hidden_states_52, hidden_states_54], Original ATen: [aten.silu, aten.convolution]
        buf218 = extern_kernels.convolution(buf216, buf217, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (1, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf219 = buf208; del buf208  # reuse
        buf220 = buf207; del buf207  # reuse
        buf221 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_42.run(buf192, arg60_1, buf218, arg68_1, buf219, buf220, buf221, 16384, 128, stream=stream0)
        buf222 = buf211; del buf211  # reuse
        buf223 = buf210; del buf210  # reuse
        buf224 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_39.run(buf219, buf220, buf221, buf222, buf223, buf224, 256, 64, stream=stream0)
        buf225 = buf213; del buf213  # reuse
        buf226 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_40.run(buf222, buf223, buf224, buf225, buf226, 32, 8, stream=stream0)
        buf229 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_43.run(buf192, arg60_1, buf218, arg68_1, buf225, buf226, arg69_1, arg70_1, buf229, 2097152, stream=stream0)
        del arg69_1
        del arg70_1
        buf230 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_56, hidden_states_57], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_34.run(arg71_1, buf230, 262144, 9, stream=stream0)
        del arg71_1
        # Topologically Sorted Source Nodes: [hidden_states_56, hidden_states_57], Original ATen: [aten.silu, aten.convolution]
        buf231 = extern_kernels.convolution(buf229, buf230, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (1, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf232 = buf221; del buf221  # reuse
        buf233 = buf220; del buf220  # reuse
        buf234 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_58], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_38.run(buf231, arg72_1, buf232, buf233, buf234, 16384, 128, stream=stream0)
        buf235 = buf224; del buf224  # reuse
        buf236 = buf223; del buf223  # reuse
        buf237 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_58], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_39.run(buf232, buf233, buf234, buf235, buf236, buf237, 256, 64, stream=stream0)
        buf238 = buf226; del buf226  # reuse
        buf239 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_58], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_40.run(buf235, buf236, buf237, buf238, buf239, 32, 8, stream=stream0)
        buf242 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_58, hidden_states_59], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_41.run(buf231, arg72_1, buf238, buf239, arg73_1, arg74_1, buf242, 2097152, stream=stream0)
        del arg72_1
        del arg73_1
        del arg74_1
        del buf231
        buf243 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_59, hidden_states_61], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_34.run(arg75_1, buf243, 262144, 9, stream=stream0)
        del arg75_1
        # Topologically Sorted Source Nodes: [hidden_states_59, hidden_states_61], Original ATen: [aten.silu, aten.convolution]
        buf244 = extern_kernels.convolution(buf242, buf243, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (1, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf245 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_36, hidden_states_38, add_4, output_tensor_4, hidden_states_43, hidden_states_45, add_5, output_tensor_5, hidden_states_46, hidden_states_47, hidden_states_52, hidden_states_54, add_6, output_tensor_6, hidden_states_59, hidden_states_61, add_7, output_tensor_7], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_div_silu_44.run(buf245, arg60_1, buf218, arg68_1, buf244, arg76_1, 2097152, stream=stream0)
        del arg60_1
        del arg68_1
        del arg76_1
        buf246 = buf234; del buf234  # reuse
        buf247 = buf233; del buf233  # reuse
        buf248 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_62], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_45.run(buf245, buf246, buf247, buf248, 16384, 128, stream=stream0)
        buf249 = buf237; del buf237  # reuse
        buf250 = buf236; del buf236  # reuse
        buf251 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_62], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_39.run(buf246, buf247, buf248, buf249, buf250, buf251, 256, 64, stream=stream0)
        buf252 = buf239; del buf239  # reuse
        buf253 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_62], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_40.run(buf249, buf250, buf251, buf252, buf253, 32, 8, stream=stream0)
        buf256 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_62, hidden_states_63], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_46.run(buf245, buf252, buf253, arg77_1, arg78_1, buf256, 2097152, stream=stream0)
        del arg77_1
        del arg78_1
        buf257 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_63, hidden_states_64], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_34.run(arg79_1, buf257, 262144, 9, stream=stream0)
        del arg79_1
        # Topologically Sorted Source Nodes: [hidden_states_63, hidden_states_64], Original ATen: [aten.silu, aten.convolution]
        buf258 = extern_kernels.convolution(buf256, buf257, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (1, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf259 = buf248; del buf248  # reuse
        buf260 = buf247; del buf247  # reuse
        buf261 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_65], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_38.run(buf258, arg80_1, buf259, buf260, buf261, 16384, 128, stream=stream0)
        buf262 = buf251; del buf251  # reuse
        buf263 = buf250; del buf250  # reuse
        buf264 = buf249; del buf249  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_65], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_39.run(buf259, buf260, buf261, buf262, buf263, buf264, 256, 64, stream=stream0)
        buf265 = buf253; del buf253  # reuse
        buf266 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_65], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_40.run(buf262, buf263, buf264, buf265, buf266, 32, 8, stream=stream0)
        buf269 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_66], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_41.run(buf258, arg80_1, buf265, buf266, arg81_1, arg82_1, buf269, 2097152, stream=stream0)
        del arg80_1
        del arg81_1
        del arg82_1
        buf270 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_66, hidden_states_68], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_34.run(arg83_1, buf270, 262144, 9, stream=stream0)
        del arg83_1
        # Topologically Sorted Source Nodes: [hidden_states_66, hidden_states_68], Original ATen: [aten.silu, aten.convolution]
        buf271 = extern_kernels.convolution(buf269, buf270, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf271, (1, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf272 = buf261; del buf261  # reuse
        buf273 = buf260; del buf260  # reuse
        buf274 = buf259; del buf259  # reuse
        # Topologically Sorted Source Nodes: [group_norm_18], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_47.run(buf245, buf271, arg84_1, buf272, buf273, buf274, 16384, 128, stream=stream0)
        buf275 = buf264; del buf264  # reuse
        buf276 = buf263; del buf263  # reuse
        buf277 = buf262; del buf262  # reuse
        # Topologically Sorted Source Nodes: [group_norm_18], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_39.run(buf272, buf273, buf274, buf275, buf276, buf277, 256, 64, stream=stream0)
        buf278 = buf266; del buf266  # reuse
        buf279 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [group_norm_18], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_40.run(buf275, buf276, buf277, buf278, buf279, 32, 8, stream=stream0)
        buf281 = reinterpret_tensor(buf269, (1, 512, 4096), (2097152, 4096, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [group_norm_18], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_48.run(buf245, buf271, arg84_1, buf278, buf279, arg85_1, arg86_1, buf281, 4096, 512, stream=stream0)
        del arg85_1
        del arg86_1
        buf282 = reinterpret_tensor(buf258, (4096, 512), (512, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [query], Original ATen: [aten.bmm]
        extern_kernels.mm(reinterpret_tensor(buf281, (4096, 512), (1, 4096), 0), reinterpret_tensor(arg87_1, (512, 512), (1, 512), 0), out=buf282)
        del arg87_1
        buf283 = reinterpret_tensor(buf218, (4096, 512), (512, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [key], Original ATen: [aten.bmm]
        extern_kernels.mm(reinterpret_tensor(buf281, (4096, 512), (1, 4096), 0), reinterpret_tensor(arg89_1, (512, 512), (1, 512), 0), out=buf283)
        del arg89_1
        buf284 = reinterpret_tensor(buf242, (4096, 512), (512, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [value], Original ATen: [aten.bmm]
        extern_kernels.mm(reinterpret_tensor(buf281, (4096, 512), (1, 4096), 0), reinterpret_tensor(arg91_1, (512, 512), (1, 512), 0), out=buf284)
        del arg91_1
        del buf281
        buf285 = reinterpret_tensor(buf282, (1, 1, 4096, 512), (2097152, 2097152, 512, 1), 0); del buf282  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_71], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_49.run(buf285, arg88_1, 2097152, stream=stream0)
        del arg88_1
        buf286 = reinterpret_tensor(buf283, (1, 1, 4096, 512), (2097152, 2097152, 512, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_71], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_49.run(buf286, arg90_1, 2097152, stream=stream0)
        del arg90_1
        buf287 = reinterpret_tensor(buf284, (1, 1, 4096, 512), (2097152, 2097152, 512, 1), 0); del buf284  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_71], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_49.run(buf287, arg92_1, 2097152, stream=stream0)
        del arg92_1
        # Topologically Sorted Source Nodes: [hidden_states_71], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf288 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf285, buf286, buf287, None, False)
        del buf285
        del buf286
        buf289 = buf288[0]
        assert_size_stride(buf289, (1, 1, 4096, 512), (2097152, 512, 512, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf289, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf288
        buf293 = reinterpret_tensor(buf287, (4096, 512), (512, 1), 0); del buf287  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_74], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf289, (4096, 512), (512, 1), 0), reinterpret_tensor(arg93_1, (512, 512), (1, 512), 0), out=buf293)
        del arg93_1
        del buf289
        buf294 = reinterpret_tensor(buf293, (1, 512, 64, 64), (2097152, 1, 32768, 512), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_66, hidden_states_68, add_8, output_tensor_8, hidden_states_77, hidden_states_78], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_silu_50.run(buf294, arg94_1, buf245, buf271, arg84_1, 2097152, stream=stream0)
        del arg84_1
        del arg94_1
        del buf245
        buf295 = buf274; del buf274  # reuse
        buf296 = buf273; del buf273  # reuse
        buf297 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_79], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_45.run(buf294, buf295, buf296, buf297, 16384, 128, stream=stream0)
        buf298 = buf277; del buf277  # reuse
        buf299 = buf276; del buf276  # reuse
        buf300 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_79], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_39.run(buf295, buf296, buf297, buf298, buf299, buf300, 256, 64, stream=stream0)
        buf301 = buf279; del buf279  # reuse
        buf302 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_79], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_40.run(buf298, buf299, buf300, buf301, buf302, 32, 8, stream=stream0)
        buf305 = buf271; del buf271  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_79, hidden_states_80], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_46.run(buf294, buf301, buf302, arg95_1, arg96_1, buf305, 2097152, stream=stream0)
        del arg95_1
        del arg96_1
        buf306 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_80, hidden_states_81], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_34.run(arg97_1, buf306, 262144, 9, stream=stream0)
        del arg97_1
        # Topologically Sorted Source Nodes: [hidden_states_80, hidden_states_81], Original ATen: [aten.silu, aten.convolution]
        buf307 = extern_kernels.convolution(buf305, buf306, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf307, (1, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf308 = buf297; del buf297  # reuse
        buf309 = buf296; del buf296  # reuse
        buf310 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_82], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_38.run(buf307, arg98_1, buf308, buf309, buf310, 16384, 128, stream=stream0)
        buf311 = buf300; del buf300  # reuse
        buf312 = buf299; del buf299  # reuse
        buf313 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_82], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_39.run(buf308, buf309, buf310, buf311, buf312, buf313, 256, 64, stream=stream0)
        buf314 = buf302; del buf302  # reuse
        buf315 = buf301; del buf301  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_82], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_40.run(buf311, buf312, buf313, buf314, buf315, 32, 8, stream=stream0)
        buf318 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_82, hidden_states_83], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_41.run(buf307, arg98_1, buf314, buf315, arg99_1, arg100_1, buf318, 2097152, stream=stream0)
        del arg100_1
        del arg98_1
        del arg99_1
        del buf307
        buf319 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_83, hidden_states_85], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_34.run(arg101_1, buf319, 262144, 9, stream=stream0)
        del arg101_1
        # Topologically Sorted Source Nodes: [hidden_states_83, hidden_states_85], Original ATen: [aten.silu, aten.convolution]
        buf320 = extern_kernels.convolution(buf318, buf319, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (1, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        del buf319
        buf321 = buf310; del buf310  # reuse
        buf322 = buf309; del buf309  # reuse
        buf323 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [sample_1], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_47.run(buf294, buf320, arg102_1, buf321, buf322, buf323, 16384, 128, stream=stream0)
        buf324 = buf313; del buf313  # reuse
        buf325 = buf312; del buf312  # reuse
        buf326 = buf311; del buf311  # reuse
        # Topologically Sorted Source Nodes: [sample_1], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_39.run(buf321, buf322, buf323, buf324, buf325, buf326, 256, 64, stream=stream0)
        del buf321
        del buf322
        del buf323
        buf327 = buf315; del buf315  # reuse
        buf328 = buf314; del buf314  # reuse
        # Topologically Sorted Source Nodes: [sample_1], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_40.run(buf324, buf325, buf326, buf327, buf328, 32, 8, stream=stream0)
        del buf324
        del buf325
        del buf326
        buf331 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [sample_1, sample_2], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_51.run(buf294, buf320, arg102_1, buf327, buf328, arg103_1, arg104_1, buf331, 2097152, stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del buf294
        del buf320
        del buf327
        del buf328
        buf332 = empty_strided_cuda((8, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [sample_2, sample_3], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_52.run(arg105_1, buf332, 4096, 9, stream=stream0)
        del arg105_1
        # Topologically Sorted Source Nodes: [sample_2, sample_3], Original ATen: [aten.silu, aten.convolution]
        buf333 = extern_kernels.convolution(buf331, buf332, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (1, 8, 64, 64), (32768, 1, 512, 8), 'torch.ops.aten.convolution.default')
        del buf331
        del buf332
        buf334 = buf333; del buf333  # reuse
        # Topologically Sorted Source Nodes: [sample_2, sample_3], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_53.run(buf334, arg106_1, 32768, stream=stream0)
        del arg106_1
        # Topologically Sorted Source Nodes: [sample_2, sample_3, enc], Original ATen: [aten.silu, aten.convolution]
        buf335 = extern_kernels.convolution(buf334, arg107_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf335, (1, 8, 64, 64), (32768, 1, 512, 8), 'torch.ops.aten.convolution.default')
        del arg107_1
        buf336 = reinterpret_tensor(buf334, (1, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf334  # reuse
        # Topologically Sorted Source Nodes: [sample_2, sample_3, enc], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_54.run(buf335, arg108_1, buf336, 8, 4096, stream=stream0)
        del arg108_1
        del buf335
        buf337 = empty_strided_cuda((1, 4, 64, 64), (16384, 4096, 64, 1), torch.bfloat16)
        buf338 = empty_strided_cuda((1, 4, 64, 64), (16384, 4096, 64, 1), torch.bfloat16)
        buf339 = empty_strided_cuda((1, 4, 64, 64), (16384, 4096, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [clamp, exp_1, mul, exp], Original ATen: [aten.clamp, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_exp_mul_55.run(buf336, buf337, buf338, buf339, 16384, stream=stream0)
    return (buf336, buf338, buf339, buf337, reinterpret_tensor(buf336, (1, 4, 64, 64), (32768, 4096, 64, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 3, 512, 512), (786432, 262144, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((128, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg15_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg16_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg17_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg18_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg19_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg21_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg22_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg23_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg24_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg25_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg26_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg27_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg28_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg29_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg30_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg31_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg33_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg35_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg36_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg37_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg39_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg41_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg42_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg43_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg44_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg45_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg46_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg47_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg48_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg49_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg50_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg51_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg52_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg53_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg54_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg55_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg56_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg57_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg58_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg59_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg60_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg61_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg62_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg63_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg64_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg65_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg66_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg67_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg71_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg72_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg73_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg75_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg76_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg77_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg78_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg79_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg80_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg81_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg82_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg83_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg84_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg85_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg86_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg87_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg88_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg89_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg90_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg91_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg92_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg93_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg94_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg95_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg96_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg97_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg98_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg99_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg100_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg101_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg102_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg103_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg104_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg105_1 = rand_strided((8, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg106_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg107_1 = rand_strided((8, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg108_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
