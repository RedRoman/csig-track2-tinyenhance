# AOT ID: ['2_inference']
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/bi/cbiligxfpc2d42rlxtljkuwvdzxrtmuj6x2ujh2paktxpsj2iwie.py
# Topologically Sorted Source Nodes: [z], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   z => convolution
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 32768, 'x': 32768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 4096*y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 4*x1), tmp0, ymask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/u4/cu4pzkwls7ubji5boeyzncfbsrxu3z7bqro5vcv5umwv5mxyokm3.py
# Topologically Sorted Source Nodes: [z], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   z => convolution
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 98312}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/na/cnan7nzk3jqqtm6aiw4r3wntr3bxxattl4i7j54teqssvjvd6733.py
# Topologically Sorted Source Nodes: [z, sample], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   sample => convolution_1
#   z => convolution
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 73728, 'x': 36864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 4*x2 + 36*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/pz/cpzayxwvbepls55753vvonyudlhi2gddqbaal5avgi3tslfnhp3q.py
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
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 5, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 393216, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/pt/cptjm2eoyvjcam7khzlndp4crj7vl27gogkbmqi4fismtiq2vgfs.py
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
    size_hints={'x': 256, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 202752, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/di/cdi7oq2x2ya5mv3ju5b3tctffg4zsdty2g5m7ist7xnk3aogonqd.py
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
    size_hints={'x': 32, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 512, 'r0_': 3072}}
)
@triton.jit
def triton_per_fused_native_group_norm_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/ef/cefvqu3c4pmdqvrfsb65e57oqhhn3dn34siegjytvmgykcvrgd33.py
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
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12585984}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/oh/cohhxagcfybsshyof6jrlx7xnaxno65jqdjlxdx3woml6dypldak.py
# Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_1 => convert_element_type_5, mul_2, sigmoid
#   hidden_states_2 => convolution_2
# Graph fragment:
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_1,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %sigmoid), kwargs = {})
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_5, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_7 = async_compile.triton('triton_poi_fused_convolution_silu_7', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 9437184, 'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/vd/cvdxkgx7snmc7awa23ydp66i57csrz62bcbizep5i5czku6zvc7q.py
# Topologically Sorted Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_2 => var_mean_2
# Graph fragment:
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_5, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_8 = async_compile.triton('triton_per_fused_native_group_norm_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 5, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 393216, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/sc/csc2c2zq34vgcsk5co7qrhav6mqqzwmlikkplc2feosf3z44tkbf.py
# Topologically Sorted Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_2 => add_6, convert_element_type_13, mul_7
# Graph fragment:
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %unsqueeze_13), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_15), kwargs = {})
#   %convert_element_type_13 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_9 = async_compile.triton('triton_poi_fused_native_group_norm_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 8388608, 'x': 8392704}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x1 + 512*y0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x1 // 16), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x1 // 16), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp22 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
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
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 4096*x1), tmp25, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/ux/cuxkxsqouruige6w7fl5zazgfcxyot6acprk6tpc2tltxmipg2pb.py
# Topologically Sorted Source Nodes: [hidden_states_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   hidden_states_9 => _scaled_dot_product_efficient_attention
# Graph fragment:
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_6, %permute_7, %permute_8, None, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_10 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_10', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12583936}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/m6/cm6hjt4hxob5imjvpbeenwanabiubygpd76nmgnhoupmk3ehzczc.py
# Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, hidden_states_15, hidden_states_16], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div]
# Source node to ATen node mapping:
#   add => add_4
#   hidden_states_15 => add_10
#   hidden_states_16 => div_1
#   hidden_states_4 => convert_element_type_11, mul_5, sigmoid_1
#   hidden_states_6 => convolution_3
#   output_tensor => div
#   sample => convolution_1
#   z => convolution
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_1 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %sigmoid_1), kwargs = {})
#   %convert_element_type_11 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5, torch.bfloat16), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_11, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %convolution_3), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_4, 1), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_22, %div), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_10, 1), kwargs = {})
triton_poi_fused_add_convolution_div_silu_11 = async_compile.triton('triton_poi_fused_add_convolution_div_silu_11', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_silu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20974592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_silu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp6 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp2 + tmp11
    tmp13 = tmp12 * tmp10
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/zp/czpschadkhomezrnsu6dd4jwl2x3q2jjvvzq5hqdqedrspkg2gwt.py
# Topologically Sorted Source Nodes: [hidden_states_17], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_17 => var_mean_3
# Graph fragment:
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_23, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_12 = async_compile.triton('triton_per_fused_native_group_norm_12', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 5, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 393216, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_12(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/3k/c3k54354xngudf6qinmn54a3irxlsmu6kyxouzh3gs3hjpfmthiq.py
# Topologically Sorted Source Nodes: [hidden_states_17, hidden_states_18], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_17 => add_12, mul_9
#   hidden_states_18 => convert_element_type_30, mul_10, sigmoid_2
# Graph fragment:
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_24, %unsqueeze_18), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %unsqueeze_21), kwargs = {})
#   %sigmoid_2 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_12,), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, %sigmoid_2), kwargs = {})
#   %convert_element_type_30 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_13 = async_compile.triton('triton_poi_fused_native_group_norm_silu_13', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12584960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/3v/c3vwxa7r7it5ycvt5pzie3tvwadh5dspitaewbrwegi4uui5nklt.py
# Topologically Sorted Source Nodes: [hidden_states_24], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_24 => var_mean_5
# Graph fragment:
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_27, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_14 = async_compile.triton('triton_per_fused_native_group_norm_14', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 5, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 393216, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_14(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/fg/cfghsln6udijtrwaw4mkt4e2ywaa5hhglmuckgs755fqhlf5zrzo.py
# Topologically Sorted Source Nodes: [hidden_states_24, hidden_states_25], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_24 => add_17, mul_15
#   hidden_states_25 => convert_element_type_42, mul_16, sigmoid_4
# Graph fragment:
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_28, %unsqueeze_30), kwargs = {})
#   %add_17 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %unsqueeze_33), kwargs = {})
#   %sigmoid_4 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_4), kwargs = {})
#   %convert_element_type_42 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_16, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_15 = async_compile.triton('triton_poi_fused_native_group_norm_silu_15', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16780288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/dx/cdxi2v4d74uzasb35adht4t4rcb44qi54a224bmqf2t2jd4b66lm.py
# Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, hidden_states_28, hidden_states_30, add_3, output_tensor_2], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div]
# Source node to ATen node mapping:
#   add_2 => add_15
#   add_3 => add_20
#   hidden_states_21 => convert_element_type_36, mul_13, sigmoid_3
#   hidden_states_23 => convolution_5
#   hidden_states_28 => convert_element_type_48, mul_19, sigmoid_5
#   hidden_states_30 => convolution_7
#   output_tensor_1 => div_2
#   output_tensor_2 => div_3
# Graph fragment:
#   %sigmoid_3 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_14,), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_14, %sigmoid_3), kwargs = {})
#   %convert_element_type_36 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_13, torch.bfloat16), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_36, %arg30_1, %arg31_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_1, %convolution_5), kwargs = {})
#   %div_2 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_15, 1), kwargs = {})
#   %sigmoid_5 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_19,), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, %sigmoid_5), kwargs = {})
#   %convert_element_type_48 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_19, torch.bfloat16), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_48, %arg37_1, %arg38_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_2, %convolution_7), kwargs = {})
#   %div_3 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_20, 1.0), kwargs = {})
triton_poi_fused_add_convolution_div_silu_16 = async_compile.triton('triton_poi_fused_add_convolution_div_silu_16', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_silu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20973568}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_silu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tmp10 * tmp5
    tl.store(in_out_ptr0 + (x2), tmp11, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/kz/ckzgeqr2hnq5vri62rtkobabhuxkbu7dxyj36uhx2wiaq2rfuhw4.py
# Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten._unsafe_index, aten.clone]
# Source node to ATen node mapping:
#   add_4 => add_25
#   add_5 => add_30
#   hidden_states_35 => convert_element_type_60, mul_25, sigmoid_7
#   hidden_states_37 => convolution_9
#   hidden_states_42 => convert_element_type_72, mul_31, sigmoid_9
#   hidden_states_44 => convolution_11
#   hidden_states_45 => _unsafe_index, clone_10, convert_element_type_73, convert_element_type_78
#   output_tensor_3 => div_4
#   output_tensor_4 => div_5
# Graph fragment:
#   %sigmoid_7 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_24,), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, %sigmoid_7), kwargs = {})
#   %convert_element_type_60 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_25, torch.bfloat16), kwargs = {})
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_60, %arg45_1, %arg46_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, %convolution_9), kwargs = {})
#   %div_4 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_25, 1.0), kwargs = {})
#   %sigmoid_9 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_29,), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, %sigmoid_9), kwargs = {})
#   %convert_element_type_72 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_31, torch.bfloat16), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_72, %arg53_1, %arg54_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_4, %convolution_11), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_30, 1.0), kwargs = {})
#   %convert_element_type_73 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_5, torch.float32), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_73, [None, None, %unsqueeze_64, %convert_element_type_77]), kwargs = {})
#   %clone_10 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index,), kwargs = {memory_format: torch.channels_last})
#   %convert_element_type_78 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_10, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_17 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 16384}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    x2 = xindex // 128
    x1 = (xindex % 128)
    y0 = yindex
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (y0 + 512*tmp8 + 32768*tmp4), ymask).to(tl.float32)
    tmp10 = tl.load(in_ptr1 + (y0 + 512*tmp8 + 32768*tmp4), ymask).to(tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tl.load(in_ptr3 + (y0 + 512*tmp8 + 32768*tmp4), ymask).to(tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tmp19 * tmp14
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr1 + (y0 + 512*x3), tmp22, ymask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/re/cree6i26wevx45q6up3snaxmuxbo6a57fbmepu7gd7ves6t6eexk.py
# Topologically Sorted Source Nodes: [hidden_states_47], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_47 => var_mean_11
# Graph fragment:
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_39, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_18 = async_compile.triton('triton_per_fused_native_group_norm_18', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 5, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 35127296, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_18(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/ty/ctysubikn63ustpc2y6ycavqzv4p7fp76iesywalx4stcy37qgua.py
# Topologically Sorted Source Nodes: [hidden_states_47], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_47 => var_mean_11
# Graph fragment:
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_39, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_19 = async_compile.triton('triton_per_fused_native_group_norm_19', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 798720, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_19(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/gv/cgvdcsxd2pmtntkn36z7v5cu4wto3zbk3cixwupdbqeqzyuuf5cp.py
# Topologically Sorted Source Nodes: [hidden_states_47], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_47 => var_mean_11
# Graph fragment:
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_39, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_20 = async_compile.triton('triton_per_fused_native_group_norm_20', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 512, 'r0_': 6144}}
)
@triton.jit
def triton_per_fused_native_group_norm_20(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/oh/coh67rtlr5liwoxggjeabcu3ymrnlp5fo3rusypwscta6ghdg2fl.py
# Topologically Sorted Source Nodes: [hidden_states_47, hidden_states_48], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_47 => add_36, mul_37
#   hidden_states_48 => convert_element_type_84, mul_38, sigmoid_10
# Graph fragment:
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_40, %unsqueeze_67), kwargs = {})
#   %add_36 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_37, %unsqueeze_70), kwargs = {})
#   %sigmoid_10 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_36,), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_36, %sigmoid_10), kwargs = {})
#   %convert_element_type_84 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_38, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_21 = async_compile.triton('triton_poi_fused_native_group_norm_silu_21', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50334720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/6z/c6zbk7j6zk6w4wjatjdhbprcn76567a64lemolhyqiq5mv3l7las.py
# Topologically Sorted Source Nodes: [hidden_states_54], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_54 => var_mean_13
# Graph fragment:
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_43, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_22 = async_compile.triton('triton_per_fused_native_group_norm_22', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 5, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 68681728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/zz/czz7n2sgwtghkicsvsleype4ufmmwvy4ufy4oplxwiyjd5hpwnhw.py
# Topologically Sorted Source Nodes: [hidden_states_54, hidden_states_55], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_54 => add_41, mul_43
#   hidden_states_55 => convert_element_type_96, mul_44, sigmoid_12
# Graph fragment:
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_44, %unsqueeze_79), kwargs = {})
#   %add_41 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %unsqueeze_82), kwargs = {})
#   %sigmoid_12 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_41,), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_41, %sigmoid_12), kwargs = {})
#   %convert_element_type_96 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_44, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_23 = async_compile.triton('triton_poi_fused_native_group_norm_silu_23', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 67112960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/2q/c2qikkp5rea7fa4peiojh3pnnhiss5milxzeu3lz57c7hz7d6tgw.py
# Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_58, hidden_states_60, add_7, output_tensor_6], Original ATen: [aten.clone, aten._to_copy, aten.convolution, aten.silu, aten.add, aten.div]
# Source node to ATen node mapping:
#   add_6 => add_39
#   add_7 => add_44
#   hidden_states_45 => clone_10, convert_element_type_78
#   hidden_states_46 => convolution_12
#   hidden_states_51 => convert_element_type_90, mul_41, sigmoid_11
#   hidden_states_53 => convolution_14
#   hidden_states_58 => convert_element_type_102, mul_47, sigmoid_13
#   hidden_states_60 => convolution_16
#   output_tensor_5 => div_6
#   output_tensor_6 => div_7
# Graph fragment:
#   %clone_10 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index,), kwargs = {memory_format: torch.channels_last})
#   %convert_element_type_78 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_10, torch.bfloat16), kwargs = {})
#   %convolution_12 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_78, %arg55_1, %arg56_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_11 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_38,), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, %sigmoid_11), kwargs = {})
#   %convert_element_type_90 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_41, torch.bfloat16), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_90, %arg63_1, %arg64_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_14), kwargs = {})
#   %div_6 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_39, 1.0), kwargs = {})
#   %sigmoid_13 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_43,), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %sigmoid_13), kwargs = {})
#   %convert_element_type_102 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_47, torch.bfloat16), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_102, %arg71_1, %arg72_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_6, %convolution_16), kwargs = {})
#   %div_7 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_44, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_clone_convolution_div_silu_24 = async_compile.triton('triton_poi_fused__to_copy_add_clone_convolution_div_silu_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_convolution_div_silu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 83889152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_convolution_div_silu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/ac/cacyqdfq2hxbzra24ougtdztjfsrfvjd6qjcnl6asenfyvz7dli7.py
# Topologically Sorted Source Nodes: [hidden_states_61], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_61 => var_mean_15
# Graph fragment:
#   %var_mean_15 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_47, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_25 = async_compile.triton('triton_per_fused_native_group_norm_25', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 5, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18350080, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_25(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/ao/caom62sfzs3mxct6f4be26kepnrfa7ekj7eny4bbimqulqdx4igh.py
# Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_62], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_61 => add_46, mul_49
#   hidden_states_62 => convert_element_type_108, mul_50, sigmoid_14
# Graph fragment:
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_48, %unsqueeze_91), kwargs = {})
#   %add_46 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, %unsqueeze_94), kwargs = {})
#   %sigmoid_14 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_46,), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_46, %sigmoid_14), kwargs = {})
#   %convert_element_type_108 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_50, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_26 = async_compile.triton('triton_poi_fused_native_group_norm_silu_26', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50333696}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/m5/cm52ojlpedz4wc257jikptizo2uhp532pbmzbxj4fnr2sqte4tx2.py
# Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten._unsafe_index, aten.clone]
# Source node to ATen node mapping:
#   add_8 => add_49
#   hidden_states_65 => convert_element_type_114, mul_53, sigmoid_15
#   hidden_states_67 => convolution_18
#   hidden_states_68 => _unsafe_index_1, clone_17, convert_element_type_115, convert_element_type_120
#   output_tensor_7 => div_8
# Graph fragment:
#   %sigmoid_15 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_48,), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %sigmoid_15), kwargs = {})
#   %convert_element_type_114 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_53, torch.bfloat16), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_114, %arg79_1, %arg80_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_7, %convolution_18), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_49, 1.0), kwargs = {})
#   %convert_element_type_115 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_8, torch.float32), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_115, [None, None, %unsqueeze_101, %convert_element_type_119]), kwargs = {})
#   %clone_17 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index_1,), kwargs = {memory_format: torch.channels_last})
#   %convert_element_type_120 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_17, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_27 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_27(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex // 131072
    x1 = ((xindex // 512) % 256)
    x0 = (xindex % 512)
    x4 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (x0 + 512*tmp8 + 65536*tmp4), None).to(tl.float32)
    tmp10 = tl.load(in_ptr1 + (x0 + 512*tmp8 + 65536*tmp4), None).to(tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp17, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/eb/cebtp7mekxqkul2qna5k75gunuo3v5l7hlfxz2iqfbotucuyppqo.py
# Topologically Sorted Source Nodes: [hidden_states_70], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_70 => var_mean_17
# Graph fragment:
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_51, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_28 = async_compile.triton('triton_per_fused_native_group_norm_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'y': 512, 'x': 256, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 67156992, 'x': 2322432, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_28(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, r0_numel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 216
    r0_numel = 347
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x2 = xindex
    y0 = (yindex % 14)
    y1 = yindex // 14
    tmp0 = r0_3 + 347*x2
    tmp1 = tl.full([1, 1, 1], 74899, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r0_3 + 347*x2 + 74899*y0
    tmp4 = tl.full([1, 1, 1], 1048576, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + (16*y1 + 512*(((r0_3 + 347*x2 + 74899*y0) % 65536)) + ((((r0_3 + 347*x2 + 74899*y0) // 65536) % 16))), r0_mask & tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (16*y1 + ((((r0_3 + 347*x2 + 74899*y0) // 65536) % 16))), r0_mask & tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
    tmp25 = tl.broadcast_to(tmp14, [YBLOCK, XBLOCK, R0_BLOCK])
    tmp26 = tl.broadcast_to(tmp19, [YBLOCK, XBLOCK, R0_BLOCK])
    tmp27 = tl.broadcast_to(tmp24, [YBLOCK, XBLOCK, R0_BLOCK])
    tmp29 = tl.where(r0_mask & xmask & ymask, tmp25, 0)
    tmp30 = tl.where(r0_mask & xmask & ymask, tmp26, 0)
    tmp31 = tl.where(r0_mask & xmask & ymask, tmp27, 0)
    tmp32, tmp33, tmp34 = triton_helpers.welford(tmp29, tmp30, tmp31, 2)
    tmp35 = tmp32[:, :, None]
    tmp36 = tmp33[:, :, None]
    tmp37 = tmp34[:, :, None]
    tl.store(out_ptr0 + (x2 + 216*y0 + 3040*y1), tmp35, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 216*y0 + 3040*y1), tmp36, xmask & ymask)
    tl.store(out_ptr2 + (x2 + 216*y0 + 3040*y1), tmp37, xmask & ymask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/ry/cryhxujoqm4rg7s42pfgm3b5sdk4uk6tzx7nsiwbbqxrsparhsgp.py
# Topologically Sorted Source Nodes: [hidden_states_70], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_70 => var_mean_17
# Graph fragment:
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_51, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_29 = async_compile.triton('triton_per_fused_native_group_norm_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10752, 'r0_': 1161216}}
)
@triton.jit
def triton_per_fused_native_group_norm_29(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 448
    r0_numel = 216
    R0_BLOCK: tl.constexpr = 256
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
    x0 = (xindex % 14)
    x1 = xindex // 14
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_2 + 216*x0 + 3040*x1), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 216*x0 + 3040*x1), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_2 + 216*x0 + 3040*x1), r0_mask & xmask, other=0.0)
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/pd/cpdukruq3pzglyaphs5arp4uv5hc62kmq4byflxma4yly3tji5a4.py
# Topologically Sorted Source Nodes: [hidden_states_70], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_70 => var_mean_17
# Graph fragment:
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_51, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_30 = async_compile.triton('triton_per_fused_native_group_norm_30', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 512, 'r0_': 5376}}
)
@triton.jit
def triton_per_fused_native_group_norm_30(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 14*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 14*x0), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 14*x0), r0_mask & xmask, other=0.0)
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/gf/cgfw2dxok7nl3lr7txfviqc4ts3y22hygbfrevpr44hth4hftzkg.py
# Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, hidden_states_70, hidden_states_71, contiguous], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten._unsafe_index, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_8 => add_49
#   contiguous => clone_20
#   hidden_states_65 => convert_element_type_114, mul_53, sigmoid_15
#   hidden_states_67 => convolution_18
#   hidden_states_68 => _unsafe_index_1, clone_17, convert_element_type_115, convert_element_type_120
#   hidden_states_69 => convolution_19
#   hidden_states_70 => add_55, mul_59
#   hidden_states_71 => convert_element_type_126, mul_60, sigmoid_16
#   output_tensor_7 => div_8
# Graph fragment:
#   %sigmoid_15 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_48,), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %sigmoid_15), kwargs = {})
#   %convert_element_type_114 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_53, torch.bfloat16), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_114, %arg79_1, %arg80_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_7, %convolution_18), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_49, 1.0), kwargs = {})
#   %convert_element_type_115 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_8, torch.float32), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_115, [None, None, %unsqueeze_101, %convert_element_type_119]), kwargs = {})
#   %clone_17 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index_1,), kwargs = {memory_format: torch.channels_last})
#   %convert_element_type_120 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_17, torch.bfloat16), kwargs = {})
#   %convolution_19 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_120, %arg81_1, %arg82_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_52, %unsqueeze_104), kwargs = {})
#   %add_55 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_59, %unsqueeze_107), kwargs = {})
#   %sigmoid_16 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_55,), kwargs = {})
#   %mul_60 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, %sigmoid_16), kwargs = {})
#   %convert_element_type_126 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_60, torch.bfloat16), kwargs = {})
#   %clone_20 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_19,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_native_group_norm_silu_31 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_native_group_norm_silu_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_native_group_norm_silu_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 335547392}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_native_group_norm_silu_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
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
    tl.store(out_ptr2 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/i2/ci2rox5sawpz6zqzxouhvo5w3zt3rklxjzjuk4f5iyjro3hc4kji.py
# Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_71 => convert_element_type_126, mul_60, sigmoid_16
#   hidden_states_72 => convolution_20
# Graph fragment:
#   %sigmoid_16 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_55,), kwargs = {})
#   %mul_60 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, %sigmoid_16), kwargs = {})
#   %convert_element_type_126 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_60, torch.bfloat16), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_126, %arg85_1, %arg86_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_32 = async_compile.triton('triton_poi_fused_convolution_silu_32', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 4718592, 'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_32(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/ry/cryc2xilwiqbkvma25wathvxeb3k3c2lwmdctqotv67xmcb2lz32.py
# Topologically Sorted Source Nodes: [hidden_states_73], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_73 => var_mean_18
# Graph fragment:
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_53, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_33 = async_compile.triton('triton_per_fused_native_group_norm_33', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 5, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_33(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/vb/cvbe7fyrlvj2nx23b2jdeftajnvebob6fffsxj7djpruoriglyti.py
# Topologically Sorted Source Nodes: [hidden_states_73], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_73 => var_mean_18
# Graph fragment:
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_53, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_34 = async_compile.triton('triton_per_fused_native_group_norm_34', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1597440, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_34(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/c7/cc7rdh2vg6pl2jmyyyzru27erbpv3kqm37cuwa2tutq65a33ivgw.py
# Topologically Sorted Source Nodes: [hidden_states_73], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_73 => var_mean_18
# Graph fragment:
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_53, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_35 = async_compile.triton('triton_per_fused_native_group_norm_35', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24576, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_35(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/zx/czxreefq5isvf2a6eds6p77hclq64vidtzjxxgejdob3ms7p6oje.py
# Topologically Sorted Source Nodes: [hidden_states_73], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_73 => var_mean_18
# Graph fragment:
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_53, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_36 = async_compile.triton('triton_per_fused_native_group_norm_36', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6656, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_36(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/qp/cqpdvfkk4fs4pej3bppptiiudpqr67tjpv5ywuttdxn6nkqe53yl.py
# Topologically Sorted Source Nodes: [hidden_states_73, hidden_states_74], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_73 => add_57, mul_62
#   hidden_states_74 => convert_element_type_132, mul_63, sigmoid_17
# Graph fragment:
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_54, %unsqueeze_110), kwargs = {})
#   %add_57 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_113), kwargs = {})
#   %sigmoid_17 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_17), kwargs = {})
#   %convert_element_type_132 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_63, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_37 = async_compile.triton('triton_poi_fused_native_group_norm_silu_37', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 100664832}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/o5/co5nsnrfdysr4gt4siaj65bxmprgh2dzob32wexrfzyhcru6xzoo.py
# Topologically Sorted Source Nodes: [hidden_states_74, hidden_states_76], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_74 => convert_element_type_132, mul_63, sigmoid_17
#   hidden_states_76 => convolution_21
# Graph fragment:
#   %sigmoid_17 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_17), kwargs = {})
#   %convert_element_type_132 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_63, torch.bfloat16), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_132, %arg89_1, %arg90_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_38 = async_compile.triton('triton_poi_fused_convolution_silu_38', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 2359296, 'x': 1179648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_38(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/wl/cwlgucitr67p6krdsa6yh6wtdnoykrzhrw2e2qft4qhle4du7pej.py
# Topologically Sorted Source Nodes: [hidden_states_77], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_77 => var_mean_19
# Graph fragment:
#   %var_mean_19 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_55, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_39 = async_compile.triton('triton_per_fused_native_group_norm_39', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 5, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/i4/ci4lbcputhpcoo4oomkpea76aw2jpp5igy55r7bqbazozjlvnlrj.py
# Topologically Sorted Source Nodes: [hidden_states_77, hidden_states_78], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_77 => add_60, mul_65
#   hidden_states_78 => convert_element_type_138, mul_66, sigmoid_18
# Graph fragment:
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_56, %unsqueeze_116), kwargs = {})
#   %add_60 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_65, %unsqueeze_119), kwargs = {})
#   %sigmoid_18 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_60,), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_60, %sigmoid_18), kwargs = {})
#   %convert_element_type_138 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_66, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_40 = async_compile.triton('triton_poi_fused_native_group_norm_silu_40', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 134219776}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/pj/cpj3hu7v2qd533cghfdyb7kqg27ocbcks66zifh3u4roa2cm7526.py
# Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, contiguous, input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_81, hidden_states_83, add_10, output_tensor_9], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten._unsafe_index, aten.clone]
# Source node to ATen node mapping:
#   add_10 => add_63
#   add_8 => add_49
#   add_9 => add_58
#   contiguous => clone_20
#   hidden_states_65 => convert_element_type_114, mul_53, sigmoid_15
#   hidden_states_67 => convolution_18
#   hidden_states_68 => _unsafe_index_1, clone_17, convert_element_type_115, convert_element_type_120
#   hidden_states_69 => convolution_19
#   hidden_states_74 => convert_element_type_132, mul_63, sigmoid_17
#   hidden_states_76 => convolution_21
#   hidden_states_81 => convert_element_type_144, mul_69, sigmoid_19
#   hidden_states_83 => convolution_24
#   input_tensor => convolution_22
#   output_tensor_7 => div_8
#   output_tensor_8 => div_9
#   output_tensor_9 => div_10
# Graph fragment:
#   %sigmoid_15 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_48,), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %sigmoid_15), kwargs = {})
#   %convert_element_type_114 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_53, torch.bfloat16), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_114, %arg79_1, %arg80_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_7, %convolution_18), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_49, 1.0), kwargs = {})
#   %convert_element_type_115 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_8, torch.float32), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_115, [None, None, %unsqueeze_101, %convert_element_type_119]), kwargs = {})
#   %clone_17 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index_1,), kwargs = {memory_format: torch.channels_last})
#   %convert_element_type_120 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_17, torch.bfloat16), kwargs = {})
#   %convolution_19 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_120, %arg81_1, %arg82_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_20 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_19,), kwargs = {memory_format: torch.contiguous_format})
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_20, %arg91_1, %arg92_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_17 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_17), kwargs = {})
#   %convert_element_type_132 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_63, torch.bfloat16), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_132, %arg89_1, %arg90_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_58 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_22, %convolution_21), kwargs = {})
#   %div_9 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_58, 1.0), kwargs = {})
#   %sigmoid_19 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_62,), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_62, %sigmoid_19), kwargs = {})
#   %convert_element_type_144 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_69, torch.bfloat16), kwargs = {})
#   %convolution_24 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_144, %arg99_1, %arg100_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_63 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_9, %convolution_24), kwargs = {})
#   %div_10 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_63, 1.0), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_41 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 167773696}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/xz/cxzpr3aoitm6wkri5rj3szkphviakqvgi3c2wgka3d5gkittrc47.py
# Topologically Sorted Source Nodes: [hidden_states_84], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_84 => var_mean_21
# Graph fragment:
#   %var_mean_21 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_59, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_42 = async_compile.triton('triton_per_fused_native_group_norm_42', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 5, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_42(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/4w/c4whepklpfgsedtfmukk4w6mmcrd7qrso62mevlcrnvg22jzvfxg.py
# Topologically Sorted Source Nodes: [hidden_states_84, hidden_states_85], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_84 => add_65, mul_71
#   hidden_states_85 => convert_element_type_150, mul_72, sigmoid_20
# Graph fragment:
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_60, %unsqueeze_128), kwargs = {})
#   %add_65 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_71, %unsqueeze_131), kwargs = {})
#   %sigmoid_20 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_65,), kwargs = {})
#   %mul_72 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_65, %sigmoid_20), kwargs = {})
#   %convert_element_type_150 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_72, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_43 = async_compile.triton('triton_poi_fused_native_group_norm_silu_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 100664320}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
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
    tmp5 = 524288.0
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/ec/cecpk52whxb44txvbgpmsikcoxqaophj4brzpso2yenoqehj5jou.py
# Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten._unsafe_index]
# Source node to ATen node mapping:
#   add_11 => add_68
#   hidden_states_88 => convert_element_type_156, mul_75, sigmoid_21
#   hidden_states_90 => convolution_26
#   hidden_states_91 => _unsafe_index_2, convert_element_type_157, convert_element_type_162
#   output_tensor_10 => div_11
# Graph fragment:
#   %sigmoid_21 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_67,), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_67, %sigmoid_21), kwargs = {})
#   %convert_element_type_156 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_75, torch.bfloat16), kwargs = {})
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_156, %arg107_1, %arg108_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_68 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_10, %convolution_26), kwargs = {})
#   %div_11 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_68, 1.0), kwargs = {})
#   %convert_element_type_157 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_11, torch.float32), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_157, [None, None, %unsqueeze_138, %convert_element_type_161]), kwargs = {})
#   %convert_element_type_162 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_2, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_convolution_div_silu_44 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_convolution_div_silu_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_convolution_div_silu_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_convolution_div_silu_44(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex // 131072
    x1 = ((xindex // 256) % 512)
    x0 = (xindex % 256)
    x4 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (x0 + 256*tmp8 + 65536*tmp4), None).to(tl.float32)
    tmp10 = tl.load(in_ptr1 + (x0 + 256*tmp8 + 65536*tmp4), None).to(tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp17, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/e3/ce3p3hasipxf7q4z2sveqgon643m2lt3oq4crcxrayqdy6ldnejl.py
# Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91, hidden_states_92], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten._unsafe_index]
# Source node to ATen node mapping:
#   add_11 => add_68
#   hidden_states_88 => convert_element_type_156, mul_75, sigmoid_21
#   hidden_states_90 => convolution_26
#   hidden_states_91 => _unsafe_index_2, convert_element_type_157, convert_element_type_162
#   hidden_states_92 => convolution_27
#   output_tensor_10 => div_11
# Graph fragment:
#   %sigmoid_21 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_67,), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_67, %sigmoid_21), kwargs = {})
#   %convert_element_type_156 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_75, torch.bfloat16), kwargs = {})
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_156, %arg107_1, %arg108_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_68 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_10, %convolution_26), kwargs = {})
#   %div_11 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_68, 1.0), kwargs = {})
#   %convert_element_type_157 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_11, torch.float32), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_157, [None, None, %unsqueeze_138, %convert_element_type_161]), kwargs = {})
#   %convert_element_type_162 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_2, torch.bfloat16), kwargs = {})
#   %convolution_27 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_162, %arg109_1, %arg110_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_convolution_div_silu_45 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_convolution_div_silu_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_convolution_div_silu_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 402653696}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_convolution_div_silu_45(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/sq/csqp63drbtxudpkn4xh6eitodbt6utfcn44acpqi3kpgexjewdpx.py
# Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_93 => var_mean_23
# Graph fragment:
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_63, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_46 = async_compile.triton('triton_per_fused_native_group_norm_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2472960, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_46(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
    xnumel = 103040
    XBLOCK: tl.constexpr = 1
    r0_numel = 652
    R0_BLOCK: tl.constexpr = 1024
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
    x1 = ((xindex // 32) % 230)
    x2 = xindex // 7360
    x0 = (xindex % 32)
    x4 = xindex
    tmp0 = r0_3 + 652*x1
    tmp1 = tl.full([1], 149797, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r0_3 + 652*x1 + 149797*x2
    tmp4 = tl.full([1], 2097152, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + (8*x0 + 256*(((r0_3 + 652*x1 + 149797*x2) % 262144)) + ((((r0_3 + 652*x1 + 149797*x2) // 262144) % 8))), r0_mask & tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
    tmp10 = tl.where(tmp6, tmp8, tmp9)
    tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = 0.0
    tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = 1.0
    tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
    tmp20 = tl.where(tmp6, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp23 = tl.broadcast_to(tmp12, [R0_BLOCK])
    tmp24 = tl.broadcast_to(tmp17, [R0_BLOCK])
    tmp25 = tl.broadcast_to(tmp22, [R0_BLOCK])
    tmp27 = tl.where(r0_mask, tmp23, 0)
    tmp28 = tl.where(r0_mask, tmp24, 0)
    tmp29 = tl.where(r0_mask, tmp25, 0)
    tmp30, tmp31, tmp32 = triton_helpers.welford(tmp27, tmp28, tmp29, 0)
    tmp33 = triton_helpers.promote_to_tensor(tmp30)
    tmp34 = triton_helpers.promote_to_tensor(tmp31)
    tmp35 = triton_helpers.promote_to_tensor(tmp32)
    tl.store(out_ptr0 + (x4), tmp33, None)
    tl.store(out_ptr1 + (x4), tmp34, None)
    tl.store(out_ptr2 + (x4), tmp35, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/oe/coelpy5f4eys357uoicdtqp5tbszgccwla2fzetwxi2prc4sb3ka.py
# Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_93 => var_mean_23
# Graph fragment:
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_63, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_47 = async_compile.triton('triton_per_fused_native_group_norm_47', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1257984, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_47(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 896
    r0_numel = 115
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
    tmp0 = tl.load(in_ptr0 + (x0 + 32*r0_2 + 3680*x1), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 32*r0_2 + 3680*x1), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 32*r0_2 + 3680*x1), r0_mask & xmask, other=0.0)
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/jr/cjrg4l5zlfgj4b2qhhwmrxxuoqblm6ex632brczf7nymlimkzbna.py
# Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_93 => var_mean_23
# Graph fragment:
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_63, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_48 = async_compile.triton('triton_per_fused_native_group_norm_48', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 21504, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_48(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/4b/c4bja4trippyjs2ytoxfhbfndluv76kq3al6gie3v4hx2igkbwtq.py
# Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_93 => var_mean_23
# Graph fragment:
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_63, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_49 = async_compile.triton('triton_per_fused_native_group_norm_49', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5888, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_49(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/zn/cznkewahcc5cpueahhgyzblpjftpayjlsjltv72q2e36jdnajdgh.py
# Topologically Sorted Source Nodes: [hidden_states_93, hidden_states_94], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_93 => add_74, mul_81
#   hidden_states_94 => convert_element_type_168, mul_82, sigmoid_22
# Graph fragment:
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_64, %unsqueeze_141), kwargs = {})
#   %add_74 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_81, %unsqueeze_144), kwargs = {})
#   %sigmoid_22 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_74,), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_74, %sigmoid_22), kwargs = {})
#   %convert_element_type_168 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_82, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_50 = async_compile.triton('triton_poi_fused_native_group_norm_silu_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 402654208}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
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
    tmp5 = 2097152.0
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/fe/cfegqvrafkriih2ksnx3sgdeyw76nzn66q23wuw6vznuhrbn3c5p.py
# Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_94 => convert_element_type_168, mul_82, sigmoid_22
#   hidden_states_95 => convolution_28
# Graph fragment:
#   %sigmoid_22 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_74,), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_74, %sigmoid_22), kwargs = {})
#   %convert_element_type_168 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_82, torch.bfloat16), kwargs = {})
#   %convolution_28 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_168, %arg113_1, %arg114_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_51 = async_compile.triton('triton_poi_fused_convolution_silu_51', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1179648, 'x': 589824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_51(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/c2/cc2yxbfvk644dj7j6pi56hckrfcplp2o7ob2avrfndr2uwsmjg2u.py
# Topologically Sorted Source Nodes: [hidden_states_96], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_96 => var_mean_24
# Graph fragment:
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_65, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_52 = async_compile.triton('triton_per_fused_native_group_norm_52', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2322432, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_52(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/tj/ctjgzxpqkn4dwrauave5ssipslemfk77yedbkwstflxxsnqzvugt.py
# Topologically Sorted Source Nodes: [hidden_states_96], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_96 => var_mean_24
# Graph fragment:
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_65, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_53 = async_compile.triton('triton_per_fused_native_group_norm_53', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1182720, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_53(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/r2/cr2o7saeplya6sqmd64rnrcp6njbchzo2jwnmazc2wcy47hib5gw.py
# Topologically Sorted Source Nodes: [hidden_states_96, hidden_states_97], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_96 => add_76, mul_84
#   hidden_states_97 => convert_element_type_174, mul_85, sigmoid_23
# Graph fragment:
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_66, %unsqueeze_147), kwargs = {})
#   %add_76 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_84, %unsqueeze_150), kwargs = {})
#   %sigmoid_23 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_76,), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_76, %sigmoid_23), kwargs = {})
#   %convert_element_type_174 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_85, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_54 = async_compile.triton('triton_poi_fused_native_group_norm_silu_54', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 201327360}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_54(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/fe/cfedchncbp6hlcczjruob7raqt4fcxc3n77ilv4crdoehcu26mof.py
# Topologically Sorted Source Nodes: [hidden_states_97, hidden_states_99], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_97 => convert_element_type_174, mul_85, sigmoid_23
#   hidden_states_99 => convolution_29
# Graph fragment:
#   %sigmoid_23 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_76,), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_76, %sigmoid_23), kwargs = {})
#   %convert_element_type_174 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_85, torch.bfloat16), kwargs = {})
#   %convolution_29 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_174, %arg117_1, %arg118_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_55 = async_compile.triton('triton_poi_fused_convolution_silu_55', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 589824, 'x': 294912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_55(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/3n/c3nwnaovvuuizai4m6nn3ragtsdquvkrrjdwphay4k7lreds5hi5.py
# Topologically Sorted Source Nodes: [hidden_states_100], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_100 => var_mean_25
# Graph fragment:
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_67, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_56 = async_compile.triton('triton_per_fused_native_group_norm_56', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2322432, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_56(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/pv/cpveaizdlpdpwsbqqbqp6aiufsrvikenxnj2gg7m7dtuhxc2wkg2.py
# Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_101], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_100 => add_79, mul_87
#   hidden_states_101 => convert_element_type_180, mul_88, sigmoid_24
# Graph fragment:
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_68, %unsqueeze_153), kwargs = {})
#   %add_79 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_87, %unsqueeze_156), kwargs = {})
#   %sigmoid_24 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_79,), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_79, %sigmoid_24), kwargs = {})
#   %convert_element_type_180 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_88, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_57 = async_compile.triton('triton_poi_fused_native_group_norm_silu_57', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 268436480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_57(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/ss/cssl5breekul4scewwwd5mq657adb7kpmklbwusqgvosm7gp5cnt.py
# Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_104, hidden_states_106, add_13, output_tensor_12], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div]
# Source node to ATen node mapping:
#   add_12 => add_77
#   add_13 => add_82
#   hidden_states_104 => convert_element_type_186, mul_91, sigmoid_25
#   hidden_states_106 => convolution_32
#   hidden_states_97 => convert_element_type_174, mul_85, sigmoid_23
#   hidden_states_99 => convolution_29
#   input_tensor_1 => convolution_30
#   output_tensor_11 => div_12
#   output_tensor_12 => div_13
# Graph fragment:
#   %convolution_30 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_27, %arg119_1, %arg120_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_23 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_76,), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_76, %sigmoid_23), kwargs = {})
#   %convert_element_type_174 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_85, torch.bfloat16), kwargs = {})
#   %convolution_29 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_174, %arg117_1, %arg118_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_77 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_30, %convolution_29), kwargs = {})
#   %div_12 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_77, 1.0), kwargs = {})
#   %sigmoid_25 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_81,), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, %sigmoid_25), kwargs = {})
#   %convert_element_type_186 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_91, torch.bfloat16), kwargs = {})
#   %convolution_32 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_186, %arg127_1, %arg128_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_82 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_12, %convolution_32), kwargs = {})
#   %div_13 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_82, 1.0), kwargs = {})
triton_poi_fused_add_convolution_div_silu_58 = async_compile.triton('triton_poi_fused_add_convolution_div_silu_58', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_silu_58', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 335545088}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_silu_58(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/fi/cfiyvezx5bnmnfeiiv2piajouqkxu4ni23ekth6ngu4xcie4hbnn.py
# Topologically Sorted Source Nodes: [hidden_states_107], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_107 => var_mean_27
# Graph fragment:
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_71, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_59 = async_compile.triton('triton_per_fused_native_group_norm_59', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2322432, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_59(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
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
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
    tmp10 = tl.where(tmp6, tmp8, tmp9)
    tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = 0.0
    tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = 1.0
    tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
    tmp20 = tl.where(tmp6, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp23 = tl.broadcast_to(tmp12, [R0_BLOCK])
    tmp24 = tl.broadcast_to(tmp17, [R0_BLOCK])
    tmp25 = tl.broadcast_to(tmp22, [R0_BLOCK])
    tmp27 = tl.where(r0_mask, tmp23, 0)
    tmp28 = tl.where(r0_mask, tmp24, 0)
    tmp29 = tl.where(r0_mask, tmp25, 0)
    tmp30, tmp31, tmp32 = triton_helpers.welford(tmp27, tmp28, tmp29, 0)
    tmp33 = triton_helpers.promote_to_tensor(tmp30)
    tmp34 = triton_helpers.promote_to_tensor(tmp31)
    tmp35 = triton_helpers.promote_to_tensor(tmp32)
    tl.store(out_ptr0 + (x4), tmp33, None)
    tl.store(out_ptr1 + (x4), tmp34, None)
    tl.store(out_ptr2 + (x4), tmp35, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/b4/cb4kir75cnctojz4b2znfrtyfkqzlv5fzx7cttzg2nuw3iqmiaae.py
# Topologically Sorted Source Nodes: [hidden_states_107, hidden_states_108], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_107 => add_84, mul_93
#   hidden_states_108 => convert_element_type_192, mul_94, sigmoid_26
# Graph fragment:
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_72, %unsqueeze_165), kwargs = {})
#   %add_84 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_93, %unsqueeze_168), kwargs = {})
#   %sigmoid_26 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_84,), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_84, %sigmoid_26), kwargs = {})
#   %convert_element_type_192 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_94, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_60 = async_compile.triton('triton_poi_fused_native_group_norm_silu_60', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_60', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 201327104}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_60(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
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
    tmp5 = 1048576.0
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/mb/cmbi7w3c3u7l4gnhod7c6o24jbmdy3ie54ehgiabm3y37fei2lhf.py
# Topologically Sorted Source Nodes: [sample_2], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   sample_2 => var_mean_29
# Graph fragment:
#   %var_mean_29 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_75, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_61 = async_compile.triton('triton_per_fused_native_group_norm_61', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2322432, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_group_norm_61(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
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
    tmp8 = tl.load(in_ptr1 + (4*x0 + 128*(((r0_3 + 347*x1 + 74899*x2) % 262144)) + ((((r0_3 + 347*x1 + 74899*x2) // 262144) % 4))), r0_mask & tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (4*x0 + ((((r0_3 + 347*x1 + 74899*x2) // 262144) % 4))), r0_mask & tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
    tmp16 = tl.where(tmp6, tmp14, tmp15)
    tmp17 = tl.full(tmp16.shape, 0, tmp16.dtype)
    tmp18 = tl.where(tmp2, tmp16, tmp17)
    tmp19 = 0.0
    tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
    tmp21 = tl.where(tmp6, tmp19, tmp20)
    tmp22 = tl.full(tmp21.shape, 0, tmp21.dtype)
    tmp23 = tl.where(tmp2, tmp21, tmp22)
    tmp24 = tl.full(tmp12.shape, 0, tmp12.dtype)
    tmp25 = tl.where(tmp6, tmp12, tmp24)
    tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
    tmp27 = tl.where(tmp2, tmp25, tmp26)
    tmp28 = tl.broadcast_to(tmp18, [R0_BLOCK])
    tmp29 = tl.broadcast_to(tmp23, [R0_BLOCK])
    tmp30 = tl.broadcast_to(tmp27, [R0_BLOCK])
    tmp32 = tl.where(r0_mask, tmp28, 0)
    tmp33 = tl.where(r0_mask, tmp29, 0)
    tmp34 = tl.where(r0_mask, tmp30, 0)
    tmp35, tmp36, tmp37 = triton_helpers.welford(tmp32, tmp33, tmp34, 0)
    tmp38 = triton_helpers.promote_to_tensor(tmp35)
    tmp39 = triton_helpers.promote_to_tensor(tmp36)
    tmp40 = triton_helpers.promote_to_tensor(tmp37)
    tl.store(out_ptr0 + (x4), tmp38, None)
    tl.store(out_ptr1 + (x4), tmp39, None)
    tl.store(out_ptr2 + (x4), tmp40, None)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/el/celx7m4pj33x2u4nqvdhkm473hm6qvwfrfg45fqsegl5olzpwrvo.py
# Topologically Sorted Source Nodes: [sample_2, sample_3], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   sample_2 => add_89, mul_99
#   sample_3 => convert_element_type_204, mul_100, sigmoid_28
# Graph fragment:
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_76, %unsqueeze_177), kwargs = {})
#   %add_89 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_99, %unsqueeze_180), kwargs = {})
#   %sigmoid_28 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_89,), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, %sigmoid_28), kwargs = {})
#   %convert_element_type_204 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_100, torch.bfloat16), kwargs = {})
triton_poi_fused_native_group_norm_silu_62 = async_compile.triton('triton_poi_fused_native_group_norm_silu_62', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_62', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 268436224}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_62(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x0 // 4), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0 // 4), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 1048576.0
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


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/wm/cwm7pwth7rfsolcr24jr5bodwvd5pwrapky7axmbogptg4a5gpb2.py
# Topologically Sorted Source Nodes: [sample_3, sample_4], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   sample_3 => convert_element_type_204, mul_100, sigmoid_28
#   sample_4 => convolution_35
# Graph fragment:
#   %sigmoid_28 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_89,), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, %sigmoid_28), kwargs = {})
#   %convert_element_type_204 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_100, torch.bfloat16), kwargs = {})
#   %convolution_35 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_204, %arg139_1, %arg140_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_63 = async_compile.triton('triton_poi_fused_convolution_silu_63', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_63', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 13824, 'x': 6912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_63(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /cache/independent_sr_cuda_20261013/compile_cache/hypir_tiles/tz/ctzlucegexqbgbihnybmwrsp6lolgy3s542gapohgkf4vobfgyek.py
# Topologically Sorted Source Nodes: [sample_3, sample_4], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   sample_3 => convert_element_type_204, mul_100, sigmoid_28
#   sample_4 => convolution_35
# Graph fragment:
#   %sigmoid_28 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_89,), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, %sigmoid_28), kwargs = {})
#   %convert_element_type_204 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_100, torch.bfloat16), kwargs = {})
#   %convolution_35 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_204, %arg139_1, %arg140_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_64 = async_compile.triton('triton_poi_fused_convolution_silu_64', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4, 'x': 262144}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_64', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '2CD0AB9B04912F8124903F28067CFAB39E58FDF44B0DD051C947CED8B36A15AC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 589826, 'x': 3145728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_64(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (y0 + 3*x1), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x1 + 262144*y0), tmp2, ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(arg1_1, (4, ), (1, ))
    assert_size_stride(arg2_1, (1, 4, 64, 64), (16384, 4096, 64, 1))
    assert_size_stride(arg3_1, (512, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg4_1, (512, ), (1, ))
    assert_size_stride(arg5_1, (512, ), (1, ))
    assert_size_stride(arg6_1, (512, ), (1, ))
    assert_size_stride(arg7_1, (512, ), (1, ))
    assert_size_stride(arg8_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg9_1, (512, ), (1, ))
    assert_size_stride(arg10_1, (512, ), (1, ))
    assert_size_stride(arg11_1, (512, ), (1, ))
    assert_size_stride(arg12_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg13_1, (512, ), (1, ))
    assert_size_stride(arg14_1, (512, ), (1, ))
    assert_size_stride(arg15_1, (512, ), (1, ))
    assert_size_stride(arg16_1, (512, 512), (512, 1))
    assert_size_stride(arg17_1, (512, ), (1, ))
    assert_size_stride(arg18_1, (512, 512), (512, 1))
    assert_size_stride(arg19_1, (512, ), (1, ))
    assert_size_stride(arg20_1, (512, 512), (512, 1))
    assert_size_stride(arg21_1, (512, ), (1, ))
    assert_size_stride(arg22_1, (512, 512), (512, 1))
    assert_size_stride(arg23_1, (512, ), (1, ))
    assert_size_stride(arg24_1, (512, ), (1, ))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg27_1, (512, ), (1, ))
    assert_size_stride(arg28_1, (512, ), (1, ))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg31_1, (512, ), (1, ))
    assert_size_stride(arg32_1, (512, ), (1, ))
    assert_size_stride(arg33_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg34_1, (512, ), (1, ))
    assert_size_stride(arg35_1, (512, ), (1, ))
    assert_size_stride(arg36_1, (512, ), (1, ))
    assert_size_stride(arg37_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg38_1, (512, ), (1, ))
    assert_size_stride(arg39_1, (512, ), (1, ))
    assert_size_stride(arg40_1, (512, ), (1, ))
    assert_size_stride(arg41_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg42_1, (512, ), (1, ))
    assert_size_stride(arg43_1, (512, ), (1, ))
    assert_size_stride(arg44_1, (512, ), (1, ))
    assert_size_stride(arg45_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg46_1, (512, ), (1, ))
    assert_size_stride(arg47_1, (512, ), (1, ))
    assert_size_stride(arg48_1, (512, ), (1, ))
    assert_size_stride(arg49_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (512, ), (1, ))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg54_1, (512, ), (1, ))
    assert_size_stride(arg55_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg56_1, (512, ), (1, ))
    assert_size_stride(arg57_1, (512, ), (1, ))
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
    assert_size_stride(arg81_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg82_1, (512, ), (1, ))
    assert_size_stride(arg83_1, (512, ), (1, ))
    assert_size_stride(arg84_1, (512, ), (1, ))
    assert_size_stride(arg85_1, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg86_1, (256, ), (1, ))
    assert_size_stride(arg87_1, (256, ), (1, ))
    assert_size_stride(arg88_1, (256, ), (1, ))
    assert_size_stride(arg89_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg90_1, (256, ), (1, ))
    assert_size_stride(arg91_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg92_1, (256, ), (1, ))
    assert_size_stride(arg93_1, (256, ), (1, ))
    assert_size_stride(arg94_1, (256, ), (1, ))
    assert_size_stride(arg95_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg96_1, (256, ), (1, ))
    assert_size_stride(arg97_1, (256, ), (1, ))
    assert_size_stride(arg98_1, (256, ), (1, ))
    assert_size_stride(arg99_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg100_1, (256, ), (1, ))
    assert_size_stride(arg101_1, (256, ), (1, ))
    assert_size_stride(arg102_1, (256, ), (1, ))
    assert_size_stride(arg103_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg104_1, (256, ), (1, ))
    assert_size_stride(arg105_1, (256, ), (1, ))
    assert_size_stride(arg106_1, (256, ), (1, ))
    assert_size_stride(arg107_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg108_1, (256, ), (1, ))
    assert_size_stride(arg109_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg110_1, (256, ), (1, ))
    assert_size_stride(arg111_1, (256, ), (1, ))
    assert_size_stride(arg112_1, (256, ), (1, ))
    assert_size_stride(arg113_1, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg114_1, (128, ), (1, ))
    assert_size_stride(arg115_1, (128, ), (1, ))
    assert_size_stride(arg116_1, (128, ), (1, ))
    assert_size_stride(arg117_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg118_1, (128, ), (1, ))
    assert_size_stride(arg119_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg120_1, (128, ), (1, ))
    assert_size_stride(arg121_1, (128, ), (1, ))
    assert_size_stride(arg122_1, (128, ), (1, ))
    assert_size_stride(arg123_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg124_1, (128, ), (1, ))
    assert_size_stride(arg125_1, (128, ), (1, ))
    assert_size_stride(arg126_1, (128, ), (1, ))
    assert_size_stride(arg127_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg128_1, (128, ), (1, ))
    assert_size_stride(arg129_1, (128, ), (1, ))
    assert_size_stride(arg130_1, (128, ), (1, ))
    assert_size_stride(arg131_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg132_1, (128, ), (1, ))
    assert_size_stride(arg133_1, (128, ), (1, ))
    assert_size_stride(arg134_1, (128, ), (1, ))
    assert_size_stride(arg135_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg136_1, (128, ), (1, ))
    assert_size_stride(arg137_1, (128, ), (1, ))
    assert_size_stride(arg138_1, (128, ), (1, ))
    assert_size_stride(arg139_1, (3, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg140_1, (3, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 4, 64, 64), (16384, 1, 256, 4), torch.bfloat16)
        # Topologically Sorted Source Nodes: [z], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg2_1, buf0, 4, 4096, stream=stream0)
        del arg2_1
        # Topologically Sorted Source Nodes: [z], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, arg0_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (1, 4, 64, 64), (16384, 1, 256, 4), 'torch.ops.aten.convolution.default')
        del arg0_1
        del buf0
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [z], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(buf2, arg1_1, 16384, stream=stream0)
        del arg1_1
        buf3 = empty_strided_cuda((512, 4, 3, 3), (36, 1, 12, 4), torch.bfloat16)
        # Topologically Sorted Source Nodes: [z, sample], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(arg3_1, buf3, 2048, 9, stream=stream0)
        del arg3_1
        # Topologically Sorted Source Nodes: [z, sample], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf2, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (1, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        del buf2
        del buf3
        buf5 = empty_strided_cuda((1, 32, 1, 1, 8, 64), (16384, 8, 16384, 16384, 1, 256), torch.float32)
        buf6 = empty_strided_cuda((1, 32, 1, 1, 8, 64), (16384, 8, 16384, 16384, 1, 256), torch.float32)
        buf7 = empty_strided_cuda((1, 32, 1, 1, 8, 64), (16384, 8, 16384, 16384, 1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_3.run(buf4, arg4_1, buf5, buf6, buf7, 16384, 128, stream=stream0)
        buf8 = empty_strided_cuda((1, 32, 1, 1, 8), (256, 8, 256, 256, 1), torch.float32)
        buf9 = empty_strided_cuda((1, 32, 1, 1, 8), (256, 8, 256, 256, 1), torch.float32)
        buf10 = empty_strided_cuda((1, 32, 1, 1, 8), (256, 8, 256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf5, buf6, buf7, buf8, buf9, buf10, 256, 64, stream=stream0)
        buf11 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 32, 32), torch.float32)
        buf12 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf8, buf9, buf10, buf11, buf12, 32, 8, stream=stream0)
        buf15 = empty_strided_cuda((1, 512, 64, 64), (2097152, 1, 32768, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states, hidden_states_1], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_6.run(buf4, arg4_1, buf11, buf12, arg6_1, arg7_1, buf15, 2097152, stream=stream0)
        del arg6_1
        del arg7_1
        buf16 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg8_1, buf16, 262144, 9, stream=stream0)
        del arg8_1
        # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2], Original ATen: [aten.silu, aten.convolution]
        buf17 = extern_kernels.convolution(buf15, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (1, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf18 = buf7; del buf7  # reuse
        buf19 = buf6; del buf6  # reuse
        buf20 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_3], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_3.run(buf17, arg9_1, buf18, buf19, buf20, 16384, 128, stream=stream0)
        buf21 = buf9; del buf9  # reuse
        buf22 = buf8; del buf8  # reuse
        buf23 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_3], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf18, buf19, buf20, buf21, buf22, buf23, 256, 64, stream=stream0)
        buf24 = buf12; del buf12  # reuse
        buf25 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_3], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf21, buf22, buf23, buf24, buf25, 32, 8, stream=stream0)
        buf28 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_6.run(buf17, arg9_1, buf24, buf25, arg10_1, arg11_1, buf28, 2097152, stream=stream0)
        del arg10_1
        del arg11_1
        del arg9_1
        buf29 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_6], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg12_1, buf29, 262144, 9, stream=stream0)
        del arg12_1
        # Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_6], Original ATen: [aten.silu, aten.convolution]
        buf30 = extern_kernels.convolution(buf28, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (1, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf31 = buf20; del buf20  # reuse
        buf32 = buf19; del buf19  # reuse
        buf33 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_8.run(buf4, arg4_1, buf30, arg13_1, buf31, buf32, buf33, 16384, 128, stream=stream0)
        buf34 = buf23; del buf23  # reuse
        buf35 = buf22; del buf22  # reuse
        buf36 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf31, buf32, buf33, buf34, buf35, buf36, 256, 64, stream=stream0)
        buf37 = buf25; del buf25  # reuse
        buf38 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf34, buf35, buf36, buf37, buf38, 32, 8, stream=stream0)
        buf40 = reinterpret_tensor(buf28, (1, 512, 4096), (2097152, 4096, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_9.run(buf4, arg4_1, buf30, arg13_1, buf37, buf38, arg14_1, arg15_1, buf40, 4096, 512, stream=stream0)
        del arg14_1
        del arg15_1
        buf41 = reinterpret_tensor(buf17, (4096, 512), (512, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [query], Original ATen: [aten.bmm]
        extern_kernels.mm(reinterpret_tensor(buf40, (4096, 512), (1, 4096), 0), reinterpret_tensor(arg16_1, (512, 512), (1, 512), 0), out=buf41)
        del arg16_1
        buf42 = empty_strided_cuda((4096, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [key], Original ATen: [aten.bmm]
        extern_kernels.mm(reinterpret_tensor(buf40, (4096, 512), (1, 4096), 0), reinterpret_tensor(arg18_1, (512, 512), (1, 512), 0), out=buf42)
        del arg18_1
        buf43 = empty_strided_cuda((4096, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [value], Original ATen: [aten.bmm]
        extern_kernels.mm(reinterpret_tensor(buf40, (4096, 512), (1, 4096), 0), reinterpret_tensor(arg20_1, (512, 512), (1, 512), 0), out=buf43)
        del arg20_1
        del buf40
        buf44 = reinterpret_tensor(buf41, (1, 1, 4096, 512), (2097152, 2097152, 512, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_10.run(buf44, arg17_1, 2097152, stream=stream0)
        del arg17_1
        buf45 = reinterpret_tensor(buf42, (1, 1, 4096, 512), (2097152, 2097152, 512, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_10.run(buf45, arg19_1, 2097152, stream=stream0)
        del arg19_1
        buf46 = reinterpret_tensor(buf43, (1, 1, 4096, 512), (2097152, 2097152, 512, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_10.run(buf46, arg21_1, 2097152, stream=stream0)
        del arg21_1
        # Topologically Sorted Source Nodes: [hidden_states_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf47 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf44, buf45, buf46, None, False)
        del buf44
        del buf45
        buf48 = buf47[0]
        assert_size_stride(buf48, (1, 1, 4096, 512), (2097152, 512, 512, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf48, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf47
        buf52 = reinterpret_tensor(buf46, (4096, 512), (512, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_12], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf48, (4096, 512), (512, 1), 0), reinterpret_tensor(arg22_1, (512, 512), (1, 512), 0), out=buf52)
        del arg22_1
        del buf48
        buf53 = reinterpret_tensor(buf52, (1, 512, 64, 64), (2097152, 1, 32768, 512), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, hidden_states_15, hidden_states_16], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_silu_11.run(buf53, arg23_1, buf4, arg4_1, buf30, arg13_1, 2097152, stream=stream0)
        del arg13_1
        del arg23_1
        del arg4_1
        del buf30
        buf54 = buf33; del buf33  # reuse
        buf55 = buf32; del buf32  # reuse
        buf56 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_17], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_12.run(buf53, buf54, buf55, buf56, 16384, 128, stream=stream0)
        buf57 = buf36; del buf36  # reuse
        buf58 = buf35; del buf35  # reuse
        buf59 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_17], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf54, buf55, buf56, buf57, buf58, buf59, 256, 64, stream=stream0)
        buf60 = buf38; del buf38  # reuse
        buf61 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_17], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf57, buf58, buf59, buf60, buf61, 32, 8, stream=stream0)
        buf64 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_17, hidden_states_18], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_13.run(buf53, buf60, buf61, arg24_1, arg25_1, buf64, 2097152, stream=stream0)
        del arg24_1
        del arg25_1
        buf65 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_19], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg26_1, buf65, 262144, 9, stream=stream0)
        del arg26_1
        # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_19], Original ATen: [aten.silu, aten.convolution]
        buf66 = extern_kernels.convolution(buf64, buf65, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (1, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf67 = buf56; del buf56  # reuse
        buf68 = buf55; del buf55  # reuse
        buf69 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_20], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_3.run(buf66, arg27_1, buf67, buf68, buf69, 16384, 128, stream=stream0)
        buf70 = buf59; del buf59  # reuse
        buf71 = buf58; del buf58  # reuse
        buf72 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_20], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf67, buf68, buf69, buf70, buf71, buf72, 256, 64, stream=stream0)
        buf73 = buf61; del buf61  # reuse
        buf74 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_20], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf70, buf71, buf72, buf73, buf74, 32, 8, stream=stream0)
        buf77 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_20, hidden_states_21], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_6.run(buf66, arg27_1, buf73, buf74, arg28_1, arg29_1, buf77, 2097152, stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del buf66
        buf78 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg30_1, buf78, 262144, 9, stream=stream0)
        del arg30_1
        # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23], Original ATen: [aten.silu, aten.convolution]
        buf79 = extern_kernels.convolution(buf77, buf78, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (1, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf80 = buf69; del buf69  # reuse
        buf81 = buf68; del buf68  # reuse
        buf82 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_24], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_14.run(buf53, buf79, arg31_1, buf80, buf81, buf82, 16384, 128, stream=stream0)
        buf83 = buf72; del buf72  # reuse
        buf84 = buf71; del buf71  # reuse
        buf85 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_24], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf80, buf81, buf82, buf83, buf84, buf85, 256, 64, stream=stream0)
        buf86 = buf74; del buf74  # reuse
        buf87 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_24], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf83, buf84, buf85, buf86, buf87, 32, 8, stream=stream0)
        buf90 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_24, hidden_states_25], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_15.run(buf53, buf79, arg31_1, buf86, buf87, arg5_1, arg32_1, buf90, 2097152, stream=stream0)
        del arg32_1
        del arg5_1
        buf91 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg33_1, buf91, 262144, 9, stream=stream0)
        del arg33_1
        # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26], Original ATen: [aten.silu, aten.convolution]
        buf92 = extern_kernels.convolution(buf90, buf91, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (1, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf93 = buf82; del buf82  # reuse
        buf94 = buf81; del buf81  # reuse
        buf95 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_27], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_3.run(buf92, arg34_1, buf93, buf94, buf95, 16384, 128, stream=stream0)
        buf96 = buf85; del buf85  # reuse
        buf97 = buf84; del buf84  # reuse
        buf98 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_27], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf93, buf94, buf95, buf96, buf97, buf98, 256, 64, stream=stream0)
        buf99 = buf87; del buf87  # reuse
        buf100 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_27], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf96, buf97, buf98, buf99, buf100, 32, 8, stream=stream0)
        buf103 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_27, hidden_states_28], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_6.run(buf92, arg34_1, buf99, buf100, arg35_1, arg36_1, buf103, 2097152, stream=stream0)
        del arg34_1
        del arg35_1
        del arg36_1
        del buf92
        buf104 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28, hidden_states_30], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg37_1, buf104, 262144, 9, stream=stream0)
        del arg37_1
        # Topologically Sorted Source Nodes: [hidden_states_28, hidden_states_30], Original ATen: [aten.silu, aten.convolution]
        buf105 = extern_kernels.convolution(buf103, buf104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (1, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        del buf103
        buf106 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, hidden_states_28, hidden_states_30, add_3, output_tensor_2], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_silu_16.run(buf106, buf79, arg31_1, buf105, arg38_1, 2097152, stream=stream0)
        del arg31_1
        del arg38_1
        del buf105
        buf107 = buf95; del buf95  # reuse
        buf108 = buf94; del buf94  # reuse
        buf109 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_12.run(buf106, buf107, buf108, buf109, 16384, 128, stream=stream0)
        buf110 = buf98; del buf98  # reuse
        buf111 = buf97; del buf97  # reuse
        buf112 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf107, buf108, buf109, buf110, buf111, buf112, 256, 64, stream=stream0)
        buf113 = buf99; del buf99  # reuse
        buf114 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf110, buf111, buf112, buf113, buf114, 32, 8, stream=stream0)
        buf117 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31, hidden_states_32], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_13.run(buf106, buf113, buf114, arg39_1, arg40_1, buf117, 2097152, stream=stream0)
        del arg39_1
        del arg40_1
        buf118 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg41_1, buf118, 262144, 9, stream=stream0)
        del arg41_1
        # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33], Original ATen: [aten.silu, aten.convolution]
        buf119 = extern_kernels.convolution(buf117, buf118, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (1, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf120 = buf109; del buf109  # reuse
        buf121 = buf108; del buf108  # reuse
        buf122 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_34], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_3.run(buf119, arg42_1, buf120, buf121, buf122, 16384, 128, stream=stream0)
        buf123 = buf112; del buf112  # reuse
        buf124 = buf111; del buf111  # reuse
        buf125 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_34], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf120, buf121, buf122, buf123, buf124, buf125, 256, 64, stream=stream0)
        buf126 = buf114; del buf114  # reuse
        buf127 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_34], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf123, buf124, buf125, buf126, buf127, 32, 8, stream=stream0)
        buf130 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_34, hidden_states_35], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_6.run(buf119, arg42_1, buf126, buf127, arg43_1, arg44_1, buf130, 2097152, stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del buf119
        buf131 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg45_1, buf131, 262144, 9, stream=stream0)
        del arg45_1
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37], Original ATen: [aten.silu, aten.convolution]
        buf132 = extern_kernels.convolution(buf130, buf131, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (1, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf133 = buf122; del buf122  # reuse
        buf134 = buf121; del buf121  # reuse
        buf135 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_38], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_14.run(buf106, buf132, arg46_1, buf133, buf134, buf135, 16384, 128, stream=stream0)
        buf136 = buf125; del buf125  # reuse
        buf137 = buf124; del buf124  # reuse
        buf138 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_38], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf133, buf134, buf135, buf136, buf137, buf138, 256, 64, stream=stream0)
        buf139 = buf127; del buf127  # reuse
        buf140 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_38], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf136, buf137, buf138, buf139, buf140, 32, 8, stream=stream0)
        buf143 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_38, hidden_states_39], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_15.run(buf106, buf132, arg46_1, buf139, buf140, arg47_1, arg48_1, buf143, 2097152, stream=stream0)
        del arg47_1
        del arg48_1
        buf144 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg49_1, buf144, 262144, 9, stream=stream0)
        del arg49_1
        # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40], Original ATen: [aten.silu, aten.convolution]
        buf145 = extern_kernels.convolution(buf143, buf144, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (1, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf146 = buf135; del buf135  # reuse
        buf147 = buf134; del buf134  # reuse
        buf148 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_41], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_3.run(buf145, arg50_1, buf146, buf147, buf148, 16384, 128, stream=stream0)
        buf149 = buf138; del buf138  # reuse
        buf150 = buf137; del buf137  # reuse
        buf151 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_41], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf146, buf147, buf148, buf149, buf150, buf151, 256, 64, stream=stream0)
        del buf146
        del buf147
        del buf148
        buf152 = buf140; del buf140  # reuse
        buf153 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_41], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf149, buf150, buf151, buf152, buf153, 32, 8, stream=stream0)
        del buf149
        del buf150
        del buf151
        buf156 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_41, hidden_states_42], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_6.run(buf145, arg50_1, buf152, buf153, arg51_1, arg52_1, buf156, 2097152, stream=stream0)
        del arg50_1
        del arg51_1
        del arg52_1
        del buf145
        buf157 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_42, hidden_states_44], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg53_1, buf157, 262144, 9, stream=stream0)
        del arg53_1
        # Topologically Sorted Source Nodes: [hidden_states_42, hidden_states_44], Original ATen: [aten.silu, aten.convolution]
        buf158 = extern_kernels.convolution(buf156, buf157, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (1, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        del buf156
        buf160 = empty_strided_cuda((1, 512, 128, 128), (8388608, 1, 65536, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten._unsafe_index, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_17.run(buf106, buf132, arg46_1, buf158, arg54_1, buf160, 512, 16384, stream=stream0)
        del arg46_1
        del arg54_1
        del buf106
        del buf132
        del buf158
        buf161 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46], Original ATen: [aten.clone, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg55_1, buf161, 262144, 9, stream=stream0)
        del arg55_1
        # Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46], Original ATen: [aten.clone, aten._to_copy, aten.convolution]
        buf162 = extern_kernels.convolution(buf160, buf161, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (1, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
        buf163 = empty_strided_cuda((1, 32, 1, 1, 16, 128), (65536, 16, 65536, 65536, 1, 512), torch.float32)
        buf164 = empty_strided_cuda((1, 32, 1, 1, 16, 128), (65536, 16, 65536, 65536, 1, 512), torch.float32)
        buf165 = empty_strided_cuda((1, 32, 1, 1, 16, 128), (65536, 16, 65536, 65536, 1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_47], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_18.run(buf162, arg56_1, buf163, buf164, buf165, 65536, 128, stream=stream0)
        buf166 = empty_strided_cuda((1, 32, 1, 1, 16), (512, 16, 512, 512, 1), torch.float32)
        buf167 = empty_strided_cuda((1, 32, 1, 1, 16), (512, 16, 512, 512, 1), torch.float32)
        buf168 = empty_strided_cuda((1, 32, 1, 1, 16), (512, 16, 512, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_47], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf163, buf164, buf165, buf166, buf167, buf168, 512, 128, stream=stream0)
        buf169 = buf153; del buf153  # reuse
        buf170 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_47], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_20.run(buf166, buf167, buf168, buf169, buf170, 32, 16, stream=stream0)
        buf173 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_47, hidden_states_48], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_21.run(buf162, arg56_1, buf169, buf170, arg57_1, arg58_1, buf173, 8388608, stream=stream0)
        del arg57_1
        del arg58_1
        buf174 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg59_1, buf174, 262144, 9, stream=stream0)
        del arg59_1
        # Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49], Original ATen: [aten.silu, aten.convolution]
        buf175 = extern_kernels.convolution(buf173, buf174, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (1, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
        buf176 = buf165; del buf165  # reuse
        buf177 = buf164; del buf164  # reuse
        buf178 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_50], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_18.run(buf175, arg60_1, buf176, buf177, buf178, 65536, 128, stream=stream0)
        buf179 = buf168; del buf168  # reuse
        buf180 = buf167; del buf167  # reuse
        buf181 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_50], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf176, buf177, buf178, buf179, buf180, buf181, 512, 128, stream=stream0)
        buf182 = buf170; del buf170  # reuse
        buf183 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_50], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_20.run(buf179, buf180, buf181, buf182, buf183, 32, 16, stream=stream0)
        buf186 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_50, hidden_states_51], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_21.run(buf175, arg60_1, buf182, buf183, arg61_1, arg62_1, buf186, 8388608, stream=stream0)
        del arg60_1
        del arg61_1
        del arg62_1
        del buf175
        buf187 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_51, hidden_states_53], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg63_1, buf187, 262144, 9, stream=stream0)
        del arg63_1
        # Topologically Sorted Source Nodes: [hidden_states_51, hidden_states_53], Original ATen: [aten.silu, aten.convolution]
        buf188 = extern_kernels.convolution(buf186, buf187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (1, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
        buf189 = buf178; del buf178  # reuse
        buf190 = buf177; del buf177  # reuse
        buf191 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_54], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_22.run(buf162, arg56_1, buf188, arg64_1, buf189, buf190, buf191, 65536, 128, stream=stream0)
        buf192 = buf181; del buf181  # reuse
        buf193 = buf180; del buf180  # reuse
        buf194 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_54], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf189, buf190, buf191, buf192, buf193, buf194, 512, 128, stream=stream0)
        buf195 = buf183; del buf183  # reuse
        buf196 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_54], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_20.run(buf192, buf193, buf194, buf195, buf196, 32, 16, stream=stream0)
        buf199 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_54, hidden_states_55], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_23.run(buf162, arg56_1, buf188, arg64_1, buf195, buf196, arg65_1, arg66_1, buf199, 8388608, stream=stream0)
        del arg65_1
        del arg66_1
        buf200 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg67_1, buf200, 262144, 9, stream=stream0)
        del arg67_1
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56], Original ATen: [aten.silu, aten.convolution]
        buf201 = extern_kernels.convolution(buf199, buf200, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (1, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
        buf202 = buf191; del buf191  # reuse
        buf203 = buf190; del buf190  # reuse
        buf204 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_57], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_18.run(buf201, arg68_1, buf202, buf203, buf204, 65536, 128, stream=stream0)
        buf205 = buf194; del buf194  # reuse
        buf206 = buf193; del buf193  # reuse
        buf207 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_57], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf202, buf203, buf204, buf205, buf206, buf207, 512, 128, stream=stream0)
        buf208 = buf196; del buf196  # reuse
        buf209 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_57], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_20.run(buf205, buf206, buf207, buf208, buf209, 32, 16, stream=stream0)
        buf212 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_57, hidden_states_58], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_21.run(buf201, arg68_1, buf208, buf209, arg69_1, arg70_1, buf212, 8388608, stream=stream0)
        del arg68_1
        del arg69_1
        del arg70_1
        del buf201
        buf213 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_58, hidden_states_60], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg71_1, buf213, 262144, 9, stream=stream0)
        del arg71_1
        # Topologically Sorted Source Nodes: [hidden_states_58, hidden_states_60], Original ATen: [aten.silu, aten.convolution]
        buf214 = extern_kernels.convolution(buf212, buf213, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (1, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
        del buf212
        buf215 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_58, hidden_states_60, add_7, output_tensor_6], Original ATen: [aten.clone, aten._to_copy, aten.convolution, aten.silu, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_convolution_div_silu_24.run(buf215, arg56_1, buf188, arg64_1, buf214, arg72_1, 8388608, stream=stream0)
        del arg56_1
        del arg64_1
        del arg72_1
        del buf188
        buf216 = buf204; del buf204  # reuse
        buf217 = buf203; del buf203  # reuse
        buf218 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_61], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_25.run(buf215, buf216, buf217, buf218, 65536, 128, stream=stream0)
        buf219 = buf207; del buf207  # reuse
        buf220 = buf206; del buf206  # reuse
        buf221 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_61], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf216, buf217, buf218, buf219, buf220, buf221, 512, 128, stream=stream0)
        buf222 = buf209; del buf209  # reuse
        buf223 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_61], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_20.run(buf219, buf220, buf221, buf222, buf223, 32, 16, stream=stream0)
        buf226 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_62], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_26.run(buf215, buf222, buf223, arg73_1, arg74_1, buf226, 8388608, stream=stream0)
        del arg73_1
        del arg74_1
        buf227 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_62, hidden_states_63], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg75_1, buf227, 262144, 9, stream=stream0)
        del arg75_1
        # Topologically Sorted Source Nodes: [hidden_states_62, hidden_states_63], Original ATen: [aten.silu, aten.convolution]
        buf228 = extern_kernels.convolution(buf226, buf227, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (1, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
        buf229 = buf218; del buf218  # reuse
        buf230 = buf217; del buf217  # reuse
        buf231 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_64], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_18.run(buf228, arg76_1, buf229, buf230, buf231, 65536, 128, stream=stream0)
        buf232 = buf221; del buf221  # reuse
        buf233 = buf220; del buf220  # reuse
        buf234 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_64], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf229, buf230, buf231, buf232, buf233, buf234, 512, 128, stream=stream0)
        del buf229
        del buf230
        del buf231
        buf235 = buf223; del buf223  # reuse
        buf236 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_64], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_20.run(buf232, buf233, buf234, buf235, buf236, 32, 16, stream=stream0)
        buf239 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_64, hidden_states_65], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_21.run(buf228, arg76_1, buf235, buf236, arg77_1, arg78_1, buf239, 8388608, stream=stream0)
        del arg76_1
        del arg77_1
        del arg78_1
        del buf228
        buf240 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg79_1, buf240, 262144, 9, stream=stream0)
        del arg79_1
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67], Original ATen: [aten.silu, aten.convolution]
        buf241 = extern_kernels.convolution(buf239, buf240, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (1, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
        del buf239
        buf242 = empty_strided_cuda((1, 512, 256, 256), (33554432, 1, 131072, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten._unsafe_index, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_27.run(buf215, buf241, arg80_1, buf242, 33554432, stream=stream0)
        del arg80_1
        del buf215
        del buf241
        buf243 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten._unsafe_index, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_7.run(arg81_1, buf243, 262144, 9, stream=stream0)
        del arg81_1
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten._unsafe_index, aten.clone]
        buf244 = extern_kernels.convolution(buf242, buf243, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (1, 512, 256, 256), (33554432, 1, 131072, 512), 'torch.ops.aten.convolution.default')
        del buf243
        buf245 = empty_strided_cuda((1, 32, 1, 1, 14, 216), (97280, 3040, 97280, 97280, 216, 1), torch.float32)
        buf246 = empty_strided_cuda((1, 32, 1, 1, 14, 216), (97280, 3040, 97280, 97280, 216, 1), torch.float32)
        buf247 = empty_strided_cuda((1, 32, 1, 1, 14, 216), (97280, 3040, 97280, 97280, 216, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_70], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_28.run(buf244, arg82_1, buf245, buf246, buf247, 448, 216, 347, stream=stream0)
        buf248 = empty_strided_cuda((1, 32, 1, 1, 14), (448, 14, 448, 448, 1), torch.float32)
        buf249 = empty_strided_cuda((1, 32, 1, 1, 14), (448, 14, 448, 448, 1), torch.float32)
        buf250 = empty_strided_cuda((1, 32, 1, 1, 14), (448, 14, 448, 448, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_70], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_29.run(buf245, buf246, buf247, buf248, buf249, buf250, 448, 216, stream=stream0)
        del buf245
        del buf246
        del buf247
        buf251 = buf236; del buf236  # reuse
        buf252 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_70], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_30.run(buf248, buf249, buf250, buf251, buf252, 32, 14, stream=stream0)
        buf255 = buf242; del buf242  # reuse
        buf270 = empty_strided_cuda((1, 512, 256, 256), (33554432, 1, 131072, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, hidden_states_70, hidden_states_71, contiguous], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten._unsafe_index, aten.clone, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_native_group_norm_silu_31.run(buf244, arg82_1, buf251, buf252, arg83_1, arg84_1, buf255, buf270, 33554432, stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del buf244
        buf256 = empty_strided_cuda((256, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_32.run(arg85_1, buf256, 131072, 9, stream=stream0)
        del arg85_1
        # Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72], Original ATen: [aten.silu, aten.convolution]
        buf257 = extern_kernels.convolution(buf255, buf256, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (1, 256, 256, 256), (16777216, 1, 65536, 256), 'torch.ops.aten.convolution.default')
        del buf255
        del buf256
        buf258 = empty_strided_cuda((1, 32, 1, 1, 16, 256), (131072, 1, 131072, 131072, 8192, 32), torch.float32)
        buf259 = empty_strided_cuda((1, 32, 1, 1, 16, 256), (131072, 1, 131072, 131072, 8192, 32), torch.float32)
        buf260 = empty_strided_cuda((1, 32, 1, 1, 16, 256), (131072, 1, 131072, 131072, 8192, 32), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_73], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_33.run(buf257, arg86_1, buf258, buf259, buf260, 131072, 128, stream=stream0)
        buf261 = empty_strided_cuda((1, 32, 1, 1, 16, 2), (1024, 1, 1024, 1024, 64, 32), torch.float32)
        buf262 = empty_strided_cuda((1, 32, 1, 1, 16, 2), (1024, 1, 1024, 1024, 64, 32), torch.float32)
        buf263 = empty_strided_cuda((1, 32, 1, 1, 16, 2), (1024, 1, 1024, 1024, 64, 32), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_73], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_34.run(buf258, buf259, buf260, buf261, buf262, buf263, 1024, 128, stream=stream0)
        buf264 = reinterpret_tensor(buf234, (1, 32, 1, 1, 16), (512, 1, 512, 512, 32), 0); del buf234  # reuse
        buf265 = reinterpret_tensor(buf233, (1, 32, 1, 1, 16), (512, 1, 512, 512, 32), 0); del buf233  # reuse
        buf266 = reinterpret_tensor(buf232, (1, 32, 1, 1, 16), (512, 1, 512, 512, 32), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_73], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_35.run(buf261, buf262, buf263, buf264, buf265, buf266, 512, 2, stream=stream0)
        buf267 = buf252; del buf252  # reuse
        buf268 = buf251; del buf251  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_73], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_36.run(buf264, buf265, buf266, buf267, buf268, 32, 16, stream=stream0)
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, contiguous, input_tensor], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten._unsafe_index, aten.clone]
        buf271 = extern_kernels.convolution(buf270, arg91_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf271, (1, 256, 256, 256), (16777216, 1, 65536, 256), 'torch.ops.aten.convolution.default')
        del arg91_1
        buf273 = empty_strided_cuda((1, 256, 256, 256), (16777216, 1, 65536, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_73, hidden_states_74], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_37.run(buf257, arg86_1, buf267, buf268, arg87_1, arg88_1, buf273, 16777216, stream=stream0)
        del arg86_1
        del arg87_1
        del arg88_1
        del buf257
        buf274 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_74, hidden_states_76], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_38.run(arg89_1, buf274, 65536, 9, stream=stream0)
        del arg89_1
        # Topologically Sorted Source Nodes: [hidden_states_74, hidden_states_76], Original ATen: [aten.silu, aten.convolution]
        buf275 = extern_kernels.convolution(buf273, buf274, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (1, 256, 256, 256), (16777216, 1, 65536, 256), 'torch.ops.aten.convolution.default')
        buf276 = buf260; del buf260  # reuse
        buf277 = buf259; del buf259  # reuse
        buf278 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_77], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_39.run(buf271, arg92_1, buf275, arg90_1, buf276, buf277, buf278, 131072, 128, stream=stream0)
        buf279 = buf263; del buf263  # reuse
        buf280 = buf262; del buf262  # reuse
        buf281 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_77], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_34.run(buf276, buf277, buf278, buf279, buf280, buf281, 1024, 128, stream=stream0)
        buf282 = buf266; del buf266  # reuse
        buf283 = buf265; del buf265  # reuse
        buf284 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_77], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_35.run(buf279, buf280, buf281, buf282, buf283, buf284, 512, 2, stream=stream0)
        buf285 = buf268; del buf268  # reuse
        buf286 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_77], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_36.run(buf282, buf283, buf284, buf285, buf286, 32, 16, stream=stream0)
        buf289 = buf273; del buf273  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_77, hidden_states_78], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_40.run(buf271, arg92_1, buf275, arg90_1, buf285, buf286, arg93_1, arg94_1, buf289, 16777216, stream=stream0)
        del arg93_1
        del arg94_1
        buf290 = buf274; del buf274  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_78, hidden_states_79], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_38.run(arg95_1, buf290, 65536, 9, stream=stream0)
        del arg95_1
        # Topologically Sorted Source Nodes: [hidden_states_78, hidden_states_79], Original ATen: [aten.silu, aten.convolution]
        buf291 = extern_kernels.convolution(buf289, buf290, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (1, 256, 256, 256), (16777216, 1, 65536, 256), 'torch.ops.aten.convolution.default')
        buf292 = buf278; del buf278  # reuse
        buf293 = buf277; del buf277  # reuse
        buf294 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_80], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_33.run(buf291, arg96_1, buf292, buf293, buf294, 131072, 128, stream=stream0)
        buf295 = buf281; del buf281  # reuse
        buf296 = buf280; del buf280  # reuse
        buf297 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_80], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_34.run(buf292, buf293, buf294, buf295, buf296, buf297, 1024, 128, stream=stream0)
        buf298 = buf284; del buf284  # reuse
        buf299 = buf283; del buf283  # reuse
        buf300 = buf282; del buf282  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_80], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_35.run(buf295, buf296, buf297, buf298, buf299, buf300, 512, 2, stream=stream0)
        buf301 = buf286; del buf286  # reuse
        buf302 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_80], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_36.run(buf298, buf299, buf300, buf301, buf302, 32, 16, stream=stream0)
        buf305 = buf289; del buf289  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_80, hidden_states_81], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_37.run(buf291, arg96_1, buf301, buf302, arg97_1, arg98_1, buf305, 16777216, stream=stream0)
        del arg96_1
        del arg97_1
        del arg98_1
        del buf291
        buf306 = buf290; del buf290  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_81, hidden_states_83], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_38.run(arg99_1, buf306, 65536, 9, stream=stream0)
        del arg99_1
        # Topologically Sorted Source Nodes: [hidden_states_81, hidden_states_83], Original ATen: [aten.silu, aten.convolution]
        buf307 = extern_kernels.convolution(buf305, buf306, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf307, (1, 256, 256, 256), (16777216, 1, 65536, 256), 'torch.ops.aten.convolution.default')
        del buf305
        buf308 = buf271; del buf271  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, contiguous, input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_81, hidden_states_83, add_10, output_tensor_9], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten._unsafe_index, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_41.run(buf308, arg92_1, buf275, arg90_1, buf307, arg100_1, 16777216, stream=stream0)
        del arg100_1
        del arg90_1
        del arg92_1
        del buf275
        buf309 = buf294; del buf294  # reuse
        buf310 = buf293; del buf293  # reuse
        buf311 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_84], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_42.run(buf308, buf309, buf310, buf311, 131072, 128, stream=stream0)
        buf312 = buf297; del buf297  # reuse
        buf313 = buf296; del buf296  # reuse
        buf314 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_84], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_34.run(buf309, buf310, buf311, buf312, buf313, buf314, 1024, 128, stream=stream0)
        buf315 = buf300; del buf300  # reuse
        buf316 = buf299; del buf299  # reuse
        buf317 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_84], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_35.run(buf312, buf313, buf314, buf315, buf316, buf317, 512, 2, stream=stream0)
        buf318 = buf302; del buf302  # reuse
        buf319 = buf301; del buf301  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_84], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_36.run(buf315, buf316, buf317, buf318, buf319, 32, 16, stream=stream0)
        buf322 = buf307; del buf307  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_84, hidden_states_85], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_43.run(buf308, buf318, buf319, arg101_1, arg102_1, buf322, 16777216, stream=stream0)
        del arg101_1
        del arg102_1
        buf323 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_86], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_38.run(arg103_1, buf323, 65536, 9, stream=stream0)
        del arg103_1
        # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_86], Original ATen: [aten.silu, aten.convolution]
        buf324 = extern_kernels.convolution(buf322, buf323, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf324, (1, 256, 256, 256), (16777216, 1, 65536, 256), 'torch.ops.aten.convolution.default')
        buf325 = buf311; del buf311  # reuse
        buf326 = buf310; del buf310  # reuse
        buf327 = buf309; del buf309  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_87], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_33.run(buf324, arg104_1, buf325, buf326, buf327, 131072, 128, stream=stream0)
        buf328 = buf314; del buf314  # reuse
        buf329 = buf313; del buf313  # reuse
        buf330 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_87], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_34.run(buf325, buf326, buf327, buf328, buf329, buf330, 1024, 128, stream=stream0)
        del buf325
        del buf326
        del buf327
        buf331 = buf317; del buf317  # reuse
        buf332 = buf316; del buf316  # reuse
        buf333 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_87], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_35.run(buf328, buf329, buf330, buf331, buf332, buf333, 512, 2, stream=stream0)
        del buf328
        del buf329
        del buf330
        buf334 = buf319; del buf319  # reuse
        buf335 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_87], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_36.run(buf331, buf332, buf333, buf334, buf335, 32, 16, stream=stream0)
        del buf331
        del buf332
        del buf333
        buf338 = buf322; del buf322  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_87, hidden_states_88], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_37.run(buf324, arg104_1, buf334, buf335, arg105_1, arg106_1, buf338, 16777216, stream=stream0)
        del arg104_1
        del arg105_1
        del arg106_1
        del buf324
        buf339 = buf323; del buf323  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_38.run(arg107_1, buf339, 65536, 9, stream=stream0)
        del arg107_1
        # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90], Original ATen: [aten.silu, aten.convolution]
        buf340 = extern_kernels.convolution(buf338, buf339, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf340, (1, 256, 256, 256), (16777216, 1, 65536, 256), 'torch.ops.aten.convolution.default')
        del buf338
        buf341 = empty_strided_cuda((1, 256, 512, 512), (67108864, 1, 131072, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_div_silu_44.run(buf308, buf340, arg108_1, buf341, 67108864, stream=stream0)
        del arg108_1
        del buf308
        del buf340
        buf342 = buf339; del buf339  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91, hidden_states_92], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_38.run(arg109_1, buf342, 65536, 9, stream=stream0)
        del arg109_1
        # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91, hidden_states_92], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten._unsafe_index]
        buf343 = extern_kernels.convolution(buf341, buf342, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf343, (1, 256, 512, 512), (67108864, 1, 131072, 256), 'torch.ops.aten.convolution.default')
        del buf342
        buf344 = buf343; del buf343  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91, hidden_states_92], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_div_silu_45.run(buf344, arg110_1, 67108864, stream=stream0)
        del arg110_1
        buf345 = empty_strided_cuda((1, 32, 1, 1, 14, 230), (103040, 1, 103040, 103040, 7360, 32), torch.float32)
        buf346 = empty_strided_cuda((1, 32, 1, 1, 14, 230), (103040, 1, 103040, 103040, 7360, 32), torch.float32)
        buf347 = empty_strided_cuda((1, 32, 1, 1, 14, 230), (103040, 1, 103040, 103040, 7360, 32), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_46.run(buf344, buf345, buf346, buf347, 103040, 652, stream=stream0)
        buf348 = empty_strided_cuda((1, 32, 1, 1, 14, 2), (896, 1, 896, 896, 64, 32), torch.float32)
        buf349 = empty_strided_cuda((1, 32, 1, 1, 14, 2), (896, 1, 896, 896, 64, 32), torch.float32)
        buf350 = empty_strided_cuda((1, 32, 1, 1, 14, 2), (896, 1, 896, 896, 64, 32), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_47.run(buf345, buf346, buf347, buf348, buf349, buf350, 896, 115, stream=stream0)
        del buf345
        del buf346
        del buf347
        buf351 = reinterpret_tensor(buf250, (1, 32, 1, 1, 14), (448, 1, 448, 448, 32), 0); del buf250  # reuse
        buf352 = reinterpret_tensor(buf249, (1, 32, 1, 1, 14), (448, 1, 448, 448, 32), 0); del buf249  # reuse
        buf353 = reinterpret_tensor(buf248, (1, 32, 1, 1, 14), (448, 1, 448, 448, 32), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_48.run(buf348, buf349, buf350, buf351, buf352, buf353, 448, 2, stream=stream0)
        buf354 = buf335; del buf335  # reuse
        buf355 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_49.run(buf351, buf352, buf353, buf354, buf355, 32, 14, stream=stream0)
        buf358 = buf341; del buf341  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_93, hidden_states_94], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_50.run(buf344, buf354, buf355, arg111_1, arg112_1, buf358, 67108864, stream=stream0)
        del arg111_1
        del arg112_1
        buf359 = empty_strided_cuda((128, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_51.run(arg113_1, buf359, 32768, 9, stream=stream0)
        del arg113_1
        # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95], Original ATen: [aten.silu, aten.convolution]
        buf360 = extern_kernels.convolution(buf358, buf359, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf360, (1, 128, 512, 512), (33554432, 1, 65536, 128), 'torch.ops.aten.convolution.default')
        del buf358
        del buf359
        buf361 = empty_strided_cuda((1, 32, 1, 1, 14, 216), (96768, 1, 96768, 96768, 6912, 32), torch.float32)
        buf362 = empty_strided_cuda((1, 32, 1, 1, 14, 216), (96768, 1, 96768, 96768, 6912, 32), torch.float32)
        buf363 = empty_strided_cuda((1, 32, 1, 1, 14, 216), (96768, 1, 96768, 96768, 6912, 32), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_96], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_52.run(buf360, arg114_1, buf361, buf362, buf363, 96768, 347, stream=stream0)
        buf364 = buf350; del buf350  # reuse
        buf365 = buf349; del buf349  # reuse
        buf366 = buf348; del buf348  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_96], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_53.run(buf361, buf362, buf363, buf364, buf365, buf366, 896, 108, stream=stream0)
        buf367 = buf353; del buf353  # reuse
        buf368 = buf352; del buf352  # reuse
        buf369 = buf351; del buf351  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_96], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_48.run(buf364, buf365, buf366, buf367, buf368, buf369, 448, 2, stream=stream0)
        buf370 = buf355; del buf355  # reuse
        buf371 = buf354; del buf354  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_96], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_49.run(buf367, buf368, buf369, buf370, buf371, 32, 14, stream=stream0)
        # Topologically Sorted Source Nodes: [input_tensor_1], Original ATen: [aten.convolution]
        buf373 = extern_kernels.convolution(buf344, arg119_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf373, (1, 128, 512, 512), (33554432, 1, 65536, 128), 'torch.ops.aten.convolution.default')
        del arg119_1
        del buf344
        buf375 = reinterpret_tensor(buf270, (1, 128, 512, 512), (33554432, 1, 65536, 128), 0); del buf270  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_96, hidden_states_97], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_54.run(buf360, arg114_1, buf370, buf371, arg115_1, arg116_1, buf375, 33554432, stream=stream0)
        del arg114_1
        del arg115_1
        del arg116_1
        del buf360
        buf376 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_97, hidden_states_99], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_55.run(arg117_1, buf376, 16384, 9, stream=stream0)
        del arg117_1
        # Topologically Sorted Source Nodes: [hidden_states_97, hidden_states_99], Original ATen: [aten.silu, aten.convolution]
        buf377 = extern_kernels.convolution(buf375, buf376, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf377, (1, 128, 512, 512), (33554432, 1, 65536, 128), 'torch.ops.aten.convolution.default')
        buf378 = buf363; del buf363  # reuse
        buf379 = buf362; del buf362  # reuse
        buf380 = buf361; del buf361  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_56.run(buf373, arg120_1, buf377, arg118_1, buf378, buf379, buf380, 96768, 347, stream=stream0)
        buf381 = buf366; del buf366  # reuse
        buf382 = buf365; del buf365  # reuse
        buf383 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_53.run(buf378, buf379, buf380, buf381, buf382, buf383, 896, 108, stream=stream0)
        buf384 = buf369; del buf369  # reuse
        buf385 = buf368; del buf368  # reuse
        buf386 = buf367; del buf367  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_48.run(buf381, buf382, buf383, buf384, buf385, buf386, 448, 2, stream=stream0)
        buf387 = buf371; del buf371  # reuse
        buf388 = buf370; del buf370  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_49.run(buf384, buf385, buf386, buf387, buf388, 32, 14, stream=stream0)
        buf391 = buf375; del buf375  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_101], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_57.run(buf373, arg120_1, buf377, arg118_1, buf387, buf388, arg121_1, arg122_1, buf391, 33554432, stream=stream0)
        del arg121_1
        del arg122_1
        buf392 = buf376; del buf376  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_55.run(arg123_1, buf392, 16384, 9, stream=stream0)
        del arg123_1
        # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102], Original ATen: [aten.silu, aten.convolution]
        buf393 = extern_kernels.convolution(buf391, buf392, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf393, (1, 128, 512, 512), (33554432, 1, 65536, 128), 'torch.ops.aten.convolution.default')
        buf394 = buf380; del buf380  # reuse
        buf395 = buf379; del buf379  # reuse
        buf396 = buf378; del buf378  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_103], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_52.run(buf393, arg124_1, buf394, buf395, buf396, 96768, 347, stream=stream0)
        buf397 = buf383; del buf383  # reuse
        buf398 = buf382; del buf382  # reuse
        buf399 = buf381; del buf381  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_103], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_53.run(buf394, buf395, buf396, buf397, buf398, buf399, 896, 108, stream=stream0)
        buf400 = buf386; del buf386  # reuse
        buf401 = buf385; del buf385  # reuse
        buf402 = buf384; del buf384  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_103], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_48.run(buf397, buf398, buf399, buf400, buf401, buf402, 448, 2, stream=stream0)
        buf403 = buf388; del buf388  # reuse
        buf404 = buf387; del buf387  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_103], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_49.run(buf400, buf401, buf402, buf403, buf404, 32, 14, stream=stream0)
        buf407 = buf391; del buf391  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_103, hidden_states_104], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_54.run(buf393, arg124_1, buf403, buf404, arg125_1, arg126_1, buf407, 33554432, stream=stream0)
        del arg124_1
        del arg125_1
        del arg126_1
        del buf393
        buf408 = buf392; del buf392  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_104, hidden_states_106], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_55.run(arg127_1, buf408, 16384, 9, stream=stream0)
        del arg127_1
        # Topologically Sorted Source Nodes: [hidden_states_104, hidden_states_106], Original ATen: [aten.silu, aten.convolution]
        buf409 = extern_kernels.convolution(buf407, buf408, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf409, (1, 128, 512, 512), (33554432, 1, 65536, 128), 'torch.ops.aten.convolution.default')
        del buf407
        buf410 = buf373; del buf373  # reuse
        # Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_104, hidden_states_106, add_13, output_tensor_12], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_silu_58.run(buf410, arg120_1, buf377, arg118_1, buf409, arg128_1, 33554432, stream=stream0)
        del arg118_1
        del arg120_1
        del arg128_1
        del buf377
        buf411 = buf396; del buf396  # reuse
        buf412 = buf395; del buf395  # reuse
        buf413 = buf394; del buf394  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_107], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_59.run(buf410, buf411, buf412, buf413, 96768, 347, stream=stream0)
        buf414 = buf399; del buf399  # reuse
        buf415 = buf398; del buf398  # reuse
        buf416 = buf397; del buf397  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_107], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_53.run(buf411, buf412, buf413, buf414, buf415, buf416, 896, 108, stream=stream0)
        buf417 = buf402; del buf402  # reuse
        buf418 = buf401; del buf401  # reuse
        buf419 = buf400; del buf400  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_107], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_48.run(buf414, buf415, buf416, buf417, buf418, buf419, 448, 2, stream=stream0)
        buf420 = buf404; del buf404  # reuse
        buf421 = buf403; del buf403  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_107], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_49.run(buf417, buf418, buf419, buf420, buf421, 32, 14, stream=stream0)
        buf424 = buf409; del buf409  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_107, hidden_states_108], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_60.run(buf410, buf420, buf421, arg129_1, arg130_1, buf424, 33554432, stream=stream0)
        del arg129_1
        del arg130_1
        buf425 = buf408; del buf408  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_109], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_55.run(arg131_1, buf425, 16384, 9, stream=stream0)
        del arg131_1
        # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_109], Original ATen: [aten.silu, aten.convolution]
        buf426 = extern_kernels.convolution(buf424, buf425, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf426, (1, 128, 512, 512), (33554432, 1, 65536, 128), 'torch.ops.aten.convolution.default')
        buf427 = buf413; del buf413  # reuse
        buf428 = buf412; del buf412  # reuse
        buf429 = buf411; del buf411  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_110], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_52.run(buf426, arg132_1, buf427, buf428, buf429, 96768, 347, stream=stream0)
        buf430 = buf416; del buf416  # reuse
        buf431 = buf415; del buf415  # reuse
        buf432 = buf414; del buf414  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_110], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_53.run(buf427, buf428, buf429, buf430, buf431, buf432, 896, 108, stream=stream0)
        buf433 = buf419; del buf419  # reuse
        buf434 = buf418; del buf418  # reuse
        buf435 = buf417; del buf417  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_110], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_48.run(buf430, buf431, buf432, buf433, buf434, buf435, 448, 2, stream=stream0)
        buf436 = buf421; del buf421  # reuse
        buf437 = buf420; del buf420  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_110], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_49.run(buf433, buf434, buf435, buf436, buf437, 32, 14, stream=stream0)
        buf440 = buf424; del buf424  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_110, hidden_states_111], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_54.run(buf426, arg132_1, buf436, buf437, arg133_1, arg134_1, buf440, 33554432, stream=stream0)
        del arg132_1
        del arg133_1
        del arg134_1
        del buf426
        buf441 = buf425; del buf425  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_55.run(arg135_1, buf441, 16384, 9, stream=stream0)
        del arg135_1
        # Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113], Original ATen: [aten.silu, aten.convolution]
        buf442 = extern_kernels.convolution(buf440, buf441, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf442, (1, 128, 512, 512), (33554432, 1, 65536, 128), 'torch.ops.aten.convolution.default')
        del buf441
        buf443 = buf429; del buf429  # reuse
        buf444 = buf428; del buf428  # reuse
        buf445 = buf427; del buf427  # reuse
        # Topologically Sorted Source Nodes: [sample_2], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_61.run(buf410, buf442, arg136_1, buf443, buf444, buf445, 96768, 347, stream=stream0)
        buf446 = buf432; del buf432  # reuse
        buf447 = buf431; del buf431  # reuse
        buf448 = buf430; del buf430  # reuse
        # Topologically Sorted Source Nodes: [sample_2], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_53.run(buf443, buf444, buf445, buf446, buf447, buf448, 896, 108, stream=stream0)
        del buf443
        del buf444
        del buf445
        buf449 = buf435; del buf435  # reuse
        buf450 = buf434; del buf434  # reuse
        buf451 = buf433; del buf433  # reuse
        # Topologically Sorted Source Nodes: [sample_2], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_48.run(buf446, buf447, buf448, buf449, buf450, buf451, 448, 2, stream=stream0)
        del buf446
        del buf447
        del buf448
        buf452 = buf437; del buf437  # reuse
        buf453 = buf436; del buf436  # reuse
        # Topologically Sorted Source Nodes: [sample_2], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_49.run(buf449, buf450, buf451, buf452, buf453, 32, 14, stream=stream0)
        del buf449
        del buf450
        del buf451
        buf456 = buf440; del buf440  # reuse
        # Topologically Sorted Source Nodes: [sample_2, sample_3], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_62.run(buf410, buf442, arg136_1, buf452, buf453, arg137_1, arg138_1, buf456, 33554432, stream=stream0)
        del arg136_1
        del arg137_1
        del arg138_1
        del buf410
        del buf442
        del buf452
        del buf453
        buf457 = empty_strided_cuda((3, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [sample_3, sample_4], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_63.run(arg139_1, buf457, 384, 9, stream=stream0)
        del arg139_1
        # Topologically Sorted Source Nodes: [sample_3, sample_4], Original ATen: [aten.silu, aten.convolution]
        buf458 = extern_kernels.convolution(buf456, buf457, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf458, (1, 3, 512, 512), (786432, 1, 1536, 3), 'torch.ops.aten.convolution.default')
        del buf456
        del buf457
        buf459 = empty_strided_cuda((1, 3, 512, 512), (786432, 262144, 512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [sample_3, sample_4], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_64.run(buf458, arg140_1, buf459, 3, 262144, stream=stream0)
        del arg140_1
        del buf458
    return (buf459, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((1, 4, 64, 64), (16384, 4096, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((512, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg15_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg16_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg17_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg18_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg19_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg20_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg21_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg22_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg23_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg24_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg25_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg26_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg27_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg28_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg29_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg30_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg31_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg32_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg33_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg34_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg35_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg36_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg37_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg38_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg39_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg40_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg41_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg42_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg43_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg44_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg45_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg46_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg47_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg48_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg49_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg50_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg51_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg52_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg53_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg54_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg55_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg56_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg57_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
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
    arg81_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg82_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg83_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg84_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg85_1 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg86_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg87_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg88_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg89_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg90_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg91_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg92_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg93_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg94_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg95_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg96_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg97_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg98_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg99_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg100_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg101_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg102_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg103_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg104_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg105_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg106_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg107_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg108_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg109_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg110_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg111_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg112_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg113_1 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg114_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg115_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg116_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg117_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg118_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg119_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg120_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg121_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg122_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg123_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg124_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg125_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg126_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg127_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg128_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg129_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg130_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg131_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg132_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg133_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg134_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg135_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg136_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg137_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg138_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg139_1 = rand_strided((3, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg140_1 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
