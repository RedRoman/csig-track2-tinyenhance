"""Owned-output, fixed-tile TensorRT runtime. No image/result cache or RNG calls."""
from pathlib import Path
import hashlib


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


class TensorRTEngine:
    def __init__(self,path,expected_sha,device):
        import torch
        import tensorrt as trt
        import torch_tensorrt  # Registers any compiler-owned TRT plugins.
        self.torch=torch;self.trt=trt;self.device=torch.device(device)
        if sha(path)!=expected_sha:raise RuntimeError('Engine SHA differs')
        self.logger=trt.Logger(trt.Logger.WARNING);trt.init_libnvinfer_plugins(self.logger,'')
        self.runtime=trt.Runtime(self.logger);self.engine=self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:raise RuntimeError('TensorRT engine deserialization failed')
        self.context=self.engine.create_execution_context()
        if self.context is None:raise RuntimeError('TensorRT context construction failed')
        names=[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs=[n for n in names if self.engine.get_tensor_mode(n)==trt.TensorIOMode.INPUT]
        self.outputs=[n for n in names if self.engine.get_tensor_mode(n)==trt.TensorIOMode.OUTPUT]
        if len(self.inputs)!=1 or len(self.outputs)!=1:raise RuntimeError('Exactly one tensor input/output required')
        self.types={trt.DataType.FLOAT:torch.float32,trt.DataType.HALF:torch.float16,trt.DataType.BF16:torch.bfloat16}
        self.input_shape=tuple(self.engine.get_tensor_shape(self.inputs[0]));self.output_shape=tuple(self.engine.get_tensor_shape(self.outputs[0]))
        self.input_dtype=self.types[self.engine.get_tensor_dtype(self.inputs[0])];self.output_dtype=self.types[self.engine.get_tensor_dtype(self.outputs[0])]
        if min(self.input_shape+self.output_shape)<=0:raise RuntimeError('Engine must have only fixed positive dimensions')

    def __call__(self,x):
        torch=self.torch
        if x.device!=self.device or x.dtype!=self.input_dtype or tuple(x.shape)!=self.input_shape:raise RuntimeError('Wrong tile shape/device/dtype for sealed engine')
        # Every call allocates its own Torch-owned output, so a later enqueue
        # cannot overwrite a previously returned tensor. Contiguity copies are
        # ordinary per-tile computation and remain inside Runner.infer timing.
        x=x.contiguous();out=torch.empty(self.output_shape,device=x.device,dtype=self.output_dtype)
        if not self.context.set_tensor_address(self.inputs[0],x.data_ptr()):raise RuntimeError('Input binding failed')
        if not self.context.set_tensor_address(self.outputs[0],out.data_ptr()):raise RuntimeError('Output binding failed')
        stream=torch.cuda.current_stream(x.device)
        if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):raise RuntimeError('TensorRT enqueue failed')
        x.record_stream(stream);out.record_stream(stream)
        return out
