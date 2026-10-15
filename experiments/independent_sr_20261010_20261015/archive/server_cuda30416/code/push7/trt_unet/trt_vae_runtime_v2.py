"""Fixed-tile TRT, dedicated non-default stream, fresh owned outputs.

The caller stream waits for all engine work before it can consume an output.
Events on the caller stream around __call__ include the complete engine work.
No synchronization is omitted from a measurement; no tensor/image cache.
"""
from pathlib import Path
import hashlib


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def contiguous_strides(shape):
    stride=1;out=[]
    for n in reversed(shape):out.append(stride);stride*=n
    return tuple(reversed(out))


class TensorRTEngine:
    def __init__(self,path,expected_sha,device):
        import torch
        import tensorrt as trt
        import torch_tensorrt  # Register compiler-owned TensorRT plugins.
        self.torch=torch;self.trt=trt;self.device=torch.device(device)
        if sha(path)!=expected_sha:raise RuntimeError('Engine SHA differs')
        with torch.cuda.device(self.device):
            self.stream=torch.cuda.Stream(device=self.device)
            if self.stream.cuda_stream==0:raise RuntimeError('Dedicated stream cannot be default')
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
        if min(self.input_shape+self.output_shape)<=0:raise RuntimeError('Only fixed positive dimensions permitted')
        self.io_records=[]
        for name,shape in [(self.inputs[0],self.input_shape),(self.outputs[0],self.output_shape)]:
            fmt=self.engine.get_tensor_format(name);strides=tuple(self.context.get_tensor_strides(name))
            if fmt!=trt.TensorFormat.LINEAR or strides!=contiguous_strides(shape):raise RuntimeError('Engine IO must be contiguous LINEAR; packed strides are unsupported')
            self.io_records.append(dict(name=name,shape=list(shape),dtype=str(self.engine.get_tensor_dtype(name)),format=str(fmt),format_description=self.engine.get_tensor_format_desc(name),strides=list(strides),location=str(self.engine.get_tensor_location(name))))
            if self.engine.get_tensor_location(name)!=trt.TensorLocation.DEVICE:raise RuntimeError('Only device IO supported')

    def __call__(self,x):
        torch=self.torch
        if x.device!=self.device or x.dtype!=self.input_dtype or tuple(x.shape)!=self.input_shape:raise RuntimeError('Wrong sealed engine tile contract')
        caller=torch.cuda.current_stream(x.device)
        # Input conversion/allocation are caller work. Worker waits for all of
        # it, including caller timing-start events, before enqueueing TensorRT.
        x=x.contiguous();out=torch.empty(self.output_shape,device=x.device,dtype=self.output_dtype)
        self.stream.wait_stream(caller)
        if not self.context.set_tensor_address(self.inputs[0],x.data_ptr()):raise RuntimeError('Input binding failed')
        if not self.context.set_tensor_address(self.outputs[0],out.data_ptr()):raise RuntimeError('Output binding failed')
        if not self.context.execute_async_v3(stream_handle=self.stream.cuda_stream):raise RuntimeError('TensorRT enqueue failed')
        # Protect fresh tensors until the worker finishes. Caller wait also
        # orders downstream consumers and caller timing-end events correctly.
        x.record_stream(self.stream);out.record_stream(self.stream)
        caller.wait_stream(self.stream)
        return out
