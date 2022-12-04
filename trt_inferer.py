import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import math
import numpy as np
import cv2

print(f"using pyTensorRT {trt.__version__}")


class MyLogger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)

    def log(self, severity, msg):
        if (severity != trt.tensorrt.ILogger.Severity.VERBOSE):
            print(msg)
            pass # Your custom logging implementation here

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTInference:
    def __init__(self, trt_model_path = "/docker/trt_models/yolov5/yolov5m6_3x768x1280_b1.engine"):
        self.trt_model_path = trt_model_path
        self.runtime = trt.Runtime(TRT_LOGGER) 
        self.batch_size = 1
        trt.init_libnvinfer_plugins(None, "") # https://github.com/onnx/onnx-tensorrt/issues/597
        self.prepare_engine()
    
    def prepare_engine(self):
        self.load_engine()
        self.set_infer_shapes()
        self.set_bindings()
    
    def load_engine(self):
        with open(self.trt_model_path, "rb") as f: 
            serialized_engine = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()
    
    def set_infer_shapes(self):
        assert (self.engine.num_bindings == self.engine.num_io_tensors == 2), "io_tensors_num != 2"
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        
        self.input_shape = (self.batch_size,*self.engine.get_tensor_shape(self.input_name)[1:])
        self.output_shape = (self.batch_size,*self.engine.get_tensor_shape(self.output_name)[1:])
        
        print("inputs shape:", self.input_shape)
        print("outputs shape:", self.output_shape)
        
        self.context.set_input_shape(self.input_name, self.input_shape)
        
    def set_bindings(self):
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        print("input_dtype", self.input_dtype)
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
        print("output_dtype", self.output_dtype)
        
        input_placeholder = np.empty(self.input_shape, dtype = self.input_dtype)
        self.output_placeholder = np.empty(self.output_shape, dtype = self.output_dtype)
        
        self.input_nbytes = input_placeholder.nbytes
        self.output_nbytes = self.output_placeholder.nbytes
        
        self.d_input = cuda.mem_alloc(self.input_nbytes)
        self.d_outputs = cuda.mem_alloc(self.output_nbytes)
        
        self.bindings = [int( self.d_input), int(self.d_outputs)]
        
        self.stream = cuda.Stream()
        self.context.set_optimization_profile_async(0, self.stream.handle) #SET THE PROFILE
        
        
    def infer(self, inputs: np.ndarray):
        # transfer input data to device
        cuda.memcpy_htod_async(self.d_input, inputs, self.stream)
        # execute model
        success = self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        if success == 0:
            print("execution failure")
        # transfer predictions back
        cuda.memcpy_dtoh_async(self.output_placeholder, self.d_outputs, self.stream)
        # syncronize threads
        self.stream.synchronize()
        
        return self.output_placeholder
    
    def load_image(self, img_p: str):
        image_np = cv2.imread(img_p)[..., ::-1] # RGB
        H, W = image_np.shape[:2]
        if (H != self.input_shape[2] or W != self.input_shape[3]):
            image_np = cv2.resize(image_np, (self.input_shape[3], self.input_shape[2]), interpolation = cv2.INTER_LINEAR)
            
        image_np = (image_np / 255).astype(self.input_dtype)
        image_np = image_np.transpose(2,0,1)[np.newaxis] # [1, 3, 768, 1280]
        return image_np.copy()
        
    
    
if __name__ == "__main__":
    
    trt_model_path = "/docker/trt_models/yolov5/yolov5m6_3x768x1280_b1.engine"

    inferer = TRTInference(trt_model_path)
    
    inputs = np.random.randn(*inferer.input_shape).astype(inferer.input_dtype)
    
    outp = inferer.infer(inputs)
    print(outp.shape)
    
    
    
        
        
