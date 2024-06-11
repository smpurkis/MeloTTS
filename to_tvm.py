import onnx
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor

onnx_model = onnx.load("/home/ubuntu/projects/oobabooga_linux/model_dec.onnx")

x_in = np.load(
    "/home/ubuntu/projects/oobabooga_linux/tts/MeloTTS/to_onnx_test/x_in.npy"
)
g_in = np.load(
    "/home/ubuntu/projects/oobabooga_linux/tts/MeloTTS/to_onnx_test/g_in.npy"
)
gt_o_output = np.load(
    "/home/ubuntu/projects/oobabooga_linux/tts/MeloTTS/to_onnx_test/o_out.npy"
)

target = "llvm"

shape_dict = {"x.3": x_in.shape, "g": g_in.shape}

print("Converting to Relay IR...")
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

print("Compiling...")
with tvm.transform.PassContext(opt_level=0):
    lib = relay.build(mod, target=target, params=params)

print("Loading on device...")
dev = tvm.device(str(target), 0)

print("Loading module...")
module = graph_executor.GraphModule(lib["default"](dev))

print("Setting inputs...")
type = "float32"
module.set_input("x.3", x_in)
module.set_input("g", g_in)

print("Running...")
module.run()

output_shape = gt_o_output.shape
tvm_output: np.ndarray = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
print("Done!")
print(f"Output: {tvm_output.shape}")


# print("Compiling Executor...")
# with tvm.transform.PassContext(opt_level=1):
#     executor = relay.build_module.create_executor(
#         "graph", mod, tvm.cpu(0), target, params
#     ).evaluate()

# print("Running...")
# dtype = "float32"
# tvm_output = executor(
#     tvm.nd.array(x_in.astype(dtype)), tvm.nd.array(g_in.astype(dtype))
# ).numpy()

# print("Done!")
# print(f"Output: {tvm_output.shape}")
