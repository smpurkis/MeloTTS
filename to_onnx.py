from pathlib import Path
from time import time

import numpy as np
from melo.api import TTS
import torch
from torch import nn
from torch.utils.mobile_optimizer import optimize_for_mobile

# Speed is adjustable
speed = 1.0

# CPU is sufficient for real-time inference.
# You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device = "cpu"  # Will automatically use GPU if available

# English
text = "Did you ever hear a folk tale about a giant turtle? It's a story about a turtle that carries the world on its back."

# Load the model
s = time()
model = TTS(language="EN", device=device)
print(f"Loaded model in {time() - s:.2f}s")
speaker_ids = model.hps.data.spk2id

model_model_dec = torch.jit.script(
    model.model.dec,
    example_inputs=(torch.zeros([1, 192, 299]), torch.zeros([1, 256, 1])),
)

# American accent
output_path = "en-us.wav"
s = time()
model.tts_to_file(text, speaker_ids["EN-US"], output_path, speed=speed)
print(f"Elapsed time: {time() - s:.2f}s")


# s = time()
# model.tts_to_file(text, speaker_ids["EN-US"], output_path, speed=speed)
# print(f"Elapsed time: {time() - s:.2f}s")

# print(f"Type before script: {type(model.model.dec)}")
# model.model.dec = torch.jit.script(
#     model.model.dec,
#     example_inputs=(torch.zeros([1, 192, 299]), torch.zeros([1, 256, 1])),
# )
# print(f"Type after script: {type(model.model.dec)}")

# s = time()
# model.tts_to_file(text, speaker_ids["EN-US"], output_path, speed=speed)
# print(f"Elapsed time: {time() - s:.2f}s")

# Path(output_path).unlink(missing_ok=True)

# ground truth for dec
x_in = np.load("/Users/user/demo_1/tts-pg/MeloTTS/to_onnx_test/x_input.npy")
g_in = np.load("/Users/user/demo_1/tts-pg/MeloTTS/to_onnx_test/g_input.npy")
gt_o_output = np.load("/Users/user/demo_1/tts-pg/MeloTTS/to_onnx_test/o_output.npy")

with torch.no_grad():
    model_model_dec.eval()
    s = time()
    torch_o_output = (
        model_model_dec(torch.tensor(x_in), torch.tensor(g_in)).detach().numpy()
    )
    print(f"Elapsed time: {time() - s:.2f}s")
    assert np.allclose(torch_o_output, gt_o_output, atol=1e-4)

    print("Scripted fp32 Torch model works correctly!")

    quantized_torch_model = torch.quantization.quantize_dynamic(
        model_model_dec,
        {
            nn.Linear,
            nn.LSTM,
            nn.GRU,
            nn.LSTMCell,
            nn.RNNCell,
            nn.GRUCell,
        },
        dtype=torch.qint8,
    )

    s = time()
    torch_o_output = (
        quantized_torch_model(torch.tensor(x_in), torch.tensor(g_in)).detach().numpy()
    )
    print(f"Quantized Elapsed time: {time() - s:.2f}s")
    assert np.allclose(torch_o_output, gt_o_output, atol=1e-4)

    print("Quantized Torch model works correctly!")

    mobile_optimized_model = optimize_for_mobile(model_model_dec)

    s = time()
    torch_o_output = (
        mobile_optimized_model(torch.tensor(x_in), torch.tensor(g_in)).detach().numpy()
    )
    print(f"Mobile optimized Elapsed time: {time() - s:.2f}s")
    assert np.allclose(torch_o_output, gt_o_output, atol=1e-4)

    print("Mobile optimized Torch model works correctly!")

    quant_mobile_optimized_model = optimize_for_mobile(quantized_torch_model)

    s = time()
    torch_o_output = (
        quant_mobile_optimized_model(torch.tensor(x_in), torch.tensor(g_in))
        .detach()
        .numpy()
    )
    print(f"Quant Mobile optimized Elapsed time: {time() - s:.2f}s")
    assert np.allclose(torch_o_output, gt_o_output, atol=1e-4)

    print("Mobile optimized Torch model works correctly!")

    # use torch.compile

    # model_compiled = torch.compile(model.model.dec)

    # s = time()
    # torch_o_output = (
    #     model_compiled(torch.tensor(x_in), torch.tensor(g_in)).detach().numpy()
    # )
    # print(f"Compiled Elapsed time: {time() - s:.2f}s")
    # assert np.allclose(torch_o_output, gt_o_output, atol=1e-4)

    # s = time()
    # torch_o_output = (
    #     model_compiled(torch.tensor(x_in), torch.tensor(g_in)).detach().numpy()
    # )
    # print(f"Compiled Elapsed time: {time() - s:.2f}s")
    # assert np.allclose(torch_o_output, gt_o_output, atol=1e-4)


# Load the ONNX model
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static
from onnxruntime.quantization.preprocess import quant_pre_process

from onnxruntime.tools.optimize_onnx_model import optimize_model


# export model.model.dec to ONNX
model_fp32_path = "model_dec.onnx"
torch.onnx.export(
    model_model_dec,
    (torch.zeros([1, 192, 299]), torch.zeros([1, 256, 1])),
    model_fp32_path,
    input_names=["x", "g"],
    output_names=["x"],
    dynamic_axes={"x": {0: "batch", 2: "length"}, "g": {0: "batch", 2: "length"}},
    opset_version=17,
)


# torch.onnx.dynamo_export(
#     model_model_dec,
#     x=torch.zeros([1, 192, 299]),
#     g=torch.zeros([1, 256, 1]),
#     # export_options=torch.onnx.ExportOptions(dynamic_shapes=True),
# ).save(model_fp32_path)


onnx_model = onnx.load(model_fp32_path)
onnx.checker.check_model(onnx_model, full_check=True)
# print(onnx_model.graph.input)

ort_session = ort.InferenceSession(model_fp32_path)

s = time()
onnx_o_output = ort_session.run(None, {"x.3": x_in, "g": g_in})[0]
print(f"fp32 Elapsed time: {time() - s:.2f}s")
assert np.allclose(onnx_o_output, gt_o_output, atol=1e-4)

print("ONNX model works correctly!")


model_opt_path = "model_dec.opt.onnx"

# optimize_model(
#     Path(model_fp32_path),
#     Path(model_opt_path),
#     level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
# )

ort_session = ort.InferenceSession(model_opt_path)

s = time()
onnx_o_output = ort_session.run(None, {"x.3": x_in, "g": g_in})[0]
print(f"fp32 optimised onnx Elapsed time: {time() - s:.2f}s")
assert np.allclose(onnx_o_output, gt_o_output, atol=1e-4)

model_quant_path = "model_dec.opt.onnx"

quant_pre_process(
    onnx.load(model_opt_path),
    model_quant_path,
    skip_optimization=False,
    skip_onnx_shape=False,
    skip_symbolic_shape=True,
    auto_merge=True,
)


quantized_onnx_model = quantize_dynamic(
    model_quant_path, model_quant_path, weight_type=QuantType.QUInt8
)
# class Reader:
#     def __init__(self, x, g):
#         self.x = x
#         self.g = g
#         self.counter = 0

#     def get_next(self):
#         if self.counter == 0:
#             self.counter += 1
#             return {"x.3": self.x, "g": self.g}
#         else:
#             return None


# quantize_static(
#     model_quant_path,
#     model_quant_path,
#     calibration_data_reader=Reader(x_in, g_in),
#     weight_type=QuantType.QUInt8,
#     activation_type=QuantType.QUInt8,
# )

ort_session = ort.InferenceSession(model_quant_path)

s = time()
onnx_o_output = ort_session.run(None, {"x.3": x_in, "g": g_in})[0]
print(f"Quantized Elapsed time: {time() - s:.2f}s")
assert np.allclose(onnx_o_output, gt_o_output, atol=1)

print("Quantized ONNX model works correctly!")

from onnxsim import simplify

onnx_model = onnx.load(model_fp32_path)
model_simp, check = simplify(onnx_model)

assert check, "Simplified ONNX model could not be validated"

# save the simplified model
simplified_model_path = "model_dec.simplified.onnx"
onnx.save(model_simp, simplified_model_path)

ort_session = ort.InferenceSession(simplified_model_path)

s = time()
onnx_o_output = ort_session.run(None, {"x.3": x_in, "g": g_in})[0]
print(f"Simplified Elapsed time: {time() - s:.2f}s")
assert np.allclose(onnx_o_output, gt_o_output, atol=1e-4)

print("Simplified ONNX model works correctly!")
