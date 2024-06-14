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
# text = "Did you ever hear a folk tale about a giant turtle? It's a story about a turtle that carries the world on its back. It's a story that's been passed down for generations. It's a story that's been told in many different cultures. It's a story that's been told in many different ways. It's a story that's been told in many different languages."
text = "It's a story that's been told in many different languages."

# Load the model
model = TTS(language="EN", device=device, use_onnx=False)
speaker_ids = model.hps.data.spk2id

model_model_dec = torch.jit.script(
    model.model.dec,
    example_inputs=(torch.zeros([1, 192, 299]), torch.zeros([1, 256, 1])),
)

# print(torch._dynamo.list_backends())
# model_compiled = torch.compile(
#     model.model.dec,
#     dynamic=False,
#     fullgraph=False,
#     backend="onnxrt",
# )

# American accent
output_path = "en-us.wav"
s = time()
model.tts_to_file(text, speaker_ids["EN-US"], output_path, speed=speed)
print(f"Elapsed time: {time() - s:.2f}s")

s = time()
model.tts_to_file(text, speaker_ids["EN-US"], output_path, speed=speed)
print(f"Elapsed time: {time() - s:.2f}s")


model = TTS(language="EN", device=device, use_onnx=True)
s = time()
model.tts_to_file(text, speaker_ids["EN-US"], output_path, speed=speed)
print(f"Elapsed time: {time() - s:.2f}s")

s = time()
model.tts_to_file(text, speaker_ids["EN-US"], output_path, speed=speed)
print(f"Elapsed time: {time() - s:.2f}s")

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
# x_in = np.random.random([1, 192, 299]).astype(np.float32)
# g_in = np.random.random([1, 256, 1]).astype(np.float32)
# gt_o_output = model.model.dec(torch.tensor(x_in), torch.tensor(g_in)).detach().numpy()


# with torch.no_grad():
#     model_model_dec.eval()
#     s = time()
#     torch_o_output = (
#         model_model_dec(torch.tensor(x_in), torch.tensor(g_in)).detach().numpy()
#     )
#     print(f"Elapsed time: {time() - s:.2f}s")
#     assert np.allclose(torch_o_output, gt_o_output, atol=1e-4)

#     print("Scripted fp32 Torch model works correctly!")

#     quantized_torch_model = torch.quantization.quantize_dynamic(
#         model_model_dec,
#         {
#             nn.Linear,
#             nn.LSTM,
#             nn.GRU,
#             nn.LSTMCell,
#             nn.RNNCell,
#             nn.GRUCell,
#         },
#         dtype=torch.qint8,
#     )

#     s = time()
#     torch_o_output = (
#         quantized_torch_model(torch.tensor(x_in), torch.tensor(g_in)).detach().numpy()
#     )
#     print(f"Quantized Elapsed time: {time() - s:.2f}s")
#     assert np.allclose(torch_o_output, gt_o_output, atol=1e-4)

#     print("Quantized Torch model works correctly!")

#     mobile_optimized_model = optimize_for_mobile(model_model_dec)

#     s = time()
#     torch_o_output = (
#         mobile_optimized_model(torch.tensor(x_in), torch.tensor(g_in)).detach().numpy()
#     )
#     print(f"Mobile optimized Elapsed time: {time() - s:.2f}s")
#     assert np.allclose(torch_o_output, gt_o_output, atol=1e-4)

#     print("Mobile optimized Torch model works correctly!")

#     quant_mobile_optimized_model = optimize_for_mobile(quantized_torch_model)

#     s = time()
#     torch_o_output = (
#         quant_mobile_optimized_model(torch.tensor(x_in), torch.tensor(g_in))
#         .detach()
#         .numpy()
#     )
#     print(f"Quant Mobile optimized Elapsed time: {time() - s:.2f}s")
#     assert np.allclose(torch_o_output, gt_o_output, atol=1e-4)

#     print("Mobile optimized Torch model works correctly!")

# use torch.compile

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
# import onnx
# import onnxruntime as ort

# # from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static
# # from onnxruntime.quantization.preprocess import quant_pre_process

# # from onnxruntime.tools.optimize_onnx_model import optimize_model


# # # export model.model.dec to ONNX
# model_fp32_path = "model_dec.onnx"
# torch.onnx.export(
#     model.model.dec,
#     (torch.zeros([1, 192, 299]), torch.zeros([1, 256, 1])),
#     model_fp32_path,
#     input_names=["x", "g"],
#     output_names=["x"],
#     dynamic_axes={"x": {0: "batch", 2: "length"}, "g": {0: "batch", 2: "length"}},
#     opset_version=17,
# )


# # torch.onnx.dynamo_export(
# #     model_model_dec,
# #     x=torch.zeros([1, 192, 299]),
# #     g=torch.zeros([1, 256, 1]),
# #     # export_options=torch.onnx.ExportOptions(dynamic_shapes=True),
# # ).save(model_fp32_path)


# onnx_model = onnx.load(model_fp32_path)
# # # onnx.checker.check_model(onnx_model, full_check=True)
# # # print(onnx_model.graph.input)

# ort_session = ort.InferenceSession(model_fp32_path)

# s = time()
# onnx_o_output = ort_session.run(
#     None, {onnx_model.graph.input[0].name: x_in, onnx_model.graph.input[1].name: g_in}
# )[0]
# print(f"fp32 Elapsed time: {time() - s:.2f}s")
# assert np.allclose(onnx_o_output, gt_o_output, atol=1e-4)

# print("ONNX model works correctly!")

# sess_opt = ort.SessionOptions()
# sess_opt.add_session_config_entry("session.intra_op.allow_spinning", "0")
# sess_opt.add_session_config_entry("session.inter_op.allow_spinning", "0")
# sess_opt.inter_op_num_threads = 0
# sess_opt.intra_op_num_threads = 0
# # sess_opt.execution_mode = ort.ExecutionMode.ORT_PARALLEL
# sess_opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
# sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# # sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

# ort_session = ort.InferenceSession(model_fp32_path, sess_opt)

# s = time()
# onnx_o_output = ort_session.run(None, {"x.3": x_in, "g": g_in})[0]
# print(f"fp32 Elapsed time: {time() - s:.2f}s")
# assert np.allclose(onnx_o_output, gt_o_output, atol=1e-4)

# print("ONNX model works correctly!")


# model_opt_path = "model_dec.opt.onnx"

# optimize_model(
#     Path(model_fp32_path),
#     Path(model_opt_path),
#     level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
# )

# ort_session = ort.InferenceSession(model_opt_path)

# s = time()
# onnx_o_output = ort_session.run(None, {"x.3": x_in, "g": g_in})[0]
# print(f"fp32 optimised onnx Elapsed time: {time() - s:.2f}s")
# assert np.allclose(onnx_o_output, gt_o_output, atol=1e-4)

# model_quant_path = "model_dec.opt.onnx"

# quant_pre_process(
#     onnx.load(model_opt_path),
#     model_quant_path,
#     skip_optimization=False,
#     skip_onnx_shape=False,
#     skip_symbolic_shape=True,
#     auto_merge=True,
# )


# quantized_onnx_model = quantize_dynamic(
#     model_quant_path, model_quant_path, weight_type=QuantType.QUInt8
# )
# # class Reader:
# #     def __init__(self, x, g):
# #         self.x = x
# #         self.g = g
# #         self.counter = 0

# #     def get_next(self):
# #         if self.counter == 0:
# #             self.counter += 1
# #             return {"x.3": self.x, "g": self.g}
# #         else:
# #             return None


# # quantize_static(
# #     model_quant_path,
# #     model_quant_path,
# #     calibration_data_reader=Reader(x_in, g_in),
# #     weight_type=QuantType.QUInt8,
# #     activation_type=QuantType.QUInt8,
# # )

# ort_session = ort.InferenceSession(model_quant_path)

# s = time()
# onnx_o_output = ort_session.run(None, {"x.3": x_in, "g": g_in})[0]
# print(f"Quantized Elapsed time: {time() - s:.2f}s")
# assert np.allclose(onnx_o_output, gt_o_output, atol=1)

# print("Quantized ONNX model works correctly!")

# from onnxsim import simplify

# onnx_model = onnx.load(model_fp32_path)
# model_simp, check = simplify(onnx_model)

# assert check, "Simplified ONNX model could not be validated"

# # save the simplified model
# simplified_model_path = "model_dec.simplified.onnx"
# onnx.save(model_simp, simplified_model_path)

# ort_session = ort.InferenceSession(simplified_model_path)

# s = time()
# onnx_o_output = ort_session.run(None, {"x.3": x_in, "g": g_in})[0]
# print(f"Simplified Elapsed time: {time() - s:.2f}s")
# assert np.allclose(onnx_o_output, gt_o_output, atol=1e-4)

# print("Simplified ONNX model works correctly!")


# import nobuco
# from nobuco import ChannelOrder, ChannelOrderingStrategy
# from nobuco.layers.weight import WeightLayer

# x_dummy = torch.zeros([1, 192, 299])
# g_dummy = torch.zeros([1, 256, 1])

# keras_model = nobuco.pytorch_to_keras(
#     model.model.dec,
#     args=[x_dummy, g_dummy],
#     input_shapes={x_dummy: (None, 192, None), g_dummy: (None, 256, None)},
#     trace_shape=True,
#     inputs_channel_order=ChannelOrder.PYTORCH,
#     outputs_channel_order=ChannelOrder.PYTORCH,
# )

# # keras_model.save("model_dec.keras")

# import keras

# # keras_model = keras.models.load_model("model_dec.keras")

# s = time()
# out = keras_model.predict({"input_1": x_in, "input_2": g_in})
# print(f"Elapsed time: {time() - s:.2f}s")
# assert np.allclose(out, gt_o_output, atol=1e-4)

# s = time()
# out = keras_model.predict({"input_1": x_in, "input_2": g_in})
# print(f"Elapsed time: {time() - s:.2f}s")
# assert np.allclose(out, gt_o_output, atol=1e-4)

# import tensorflow as tf


# # def representative_dataset():
# #     for _ in range(10):
# #         yield [
# #             np.random.rand(1, 192, np.random.randint(200, 600)).astype(np.float32),
# #             np.random.rand(1, 256, 1).astype(np.float32),
# #         ]


# converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# # converter.representative_dataset = representative_dataset
# converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS,
#     # tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
#     # tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
# ]
# # converter.inference_input_type = tf.int8
# # converter.inference_output_type = tf.int8

# tflite_model = converter.convert()
# with open("model_dec.tflite", "wb") as f:
#     f.write(tflite_model)

# tflite_model = Path("model_dec.tflite").read_bytes()

# interpreter = tf.lite.Interpreter(model_content=tflite_model, num_threads=4)
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# interpreter.resize_tensor_input(input_details[0]["index"], x_in.shape)
# interpreter.resize_tensor_input(input_details[1]["index"], g_in.shape)

# interpreter.allocate_tensors()

# interpreter.set_tensor(input_details[0]["index"], x_in)
# interpreter.set_tensor(input_details[1]["index"], g_in)

# s = time()
# interpreter.invoke()
# tflite_o_output = interpreter.get_tensor(output_details[0]["index"])
# print(f"Elapsed time: {time() - s:.2f}s")
# assert np.allclose(tflite_o_output, gt_o_output, atol=1e-4)

# s = time()
# interpreter.invoke()
# tflite_o_output = interpreter.get_tensor(output_details[0]["index"])
# print(f"Elapsed time: {time() - s:.2f}s")
# assert np.allclose(tflite_o_output, gt_o_output, atol=1e-4)
