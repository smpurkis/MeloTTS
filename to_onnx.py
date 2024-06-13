from time import time

from melo.api import TTS
import torch

# Speed is adjustable
speed = 1.0

# CPU is sufficient for real-time inference.
# You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device = "cpu"  # Will automatically use GPU if available

# English
text = "Did you ever hear a folk tale about a giant turtle?"

# Load the model
s = time()
model = TTS(language="EN", device=device, use_onnx=False)
print(f"Loaded model in {time() - s:.2f}s")
speaker_ids = model.hps.data.spk2id
output_path = "en-us.wav"


# model_scripted = torch.jit.script(model.model)

# inputs = model.get_inputs(text, speaker_ids["EN-US"], output_path, speed=speed)

# out = model_scripted(**inputs)

# American accent
s = time()
model.tts_to_file(text, speaker_ids["EN-US"], output_path, speed=speed)
print(f"Elapsed time: {time() - s:.2f}s")

# s = time()
# model.tts_to_file(text, speaker_ids["EN-US"], output_path, speed=speed)
# print(f"Elapsed time: {time() - s:.2f}s")


# model = TTS(language="EN", device=device, use_onnx=True)
# s = time()
# model.tts_to_file(text, speaker_ids["EN-US"], output_path, speed=speed)
# print(f"Elapsed time: {time() - s:.2f}s")

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

# # ground truth for dec
# x_in = np.load("/Users/user/demo_1/tts-pg/MeloTTS/to_onnx_test/x_input.npy")
# g_in = np.load("/Users/user/demo_1/tts-pg/MeloTTS/to_onnx_test/g_input.npy")
# gt_o_output = np.load("/Users/user/demo_1/tts-pg/MeloTTS/to_onnx_test/o_output.npy")

# with torch.no_grad():
#     model_model_dec.eval()
#     s = time()
#     torch_o_output = (
#         model_model_dec(torch.tensor(x_in), torch.tensor(g_in)).detach().numpy()
#     )
#     print(f"Elapsed time: {time() - s:.2f}s")
#     assert np.allclose(torch_o_output, gt_o_output, atol=1e-4)

#     print("Scripted fp32 Torch model works correctly!")


# # Load the ONNX model
# import onnx
# import onnxruntime as ort
# from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static
# from onnxruntime.quantization.preprocess import quant_pre_process

# from onnxruntime.tools.optimize_onnx_model import optimize_model


# # export model.model.dec to ONNX
# model_fp32_path = "model_dec.onnx"
# torch.onnx.export(
#     model_model_dec,
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
# onnx.checker.check_model(onnx_model, full_check=True)
# # print(onnx_model.graph.input)

# ort_session = ort.InferenceSession(model_fp32_path)

# s = time()
# onnx_o_output = ort_session.run(None, {"x.3": x_in, "g": g_in})[0]
# print(f"fp32 Elapsed time: {time() - s:.2f}s")
# assert np.allclose(onnx_o_output, gt_o_output, atol=1e-4)
