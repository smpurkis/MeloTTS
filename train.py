from pathlib import Path
import random
import shutil
from time import time

import numpy as np
from melo.api import TTS
from melo.models import Generator
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import wandb
import re

# # Speed is adjustable
speed = 1.0

# CPU is sufficient for real-time inference.
# You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device_str = "gpu" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
# # CPU is sufficient for real-time inference.
# # You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device = "cpu"  # Will automatically use GPU if available

# # English
# # text = "Did you ever hear a folk tale about a giant turtle? It's a story about a turtle that carries the world on its back."
# # texts = [text]


def parse_text(text):
    # remove ', ", ”, “
    text = text.replace("'", "")
    text = text.replace('"', "")
    text = text.replace("”", "")
    text = text.replace("“", "")

    # remove all \n
    text = text.replace("\n", "")

    # split by ., !, ?, ;, ',' using regex
    text = re.split(
        r"(?<!\d)\.(?!\d)|(?<!\d)!(?!\d)|(?<!\d)\?(?!\d)|(?<!\d);(?!\d)",
        text,
    )

    texts = [t for t in text if t.strip() != ""]

    # join the texts to make sure the length of the text is less than target_word_length
    target_word_length = 15
    min_word_length = 6
    max_word_length = 30
    joined_texts = []
    joined_text = ""
    lens = {}
    for t in texts:
        combined_length = len(joined_text.split()) + len(t.split())
        if (
            combined_length <= target_word_length
            or len(t.split()) < min_word_length
            and combined_length <= max_word_length
        ):
            joined_text += " " + t
        else:
            joined_length = len(joined_text.split())

            if min_word_length <= joined_length <= max_word_length:

                # ensure all commas are followed by a space
                joined_text = re.sub(r",(?=[^\s])", ", ", joined_text)

                joined_texts.append(joined_text.strip())

            if joined_length in lens:
                lens[joined_length] += 1
            else:
                lens[joined_length] = 1

            joined_text = t

    lens = sorted(lens.items(), key=lambda x: x[0])

    shortest_s = min(joined_texts, key=lambda x: len(x))
    longest_s = max(joined_texts, key=lambda x: len(x))
    return joined_texts


# texts = parse_text(Path("bible-web.txt").read_text())

# # Load the model
s = time()
model = TTS(language="EN", device=device_str, use_onnx=False)
print(f"Loaded model in {time() - s:.2f}s")
speaker_ids = model.hps.data.spk2id
output_path = "/tmp/en-default.wav"
model.model.dec_training = []
model.tts_to_file(
    "Now those who sealed were: Nehemiah the governor, the son of Hacaliah, and Zedekiah, Seraiah, Azariah, Jeremiah, Pashhur, Amariah, Malchijah, Hattush, Shebaniah, Malluch, Harim, Meremoth, Obadiah, Daniel, Ginnethon, Baruch, Meshullam, Abijah, Mijamin, Maaziah, Bilgai, and Shemaiah",
    speaker_ids["EN-Default"],
    "en-default.wav",
    speed=speed,
)

generator_params = dict(
    initial_channel=192,  # 192
    resblock="1",  # 1
    resblock_kernel_sizes=[
        5,
        11,
        #    3
    ],  # [3, 7, 11]
    resblock_dilation_sizes=[
        [1, 3, 5],
        [1, 3, 5],
        # [1, 3, 5],
    ],  # [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    upsample_rates=[2, 4, 4, 4, 4],  # [8, 8, 2, 2, 2]
    upsample_initial_channel=512,  # 512
    upsample_kernel_sizes=[4, 8, 16, 4, 4],  # [16, 16, 8, 2, 2]
    gin_channels=256,  # 256
)

distilled_generator = Generator(**generator_params).eval()

total_params_dec = sum(p.numel() for p in model.model.dec.parameters())
total_params_dec_distilled = sum(p.numel() for p in distilled_generator.parameters())
print(f"Total params in dec: {total_params_dec}")
print(f"Total params in dec distilled: {total_params_dec_distilled}")
print(f"Ratio: {total_params_dec_distilled / total_params_dec:.2f}")

# out_distilled = distilled_generator(
#     torch.randn([1, 192, 299]), torch.randn([1, 256, 1])
# )
s = time()
out_distilled = distilled_generator(
    torch.randn([1, 192, 299]), torch.randn([1, 256, 1])
)
total_time_distilled = time() - s

model.model.dec = model.model.dec.eval()
s = time()
out = model.model.dec(torch.randn([1, 192, 299]), torch.randn([1, 256, 1]))
total_time = time() - s

print(f"Elapsed time distilled: {total_time_distilled:.2f}s")
print(f"Elapsed time: {total_time:.2f}s")
print(f"Time ratio: {total_time_distilled / total_time:.2f}")
assert out.shape == out_distilled.shape, f"{out.shape} != {out_distilled.shape}"
