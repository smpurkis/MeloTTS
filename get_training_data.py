from pathlib import Path
import shutil
from time import time

import numpy as np
from melo.api import TTS
from melo.models import Generator
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# Speed is adjustable
speed = 1.0

# CPU is sufficient for real-time inference.
# You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device = "cpu"  # Will automatically use GPU if available

# English
# text = "Did you ever hear a folk tale about a giant turtle? It's a story about a turtle that carries the world on its back."
# texts = [text]
texts = Path("bible-web.txt").read_text().split("\n")

# Load the model
s = time()
model = TTS(language="EN", device=device, use_onnx=False)
print(f"Loaded model in {time() - s:.2f}s")
speaker_ids = model.hps.data.spk2id
output_path = "/tmp/en-default.wav"
model.model.dec_training = []
# model.tts_to_file("blah", speaker_ids["EN-US"], output_path, speed=speed)


distilled_generator = Generator(
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
).eval()

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

# use the weights of the last resblock of the original model on
# the distilled model
# distilled_generator.resblocks[-1].load_state_dict(
#     model.model.dec.resblocks[-1].state_dict()
# )

# training

epochs = 10

batch_size = 10
texts = texts[:1000]

training = True

optimizer = torch.optim.Adam(distilled_generator.parameters(), lr=1e-4)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    steps = len(texts) // batch_size

    for step in tqdm(range(steps), desc="Steps"):
        training = step % 10 != 0
        print(f"Epoch {epoch + 1}/{epochs}, Step {step + 1}/{steps}")

        batch = []

        model.model.dec_training = []

        # generate the batch of data
        for i in tqdm(
            range(batch_size),
            desc=f"Batch generate {'training' if training else 'evaluation'}",
            leave=False,
        ):
            text = texts[step * batch_size + i]

            model.model.dec_training.append({"text": text})
            model.tts_to_file(
                text, speaker_ids["EN-Default"], output_path, speed=speed, quiet=True
            )

            # load batch of data to train on
            point = model.model.dec_training.pop(0)
            x_in = np.load(point["x_in_path"])
            x_in = torch.tensor(x_in, dtype=torch.float32).to(device)

            g_in = np.load(point["g_in_path"])
            g_in = torch.tensor(g_in, dtype=torch.float32).to(device)

            o_out = np.load(point["o_out_path"])
            o_out = torch.tensor(o_out, dtype=torch.float32).to(device)

            batch.append((x_in, g_in, o_out))

        # clear the numpy files
        for point in model.model.dec_training:
            Path(point["x_in_path"]).unlink(missing_ok=True)
            Path(point["g_in_path"]).unlink(missing_ok=True)
            Path(point["o_out_path"]).unlink(missing_ok=True)
        shutil.rmtree("dec_distill/data", ignore_errors=True)
        Path("dec_distill/data").mkdir(exist_ok=True)

        # train on batch
        if training:
            distilled_generator.train()
        else:
            distilled_generator.eval()

        # run the whole batch through then run backwards and optimize
        batch_loss = 0
        optimizer.zero_grad()  # zero the gradient buffers

        for x_in, g_in, o_out in tqdm(
            batch, desc=f"Batch {'training' if training else 'evaluation'}", leave=False
        ):
            output = distilled_generator(x_in, g_in)  # forward pass
            loss = F.mse_loss(output, o_out)
            loss = loss * torch.sqrt(
                torch.tensor(len(output)).float()
            )  # weight the loss
            if training:
                loss.backward()  # backpropagation
            batch_loss += loss.item()

        if training:
            optimizer.step()  # does the update

        print(
            f"{'Training' if training else 'Evaluation'} Batch Loss: {batch_loss / len(batch)}"
        )

    # save model
    print("Saving model...")
    torch.save(
        distilled_generator.state_dict(),
        f"dec_distill/checkpoints/model_{epoch}_loss_{batch_loss}.pt",
    )
