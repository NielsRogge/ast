from multiprocessing import dummy
import os
from huggingface_hub import hf_hub_download

from datasets import load_dataset

import torch
import torchaudio

from models import ASTModel


def make_features(waveform, mel_bins, target_length=1024):
    waveform = torch.from_numpy(waveform).unsqueeze(0)

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=16000, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
        frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-6.845978)) / (5.5654526 * 2)
    return fbank.float()

# create dummy audio
dataset = load_dataset("speech_commands", "v0.02", split="validation")
waveform = dataset[0]["audio"]["array"]

feats = make_features(waveform, mel_bins=128, target_length=128) # shape(128, 128)
dummy_input = feats.expand(1, 128, 128)  # (batch_size, time, freq)  

# create an AST model
model = ASTModel(label_dim=35, fstride=10, tstride=10, input_fdim=128, input_tdim=128, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)

# load pretrained weights
state_dict = torch.hub.load_state_dict_from_url("https://www.dropbox.com/s/q0tbqpwv44pquwy/speechcommands_10_10_0.9812.pth?dl=1", map_location="cpu")

for key in list(state_dict.keys()):
    new_key = key.replace("module.", "")
    state_dict[new_key] = state_dict.pop(key)

model.load_state_dict(state_dict)

print("Shape of inputs:", dummy_input.shape)
print("First values of inputs:", dummy_input[0,:3,:3])

with torch.no_grad():
    output = model(dummy_input)

# output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
print("Shape of the logits:", output.shape)
print("Predicted class:", output.argmax(-1))
print("First values of logits:", output[0, :3])