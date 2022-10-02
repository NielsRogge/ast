import os
from huggingface_hub import hf_hub_download

import torch
import torchaudio

from models import ASTModel


def make_features(wav_name, mel_bins, target_length=1024):
    waveform, sr = torchaudio.load(wav_name)

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
        frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank

# create dummy audio
filepath = hf_hub_download(repo_id="nielsr/audio-spectogram-transformer-checkpoint",
                           filename="sample_audio.flac",
                           repo_type="dataset")

feats = make_features('/content/ast/sample_audios/sample_audio.flac', mel_bins=128) # shape(1024, 128)
dummy_input = feats.expand(1, 1024, 128)                       

# create an AST model
model = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)

# load pretrained weights
filepath = hf_hub_download(repo_id="nielsr/audio-spectogram-transformer-checkpoint",
                             filename="audioset_10_10_0.4593.pth",
                             repo_type="dataset")
state_dict = torch.load(filepath, map_location="cpu")

for key in list(state_dict.keys()):
    new_key = key.replace("module.", "")
    state_dict[new_key] = state_dict.pop(key)

model.load_state_dict(state_dict)

with torch.no_grad():
    output = model(dummy_input)

# output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
print("Shape of the logits:", output.shape)
print("Predicted class:", output.argmax(-1)