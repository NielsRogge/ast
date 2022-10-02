import torch
from models import ASTModel

from huggingface_hub import hf_hub_download

# assume each input spectrogram has 100 time frames
input_tdim = 100
# assume the task has 527 classes
label_dim = 527
# create a pseudo input: a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
test_input = torch.rand([10, input_tdim, 128])

# create an AST model
model = ASTModel(label_dim=label_dim, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False)

# load pretrained weights
filepath = hf_hub_download(repo_id="nielsr/audio-spectogram-transformer-checkpoint",
                             filename="audioset_10_10_0.4593.pth",
                             repo_type="dataset")
state_dict = torch.load(filepath, map_location="cpu")
model.load_state_dict(state_dict)

with torch.no_grad():
    output = model(test_input)

# output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
print(output.shape)