"""Params for ADDA."""

PAD = 0
UNK = 1
SOS = 2
EOS = 3

# params for dataset and data loader
data_root = "data"
process_root = "data/processed"

# params for source dataset
src_encoder_restore = "snapshots/ADDA-source-encoder-final.pt"
src_classifier_restore = "snapshots/ADDA-source-classifier-final.pt"

# params for target dataset
tgt_encoder_restore = "snapshots/ADDA-target-encoder-final.pt"

# params for setting up models
model_root = "snapshots"
d_model_restore = "snapshots/ADDA-critic-final.pt"

# params for training network
num_gpu = 1
manual_seed = None

# params for optimizing models
c_learning_rate = 5e-5
d_learning_rate = 1e-5
beta1 = 0.9
beta2 = 0.999

n_vocab = 30522
hidden_size = 768
intermediate_size = 3072
embed_dim = 300
kernel_num = 20
kernel_sizes = [3, 4, 5]
pretrain = True
embed_freeze = True
class_num = 2
dropout = 0.1
num_labels = 2
d_hidden_dims = 384
d_output_dims = 2
