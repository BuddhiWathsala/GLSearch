import torch
import klepto
from config import FLAGS
import os.path
from torch.utils.data import Dataset as TorchDataset


class OurModelData(TorchDataset):
    """Stores a list of graph id pairs with known pairwise results."""

    def __init__(self, dataset, num_node_feat):
        self.dataset, self.num_node_feat = dataset, num_node_feat
        gid_pairs = list(self.dataset.pairs.keys())
        self.gid1gid2_list = torch.tensor(
            sorted(gid_pairs),
            device=FLAGS.device)  # takes a while to move to GPU

    def __len__(self):
        return len(self.gid1gid2_list)

    def __getitem__(self, idx):
        return self.gid1gid2_list[idx]


root = "/workspace/GLSearch/model/OurMCS/"
filename = "mutag_train_test_mcs_bfs_one_hot_local_degree_profile_None.klepto"
path = root + filename

print(os.path.isfile(path))
print(os.path.isfile(root + "config.py"))
print(os.path.isfile(root + "test.py"))
file = klepto.archives.file_archive(path)
file.load()
print("Reproduce the issue https://github.com/DerekQXu/GLSearch/issues/2")
print(file)
