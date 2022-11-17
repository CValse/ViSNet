from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_warn

from transformer_m_visnet.utils import MissingLabelException
from pytorch_lightning.utilities import rank_zero_only
from transformer_m_visnet.datasets.utils import collator
from transformer_m_visnet.datasets import GlobalPygPCQM4Mv2Dataset
from functools import partial

class DataModule(LightningDataModule):
    def __init__(self, hparams, dataset=None):
        super(DataModule, self).__init__()
        
        self.hparams.update(hparams.__dict__) if hasattr(hparams, "__dict__") else self.hparams.update(hparams)
        
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
        
        self.dataset = dataset
        if self.dataset is None:
            self.dataset = GlobalPygPCQM4Mv2Dataset(root=self.hparams["dataset_root"], AddHs=self.hparams['AddHs'])

    def split_compute(self):
            
        split_idx = self.dataset.get_idx_split()
        
        print(f"train {len(split_idx['train'])}, val {len(split_idx['valid'])}, test {len(split_idx[self.hparams.get('inference_dataset', 'valid')])}")

        self.train_dataset = self.dataset.index_select(split_idx["train"])
        self.val_dataset = self.dataset.index_select(split_idx["valid"])
        self.test_dataset = self.dataset.index_select(split_idx[self.hparams.get("inference_dataset", "valid")])

        if self.hparams["standardize"] and self.hparams["task"] == 'train':
            self._standardize()

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, "val")

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        
        store_dataloader = (store_dataloader and not self.hparams["reload"])
        if stage in self._saved_dataloaders and store_dataloader:
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False

        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            collate_fn=partial(collator, max_node=256, multi_hop_max_dist=5, spatial_pos_max=1024),
        )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl
    
    @rank_zero_only
    def _standardize(self):
        
        def get_label(batch):
            if batch['y'] is None:
                raise MissingLabelException()
            return batch['y'].squeeze().clone()

        data = tqdm(self._get_dataloader(self.train_dataset, "val", store_dataloader=False), desc="computing mean and std",)
        
        try:
            ys = torch.cat([get_label(batch) for batch in data])
        except MissingLabelException:
            rank_zero_warn("Standardize is true but failed to compute dataset mean and standard deviation.")
            return None

        # compute mean and standard deviation
        self._mean = ys.mean(dim=0)
        self._std = ys.std(dim=0)
