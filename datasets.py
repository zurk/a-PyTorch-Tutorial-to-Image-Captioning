import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, epoch_per_dataset=1, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']
        self.img_paths = self.h["paths"]

        # Captions per image
        self.captions_per_image = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        self.epoch_per_dataset = epoch_per_dataset
        self.current_dataset_chunk = 0
        self.dataset_chunk_size = len(self.captions) // self.epoch_per_dataset

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        assert i <= len(self)
        i += self.dataset_chunk_size * self.current_dataset_chunk
        image_index = i // self.captions_per_image
        img = torch.FloatTensor(self.imgs[image_index] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':
            return img, caption, caplen

        # For validation or testing, also return all 'captions_per_image' captions to find
        # BLEU-4 score
        start = image_index * self.captions_per_image
        all_captions = torch.LongTensor(self.captions[start:start + self.captions_per_image])
        if self.split == 'VAL':
            return img, caption, caplen, all_captions

        path = torch.CharTensor(list(self.img_paths[i]))
        return path, img, caption, caplen, all_captions

    def __len__(self):
        if self.current_dataset_chunk == self.epoch_per_dataset - 1:
            return len(self.captions) - self.dataset_chunk_size * self.current_dataset_chunk
        return self.dataset_chunk_size

    def increment_chunk(self):
        self.current_dataset_chunk += 1
        self.current_dataset_chunk = self.current_dataset_chunk % self.epoch_per_dataset