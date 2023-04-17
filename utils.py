import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]


def train_val_dataset(dataset, val_split=0.20, random_state=42, shuffle=True, stratify=None,
                      train_transform=None, val_transform=None):
    if stratify:
        stratify = [x[1] for x in dataset]
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))),
        test_size=val_split,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify
    )
    datasets = {}
    # datasets['train'] = ImageTransformer(Subset(dataset, train_idx), transform=train_transform)
    train_idx = set(train_idx)
    needed_images_train = [os.path.basename(sample[0]) for i, sample in enumerate(dataset.samples) if i in train_idx]
    datasets['train'] = ImageTransformer(
        ImageFolder2(dataset.image_folder, dataset.info_file, needed_images_train),
        transform=train_transform
    )
    datasets['train'].classes = dataset.classes

    # datasets['val'] = ImageTransformer(Subset(dataset, val_idx), transform=val_transform)
    val_idx = set(val_idx)
    needed_images_val = [os.path.basename(sample[0]) for i, sample in enumerate(dataset.samples) if i in val_idx]
    datasets['val'] = ImageTransformer(
        ImageFolder2(dataset.image_folder, dataset.info_file, needed_images_val),
        transform=val_transform
    )
    datasets['val'].classes = dataset.classes

    return datasets


def plot_from_batch_generator(batch_gen):
    data_batch, label_batch = next(iter(batch_gen))
    grid_size = (3, 3)
    f, axarr = plt.subplots(*grid_size)
    f.set_size_inches(15, 10)
    class_names = batch_gen.dataset.classes
    for i in range(grid_size[0] * grid_size[1]):
        batch_image_ndarray = np.transpose(data_batch[i].numpy(), [1, 2, 0])
        src = np.clip(image_std * batch_image_ndarray + image_mean, 0, 1)

        sample_title = 'Label = %d (%s)' % (label_batch[i], class_names[label_batch[i]])
        axarr[i // grid_size[0], i % grid_size[0]].imshow(src)
        axarr[i // grid_size[0], i % grid_size[0]].set_title(sample_title)
    pass


class ImageTransformer:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample, target = self.dataset[index]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


class ImageFolder2:
    def __init__(self, image_folder, info_file, needed_images=None, is_test=False):
        self.image_folder = image_folder
        self.info_file = info_file
        self.is_test = is_test

        if not needed_images:
            needed_images = set()
        else:
            needed_images = set(needed_images)

        self._load_info_(needed_images)
        self._load_samples_(needed_images)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self._load_sample_(path)

        return sample, target

    def _load_info_(self, needed_images):
        if self.is_test:
            return

        data_df = pd.read_csv(self.info_file)

        class_names = sorted(data_df.label.unique())
        self.class_i_to_label = {}
        self.class_label_to_i = {}

        self.classes = []
        for i, name in enumerate(class_names):
            self.class_i_to_label[i] = name
            self.class_label_to_i[name] = i
            self.classes.append(name)

        self.image_id_to_label = {}
        for row in zip(data_df['image_id'], data_df['label']):
            if len(needed_images) > 0:
                if row[0] in needed_images:
                    self.image_id_to_label[row[0]] = row[1]
            else:
                self.image_id_to_label[row[0]] = row[1]

    def _load_samples_(self, needed_images):
        images_paths = glob.glob(os.path.join(self.image_folder, '*'))

        if self.is_test:
            self.samples = [(
                path,
                path
            ) for path in images_paths]
            return

        if len(needed_images) > 0:
            self.samples = [(
                path,
                self.class_label_to_i[self.image_id_to_label[os.path.basename(path)]]
            ) for path in images_paths if os.path.basename(path) in needed_images]
        else:
            self.samples = [(
                path,
                self.class_label_to_i[self.image_id_to_label[os.path.basename(path)]]
            ) for path in images_paths]

    @classmethod
    def _load_sample_(cls, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
