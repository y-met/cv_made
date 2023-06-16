import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.functional import ctc_loss, log_softmax

from nltk import edit_distance

from tqdm.notebook import tqdm


class Resize(object):

    def __init__(self, size=(320, 64)):
        self.size = size

    def __call__(self, item):
        interpolation = cv2.INTER_AREA if self.size[0] < item["image"].shape[1] else cv2.INTER_LINEAR
        item["image"] = cv2.resize(item["image"], self.size, interpolation=interpolation)

        return item


class RecognitionDataset(Dataset):
    def __init__(self, df, char_to_id, folder='..data_2/train/train', transforms=None):
        super(RecognitionDataset, self).__init__()
        self.df = df

        self.image_names = df['Id'].to_list()
        self.texts = df['Expected'].to_list()

        self.char_to_id = char_to_id
        self.folder = folder
        self.transforms = transforms

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        image = cv2.imread(f'{self.folder}/{self.image_names[item]}').astype(np.float32) / 255.
        text = str(self.texts[item])
        seq = [self.char_to_id[c] for c in text if c in self.char_to_id]
        seq_len = len(seq)

        output = dict(image=image, seq=seq, seq_len=seq_len, text=text)

        if self.transforms is not None:
            output = self.transforms(output)

        return output


def collate_fn(batch):
    images, seqs, seq_lens, texts = [], [], [], []

    for item in batch:
        images.append(torch.from_numpy(item["image"]).permute(2, 0, 1).float())
        seqs.extend(item["seq"])
        seq_lens.append(item["seq_len"])
        texts.append(item["text"])

    images = torch.stack(images)
    seqs = torch.Tensor(seqs).int()
    seq_lens = torch.Tensor(seq_lens).int()

    batch = {"image": images, "seq": seqs, "seq_len": seq_lens, "text": texts}

    return batch


def pred_to_string(pred, abc):
    seq = []
    for i in range(len(pred)):
        label = np.argmax(pred[i])
        seq.append(label)
    out = []
    for i in range(len(seq)):
        if len(out) == 0:
            if seq[i] != 0:
                out.append(seq[i])
        else:
            if seq[i] != 0 and seq[i] != seq[i - 1]:
                out.append(seq[i])
    out = ''.join([abc[c] for c in out])
    return out


def decode(pred, abc):
    pred = pred.permute(1, 0, 2).cpu().data.numpy()
    outputs = []
    for i in range(len(pred)):
        outputs.append(pred_to_string(pred[i], abc))
    return outputs


def load_checkpoint(checkpoint_path):
    with open(checkpoint_path, 'rb') as checkpoint_file:
        model = torch.load(checkpoint_file)

    return model


def train_model(device, crnn, optimizer,
                train_dataloader, val_dataloader, id_to_char,
                num_epochs=10, checkpoint_name='models/moodel'):
    general_val_loss = 10000
    general_lev_mean = 10000

    for i, epoch in enumerate(range(num_epochs)):
        # train
        crnn.train()
        epoch_losses = []

        for j, b in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
            images = b["image"].to(device)
            seqs_gt = b["seq"]
            seq_lens_gt = b["seq_len"]

            seqs_pred = crnn(images).cpu()
            log_probs = log_softmax(seqs_pred, dim=2)

            seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

            loss = ctc_loss(log_probs=log_probs,
                            targets=seqs_gt,
                            input_lengths=seq_lens_pred,
                            target_lengths=seq_lens_gt)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(crnn.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_losses.append(loss.item())

        print(i, 'train:', np.mean(epoch_losses))

        # eval
        crnn.eval()
        epoch_losses = []

        sum_distances = 0
        count_distances = 0

        for j, b in enumerate(tqdm(val_dataloader, total=len(val_dataloader))):
            images = b["image"].to(device)
            seqs_gt = b["seq"]
            seq_lens_gt = b["seq_len"]

            seqs_pred = crnn(images).cpu()
            log_probs = log_softmax(seqs_pred, dim=2)

            seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

            loss = ctc_loss(log_probs=log_probs,
                            targets=seqs_gt,
                            input_lengths=seq_lens_pred,
                            target_lengths=seq_lens_gt)

            epoch_losses.append(loss.item())

            t_decodes = decode(seqs_pred, id_to_char)
            t_texts = b["text"]

            for dec, text in zip(t_decodes, t_texts):
                sum_distances += edit_distance(dec, text)
                count_distances += 1

        val_loss = np.mean(epoch_losses)
        print(i, 'validation loss:', val_loss)

        lev_mean = sum_distances / count_distances
        print(i, 'validation lev_mean:', lev_mean)

        if val_loss < general_val_loss and checkpoint_name is not None:
            general_val_loss = val_loss
            with open(checkpoint_name, 'wb') as checkpoint_file:
                torch.save(crnn, checkpoint_file)

        if lev_mean < general_lev_mean and checkpoint_name is not None:
            general_lev_mean = lev_mean
            with open(f'{checkpoint_name}_lev_mean', 'wb') as checkpoint_file:
                torch.save(crnn, checkpoint_file)


def get_predictions(device, crnn, test_dataloader, id_to_char, submission_df, result_path):
    submission_df = submission_df.copy(deep=True)
    predictions = []

    crnn.eval()
    for j, b in enumerate(tqdm(test_dataloader, total=len(test_dataloader))):
        images = b["image"].to(device)

        seqs_pred = crnn(images).cpu()

        t_decodes = decode(seqs_pred, id_to_char)
        predictions.extend(t_decodes)

    submission_df['Predicted'] = predictions
    if 'Expected' in submission_df.columns:
        submission_df.drop(columns=['Expected'], inplace=True)
    submission_df.to_csv(result_path, index=False)

    return submission_df




