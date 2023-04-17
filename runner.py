import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from collections import defaultdict

from sklearn.metrics import f1_score, accuracy_score
from IPython.display import clear_output


class Runner:
    def __init__(self, model, opt, device, checkpoint_name=None, saved_checkpoint=None, scheduler=None):
        self.model = model
        self.opt = opt
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_name = checkpoint_name
        if saved_checkpoint:
            self.load_checkpoint(saved_checkpoint)

        self.epoch = 0
        self.output = None
        self.metrics = None
        self._global_step = 0
        self._set_events()
        self._top_val_accuracy = -1
        self.log_dict = {
            "train": [],
            "val": [],
            "test": []
        }

    def _set_events(self):
        self._phase_name = ''
        self.events = {
            "train": defaultdict(list),
            "val": defaultdict(list),
            "test": defaultdict(list)
        }

    def _reset_events(self, event_name):
        self.events[event_name] = defaultdict(list)

    def forward(self, img_batch, **kwargs):
        logits = self.model(img_batch)
        output = {
            "logits": logits,
        }
        return output

    def _run_batch(self, batch):
        X_batch, y_batch = batch

        self._global_step += len(y_batch)

        X_batch = X_batch.to(self.device)

        self.output = self.forward(X_batch)

    def _run_epoch(self, loader, train_phase=True, output_log=False, **kwargs):
        self.model.train(train_phase)

        _phase_description = 'Training' if train_phase else 'Evaluation'
        for batch in tqdm(loader, desc=_phase_description, leave=False):
            self._run_batch(batch)

            with torch.set_grad_enabled(train_phase):
                loss = self.run_criterion(batch)

            if train_phase:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

        self.log_dict[self._phase_name].append(np.mean(self.events[self._phase_name]['loss']))

        if output_log:
            self.output_log(**kwargs)

    def train(self, train_loader, val_loader, n_epochs, model=None, opt=None, scheduler=None, **kwargs):
        self.opt = (opt or self.opt)
        self.scheduler = (scheduler or self.scheduler)
        self.model = (model or self.model)

        for _epoch in range(n_epochs):
            start_time = time.time()
            self.epoch += 1
            print(f"epoch {self.epoch:3d}/{n_epochs:3d} started")

            self._set_events()
            self._phase_name = 'train'
            self._run_epoch(train_loader, train_phase=True)

            print(f"epoch {self.epoch:3d}/{n_epochs:3d} took {time.time() - start_time:.2f}s")

            self._phase_name = 'val'
            self.validate(val_loader, **kwargs)
            self.save_checkpoint()
            if self.scheduler:
                self.scheduler.step(self.metrics['loss'])

    @torch.no_grad()
    def validate(self, loader, model=None, phase_name='val', **kwargs):
        self._phase_name = phase_name
        self._reset_events(phase_name)
        self._run_epoch(loader, train_phase=False, output_log=True, **kwargs)
        return self.metrics

    def _get_file_name(self, loader, index):
        file_path = loader.dataset.dataset.samples[index][0]
        i_sep = file_path.rfind(os.path.sep)
        return file_path[i_sep + 1:]

    @torch.no_grad()
    def get_predictions(self, loader, device, classes, model=None):
        phase_description = 'prediction'
        if model is not None:
            self.model = model.to(device)
        self.device = device

        predictions = []
        file_names = []

        for i, batch in tqdm(enumerate(loader), desc=phase_description, leave=False):
            X_batch, y_batch = batch
            self._global_step += len(X_batch)
            X_batch = X_batch.to(self.device)

            out = self.forward(X_batch)

            _, y_pred_tags = torch.max(torch.log_softmax(out['logits'], dim=1), dim=1)
            scores = y_pred_tags.detach().cpu().numpy().tolist()
            predictions.extend([classes[x] for x in scores])
            file_names.extend(y_batch)

        predictions = pd.DataFrame({'image_id': file_names, 'label': predictions})
        predictions['image_id'] = predictions['image_id'].apply(lambda x: os.path.basename(x))

        return predictions

    def run_criterion(self, batch):
        X_batch, label_batch = batch
        label_batch = label_batch.to(self.device)

        logit_batch = self.output['logits']
        loss = F.cross_entropy(logit_batch, label_batch)

        _, y_pred_tags = torch.max(torch.log_softmax(logit_batch, dim=1), dim=1)
        scores = y_pred_tags.detach().cpu().numpy().tolist()
        labels = label_batch.detach().cpu().numpy().ravel().tolist()

        self.events[self._phase_name]['loss'].append(loss.detach().cpu().numpy())
        self.events[self._phase_name]['scores'].extend(scores)
        self.events[self._phase_name]['labels'].extend(labels)

        return loss

    def save_checkpoint(self):
        val_accuracy = self.metrics['f1-micro']
        if val_accuracy > self._top_val_accuracy and self.checkpoint_name is not None:
            self._top_val_accuracy = val_accuracy
            with open(self.checkpoint_name, 'wb') as checkpoint_file:
                torch.save(self.model, checkpoint_file)

    def load_checkpoint(self, checkpoint_path):
        with open(checkpoint_path, 'rb') as checkpoint_file:
            self.model = torch.load(checkpoint_file)
            self.model = self.model.to(self.device)

    def output_log(self, **kwargs):
        scores = np.array(self.events[self._phase_name]['scores'])
        labels = np.array(self.events[self._phase_name]['labels'])

        assert len(labels) > 0, print('Label list is empty')
        assert len(scores) > 0, print('Score list is empty')
        assert len(labels) == len(scores), print('Label and score lists are of different size')

        visualize = kwargs.get('visualize', False)
        #         if visualize:
        #             clear_output()

        self.metrics = {
            "loss": np.mean(self.events[self._phase_name]['loss']),
            "accuracy": accuracy_score(labels, scores),
            "f1-weighted": f1_score(labels, scores, average='weighted'),
            "f1-micro": f1_score(labels, scores, average='micro')
        }

        print(f'{self._phase_name}: ', end='')
        print(' | '.join([f'{k}: {v:.4f}' for k, v in self.metrics.items()]))

        self.save_checkpoint()

        if visualize:
            fig = plt.figure(figsize=(15, 5))
            ax1 = fig.add_subplot(1, 2, 1)

            ax1.plot(self.log_dict['train'], color='b', label='train')
            ax1.plot(self.log_dict['val'], color='c', label='val')
            ax1.legend()
            ax1.set_title('Train/val loss.')

            plt.show()


