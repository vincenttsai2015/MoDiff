import os
import random
import time
from tqdm import tqdm, trange
import numpy as np
import torch

from utils.loader import load_seed, load_device, load_data_TD_train_comp, load_model_params, load_model_optimizer, \
    load_ema, load_batch, load_batch2, load_loss_fn4DT
from utils.logger import Logger, set_log, start_log, train_log


class Trainer_G_DT_comp(object):
    def __init__(self, config):
        super(Trainer_G_DT_comp, self).__init__()

        self.config = config
        self.config.data.file1 = f'sampled_{config.scale}/motif/G0_mot'
        self.config.data.file2 = f'sampled_{config.scale}/motif/G1_mot'
        print("self.config:",self.config)
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)

        self.seed = load_seed(self.config.seed)
        print("seed:", self.seed)
        self.device = [0] #load_device()
        self.train_loader_G0, self.train_loader_G1, self.test_loader_G0, self.test_loader_G1 = load_data_TD_train_comp(self.config)
        self.params_x, self.params_adj = load_model_params(self.config)
        self.train_loss = [2000] * 1

    def train(self, ts):
        self.config.exp_name = ts
        self.ckpt = f'{ts}'
        print('\033[91m' + f'{self.ckpt}' + '\033[0m')

        # -------- Load models, optimizers, ema --------
        self.model_x, self.optimizer_x, self.scheduler_x = load_model_optimizer(self.params_x, self.config.train, self.device)
        self.model_adj, self.optimizer_adj, self.scheduler_adj = load_model_optimizer(self.params_adj, self.config.train, self.device)
        self.ema_x = load_ema(self.model_x, decay=self.config.train.ema)
        self.ema_adj = load_ema(self.model_adj, decay=self.config.train.ema)

        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)
        start_log(logger, self.config)
        train_log(logger, self.config)

        self.loss_fn = load_loss_fn4DT(self.config)

        # -------- Training --------
        for epoch in tqdm(range(self.config.train.num_epochs), desc='[Epoch]', position=1, leave=False, mininterval=10):

            self.train_x, self.train_adj, self.test_x, self.test_adj  = [], [], [], []
            t_start = time.time()

            self.model_x.train()
            self.model_adj.train()

            for _, (train_b_0, train_b_1) in enumerate(zip(self.train_loader_G0, self.train_loader_G1)):
                self.optimizer_x.zero_grad()
                self.optimizer_adj.zero_grad()
                x_0, adj_0, u_0, la_0 = load_batch2(train_b_0, self.device)
                la_1 = train_b_1[3].to(f'cuda:{self.device[0]}')

                loss_subject = (x_0, adj_0, u_0, la_0, la_1)

                loss_x, loss_adj = self.loss_fn(self.model_x, self.model_adj, *loss_subject)
                loss_x.backward()
                loss_adj.backward()

                if torch.isnan(loss_adj):   print("nan~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                torch.nn.utils.clip_grad_norm_(self.model_x.parameters(), self.config.train.grad_norm)
                torch.nn.utils.clip_grad_norm_(self.model_adj.parameters(), self.config.train.grad_norm)

                self.optimizer_x.step()
                self.optimizer_adj.step()

                # -------- EMA update --------
                self.ema_x.update(self.model_x.parameters())
                self.ema_adj.update(self.model_adj.parameters())

                self.train_x.append(loss_x.item())
                self.train_adj.append(loss_adj.item())

            if self.config.train.lr_schedule:
                self.scheduler_x.step()
                self.scheduler_adj.step()

            self.model_x.eval()
            self.model_adj.eval()
            for _,  (test_b_0, test_b_1) in enumerate(zip(self.test_loader_G0, self.test_loader_G1)):
                x_0, adj_0, u_0, la_0 = load_batch2(test_b_0, self.device)
                u_1 = test_b_1[2].to(f'cuda:{self.device[0]}')
                la_1 = test_b_1[3].to(f'cuda:{self.device[0]}')
                loss_subject = (x_0, adj_0, u_0, la_0, la_1)

                with torch.no_grad():
                    self.ema_x.store(self.model_x.parameters())
                    self.ema_x.copy_to(self.model_x.parameters())
                    self.ema_adj.store(self.model_adj.parameters())
                    self.ema_adj.copy_to(self.model_adj.parameters())

                    loss_x, loss_adj = self.loss_fn(self.model_x, self.model_adj, *loss_subject)
                    self.test_x.append(loss_x.item())
                    self.test_adj.append(loss_adj.item())

                    self.ema_x.restore(self.model_x.parameters())
                    self.ema_adj.restore(self.model_adj.parameters())

            mean_train_x = np.mean(self.train_x)
            mean_train_adj = np.mean(self.train_adj)
            mean_test_x = np.mean(self.test_x)
            mean_test_adj = np.mean(self.test_adj)

            # -------- Log losses --------
            logger.log(f'{epoch + 1:03d} | {time.time() - t_start:.2f}s | '
                       f'test x: {mean_test_x:.3e} | test adj: {mean_test_adj:.3e} | '
                       f'train x: {mean_train_x:.3e} | train adj: {mean_train_adj:.3e} | ', verbose=False)

            # -------- Save checkpoints --------
            if epoch % self.config.train.num_epochs == self.config.train.num_epochs  - 1: #self.config.train.save_interval
                torch.save({
                    'model_config': self.config,
                    'params_x': self.params_x,
                    'params_adj': self.params_adj,
                    'x_state_dict': self.model_x.state_dict(),
                    'adj_state_dict': self.model_adj.state_dict(),
                    'ema_x': self.ema_x.state_dict(),
                    'ema_adj': self.ema_adj.state_dict()
                }, f'./checkpoints/{self.config.data.data}/{self.ckpt}.pth')

            if epoch % self.config.train.print_interval == self.config.train.print_interval - 1:
                tqdm.write(f'[EPOCH {epoch + 1:04d}] test adj: {mean_test_adj:.3e} | train adj: {mean_train_adj:.3e} | test x: {mean_test_x:.3e} | train x: {mean_train_x:.3e}')
        print(' ')
        return self.ckpt