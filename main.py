import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from models.stgcn import STGCN
from utils.utils import generate_dataset, load_metr_la_data, get_normalized_adj,load_pems_m_data

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils.summary import TensorboardSummary
from utils.logger import Logger
from tqdm import trange

use_gpu = False


class Trainer():
    def __init__(self, resume, device, weight_path, batch_size, dataset, epochs):
        torch.manual_seed(7)
        self.resume = resume
        self.device = device
        if not self.resume:
            self.init_routes()
        # Training params
        self.start_epoch = 0
        self.num_timesteps_input = 12
        self.num_timesteps_output = 3

        self.epochs = epochs
        self.batch_size = batch_size

        # Tools
        self.summary = TensorboardSummary(os.path.join('result', 'events'))

        # Model
        self.model = None
        self.optimizer = None
        self.loss_criterion = None


        # Eval measures
        self.best_mae = 10000
        self.best_loss = 10000





    def load_data(self):
        print("LOAD DATA...")
        self.data = load_metr_la_data()
        # A, X, means, stds = load_pems_m_data()

        print("==LOAD DATA SUCEESS: A-{0} X-{1} means-{2} stds-{3}".format(self.data.A.shape, self.data.X.shape, self.data.means, self.data.stds))

        split_line1 = int(self.data.X.shape[2] * 0.8)
        split_line2 = int(self.data.X.shape[2] * 0.9)

        train_original_data = self.data.X[:, :, :split_line1]
        val_original_data = self.data.X[:, :, split_line1:split_line2]
        test_original_data = self.data.X[:, :, split_line2:]

        self.training_input, self.training_target = generate_dataset(train_original_data,
                                                           num_timesteps_input=self.num_timesteps_input,
                                                           num_timesteps_output=self.num_timesteps_output)
        self.val_input, self.val_target = generate_dataset(val_original_data,
                                                 num_timesteps_input=self.num_timesteps_input,
                                                 num_timesteps_output=self.num_timesteps_output)
        self.test_input, self.test_target = generate_dataset(test_original_data,
                                                   num_timesteps_input=self.num_timesteps_input,
                                                   num_timesteps_output=self.num_timesteps_output)

        self.A_wave = get_normalized_adj(self.data.A)
        self.A_wave = torch.from_numpy(self.A_wave)

        self.A_wave = self.A_wave.to(device=args.device)

    def init_routes(self, basename='result'):
        if os.path.exists(basename):
            import shutil
            shutil.rmtree(basename)
        os.mkdir(basename)
        os.mkdir(os.path.join(basename, 'weights'))
        os.mkdir(os.path.join(basename, 'logs'))
        os.mkdir(os.path.join(basename, 'events'))
        os.mkdir(os.path.join(basename, 'graphs'))

    def resume_weight(self):
        last_weight = 'result/weights/last.pt'
        chkpt = torch.load(last_weight, map_location=self.device)

        self.model.load_state_dict(chkpt['model'])

        self.start_epoch = chkpt['epoch'] + 1
        if chkpt['optimizer'] is not None:
            self.optimizer.load_state_dict(chkpt['optimizer'])
            self.best_mae = chkpt['best_mae']
        del chkpt

        print("resume model weights from : {}".format(last_weight))

    def save_checkpoint(self, state, is_best=False):
        torch.save(state,'result/weights/last.pt')
        if is_best:
            torch.save({
                "epoch": state["epoch"],
                # "best_loss": self.best_loss,
                "best_mae": self.best_mae,
                "model": self.model,
                "optimizer": self.optimizer
            }, 'result/weights/best.pt')


    def run(self):
        self.load_data()
        print("==PROCESS DATA FINISHED: A_wave-{0}".format(self.A_wave.shape))
        print("DEFINE MODEL")

        self.model = STGCN(self.A_wave.shape[0],
                    self.training_input.shape[3],
                    self.num_timesteps_input,
                    self.num_timesteps_output).to(device=args.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0.0005)
        self.loss_criterion = nn.MSELoss()
        if self.resume:
            self.resume_weight()
        training_losses = []
        validation_losses = []
        validation_maes = []
        validation_rmse = []
        validation_r2 = []
        print("BEGIN TRAINING")
        for epoch in range(self.start_epoch, self.epochs):
            print("Epoch: {}".format(epoch))
            loss = self.train_epoch(epoch, self.training_input, self.training_target,
                               batch_size=self.batch_size)
            training_losses.append(loss)

            # Run validation
            with torch.no_grad():
                self.model.eval()
                self.val_input = self.val_input.to(device=self.device)
                self.val_target = self.val_target.to(device=self.device)

                out = self.model(self.A_wave, self.val_input)
                val_loss = self.loss_criterion(out, self.val_target).to(device="cpu")
                validation_losses.append(np.asscalar(val_loss.detach().numpy()))

                out_unnormalized = out.detach().cpu().numpy() * self.data.stds[0] + self.data.means[0]
                target_unnormalized = self.val_target.detach().cpu().numpy() * self.data.stds[0] + self.data.means[0]
                mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
                rmse = np.sqrt(mean_squared_error(out_unnormalized.reshape(-1, 1), target_unnormalized.reshape(-1,1)))
                r2 = r2_score(out_unnormalized.reshape(-1, 1), target_unnormalized.reshape(-1,1))
                validation_maes.append(mae)
                validation_r2.append(r2)
                validation_rmse.append(rmse)

                out = None
                self.val_input = self.val_input.to(device="cpu")
                self.val_target = self.val_target.to(device="cpu")
            print("Epoch: {}".format(epoch))
            print("Training loss: {}".format(training_losses[-1]))
            print("Validation loss: {}".format(validation_losses[-1]))
            print("Validation MAE: {}".format(validation_maes[-1]))
            print("Training RMSE: {}".format(validation_rmse[-1]))
            print("Validation R2: {}".format(validation_r2[-1]))

            self.summary.writer.add_scalar("val/loss", loss, epoch)
            self.summary.writer.add_scalar("val/mae", validation_maes[-1], epoch)
            self.summary.writer.add_scalar("val/rmse", rmse, epoch)
            self.summary.writer.add_scalar("val/R2", r2, epoch)

            checkpoint_path = "checkpoints/"
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            with open("checkpoints/losses.pk", "wb") as fd:
                pk.dump((training_losses, validation_losses, validation_maes, validation_rmse, validation_r2), fd)
            self.save_checkpoint({
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,

                })
            if(mae < self.best_mae):
                self.save_checkpoint({
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "best_mae": mae
                }, is_best=True)

    def train_epoch(self, epoch, training_input, training_target, batch_size):
        """
        Trains one epoch with the given data.
        :param training_input: Training inputs of shape (num_samples, num_nodes,
        num_timesteps_train, num_features).
        :param training_target: Training targets of shape (num_samples, num_nodes,
        num_timesteps_predict).
        :param batch_size: Batch size to use during training.
        :return: Average loss for this epoch.
        """
        permutation = torch.randperm(training_input.shape[0])

        epoch_training_losses = []
        with trange(self.start_epoch, training_input.shape[0], batch_size) as t:
            for i in t:
                t.set_description("Traing details: ")
                self.model.train()
                self.optimizer.zero_grad()

                indices = permutation[i:i + batch_size]
                X_batch, y_batch = training_input[indices], training_target[indices]
                X_batch = X_batch.to(device=self.device)
                y_batch = y_batch.to(device=self.device)

                out = self.model(self.A_wave, X_batch)
                loss = self.loss_criterion(out, y_batch)
                loss.backward()
                self.summary.writer.add_scalar("train/iteration_loss", loss, i + self.training_input.shape[0] * epoch)
                self.optimizer.step()
                epoch_training_losses.append(loss.detach().cpu().numpy())
                t.set_postfix(iteration=i, loss=loss)
                # print("iteration:{0}, loss: {1}".format(i, loss))
        loss = sum(epoch_training_losses)/len(epoch_training_losses)
        self.summary.writer.add_scalar("train/total_loss", loss, epoch)
        return loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume training')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Epochs')
    parser.add_argument('--enable-cuda', action='store_true', default=True,
                        help='Enable CUDA')
    parser.add_argument('--weight-path', type=str, default='result/weights/last.pt',
                        help='Weight path')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')


    args = parser.parse_args()
    args.device = None
    if args.enable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    Trainer(
        resume=args.resume,
        device=args.device,
        weight_path=args.weight_path,
        batch_size=args.batch_size,
        dataset='melr',
        epochs=args.epochs
    ).run()

