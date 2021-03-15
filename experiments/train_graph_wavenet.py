import os
import sys
sys.path.append(os.path.join(os.getcwd()))
print(sys.path)
import torch
import numpy as np
import argparse
import time
from utils import utils_wavenet as util
import matplotlib.pyplot as plt
from experiments.engine_wavenet import trainer
from utils.summary import TensorboardSummary
from tqdm import trange, tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='/media/csu2020/5ccb40f7-7d27-43c0-a2be-9070bdad41db/csudx/Documents/Traffic/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='/home/csu2020/Zhuge/Traffic/Graph-WaveNet/data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=128,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='checkpoints/metr',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

args = parser.parse_args()


class Trainer():

    def __init__(self):
        self.init_routes()
        self.summary = TensorboardSummary(os.path.join('result_wavenet', 'events')) #writer.add_scalar("train/mape", mape, epoch)

    def init_routes(self, basename='result_wavenet'):
        print("Init: {}".format(basename))
        if os.path.exists(basename):
            import shutil
            shutil.rmtree(basename)
        os.mkdir(basename)
        os.mkdir(os.path.join(basename, 'weights'))
        os.mkdir(os.path.join(basename, 'logs'))
        os.mkdir(os.path.join(basename, 'events'))
        os.mkdir(os.path.join(basename, 'graphs'))

    def main(self):
        """
        TODO: there are some bugs in vissualization of performance curves: there's sth wrong in the iteration dimention
        :return:
        """
        #set seed
        #torch.manual_seed(args.seed)
        #np.random.seed(args.seed)
        #load data
        device = torch.device(args.device)
        # id, sensor_id_to_ind, adj; What is sensor_id_to_ind
        sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype) # by default adjtype is 'doubletransition'
        dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
        scaler = dataloader['scaler']
        supports = [torch.tensor(i).to(device) for i in adj_mx]

        print(args)
       # if os.path.exists(args.save):
        #    os.mkdir(args.save)
        if args.randomadj:
            adjinit = None
        else:
            adjinit = supports[0]

        if args.aptonly:
            supports = None



        engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                             args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                             adjinit)


        print("Start training...",flush=True)
        his_loss =[]
        val_time = []
        train_time = []
        for i in range(1,args.epochs+1):
            #if i % 10 == 0:
                #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
                #for g in engine.optimizer.param_groups:
                    #g['lr'] = lr
            train_loss = []
            train_mape = []
            train_rmse = []
            t1 = time.time()
            dataloader['train_loader'].shuffle()
            for iter, (x, y) in tqdm(enumerate(dataloader['train_loader'].get_iterator())):
                trainx = torch.Tensor(x).to(device)
                trainx= trainx.transpose(1, 3)
                trainy = torch.Tensor(y).to(device)
                trainy = trainy.transpose(1, 3)
                metrics = engine.train(trainx, trainy[:,0,:,:])
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
                self.summary.writer.add_scalar("iteration_train/loss(MAE)", metrics[0], (i-1)*dataloader['train_loader'].size+iter)
                self.summary.writer.add_scalar("iteration_train/mape", metrics[1], (i-1)*dataloader['train_loader'].size+iter)
                self.summary.writer.add_scalar("iteration_train/rmse", metrics[2], (i-1)*dataloader['train_loader'].size+iter)
                if iter % args.print_every == 0 :
                    log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
            # loss = sum(train_loss) / len(train_loss)
            # mape = sum(train_mape) / len(train_mape)
            # rmse = sum(train_rmse) / len(train_rmse)

            # self.summary.writer.add_scalar("train/total_loss", loss, i)
            # self.summary.writer.add_scalar("train/total_mape", mape, i)
            # self.summary.writer.add_scalar("train/total_rmse", rmse, i)
            t2 = time.time()
            train_time.append(t2-t1)
            #validation
            valid_loss = []
            valid_mape = []
            valid_rmse = []


            s1 = time.time()
            for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)
                testy = torch.Tensor(y).to(device)
                testy = testy.transpose(1, 3)
                metrics = engine.eval(testx, testy[:,0,:,:])
                valid_loss.append(metrics[0])
                valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2])
                self.summary.writer.add_scalar("iteration_val/loss(MAE)", metrics[0], (i-1)*dataloader['val_loader'].size+iter)
                self.summary.writer.add_scalar("iteration_val/mape", metrics[1], (i-1)*dataloader['val_loader'].size+iter)
                self.summary.writer.add_scalar("iteration_val/rmse", metrics[2], (i-1)*dataloader['val_loader'].size+iter)
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(log.format(i,(s2-s1)))
            val_time.append(s2-s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)
            self.summary.writer.add_scalar("train/total_loss", mtrain_loss, i)
            self.summary.writer.add_scalar("train/total_mape", mtrain_mape, i)
            self.summary.writer.add_scalar("train/total_rmse", mtrain_rmse, i)
            self.summary.writer.add_scalar("val/total_loss", mvalid_loss, i)
            self.summary.writer.add_scalar("val/total_mape", mvalid_mape, i)
            self.summary.writer.add_scalar("val/total_rmse", mvalid_rmse, i)
            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
            #torch.save(engine.model.state_dict(), os.path.join(args.save, "_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth"))
            torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

        #testing
        bestid = np.argmin(his_loss)
        engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))


        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1,3)[:,0,:,:]

        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1,3)
            with torch.no_grad():
                preds = engine.model(testx).transpose(1,3)
            outputs.append(preds.squeeze())

        yhat = torch.cat(outputs,dim=0)
        yhat = yhat[:realy.size(0),...]


        print("Training finished")
        print("The valid loss on best model is", str(round(his_loss[bestid],4)))


        amae = []
        amape = []
        armse = []
        for i in range(12):
            pred = scaler.inverse_transform(yhat[:,:,i])
            real = realy[:,:,i]
            metrics = util.metric(pred,real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])

        log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
        torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")



if __name__ == "__main__":
    t1 = time.time()
    Trainer().main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
