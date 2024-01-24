from src.sumformer import * 
from src.transformer import *
from src.dataset import *
from src.data_representation import Batch

from torch.optim import Adam

from torch.nn import CrossEntropyLoss, NLLLoss
from torch.utils.data import DataLoader

import argparse
import os
import yaml
import numpy as np
import csv
import logging
from tqdm import tqdm, trange
import geomloss

from torch.utils.tensorboard.writer import SummaryWriter


output_dir = '/data/sam/coreset/'

cross_entropy_loss = CrossEntropyLoss()
sinkhorn_loss = geomloss.SamplesLoss(loss='sinkhorn')

def npz_to_batches(raw_data, batch_size=128):

    batch_list = []
    gt_list = []
    in_batch =[]
    in_batch_gt = []
    count = 0

    for i in range(raw_data.shape[0]):
        ptset = raw_data[i][:, :-1]
        cvx_hull_idx = np.where(raw_data[i][:, -1] == 1.0)
        in_batch.append(torch.tensor(ptset, dtype=torch.float))
        in_batch_gt.append(torch.tensor(ptset[cvx_hull_idx], dtype=torch.float))
        if (count != 0 and count % (batch_size-1) == 0) or i == raw_data.shape[0] - 1:
            batch = Batch.from_list(in_batch, order = 1)
            batch_list.append(batch)
            gt_list.append(in_batch_gt)
            in_batch = []
            in_batch_gt = []
        count += 1
    return batch_list, gt_list

def get_approx_chull(probabilities, batch):
    hulls = []
    start = 0
    for num in batch.n_nodes:
        end = start + num
        ptset = batch.data[start:end]
        ptset_probs = probabilities.data[start:end]
        hull_approx = torch.mm(ptset_probs.T, ptset)
        hulls.append(hull_approx)
        start = end
    return hulls

def format_log_dir(output_dir, 
                   dataset_name, 
                   modelname, 
                   loss_func, 
                   layer_type,
                   trial):
    log_dir = os.path.join(output_dir, 
                           'models', 
                           dataset_name, 
                           layer_type)
    
    log_dir = os.path.join(log_dir, loss_func, modelname, trial)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

# TODO: Given P and an approximate Q and n directions, we should compute
# \sum_{i = 1}^n abs(max_p <u_i, p> - max_q <u_i, q>) + \sum_i^n abs(min_p <u_i, q> - min_q <u_i, q>)
def direction_loss(P, Q, n, in_dim=3):
    directions = torch.rand((n, in_dim), dtype=torch.float)
    
    return 0

# TODO: Cross entropy loss for n directions
def cross_entropy_loss():
    return 0

def compute_test_error(model, test_dataloader, test_gt, test_sz, device='cuda:0'):
    count = 0
    loss = 0.0
    for batch in test_dataloader:
        batch = batch.to(device)
        out = F.softmax(model(batch), dim=1)
        hulls = get_approx_chull(out, batch)
        gt_p_batch = test_gt[count]

        for i in range(len(gt_p_batch)):
            ground_truth = gt_p_batch[i].to(device)
            loss += sinkhorn_loss(hulls[i], ground_truth).detach()
        count += 1
    loss = loss/test_sz
    return loss

def train(modeltype, config, train_dataloader, train_gt, test_dataloader, test_gt, device, log_dir,
          epochs=100, lr=0.001, activation='LeakyReLU', test_sz = 300, save_freq=20):

    # Initialize model
    model = globals()[modeltype](**config)
    model.to(device)

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    record_dir = os.path.join(log_dir, 'record/')

    # Initialize logs
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    print("logging losses and intermediary models to:", record_dir)

    log_file = os.path.join(record_dir, 'training_log.log')
    logging.basicConfig(level=logging.INFO, filename=log_file)

    logging.info(f'Model type:{modeltype}')
    logging.info('Sample dict log: %s',config)

    writer = SummaryWriter(log_dir=record_dir)

    for epoch in trange(epochs):
        count = 0
        total_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch = batch.to(device)
            out = F.softmax(model(batch), dim=1)
            hulls = get_approx_chull(out, batch)
            gt_p_batch = train_gt[count]

            loss = 0.0
            for i in range(len(gt_p_batch)):
                ground_truth = gt_p_batch[i].to(device)
                loss += sinkhorn_loss(hulls[i], ground_truth)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            count +=1
        # compute test error
        writer.add_scalar('train/mse_loss', total_loss/count, epoch)
        if epoch % save_freq == 0:
            path = os.path.join(record_dir, f'model_{epoch}.pt')
            torch.save(model.state_dict(), path)
    
    test_err = compute_test_error(model, test_dataloader, test_gt, 300)
    print("test error:", test_err)
    path = os.path.join(log_dir, 'final_model.pt')
    print("saving model to:", path)
    torch.save(model.state_dict(), path)

    return test_err

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, help='Needed to construct output directory')
    parser.add_argument('--datafile', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--configs', type=str, default='configs/config-base.yml')
    parser.add_argument('--device', type=str)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--trial', type=str)
    parser.add_argument('--layer-type', type=str)

    args = parser.parse_args()

    # Load data 
    train_file = os.path.join(output_dir, 'data', args.datafile)

    raw_data = np.load(train_file)

    train_batches, train_gt = npz_to_batches(raw_data[:2700], args.batch_size)
    test_batches, test_gt = npz_to_batches(raw_data[2700:], args.batch_size)

    # Load model configs
    with open(args.configs, 'r') as file:
        model_configs = yaml.safe_load(file)
    
    loss_data = []
    
    for modelname in model_configs:
        log_dir = format_log_dir(output_dir, 
                                args.dataset_name, 
                                modelname, 
                                'sinkhorn', 
                                args.layer_type,
                                args.trial)
        config=model_configs[modelname]
        print(config)

        output = train(args.layer_type, config, train_batches,train_gt, 
                       test_batches, test_gt, args.device, log_dir, 
                       epochs=args.epochs)
        
        loss_data.append({'modelname':modelname, 'loss':output.item()})

    # Keep track of test losses for each configuration
    modeltype = args.layer_type
    fieldnames = ['modelname', 'loss']
    csv_file = os.path.join('output', 
                            args.dataset_name, 
                            args.layer_type,
                            'sinkhorn',
                            args.trial)
    if not os.path.exists(csv_file):
        os.makedirs(csv_file)
    csv_file = csv_file + f'/{modeltype}.csv'
    csvfile = open(csv_file, 'w', newline='')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in loss_data:
        writer.writerow(row)
    csvfile.close()
    print(f'Data has been written to {csv_file}')

    return 

if __name__ == '__main__':
    main()