from ctypes.wintypes import MAX_PATH
from src.sumformer import * 
from src.transformer import *
from src.dataset import *
from src.data_representation import Batch

from torch.optim import Adam

from torch.nn import CrossEntropyLoss, NLLLoss
from torch.utils.data import DataLoader
from geotorch.sphere import uniform_init_sphere_ as unif_sphere
from scipy.spatial.distance import directed_hausdorff


import argparse
import os
import yaml
import numpy as np
import csv
import logging
from tqdm import tqdm, trange
import geomloss

from torch.utils.tensorboard.writer import SummaryWriter


output_dir = '/data/oren/coreset/'
input_dir = '/data/oren/coreset/'

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

def get_approx_chull(output, od):
    hulls = []
    start = 0
    
    while (start + 8) <= output.data.shape[0]:
    # for num in output.n_nodes:
        
        end = start + od
        hull_approx = output.data[start:end]
        hulls.append(hull_approx)
        start = end

    print(len(hulls))
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
def direction_loss(p, q, directions, n=100, in_dim=3, device='cuda:0'):
    # directions = unif_sphere(torch.zeros(n,in_dim))
    directions = directions.t().double().to(device)
    p, q = p.to(device), q.to(device)

    
    changes_p = p.batch1[:-1] != p.batch1[1:]
    changes_q = q.batch1[:-1] != q.batch1[1:]
    idx_change_p = (torch.nonzero(changes_p) + 1).squeeze().cpu()
    idx_change_q = (torch.nonzero(changes_q) + 1).squeeze().cpu()


    idx_change_p = torch.cat((idx_change_p, torch.tensor([len(p.data)])))
    idx_change_q = torch.cat((idx_change_q, torch.tensor([len(q.data)])))




    lengths_p = idx_change_p - torch.cat((torch.tensor([0]), idx_change_p[:-1]))
    lengths_q = idx_change_q - torch.cat((torch.tensor([0]), idx_change_q[:-1]))

    p_slices = torch.split(p.data, lengths_p.tolist())
    q_slices = torch.split(q.data, lengths_q.tolist())


    p_batch = torch.stack(p_slices) #only works if all point sets same size
    q_batch = torch.stack(q_slices) #only works if all point sets same size


    projections_q = torch.matmul(q_batch.double(), directions)
    projections_p = torch.matmul(p_batch.double(), directions)
    

    max_q = torch.max(projections_q, dim=1)[0]  
    min_q = torch.min(projections_q, dim=1)[0]  
    max_p = torch.max(projections_p, dim=1)[0]
    min_p = torch.min(projections_p, dim=1)[0]

    
    diff_max = torch.abs(max_q - max_p)
    diff_min = torch.abs(min_q - min_p)

    losses = torch.sum(diff_max + diff_min, dim=1)


    return torch.mean(losses)

# TODO: Cross entropy loss for n directions
def cross_entropy_loss():
    return 0

def compute_test_error(model, test_dataloader, test_gt, test_sz, device='cuda:0'):
    count = 0
    loss = torch.zeros(1).to(device)
    directions = unif_sphere(torch.zeros(50, 2)) #hardcoding size of point set and in_dim
    for batch in test_dataloader:

        batch = batch.to(device)
        out = model(batch) #new model
        batch_len = out.data.shape[0]

        out = out.view(batch_len * 8, 2) #hardcoding for now: out.view(batch_size * output_dim, 2) 

        # hulls = Batch.from_list(get_approx_chull(out, od=8), order = 1) #hardcoding output dim
      
        

        loss += direction_loss(out, batch, directions, n=50, in_dim=2, device = device) #hardcoding in_dim -- change later
        count += 1
    loss = loss/test_sz
    return loss


def gen_model_output(model, train_dataloader, test_dataloader, log_dir, epoch, device='cuda:0'):
    model.to(device)

    train_hulls = []
    test_hulls = []

    for batch in train_dataloader:
        batch = batch.to(device)
        out = model(batch) #new model

        train_hulls += [tensor.cpu().detach().numpy() for tensor in get_approx_chull(out, batch)]
            
    model_output_dir = os.path.join(log_dir, 'output')
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)


    train_hull_file = f'chull_train_e{epoch}'
    np.save(os.path.join(model_output_dir, train_hull_file), np.array(train_hulls))
    print(f'output saved to {train_hull_file}')

    for batch in test_dataloader:
        batch = batch.to(device)
        out = model(batch)

        test_hulls += [tensor.cpu().detach().numpy() for tensor in get_approx_chull(out, batch)]
            
       
    test_hull_file = f'chull_test_e{epoch}'
    np.save(os.path.join(model_output_dir, test_hull_file), np.array(test_hulls))
    print(f'output saved to {test_hull_file}')


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
        
        #generating directions -- hardcoding for 2d
        angles = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)

        x_tensor = torch.tensor(x)
        y_tensor = torch.tensor(y)
        directions = torch.stack([x_tensor, y_tensor], dim=1)


        for batch in train_dataloader:


            optimizer.zero_grad()
            batch = batch.to(device)
            out = model(batch)
            batch_len = out.data.shape[0]
            out = out.view(batch_len * 8, 2) #hardcoding od for now: out.view(batch_len * output_dim, 2) 
            # hulls = Batch.from_list(get_approx_chull(out, od=8), order = 1) #hardcoding output dim
            
                        
            loss = direction_loss(out, batch, directions, n=50, in_dim=2, device = device) #hardcoding in_dim -- change later
            
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            count +=1


        # compute test error
        writer.add_scalar('train/mse_loss', total_loss/count, epoch)
        if epoch % save_freq == 0:
            path = os.path.join(record_dir, f'model_{epoch}.pt')
            torch.save(model.state_dict(), path)

            #saving output
            # gen_model_output(model, train_dataloader, test_dataloader, log_dir, epoch, device)

            # hulls = [tensor.cpu().detach().numpy() for tensor in get_approx_chull(out, batch)]
            # print(np.array(hulls))
            
            # model_output_dir = os.path.join(log_dir, 'output')
            # if not os.path.exists(model_output_dir):
            #     os.makedirs(model_output_dir)
            # np.save(os.path.join(model_output_dir, f'chull_test_e{epoch}'), np.array(hulls))
            # print('output saved')
            # hulls = [torch.tensor(arr) for arr in hulls]
            # hulls = Batch.from_list(hulls, order = 1) #casting back to batch
    
    test_err = compute_test_error(model, test_dataloader, test_gt, 300, device = device)
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
                                'direction', 
                                args.layer_type,
                                args.trial)
        config=model_configs[modelname]
        print(config)

        output = train(args.layer_type, config, train_batches,train_gt, 
                       test_batches, test_gt, args.device, log_dir, 
                       epochs=args.epochs, save_freq = 20) #added save_freq
        
        loss_data.append({'modelname':modelname, 'loss':output.item()})

    # Keep track of test losses for each configuration
    modeltype = args.layer_type
    fieldnames = ['modelname', 'loss']
    csv_file = os.path.join('output', 
                            args.dataset_name, 
                            args.layer_type,
                            'direction',
                            args.trial)
    if not os.path.exists(csv_file):
        os.makedirs(csv_file)
    csv_file = csv_file + f'/{modeltype}.csv'
    csvfile = open(csv_file, 'w', newline='')
    csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
    csvwriter.writeheader()
    for row in loss_data:
        csvwriter.writerow(row)
    csvfile.close()
    print(f'Data has been written to {csv_file}')

    return 

if __name__ == '__main__':
    main()