from ctypes.wintypes import MAX_PATH
from src.sumformer import * 
from src.transformer import *
from src.dataset import *
from src.data_representation import Batch
from eval_utils import directional_width, directional_err

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
import json
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


def json_to_batches(raw_data, batch_size=128):
    batch_list = []
    ground_truth_list = []  # List for single-vector ground truth
    in_batch = []
    in_batch_ground_truth = []
    count = 0

    for i in range(len(raw_data)):
        # Extract point set, center, and radius
        ptset = raw_data[i]['pointset']
        center = raw_data[i]['meb_center']  # List of d values
        radius = raw_data[i]['meb_radius']  # Scalar value

        in_batch.append(torch.tensor(ptset, dtype=torch.float))
        
        # Combine center and radius into a single tensor
        ground_truth = torch.tensor(center + [radius], dtype=torch.float)  # [c1, c2, ..., cd, r]
        in_batch_ground_truth.append(ground_truth)
        
        # Batch management
        if (count != 0 and count % (batch_size - 1) == 0) or i == len(raw_data) - 1:
            batch = Batch.from_list(in_batch, order=1)  # Your custom Batch object
            batch_list.append(batch)
            ground_truth_list.append(torch.stack(in_batch_ground_truth))  # Stack tensors into one batch
            
            # Reset temporary lists
            in_batch = []
            in_batch_ground_truth = []

        count += 1

    return batch_list, ground_truth_list

def get_approx_chull(probabilities, batch):
    hulls = []
    start = 0
    for num in batch.n_nodes:
        end = start + num
        ptset = batch.data[start:end]
        ptset_probs = probabilities.data[start:end]
        # print(f'probs (from softmax): {ptset_probs}')
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

def centroid_distance(a, b):
    centroid_a = torch.mean(a, axis=0)
    distances = torch.norm(b - centroid_a, dim=1)
    
    return torch.mean(distances)


# TODO: Given P and an approximate Q and n directions, we should compute
# \sum_{i = 1}^n abs(max_p <u_i, p> - max_q <u_i, q>) + \sum_i^n abs(min_p <u_i, q> - min_q <u_i, q>)
def direction_loss(p, q, directions = None, n=100, in_dim=3, device='cuda:0'):
    if directions is None:
        directions = unif_sphere(torch.zeros(n,in_dim)).T.to(device) #random directions
    
    proj_p = torch.matmul(p, directions)
    proj_q = torch.matmul(q, directions)


    # print(f'batch1 shape: {proj_p.batch1.shape}')

    max_p = global_max_pool(x = proj_p.data, batch = proj_p.batch1) #add batch indicator manually
    max_q = global_max_pool(x = proj_q.data, batch = proj_q.batch1)

    min_p = -1 * global_max_pool(x = -1 * proj_p.data, batch = proj_p.batch1)
    min_q = -1 * global_max_pool(x = -1 * proj_q.data, batch = proj_q.batch1)

    diff_max = torch.abs(max_q - max_p)
    diff_min = torch.abs(min_q - min_p)

    losses = torch.sum(diff_max + diff_min, dim=1)
    
    return torch.mean(losses) #- centroid_distance(q.data, p.data)

# TODO: Cross entropy loss for n directions
def cross_entropy_loss():
    return 0

def compute_test_error(model, directions, test_dataloader, test_gt, test_sz, device='cuda:0'):
    try:
        count = 0
        loss = 0
        for batch in test_dataloader:
            batch = batch.to(device)

            if isinstance(model, ConvexHullEncoderTransformer):
                out, attn_maps = model(batch)
                #reshaping to apply softmax setwise
                out = out.data.view(-1, 25, out.data.size(-1)) #todo: hardcoding ptset size
                out = F.softmax(out, dim=1)
                out = out.view(-1, out.size(-1))

                hulls = Batch.from_list(get_approx_chull(out, batch), order = 1).to(device)
        
                loss += direction_loss(hulls, batch, directions=directions, in_dim = 2, device = device).detach() #todo: hardcoding input_dim
                

            if isinstance(model, ConvexHullNN) or isinstance(model, ConvexHullNNTransformer):
                out = model(batch) #old model

                #reshaping to apply softmax setwise
                out = out.data.view(-1, 25, out.data.size(-1)) #todo: hardcoding ptset size
                out = F.softmax(out, dim=1)
                out = out.view(-1, out.size(-1))

                hulls = Batch.from_list(get_approx_chull(out, batch), order = 1).to(device)
        
                loss += direction_loss(hulls, batch, directions=directions, in_dim = 2, device = device).detach() #todo: hardcoding in_dim

            else:
                out = model(batch)
                batch_size = int(batch.data.shape[0] / 50) #hardcoding ptset_size
                n_nodes = torch.full((batch_size,), 50).to(device) #hardcoding output_dim for now
                out =  Batch.from_batched(out, n_nodes = n_nodes, order = 1)

                loss = direction_loss(out, batch, directions=directions, n=50, in_dim=10, device = device) #todo: hardcoding in_dim 


            count += 1
        loss = loss/test_sz
        return loss
    except:
        return None


def train(modeltype, config, train_dataloader, train_gt, test_dataloader, test_gt, device, log_dir, epd,
          epochs=100, lr=0.001, activation='LeakyReLU', test_sz = 300, save_freq=20, args=None):



    # Initialize model
    model = globals()[modeltype](**config)
    model.to(device)

    # Freeze processor weights
    for param in model.processor.parameters():
        param.requires_grad = False

    trainable_params = list(model.encoder.parameters()) + list(model.decoder.parameters()) #only encoder and decoder params

    # Initialize optimizer
    optimizer = Adam(trainable_params, lr=lr)



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

    # if not epd:
    # # generating evenly spaced directions -- hardcoding for 2d
    angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)
    directions = torch.stack([x_tensor, y_tensor], dim=1)
    directions = directions.t().float().to(device)
    directions = None

    for epoch in trange(epochs):
        count = 0
        total_loss = 0.0


        for batch in train_dataloader:

            optimizer.zero_grad()
            batch = batch.to(device)



            if epd:
               
                out = model(batch)
                print(out.shape)
                # out = out.view(-1, 50, out.data.size(-1)) #meb ptset size = 50
                # out = F.softmax(out, dim=1)
                # out = out.view(-1, out.data.size(-1))

                # hulls = Batch.from_list(get_approx_chull(out, batch), order = 1).to(device)
                
                #todo: check loss function
                loss = direction_loss(out, batch, directions=directions, n=200, in_dim=3, device = device) #todo: hardcoding in_dim



            elif isinstance(model, ConvexHullNN) or isinstance(model, ConvexHullNNTransformer):
                out = model(batch)
                # if isinstance(model, ConvexHullNNTransformer):
                #     out, attn_maps = model(batch)
                # else:
                #     out = model(batch)

                #reshaping to apply softmax setwise
                out = out.view(-1, 25, out.data.size(-1))
                out = F.softmax(out, dim=1)
                out = out.view(-1, out.data.size(-1))

                hulls = Batch.from_list(get_approx_chull(out, batch), order = 1).to(device)
                
                loss = direction_loss(hulls, batch, directions=directions, n=200, in_dim=2, device = device) #todo: hardcoding in_dim

            else:
                out = model(batch)
                batch_size = int(batch.data.shape[0] / 25) #hardcoding
                n_nodes = torch.full((batch_size,), 25).to(device) #hardcoding output_dim for now
                out =  Batch.from_batched(out, n_nodes = n_nodes, order = 1)


                loss = direction_loss(out, batch, directions, n=200, in_dim=5, device = device) #todo: hardcoding in_dim -- change later
                
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            count +=1


        # compute test error
        writer.add_scalar('train/mse_loss', total_loss/count, epoch)
        if epoch % save_freq == 0:
            path = os.path.join(record_dir, f'model_{epoch}.pt')
            torch.save(model.state_dict(), path)
    
    
    path = os.path.join(log_dir, 'final_model.pt')
    print("saving model to:", path)
    torch.save(model.state_dict(), path)

    test_err = compute_test_error(model, directions, test_dataloader, test_gt, 300, device = device)
    print("test error:", test_err)

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
    parser.add_argument('--epd', type=lambda x: x.lower() == 'true', default=False, help='Set EPD to True or False')
    parser.add_argument('--processor-path', type=str, help='Path to pre-trained processor weights')

    
    

    args = parser.parse_args()
   

    # Load data 
    train_file = os.path.join(output_dir, 'data', args.datafile)



    raw_data = json.load(open(train_file))

    train_batches, train_gt = json_to_batches(raw_data[:2400], args.batch_size) #dataset size = 3000
    test_batches, test_gt = json_to_batches(raw_data[2400:], args.batch_size)

   

    print(args)

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


        output = train(args.layer_type, config, train_batches, train_gt, 
                    test_batches, test_gt, args.device, log_dir, 
                    epochs=args.epochs, epd = args.epd, save_freq = 20, args=args) #added save_freq and args
        
        loss_data.append({'modelname':modelname, 'loss':output})

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