from src.sumformer import * 
from src.transformer import *
from src.dataset import *

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

output_dir = '/data/sam/coreset/'

cross_entropy_loss = CrossEntropyLoss()

def npz_to_dataset(npz_obj):
    raise NotImplementedError("TODO: implement numpy to CoresetDataset")

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


def train(modeltype, config, train_dataloader,test_dataloader, device, log_dir,
          epochs=100, lr=0.001, activation='LeakyReLU'):

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
    # TODO: add more logging details

    # TODO: Add tensorboard logging
    for epoch in tqdm(epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            pointsets = batch[0].to(device)
            coreset_masks = batch[1].to(device)

            predicted_masks = model(pointsets) # TODO: this mask should have the same size as the coreset_mask

            loss = cross_entropy_loss(predicted_masks, coreset_masks)
            loss = loss/len(batch)

            loss.backward()
            optimizer.step()

    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, help='Needed to construct output directory')
    parser.add_argument('--train-data', type=str)
    parser.add_argument('--test-data', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--configs', type=str, default='configs/config-base.yml')
    parser.add_argument('--device', type=str)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--trial', type=str)
    parser.add_argument('--layer_type', type=str)

    args = parser.parse_args()

    # Load data 
    train_file = os.path.join(output_dir, 'data', args.train_data)
    test_file = os.path.join(output_dir, 'data', args.test_data)

    train_data = np.load(train_file, allow_pickle=True)
    test_data = np.load(test_file, allow_pickle=True)

    train_dataset = npz_to_dataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = npz_to_dataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False)

    # Load model configs
    with open(args.config, 'r') as file:
        model_configs = yaml.safe_load(file)
    
    loss_data = []
    
    for modelname in model_configs:
        log_dir = format_log_dir(output_dir, 
                                args.dataset_name, 
                                modelname, 
                                args.loss, 
                                args.layer_type,
                                args.trial)
        config=model_configs[modelname]

        output = train(args.layer_type, config, train_dataloader, test_dataloader, args.device, log_dir)
        loss_data.append({'modelname':modelname, 'loss':output[-1]})

    # Keep track of validation losses for each configuration
    if 'base' in args.config:
        print("finished training example model")
        return 0
    modeltype = args.layer_type
    fieldnames = ['modelname', 'loss']
    csv_file = os.path.join('output', 
                            args.dataset_name, 
                            args.layer_type,
                            args.loss,
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