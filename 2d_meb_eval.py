import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from eval_utils import *
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import pandas as pd
import json
from collections import defaultdict
import re
import yaml
import argparse



from src.sumformer import *


def count_pts_outside(points, center, radius):
    """
    Count the number of points outside the estimated minimum enclosing ball.
    """
    distances_squared = torch.sum((points - center) ** 2, axis=1)
    radius_squared = radius ** 2
    
    # Count points whose squared distance is greater than the squared radius
    count_outside = torch.sum(distances_squared > radius_squared)
    
    return count_outside

def meb_eval(gt, preds, pts):
    center = preds[:-1].detach() #pred
    rad = preds[-1].detach()

    gt_center = gt[:-1]
    gt_rad = gt[-1]

    center_diff = np.linalg.norm(gt_center - center)
    rad_diff = abs(gt_rad - rad).item() / gt_rad

    num_outside = count_pts_outside(pts, center, rad).item() / len(pts)

    #return (center_diff + rad_diff).item()
    return float(center_diff), float(rad_diff), float(num_outside)
    
def evaluate(batches, gt, model, device = 'cpu'):
    errs = []
    model.eval()

    # device = 'cpu'
    model = model.to(device)
    
    for idx, batch in tqdm(enumerate(batches)):
        batch = batch.to(device)
        out = model(batch)
    
        start = 0
        for i, num in enumerate(batch.n_nodes):
            end = start + num
            ptset = batch.data[start:end]
            
            preds = out[i].cpu()
            ground_truth = gt[idx][i]
            
            errs.append(meb_eval(ground_truth, preds, ptset.cpu()))
            start = end

    df = pd.DataFrame(errs, columns=["center error", "radius error", "proportion outside"])
    return (df['center error'].mean(), df['center error'].std(), df['radius error'].mean(), df['radius error'].std(),
            df['proportion outside'].mean(), df['proportion outside'].std())

def extract_hyperparameters(filepath):
    # Define the regular expression pattern
    pattern = r"(?P<name>[a-z-]+)(?P<value>\d+)"
    
    # Use defaultdict to store each hyperparameter as a list of values
    hyperparameters = defaultdict(list)
    
    # Use finditer to get all matches in the filepath
    matches = re.finditer(pattern, filepath)
    
    # Iterate through matches and store them in the dictionary
    for match in matches:
        name = match.group("name")  # hyperparameter name
        value = int(match.group("value"))  # convert value to integer
        hyperparameters[name].append(value)
    
    return dict(hyperparameters)    

def define_model(params, fp):
    model = EncoderProcessDecoder(**params)
    # model = ShapeFittingBaseline(**params)
    
    state_dict_path = os.path.join(fp, 'final_model.pt')
    # state_dict_path = os.path.join(fp, 'record', 'model_40000.pt')


    model.load_state_dict(torch.load(state_dict_path, map_location = 'cpu'), strict = False)
    return model


def get_models(yml_file_path):
    with open(yml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    return list(data.items())  # Return a list of (model_name, config_dict) tuples


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default = 'model1_3d_uniform')
    parser.add_argument('--model_specs', type=str)
    parser.add_argument('--device', type=str, default = 'cpu')

    args = parser.parse_args()

    experiment = args.experiment
    device = args.device


    filepaths = get_models(args.model_specs)


    results = []

   ### Synthetic data
    unif = json.load(open('/data/oren/coreset/data/meb_2d_200_test.json'))
    ellipses = json.load(open('/data/oren/coreset/data/meb_2d_ellipse_200_test.json'))
    fish = json.load(open('/data/oren/coreset/data/upsampled_fish_meb_test.json'))



    unif_errs = {}
    ellipse_errs = {}
    fish_errs = {}


    batches_unif, gt_unif = json_to_batches(unif, 128)
    batches_ellipse, gt_ellipse = json_to_batches(ellipses, 128)
    batches_fish, gt_fish = json_to_batches(fish, 128)

    # for path, params in tqdm(zip(fps, config_list)):
    for (path, params) in tqdm(filepaths):

        model_fp = os.path.join('EncoderProcessDecoder/mse', path)
        # model_fp = os.path.join('EncoderProcessDecoder/mse', path)
        data_fp = 'min_enclosing_ball'
        fp = os.path.join('/data/oren/coreset/models', data_fp, model_fp, experiment)


        model = define_model(params = params, fp = fp)


        result_unif = evaluate(batches_unif, gt_unif, model, device = device)
        result_ellipse = evaluate(batches_ellipse, gt_ellipse, model, device = device)
        result_fish = evaluate(batches_fish, gt_fish, model, device = device)

        unif_errs[path] = result_unif
        ellipse_errs[path] = result_ellipse
        fish_errs[path] = result_fish



    data = {
        'model': list(unif_errs.keys()),
        'experiment': experiment,
        'circle center err': [val[0] for val in unif_errs.values()],
        'circle center std': [val[1] for val in unif_errs.values()],
        'circle radius err': [val[2] for val in unif_errs.values()],
        'circle radius std': [val[3] for val in unif_errs.values()],
        'circle prop outside err': [val[4] for val in unif_errs.values()],
        'circle prop outsde std': [val[5] for val in unif_errs.values()],


        'ellipse center err': [val[0] for val in ellipse_errs.values()],
        'ellipse center std': [val[1] for val in ellipse_errs.values()],
        'ellipse radius err': [val[2] for val in ellipse_errs.values()],
        'ellipse radius std': [val[3] for val in ellipse_errs.values()],
        'ellipse prop outside err': [val[4] for val in ellipse_errs.values()],
        'ellipse prop outsde std': [val[5] for val in ellipse_errs.values()],
        
        'fish center err': [val[0] for val in fish_errs.values()],
        'fish center std': [val[1] for val in fish_errs.values()],
        'fish radius err': [val[2] for val in fish_errs.values()],
        'fish radius std': [val[3] for val in fish_errs.values()],
        'fish prop outside err': [val[4] for val in fish_errs.values()],
        'fish prop outsde std': [val[5] for val in fish_errs.values()]
        }


    df = pd.DataFrame(data)
    df.to_csv(f'/data/oren/coreset/out/{experiment}_meb_results_chkpt.csv', index = False)



if __name__ == "__main__":
    main()