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


def count_pts_outside(points, center, inner_radius, outer_radius):
    """
    Count number of points outside the estimated minimum enclosing ellipse.
    
    points: [N, 2]
    center: [2]
    r_min: scalar (minor axis)
    r_maj: scalar (major axis)
    angle: scalar (radians), angle from x-axis to major axis
    """
    distances_squared = torch.sum((points) ** 2, axis=1)
    inner_squared = inner_radius ** 2
    outer_squared = outer_radius ** 2

    # Points outside annulus; distance squared < inner^2 or > outer^2
    outside_mask = (distances_squared < inner_squared) | (distances_squared > outer_squared)
    count_outside = torch.sum(outside_mask)

    return count_outside

def mea_eval(gt, preds, pts):
    # center = preds[:-1].detach() #pred
    # rad = preds[-1].detach()

    # center = preds[:-2].detach()
    rad_in = preds[0].detach()
    rad_out = preds[1].detach()

    # gt_center = gt[:-2]
    gt_rad_in = gt[0].detach()
    gt_rad_out = gt[1].detach()

    # center_diff = np.linalg.norm(gt_center - center)
    rad_diff_in = abs(gt_rad_in - rad_in).item() / gt_rad_in
    rad_diff_out = abs(gt_rad_out - rad_out).item() / gt_rad_out


    width_pred = rad_out - rad_in
    width_gt = gt_rad_out - gt_rad_in
    width_diff = abs(width_gt - width_pred).item() / width_gt

    num_outside = count_pts_outside(pts, None, rad_in, rad_out).item() / len(pts)

    return float(0), float(rad_diff_in), float(rad_diff_out), float(num_outside), float(width_diff)
    
def evaluate(batches, gt, model, device = 'cpu'):
    errs = []
    
    model = model.to(device)
    for idx, batch in tqdm(enumerate(batches)):
        batch = batch.to(device)
        out = model(batch)
    
        start = 0
        for i, num in enumerate(batch.n_nodes):
            end = start + num
            ptset = batch.data[start:end]
            
            preds = out[i]
            ground_truth = gt[idx][i]
            
            errs.append(mea_eval(ground_truth, preds, ptset))
            start = end

    df = pd.DataFrame(errs, columns=["center error", "inner rad error", "outer rad error", "proportion outside", "width error"])
    return (df['center error'].mean(), df['center error'].std(), df['inner rad error'].mean(), df['inner rad error'].std(), 
            df['outer rad error'].mean(), df['outer rad error'].std(),
            df['proportion outside'].mean(), df['proportion outside'].std(), df['width error'].mean(), df['width error'].std())

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
    
    state_dict_path = os.path.join(fp, 'final_model.pt')
    
    model.load_state_dict(torch.load(state_dict_path), strict = False)
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

    device = args.device
    experiment = args.experiment


    filepaths = get_models(args.model_specs)


    results = []

   ### Synthetic data
    annulus = json.load(open('/data/oren/coreset/data/centered_annular_mea_test.json'))
    fish = json.load(open('/data/oren/coreset/data/normalized_upsampled_fish_mea_test.json'))



    annulus_errs = {}
    ellipse_errs = {}
    fish_errs = {}


    batches_annulus, gt_annulus = json_to_batches_annulus(annulus, 128)
    batches_fish, gt_fish = json_to_batches_annulus(fish, 128)

    # for path, params in tqdm(zip(fps, config_list)):
    for (path, params) in tqdm(filepaths):

        model_fp = os.path.join('EncoderProcessDecoder/mse', path)
        data_fp = 'min_enclosing_annulus'
        fp = os.path.join('/data/oren/coreset/models', data_fp, model_fp, experiment)


        model = define_model(params = params, fp = fp)


        result_annulus = evaluate(batches_annulus, gt_annulus, model, device = device)
        result_fish = evaluate(batches_fish, gt_fish, model, device = device)

        annulus_errs[path] = result_annulus
        fish_errs[path] = result_fish



    data = {
        'model': list(annulus_errs.keys()),
        'experiment': experiment,
        'annulus center err': [val[0] for val in annulus_errs.values()],
        'annulus center std': [val[1] for val in annulus_errs.values()],
        'annulus inner radius err': [val[2] for val in annulus_errs.values()],
        'annulus inner radius std': [val[3] for val in annulus_errs.values()],
        'annulus outer radius err': [val[4] for val in annulus_errs.values()],
        'annulus outer radius std': [val[5] for val in annulus_errs.values()],
        'annulus prop outside err': [val[6] for val in annulus_errs.values()],
        'annulus prop outsde std': [val[7] for val in annulus_errs.values()],
        'annulus width err': [val[8] for val in annulus_errs.values()],
        'annulus width std': [val[9] for val in annulus_errs.values()],
        
        'fish center err': [val[0] for val in fish_errs.values()],
        'fish center std': [val[1] for val in fish_errs.values()],
        'fish inner radius err': [val[2] for val in fish_errs.values()],
        'fish inner radius std': [val[3] for val in fish_errs.values()],
        'fish outer radius err': [val[4] for val in fish_errs.values()],
        'fish outer radius std': [val[5] for val in fish_errs.values()],
        'fish prop outside err': [val[6] for val in fish_errs.values()],
        'fish prop outsde std': [val[7] for val in fish_errs.values()],
        'fish width err': [val[8] for val in fish_errs.values()],
        'fish width std': [val[9] for val in fish_errs.values()]
        }


    df = pd.DataFrame(data)
    df.to_csv(f'/data/oren/coreset/out/{experiment}_mea_results_chkpt.csv', index = False)
    print(f'Saved results to /data/oren/coreset/out/{experiment}_mea_results_chkpt.csv')



if __name__ == "__main__":
    main()