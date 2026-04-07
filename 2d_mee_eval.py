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


def count_pts_outside(points, r_min, r_maj, angle, center = None):
    """
    Count number of points outside the estimated minimum enclosing ellipse.

    Assumes ellipse is centered at [0, 0].

    points: [N, 2]
    r_min: scalar (minor axis)
    r_maj: scalar (major axis)
    angle: scalar (radians), angle from x-axis to major axis
    """

    # if center is not None:
    #     points = points - center


    c, s = torch.cos(-1 * angle), torch.sin(-1 * angle)
    R = torch.tensor([[c, -s],
                      [s,  c]], device=points.device)

    # Rotate points to align with ellipse axes
    rotated = points @ R.T  # shape [N, 2]

    # Normalize by axis lengths

    normed = rotated / torch.tensor([r_maj, r_min], device=points.device)

    # Check ellipse constraint
    squared_norms = torch.sum(normed ** 2, dim=1)
    count_outside = torch.sum(squared_norms > 1.0)

    return count_outside

def mee_eval(gt, preds, pts, device = 'cpu'):
    # center = preds[:-1].detach() #pred
    # rad = preds[-1].detach()

    # center = preds[:-3].detach() #pred
    rad_maj = preds[-3].detach()
    rad_min = preds[-2].detach()
    angle = preds[-1].detach()

    gt_center = gt[:-3].to(device)
    gt_rad_maj = gt[-3].detach()
    gt_rad_min = gt[-2].detach()
    gt_angle = gt[-1].detach()

    # center_diff = np.linalg.norm(gt_center - center)
    rad_diff_min = abs(rad_min - gt_rad_min).item() / gt_rad_min
    rad_diff_maj = abs(rad_maj - gt_rad_maj).item() / gt_rad_maj
    angle_diff = abs(angle - gt_angle).item() / gt_angle

    num_outside = count_pts_outside(pts, rad_min, rad_maj, angle, center = gt_center).item() / len(pts)

    return float(0), float(rad_diff_min), float(rad_diff_maj), float(angle_diff), float(num_outside)
    
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
            
            errs.append(mee_eval(ground_truth, preds, ptset, device = device))
            start = end

    df = pd.DataFrame(errs, columns=["center error", "rad min error", "rad maj error", "angle error", "proportion outside"])
    return (df['center error'].mean(), df['center error'].std(), df['rad min error'].mean(), df['rad min error'].std(), 
            df['rad maj error'].mean(), df['rad maj error'].std(), df['angle error'].mean(), df['angle error'].std(), 
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
    
    # state_dict_path = os.path.join(fp, 'final_model.pt')
    state_dict_path = os.path.join(fp, 'record', 'model_17000.pt')
    
    model.load_state_dict(torch.load(state_dict_path), strict = False)
    return model


def get_models(yml_file_path):
    with open(yml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    return list(data.items())  # Return a list of (model_name, config_dict) tuples


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--model_specs', type=str)
    parser.add_argument('--device', type=str, default = 'cpu')
    args = parser.parse_args()

    device = args.device
    experiment = args.experiment


    filepaths = get_models(args.model_specs)


    results = []

   ### Synthetic data
    unif100 = json.load(open('/data/oren/coreset/data/unif100_mee_test.json'))
    ellipses = json.load(open('/data/oren/coreset/data/centered_skinny_unif100_mee_test.json'))
    # fish = json.load(open('/data/oren/coreset/data/centered_upsampled_fish_mee_test.json'))
    fish = json.load(open('/data/oren/coreset/data/normalized_upsampled_fish_mee_test.json'))




    unif_errs = {}
    ellipse_errs = {}
    fish_errs = {}


    batches_unif, gt_unif = json_to_batches_ellipse(unif100, 128)
    batches_ellipse, gt_ellipse = json_to_batches_ellipse(ellipses, 128)
    batches_fish, gt_fish = json_to_batches_ellipse(fish, 128)

    # for path, params in tqdm(zip(fps, config_list)):
    for (path, params) in tqdm(filepaths):

        model_fp = os.path.join('EncoderProcessDecoder/mse', path)
        data_fp = 'min_enclosing_ellipse'
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
        'circle min rad err': [val[2] for val in unif_errs.values()],
        'circle min rad std': [val[3] for val in unif_errs.values()],
        'circle maj rad err': [val[4] for val in unif_errs.values()],
        'circle maj rad std': [val[5] for val in unif_errs.values()],
        'circle angle err': [val[6] for val in unif_errs.values()],
        'circle angle std': [val[7] for val in unif_errs.values()],
        'circle prop outside err': [val[8] for val in unif_errs.values()],
        'circle prop outsde std': [val[9] for val in unif_errs.values()],
        'ellipse center err': [val[0] for val in ellipse_errs.values()],
        'ellipse center std': [val[1] for val in ellipse_errs.values()],
        'ellipse min rad err': [val[2] for val in ellipse_errs.values()],
        'ellipse min rad std': [val[3] for val in ellipse_errs.values()],
        'ellipse maj rad err': [val[4] for val in ellipse_errs.values()],
        'ellipse maj rad std': [val[5] for val in ellipse_errs.values()],
        'ellipse angle err': [val[6] for val in ellipse_errs.values()],
        'ellipse angle std': [val[7] for val in ellipse_errs.values()],
        'ellipse prop outside err': [val[8] for val in ellipse_errs.values()],
        'ellipse prop outsde std': [val[9] for val in ellipse_errs.values()],
        'fish center err': [val[0] for val in fish_errs.values()],
        'fish center std': [val[1] for val in fish_errs.values()],
        'fish min rad err': [val[2] for val in fish_errs.values()],
        'fish min rad std': [val[3] for val in fish_errs.values()],
        'fish maj rad err': [val[4] for val in fish_errs.values()],
        'fish maj rad std': [val[5] for val in fish_errs.values()],
        'fish angle err': [val[6] for val in fish_errs.values()],
        'fish angle std': [val[7] for val in fish_errs.values()],
        'fish prop outside err': [val[8] for val in fish_errs.values()],
        'fish prop outsde std': [val[9] for val in fish_errs.values()],
        }


    df = pd.DataFrame(data)
    df.to_csv(f'/data/oren/coreset/out/{experiment}_mee_results_chkpt_17k.csv', index = False)



if __name__ == "__main__":
    main()