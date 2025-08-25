import torch
from torch import nn, Tensor
import os
import yaml
import json
from tqdm import tqdm

import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from matplotlib.path import Path
from scipy.spatial import ConvexHull
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import wasserstein_distance
from geotorch.sphere import uniform_init_sphere_ as unif_sphere
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.sumformer import *
from src.transformer import *

def get_models(yml_file_path):
    with open(yml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    return list(data.items())

def get_params(model_specs, model_idx, task = 'meb', model_type = 'epd', experiment = None):
    filepaths = get_models(model_specs) ##This function will read in all models in this file
    
    path, params = filepaths[model_idx]

    if model_type == 'epd':
        model_fp = os.path.join('EncoderProcessDecoder/mse', path)
    elif model_type == 'baseline':
        model_fp = os.path.join('ShapeFittingBaseline/mse', path)
    else:
        model_fp = os.path.join('ShapeFittingDirect/mse', path)

    if task == 'mee':
        data_fp = 'min_enclosing_ellipse'
    elif task == 'mea':
        data_fp = 'min_enclosing_annulus'
    else:
        data_fp = 'min_enclosing_ball'

    fp = os.path.join('/data/oren/coreset/models', data_fp, model_fp, experiment)


    return params, fp


def define_model_epd(params, fp, inter = False, epoch = None):
    model = EncoderProcessDecoder(**params)

    if not inter:
        try:
            state_dict_path = os.path.join(fp, 'final_model.pt')
        except Exception as e:
            print(e + '\n' + '*' * 40)
            print('Failed to load final state dict. Perhaps you wanted an intermediate file (pass inter = True and epoch = epoch number)')
    else:
        state_dict_path = os.path.join(fp, 'record', f'model_{epoch}.pt')

    model.load_state_dict(torch.load(state_dict_path, map_location = 'cpu'), strict = False)
    return model

def define_model_baseline(params, fp, inter = False, epoch = None):
    model = ShapeFittingBaseline(**params)

    if not inter:
        try:
            state_dict_path = os.path.join(fp, 'final_model.pt')
        except Exception as e:
            print(e + '\n' + '*' * 40)
            print('Failed to load final state dict. Perhaps you wanted an intermediate file (pass inter = True and epoch = epoch number)')
    else:
        state_dict_path = os.path.join(fp, 'record', f'model_{epoch}.pt')

    model.load_state_dict(torch.load(state_dict_path, map_location = 'cpu'), strict = False)
    return model

def define_model_direct_extent(params, fp):
    model = ShapeFittingDirect(**params)
    
    state_dict_path = os.path.join(fp, 'final_model.pt')
    
    model.load_state_dict(torch.load(state_dict_path), strict = False)
    return model


def make_preds(batches, gt, model, device = 'cpu'):
    preds = []
    pointsets = []
    model.eval()

    model = model.to(device)
    
    for idx, batch in tqdm(enumerate(batches)):
        batch = batch.to(device)
        out = model(batch)
    
        start = 0
        for i, num in enumerate(batch.n_nodes):
            end = start + num
            ptset = batch.data[start:end]
            
            pred = out[i].cpu()
            # ground_truth = gt[idx][i]

            preds.append(pred)
            pointsets.append(ptset)
            
            start = end

    return pointsets, preds




def npz_to_batches(raw_data, batch_size=128, precision = torch.float):

    batch_list = []
    gt_list = []
    in_batch =[]
    in_batch_gt = []
    count = 0

    for i in range(raw_data.shape[0]):
        ptset = raw_data[i][:, :-1]
       
        cvx_hull_idx = np.where(raw_data[i][:, -1] == 1.0)

        in_batch.append(torch.tensor(ptset, dtype=precision))
        in_batch_gt.append(torch.tensor(ptset[cvx_hull_idx], dtype=precision))
        if (count != 0 and count % (batch_size-1) == 0) or i == raw_data.shape[0] - 1:
            batch = Batch.from_list(in_batch, order = 1)
            batch_list.append(batch)
            gt_list.append(in_batch_gt)
            in_batch = []
            in_batch_gt = []
        count += 1

    return batch_list, gt_list

def json_to_batches_ellipse(raw_data, batch_size=128):
    batch_list = []
    ground_truth_list = []  # List for single-vector ground truth
    in_batch = []
    in_batch_ground_truth = []
    count = 0

    for i in range(len(raw_data)):
        try:
            # Extract point set, center, and radius
            ptset = raw_data[i]['pointset']
            center = raw_data[i]['center']  # List of d values
            maj_radius = raw_data[i]['major_radius']  # Scalar value
            min_radius = raw_data[i]['minor_radius']
            angle = raw_data[i]['angle']
        except:
            continue

        in_batch.append(torch.tensor(ptset, dtype=torch.float))
        
        # Combine center and radius into a single tensor
        ground_truth = torch.tensor(center + [maj_radius, min_radius, angle], dtype=torch.float)  # [c1, c2, ..., cd, r]
        in_batch_ground_truth.append(ground_truth)
        
        # Batch management
        if (count != 0 and count % (batch_size - 1) == 0) or i == len(raw_data) - 1:
            batch = Batch.from_list(in_batch, order=1)
            batch_list.append(batch)
            ground_truth_list.append(torch.stack(in_batch_ground_truth))  # Stack tensors into one batch
            
            # Reset temporary lists
            in_batch = []
            in_batch_ground_truth = []

        count += 1

    return batch_list, ground_truth_list

def json_to_batches_annulus(raw_data, batch_size=128):
    batch_list = []
    ground_truth_list = []  # List for single-vector ground truth
    in_batch = []
    in_batch_ground_truth = []
    count = 0

    for i in range(len(raw_data)):
        # Extract point set, center, and radius
        ptset = raw_data[i]['Pointset']
        center = raw_data[i]['Center']  # List of d values
        radius_1 = raw_data[i]['Inner radius']  # Scalar value
        radius_2 = raw_data[i]['Outer radius']  # Scalar value

        in_batch.append(torch.tensor(ptset, dtype=torch.float))
        
        # Combine center and radius into a single tensor
        ground_truth = torch.tensor([radius_1, radius_2], dtype=torch.float)  # used to be center + [radius_1, radius_2] -- [c1, c2, ..., cd, r]
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
            batch = Batch.from_list(in_batch, order=1)
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
        hull_approx = torch.mm(ptset_probs.T, ptset)

        hulls.append(hull_approx)
        start = end
    return hulls



def sort_points_by_angle(points):
    centroid = np.mean(points, axis=0)
    
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    
    # Sort points by angles
    return points[np.argsort(angles)]


def directional_width(P, u):
    prods = P @ u
    
    return max(prods) - min(prods)

def directional_err(Q, P, in_dim=2, n=1000):
    Q, P = torch.tensor(Q, dtype=torch.float32), torch.tensor(P, dtype=torch.float32) # cast inputs to tensor
    if Q.numel() == 0:
        return None

    directions = unif_sphere(torch.zeros(n, in_dim)) # n random directions on the unit sphere
    
    # Convert directions to a tensor
    directions = torch.tensor(directions, dtype=torch.float32)
    
    # Compute directional projections for both P and Q for all directions
    P_proj = torch.matmul(P, directions.T)  # Shape: [P_size, n]
    Q_proj = torch.matmul(Q, directions.T)  # Shape: [Q_size, n]
    
    # Compute the width for each direction (max - min)
    P_widths = torch.max(P_proj, dim=0).values - torch.min(P_proj, dim=0).values
    Q_widths = torch.max(Q_proj, dim=0).values - torch.min(Q_proj, dim=0).values
    
    # Compute the error for each direction and find the maximum error
    errors = torch.abs(P_widths - Q_widths) / P_widths
    max_error = torch.max(errors)
    
    return max_error.item()

def directional_err_mean(Q, P, in_dim=2, n=1000):
    Q, P = torch.tensor(Q, dtype=torch.float32), torch.tensor(P, dtype=torch.float32) # cast inputs to tensor
    directions = unif_sphere(torch.zeros(n, in_dim)) # n random directions on the unit sphere
    
    # Convert directions to a tensor
    directions = torch.tensor(directions, dtype=torch.float32)
    
    # Compute directional projections for both P and Q for all directions
    P_proj = torch.matmul(P, directions.T)  # Shape: [P_size, n]
    Q_proj = torch.matmul(Q, directions.T)  # Shape: [Q_size, n]
    
    # Compute the width for each direction (max - min)
    P_widths = torch.max(P_proj, dim=0).values - torch.min(P_proj, dim=0).values
    Q_widths = torch.max(Q_proj, dim=0).values - torch.min(Q_proj, dim=0).values
    
    # Compute the error for each direction and find the maximum error
    errors = torch.abs(P_widths - Q_widths) / P_widths
    mean_error = torch.mean(errors)
    
    return mean_error.item()

# def directional_err(Q, P, in_dim=2, n=1000):
#     directions = unif_sphere(torch.zeros(n, in_dim)).numpy()
#     max_width = -float('inf')
    
#     for u in directions:
#         num = abs(directional_width(P, u) - directional_width(Q, u))
#         denom = directional_width(P, u)
        
#         err = num / denom
        
#         if err >= max_width:
#             max_width = err

#     return max_width


def wasserstein_2d(A, B):
    A_x, A_y = A[:, 0], A[:, 1]
    B_x, B_y = B[:, 0], B[:, 1]
    
    
    distance_x = wasserstein_distance(A_x, B_x)
    distance_y = wasserstein_distance(A_y, B_y)

    return np.sqrt(distance_x**2 + distance_y**2)

def wasserstein_nd(A, B):
    # Ensure A and B have the same shape
    assert A.shape[1] == B.shape[1], "Point sets must have the same number of dimensions"
    
    # Number of dimensions
    D = A.shape[1]
    
    # Compute Wasserstein distance for each dimension
    distances = [wasserstein_distance(A[:, d], B[:, d]) for d in range(D)]
    
    # Combine distances using the Euclidean norm
    return np.sqrt(np.sum(np.square(distances)))


def avg_err(chull, ptsets, in_dim=2): #(chull, ptsets, in_dim=2, trial = None, train = None, eval = None)
    errs = []
    for i in range(len(chull)):
        points = ptsets[i]
    
        errs.append(directional_err(chull[i], points, in_dim = in_dim, n=1000))
        
    return np.mean(errs), np.std(errs)

def avg_err_nan(chull, ptsets, in_dim=2): #(chull, ptsets, in_dim=2, trial = None, train = None, eval = None)
    errs = []
    empty = 0
    for i in range(len(chull)):
        points = ptsets[i]
    
        if len(chull[i]) > 0:
            errs.append(directional_err(chull[i], points, in_dim = in_dim, n=1000))
        else:
            empty += 1
        
    return np.mean(errs), np.std(errs), empty

def avg_wasserstein(chull, gt_hulls):
    distances = []
    for i in range(len(chull)):
        distances.append(wasserstein_2d(chull[i], gt_hulls[i]))
    
    return np.mean(distances)

def avg_wasserstein_nd(chull, gt_hulls):
    distances = []
    for i in range(len(chull)):
        distances.append(wasserstein_nd(chull[i], gt_hulls[i]))
    
    return np.mean(distances)

def plot_hull(chull, gt_hulls, raw_data, i):
    
    sorted_chull = sort_points_by_angle(chull[i])
    # sorted_gt = sort_points_by_angle(gt_hulls[i])
   
    pt_set = raw_data[i]
   
    
    sorted_chull = np.vstack([sorted_chull, sorted_chull[0]])



    common_points = []
    for pt in pt_set:
        if np.any(np.sqrt(np.sum(np.square(chull[i] - pt), axis=1)) < 0.01):
            common_points.append(pt)
    common_points = np.array(common_points)

    # Plotting
    plt.figure(figsize=(8, 6))
    
    
    mask = np.array([np.any(np.sqrt(np.sum(np.square(chull[i] - pt), axis=1)) < 0.1) for pt in pt_set])
    # plt.plot(sorted_chull[:, 0], sorted_chull[:, 1], 'r-', linewidth=2, label='Approx. Convex Hull', marker = 'x')
    plt.scatter(pt_set[~mask][:, 0], pt_set[~mask][:, 1], color='orange', label='Data Points')
    

    # Plot common points in yellow if there are any
    if len(common_points) > 0:
        plt.scatter(common_points[:, 0], common_points[:, 1], color='black', label='Common Points')

    
    # plt.plot(sorted_gt[:, 0], sorted_gt[:, 1], linewidth=2, color='green', label='Ground Truth Hull')


    
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Estimated Convex Hull (test set)')
        
    plt.legend(loc='lower right', bbox_to_anchor=(1.3, 0))
    plt.show()

   
    # print(f'The estimated hull contains {prop * 100} percent of the points')
    # distance = directed_hausdorff(chull[i], pt_set[gt_hull.vertices])[0]
    # print(f'Hausdorff distance between the approx convex hull and true chull: {distance}')
    print(f'Directional Width Error is {directional_err(sorted_chull, pt_set, 2, n=1000)}')


def plot_hull_3d(dataset, chull, gt_hull, i):
    pt_set = dataset[i]
    approx_hull_points = chull[i]
    true_hull_points = gt_hull[i]
    
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the data points
    ax.scatter(pt_set[:, 0], pt_set[:, 1], pt_set[:, 2], color='orange', label='Data Points', s=10)
    
    # Plot the approximated convex hull as a wireframe
    if len(approx_hull_points) > 0:
        approx_hull = ConvexHull(approx_hull_points)
        for simplex in approx_hull.simplices:
            ax.plot(approx_hull_points[simplex, 0], approx_hull_points[simplex, 1], approx_hull_points[simplex, 2], 'r-', linewidth=2)
    
    # Plot the true convex hull as a wireframe
    if len(true_hull_points) > 0:
        true_hull = ConvexHull(true_hull_points)
        for simplex in true_hull.simplices:
            ax.plot(true_hull_points[simplex, 0], true_hull_points[simplex, 1], true_hull_points[simplex, 2], 'g-', linewidth=2)
    
    # Highlight common points between the approximate and true hulls
    common_points = []
    for pt in pt_set:
        if np.any(np.sqrt(np.sum(np.square(approx_hull_points - pt), axis=1)) < 0.01):
            common_points.append(pt)
    common_points = np.array(common_points)
    
    if len(common_points) > 0:
        ax.scatter(common_points[:, 0], common_points[:, 1], common_points[:, 2], color='black', label='Common Points', s=50)

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.3)
    
    # Labels and titles
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Estimated Convex Hull vs Ground Truth')
    
    # Show legend
    ax.legend()
    
    plt.show()

    print(f'Directional Width Error is {directional_err(approx_hull_points, pt_set, 3, n=1000)}')


def plot_prob_dist(n, out, input_num, od = 8):

    start = input_num * n
    end = start + n
    
    probs = out.data[start:end].detach().cpu()

    fig, axes = plt.subplots(1, od, figsize=(15, 5), sharey=True)
    
    for i in range(od):
        axes[i].hist(probs[:, i], bins=20, edgecolor='black')
        axes[i].set_title(f'Output Point {i}')
    
    plt.tight_layout()
    plt.show()

def ptwise_prob_dist(outputs, ptsets, ex, n, od = 8):
    probs = outputs[ex].detach().cpu()
    pts = np.round(ptsets[ex] * 100) / 100
    
    # fig, axes = plt.subplots(1, 8, figsize=(15, 5), sharey=True)

   
    n_cols = 2
    n_rows = od // n_cols

    # Create subplots with 4 rows and 2 columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    axes = axes.flatten()
    
    for i in range(od):
        axes[i].bar(range(n), probs[:, i].numpy())
        axes[i].set_xticks(range(n))  # Set the x-tick positions
        axes[i].set_xticklabels(pts, rotation=90)

        axes[i].set_title(f'Output Direction {i}')
    
    plt.tight_layout()
    plt.show()

def probs_heatmap(outputs, ptsets, hulls, ex, n):
    probs = outputs[ex].detach().cpu()
    pts = np.round(ptsets[ex] * 100) / 100
    directions = np.round(hulls[ex] * 100) / 100

    plt.figure(figsize = (8, 10))
    # plt.figure(figsize = (8, 15))
    

    sns.heatmap(probs.numpy(), cmap='Blues', annot=False, 
                cbar=True, linewidths=0.5, linecolor='black',
                yticklabels=pts, xticklabels=directions)
    
    plt.ylabel('Input Points')
    plt.xlabel('Directions')
    plt.title('Probability Heatmap')