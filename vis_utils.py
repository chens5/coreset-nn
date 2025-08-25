import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import math
import torch


def plot_meb_2d(pts, predictions, save_plot = None):
    """
    Plots a 2D minimum enclosing ball.

    Args:
        pts (torch.Tensor or np.ndarray): A tensor or array of 2D points.
        predictions (torch.Tensor or np.ndarray): A tensor or array containing the center and radius of the MEB.
    """
    if isinstance(predictions, torch.Tensor):
        ball = predictions.detach().numpy()
    else:
        ball = predictions

    if isinstance(pts, torch.Tensor):
        pts = pts.detach().numpy()

    center, radius = ball[:-1], ball[-1]

    # Create a 2D plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the points
    ax.scatter(pts[:, 0], pts[:, 1], label="Points", color="blue", s=30)

    # Plot the center of the MEB
    ax.scatter(center[0], center[1], color="red", label="MEB Center", s=50)

    # Draw the MEB circle
    circle = plt.Circle((center[0], center[1]), radius, color="green", alpha=0.2, label="MEB")
    ax.add_artist(circle)

    # Ensure the circle is fully visible
    margin = 0.1 * radius
    ax.set_xlim([center[0] - radius - margin, center[0] + radius + margin])
    ax.set_ylim([center[1] - radius - margin, center[1] + radius + margin])

    # Ensure equal axis scaling
    ax.set_aspect('equal', adjustable='box')

    # Add labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Minimum Enclosing Ball in 2D")
    ax.legend()
    ax.grid(True)

    if save_plot:
        plt.savefig(f"{save_plot}")

    plt.show()

def plot_meb_3d(pts, predictions, save_plot = None):
    ball = predictions.detach()

    center, radius = ball[:-1], ball[-1]

    # Create a 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], label="Points", color="blue", s=30)

    # Plot the center of the MEB
    ax.scatter(center[0], center[1], center[2], color="red", label="MEB Center", s=50)

    # Draw the MEB sphere
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = radius * np.cos(u) * np.sin(v) + center[0]
    y = radius * np.sin(u) * np.sin(v) + center[1]
    z = radius * np.cos(v) + center[2]
    ax.plot_surface(x, y, z, color="green", alpha=0.2)

    # Ensure center is in the middle and sphere is fully visible
    margin = 0.1 * radius  # Add a margin for better visibility
    ax.set_xlim([center[0] - radius - margin, center[0] + radius + margin])
    ax.set_ylim([center[1] - radius - margin, center[1] + radius + margin])
    ax.set_zlim([center[2] - radius - margin, center[2] + radius + margin])

    # Ensure equal axis scaling
    ax.set_box_aspect([1, 1, 1])  

    # Add labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Minimum Enclosing Ball in 3D")
    # ax.legend()

    if save_plot:
        plt.savefig(f"{save_plot}")
        
    plt.show()
    


def plot_min_enclosing_ellipse(points, preds, degrees = False, ax=None, save_plot=None):

    if isinstance(preds, torch.Tensor):
        preds_np = preds.detach().cpu().numpy()
    else:
        preds_np = np.asarray(preds)

    if isinstance(points, torch.Tensor):
        points_np = points.detach().cpu().numpy()
    else:
        points_np = np.asarray(points)

    # --- 2. Extract Ellipse Parameters ---
    # Unpack parameters from the prediction array
    rad_maj = preds_np[-3]
    rad_min = preds_np[-2]
    theta = preds_np[-1]

    

    if degrees:
        theta = math.radians(theta)
    

    if ax is None:
        fig, ax = plt.subplots()

    # Plot point cloud
    ax.scatter(points[:, 0], points[:, 1], color='blue', label='Points', alpha=0.6)

    # Plot ellipse
    ellipse_patch = Ellipse(
        xy= [0, 0], # Assume input is centered
        width=2 * rad_maj,
        height=2 * rad_min,
        angle=np.degrees(theta),
        edgecolor='red',
        facecolor='none',
        linewidth=2,
        label='Min Enclosing Ellipse'
    )
    ax.add_patch(ellipse_patch)

    ax.set_aspect('equal')
    ax.set_title("Minimum Enclosing Ellipse")
    # ax.legend(loc='upper right')

    if save_plot:
        plt.savefig(f"{save_plot}")
    
    plt.show()


def plot_annulus(points, preds, save_plot=None):
    """
    Plots the given 2D points and the minimum enclosing annulus.

    Args:
        points (list of tuples): List of (x, y) points.
        center (list or tuple): Coordinates of the annulus center [x, y].
        inner_radius (float): Radius of the inner circle.
        outer_radius (float): Radius of the outer circle.
    """
    points = np.array(points)
    cx, cy = 0, 0

    inner_radius, outer_radius = preds[0], preds[1]

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], color='blue', label='Points')

    # Plot the center
    ax.plot(cx, cy, 'ro', label='Center')

    # Draw outer circle
    outer_circle = plt.Circle((cx, cy), outer_radius, color='green', fill=False, linestyle='--', label='Outer radius')
    ax.add_artist(outer_circle)

    # Draw inner circle
    inner_circle = plt.Circle((cx, cy), inner_radius, color='red', fill=False, linestyle=':', label='Inner radius')
    ax.add_artist(inner_circle)

    # Set limits
    padding = outer_radius + 1
    ax.set_xlim(cx - padding, cx + padding)
    ax.set_ylim(cy - padding, cy + padding)

    # ax.legend()
    ax.set_title("Minimum Enclosing Annulus")
    plt.grid(True)

    if save_plot:
        plt.savefig(f"{save_plot}")
        
    plt.show()