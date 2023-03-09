import numpy as np
import torch
import cv2
import torch.nn as nn

def cv2Hough(img, rho, theta, threshold, minLineLength, maxLineGap):
    # img: input image
    # rho: distance resolution of the accumulator in pixels
    # theta: angle resolution of the accumulator in radians
    # threshold: accumulator threshold parameter. Only those lines are returned that get enough votes
    # minLineLength: minimum line length. Line segments shorter than that are rejected
    # maxLineGap: maximum allowed gap between points on the same line to link them

    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply canny edge detection
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    # apply hough transform
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength, maxLineGap)

    # convert to tensor
    lines = torch.from_numpy(lines)

    return lines


def get_r_theta_target_array(m_target, b_target, theta_grid, r_grid):
    theta_target = torch.atan(-1 / (m_target + 1e-6)) 
    r_target = b_target / torch.sin(theta_target)

    # get relevant idx
    theta_target_idx = torch.argmin(torch.abs(theta_grid - theta_target), dim=0)
    r_target_idx = torch.argmin(torch.abs(r_grid - r_target), dim=0)

    # get target array
    r_theta_target_array = torch.zeros(grid_size, grid_size)
    r_theta_target_array[theta_target_idx, r_target_idx] = 1

    return r_theta_target_array

def calc_kernels(r_theta, r_grid):
    # kernel method for calculating r-theta density
    num_corners, grid_size = r_theta.shape
    r_theta_ = r_theta.unsqueeze(2).repeat(1, 1, grid_size)
    r_grid_ = r_grid.unsqueeze(0).unsqueeze(0).repeat(num_corners, grid_size, 1)
    r_theta_diff_kernel = (r_theta_ - r_grid_) ** 2

    # test kernel method
    # assert r_theta_diff_kernel[0,0,0] == (r_theta[0,0] - r_grid[0]) ** 2
    # assert r_theta_diff_kernel[1,0,0] == (r_theta[1,0] - r_grid[0]) ** 2
    # assert r_theta_diff_kernel[1,1,0] == (r_theta[1,1] - r_grid[0]) ** 2
    # assert r_theta_diff_kernel[1,1,1] == (r_theta[1,1] - r_grid[1]) ** 2
    # assert r_theta_diff_kernel[0,1,2] == (r_theta[0,1] - r_grid[2]) ** 2

    return r_theta_diff_kernel

def predict_lines(r_theta_density, line_perdiction_threshold, theta_grid, r_grid):
    # make predictions
    theta_idx, r_idx = torch.where(r_theta_density > line_perdiction_threshold) # this is the sampling operation, can use a random number
    theta_lines, r_lines = theta_grid[theta_idx].squeeze(-1), r_grid[r_idx]

    # line parameterization
    m = -torch.cos(theta_lines) / torch.sin(theta_lines)
    b = r_lines / torch.sin(theta_lines)

    return m, b


class PredictionHead(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, hidden_size=8, single_layer=False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.hidden_size = hidden_size
        self.single_layer = single_layer

        if self.single_layer:
            self.nn = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, bias=bias),
                nn.Sigmoid(),
            )
            self.pre_norm = None
            self.pre_norm_skip = None
        else:
            self.pre_norm = nn.BatchNorm2d(in_channels)
            self.pre_norm_skip = nn.BatchNorm2d(in_channels)
            self.nn = nn.Sequential(
                nn.Conv2d(in_channels, hidden_size, bias=bias),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, out_channels, bias=bias),
                nn.Sigmoid(),
            )
            self.post_norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        if self.single_layer:
            x = self.nn(x)
            return x
        else:
            x = self.post_norm(self.nn(self.pre_norm(x)) + self.pre_norm_skip(x))
            return x

if __name__ == "__main__":

    grid_size_ = 64
    xy_threshold = 0.6 # learned parameter
    intersection_threshold = 0.04 # learned parameter
    temperature = 0.1 # learned parameter
    line_perdiction_threshold = 0.5 # fixed hyperparameter
    head = PredictionHead(1, 1, single_layer=True)

    # dummy activations
    xy_activations = torch.zeros(grid_size_, grid_size_) # activations after sigmoid, between 0,1
    xy_activations[2, 2] = 1
    xy_activations[4, 4] = 1
    xy_activations[5, 5] = 1
    xy_activations[21, 21] = 1
    print(xy_activations)

    # make grid
    grid_size = xy_activations.shape[0]
    x_grid = torch.arange(0, grid_size) / grid_size
    y_grid = torch.arange(0, grid_size) / grid_size
    step_size = 2 * np.pi / grid_size
    theta_grid = torch.arange(0, 2 * np.pi, step_size).unsqueeze(1)
    theta_grid += step_size / 2
    r_grid = torch.arange(0, grid_size) / grid_size

    # quantize activations to 0 or 1, for strong activations
    # xy_activations -= xy_threshold
    # hard_xy_activations = (xy_activations - (xy_activations > 0).int()).detach() + xy_activations
    # hard_xy_activations += xy_threshold

    # select\sample indices of strong activations
    x_corners_idx, y_corners_idx = torch.where(xy_activations > xy_threshold) # this is the sampling operation, can use a random number
    
    # get relevant x,y, coordinations and activationsactivations
    x_corners, y_corners = x_grid[x_corners_idx], y_grid[x_corners_idx]
    selected_activations = xy_activations[x_corners_idx, y_corners_idx]

    # Hough transform
    r_theta = x_corners * torch.cos(theta_grid) + y_corners * torch.sin(theta_grid)

    # multiply by activations to propogate gradients
    r_theta = (selected_activations * r_theta).permute(1, 0)

    # calculate density
    r_theta_diff_kernel = calc_kernels(r_theta, r_grid)
    weights = selected_activations.unsqueeze(-1).unsqueeze(-1)
    weighted_kernels = weights * torch.exp(-r_theta_diff_kernel / temperature)
    kernel_density = torch.sum(weighted_kernels, dim=0)

    # r_theta_activations = torch.sigmoid((intersection_threshold - r_theta_sum_kernel) / temperature)
    r_theta_activations = head(kernel_density.unsqueeze(0).unsqueeze(0))

    # predict - make this an internal class
    m, b = predict_lines(r_theta_activations, line_perdiction_threshold, theta_grid, r_grid)

    a = 1