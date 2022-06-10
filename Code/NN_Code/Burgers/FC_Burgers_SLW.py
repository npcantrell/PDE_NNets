# Solving Burgers Equation Using a Fully Connected NN
# The Initial Condition is a Slanted Wave IC
# Authors: Ryan Kramlich & Nicholas Cantrell 
# ----------------------------------------------------------------------



# ----------------------------------------------------------------------
# Imports:
# ----------------------------------------------------------------------
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go

import torch 
import torchvision 
from torch import nn
import tqdm
import torch.nn.functional as F 
from matplotlib import pyplot as plt
from math import floor, ceil

import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
import glob
from PIL import Image
from torchvision import transforms
from sklearn import preprocessing
import time

# Setting GPU if available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Loading Data: 
# ----------------------------------------------------------------------

# Turn .mat file into dictionary
def add_todict(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = add_todict(elem)
        else:
            dict[strg] = elem
    return dict

def convert_to_dictonary(dict):
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = add_todict(dict[key])
    return dict

data = sio.loadmat('../../Data/Generated_Data/Burgers/FC_slantedwave_data.mat', struct_as_record=False, squeeze_me=True)
data_dict = convert_to_dictonary(data)

u = data_dict['u']
x = data_dict['x']
t = data_dict['t']

print("u shape:", np.shape(u))
print("x shape:", np.shape(x))
print("t shape:", np.shape(t))

X, T = np.meshgrid(x, t)
N = len(x)
nu = 0.1


# Plotting inital conditions: 
plt.plot(x, u[:, 0], label="u(x, 0)")
plt.title("Inital Condition")
plt.legend()
plt.show()


# ----------------------------------------------------------------------
# Building Network Input Data for BC, IC and Gradient Behavior 
# ----------------------------------------------------------------------

# Initial Condition
def IC(Xinp):
  Xinp = Xinp.reshape(len(Xinp), )
  u0 = np.zeros(Xinp.shape)

  for i in range(0, len(Xinp)):
    if Xinp[i] < 0:
      u0[i] = 1
    elif Xinp[i] > 1:
      u0[i] = 0
    else:
      u0[i] = 1 - Xinp[i]

  return u0 

def get_IC(size): 
  x = np.linspace(-2, 6, size)

  u0 = IC(x)
  u0 = np.expand_dims(u0, axis=1)
  X_ic = np.column_stack((x, np.zeros(x.shape)))

  X_ic = torch.from_numpy(X_ic).type(torch.Tensor)
  u0 = torch.from_numpy(u0).type(torch.Tensor)
  
  return X_ic.to(device), u0.to(device)

# Boundary Conditions 
def get_BC(size=201):
  x = np.linspace(-2, 6, size)
  t = np.linspace(0, 1, size)

  X_left = np.column_stack((np.zeros(x.shape), t))
  X_right = np.column_stack((np.zeros(x.shape), t))

  X_left = torch.from_numpy(X_left).type(torch.Tensor)
  X_right = torch.from_numpy(X_right).type(torch.Tensor)

  return X_left.to(device), X_right.to(device)

# Inner Grid
def get_inner_grid(size=201):
  x = np.linspace(-2, 6, size)
  t = np.linspace(0, 1, size)

  X_train = np.column_stack((x, t))
  X_train = torch.from_numpy(X_train).type(torch.Tensor).to(device)

  return X_train


# ----------------------------------------------------------------------
# Neural Network
# ----------------------------------------------------------------------
class Network(nn.Module):
  def __init__(self, input_dim):
    super().__init__()

    self.layer1 = nn.Linear(input_dim, 1024)
    nn.init.xavier_normal_(self.layer1.weight.data, gain=1.0)
    nn.init.zeros_(self.layer1.bias.data)

    self.layer2 = nn.Linear(1024, 2048)
    nn.init.xavier_normal_(self.layer2.weight.data, gain=1.0)
    nn.init.zeros_(self.layer2.bias.data)

    self.layer3 = nn.Linear(2048, 2048)
    nn.init.xavier_normal_(self.layer3.weight.data, gain=1.0)
    nn.init.zeros_(self.layer3.bias.data)

    self.layer4 = nn.Linear(2048, 1024)
    nn.init.xavier_normal_(self.layer4.weight.data, gain=1.0)
    nn.init.zeros_(self.layer4.bias.data)

    self.layer5 = nn.Linear(1024, 1024)
    nn.init.xavier_normal_(self.layer5.weight.data, gain=1.0)
    nn.init.zeros_(self.layer5.bias.data)

    self.layer6 = nn.Linear(1024, 512)
    nn.init.xavier_normal_(self.layer6.weight.data, gain=1.0)
    nn.init.zeros_(self.layer6.bias.data)

    self.layer7 = nn.Linear(512, 256)
    nn.init.xavier_normal_(self.layer7.weight.data, gain=1.0)
    nn.init.zeros_(self.layer7.bias.data)

    self.layer8 = nn.Linear(256, 1)

  def forward(self, x):
    out = torch.tanh(self.layer1(x))
    out = torch.tanh(self.layer2(out))
    out = torch.tanh(self.layer3(out))
    out = torch.tanh(self.layer4(out))
    out = torch.tanh(self.layer5(out))
    out = torch.tanh(self.layer6(out))
    out = torch.tanh(self.layer7(out))

    out = self.layer8(out)

    return out


# ----------------------------------------------------------------------
# Training and Evaluation of Network
# ----------------------------------------------------------------------
def run_network(size):
  # Creating Model
  model = Network(2).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.00135)
  num_epochs = 1200
  loss_fn = torch.nn.MSELoss()

  total_losses = []
  ic_losses = []
  bc_losses = [] 
  inner_losses = []
  val_losses = []

  n = size # grid size for x and t 
  lambd = 0.001
  print("Training Results for Grid Sizes of:", size)
  print("------------------------------------------\n")
  for epoch in range(num_epochs):

    # Training Set
    # -----------------
    # Gradients 
    X_train = get_inner_grid(n)
    X_train.requires_grad_(True)
    u_pred = model(X_train).to(device)

    u_t, = torch.autograd.grad(
              u_pred, X_train, 
              grad_outputs=torch.ones_like(u_pred),
              retain_graph=True,
              create_graph=True)

    u_t = u_t[:, 1]
          
    u_x, = torch.autograd.grad(
        u_pred, X_train, 
        grad_outputs=torch.ones_like(u_pred),
        retain_graph=True,
        create_graph=True)

    u_x = u_x[:, 0]
    
    u_xx, = torch.autograd.grad(
        u_x, X_train, 
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True)
    
    u_xx = u_xx[:, 0]

    u_pred = u_pred.reshape(n,)
    inner_loss = loss_fn(u_t, -(u_pred * u_x) + nu*u_xx)

    # Initial Condition 
    X_ic, u0 = get_IC(n)
    u_pred = model(X_ic).to(device)

    ic_loss = loss_fn(u_pred, u0)

    # Boundary Conditions 
    X_left, X_right = get_BC(n)
    u_left = model(X_left).to(device)
    u_right = model(X_right).to(device)

    bc_loss = loss_fn(u_left, u_right)

    # Updating
    optimizer.zero_grad()
    loss = inner_loss + ic_loss + bc_loss
    loss.backward()
    optimizer.step()

    total_losses.append(loss.item())
    ic_losses.append(ic_loss.item())
    bc_losses.append(bc_loss.item())
    inner_losses.append(inner_loss.item())


    # Validation Set
    # ---------------------
    n_val = int(0.2*size)
    X_val = get_inner_grid(n_val)
    X_val.requires_grad_(True)
    u_pred_val = model(X_val).to(device)

    v_t, = torch.autograd.grad(
              u_pred_val, X_val, 
              grad_outputs=torch.ones_like(u_pred_val),
              retain_graph=True,
              create_graph=True)

    v_t = v_t[:, 1]
          
    v_x, = torch.autograd.grad(
        u_pred_val, X_val, 
        grad_outputs=torch.ones_like(u_pred_val),
        retain_graph=True,
        create_graph=True)

    v_x = v_x[:, 0]
    
    v_xx, = torch.autograd.grad(
        v_x, X_val, 
        grad_outputs=torch.ones_like(v_x),
        retain_graph=True,
        create_graph=True)
    
    v_xx = v_xx[:, 0]

    u_pred_val = u_pred_val.reshape(n_val,)
    inner_loss_val = loss_fn(v_t, -(u_pred_val * v_x) + nu*v_xx)

    # Initial Condition 
    X_ic_v, v0 = get_IC(n_val)
    u_pred_val = model(X_ic_v).to(device)

    ic_loss_val = loss_fn(u_pred_val, v0)

    # Boundary Conditions 
    X_left_val, X_right_val = get_BC(n_val)
    u_left_val = model(X_left_val).to(device)
    u_right_val = model(X_right_val).to(device)

    bc_loss_val = loss_fn(u_left_val, u_right_val)

    # Updating
    optimizer.zero_grad()
    loss_val = inner_loss_val + ic_loss_val + bc_loss_val
    loss_val.backward()
    optimizer.step()

    val_losses.append(loss_val.item())


    if epoch % 100 == 0:
      print("Epoch:", epoch, "    IC Loss:               ", ic_loss.item())
      print("Epoch:", epoch, "    Inner PDE Loss:        ", inner_loss.item())
      print("Epoch:", epoch, "    BC Loss:               ", bc_loss.item())
      print("Epoch:", epoch, "    Total Loss:            ", loss.item())
      print("Epoch:", epoch, "    Total Validation Loss: ", loss_val.item(), "\n")


  print("\nFinal Training Loss:   ", loss.item())
  print("Final Validation Loss: ", loss_val.item(),"\n")


  # Plotting Training Losses
  plt.plot(total_losses, label="Overall Training Loss")
  plt.plot(inner_losses, label="Gradient Training Loss")
  plt.plot(ic_losses, label="IC Training Loss")
  plt.plot(bc_losses, label="BC Training Loss")
  plt.plot(val_losses, label="Overall Validation Loss")
  plt.yscale('log')

  plt.title("Training and Validation Losses for Size: " + str(size))
  plt.ylabel("Log of Loss")
  plt.xlabel("Epochs")
  plt.legend()

  plt.show()


  # Evaluating the Model:
  # ---------------------
  def eval_model(N, t):
    x = torch.linspace(-2, 6, N)
    X = torch.stack([x, torch.ones(N)*t], axis=-1).to(device)

    model.eval()
    with torch.no_grad():
      sol = model(X).to(device)

    return x.numpy(), sol.detach().cpu().numpy().ravel()


  U_Pred = []
  tvals = np.linspace(0, 1, 201)
  M = len(tvals)
  for i in tvals:
    x, u_sol = eval_model(N=201, t=i)
    U_Pred.append(u_sol)

  U_Pred = np.array(U_Pred) 
  # Printing Test Error:
  U_Pred = np.array(U_Pred) 
  u = data_dict['u']

  test_loss = loss_fn(torch.from_numpy(u.T).type(torch.Tensor).to(device), 
                      torch.from_numpy(U_Pred).type(torch.Tensor).to(device))

  print("Size: ", size)
  print("Evaluation Error:", test_loss.item())

  return U_Pred, test_loss.item()


# ----------------------------------------------------------------------
# Running Training and Evaluation Functions:
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Size 10000 Training:
    full_grid = 10000
    U_full_grid, errorfull_grid = run_network(full_grid)

    # Size 5000 Training:
    half_grid = 5000
    U_half_grid, errorhalf_grid = run_network(half_grid)

    # Size 2500 Training:
    quarter_grid = 2500
    U_quarter_grid, errorquarter_grid = run_network(quarter_grid)


    # Plotting Plotting Predicted vs. True Solution  
    u = data_dict['u']
    x = data_dict['x']
    t = data_dict['t']

    xspan, tspan = np.meshgrid(x, t)
    u_true = u
    fig = plt.figure(figsize=(14, 12))

    # True solution plot 
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot_surface(xspan, tspan, u_true.T, cmap='plasma')
    ax.set_xlabel('x',fontsize=12)
    ax.set_ylabel('t',fontsize=12)
    ax.set_zlabel('u(x,t)',fontsize=12)
    ax.set_title("True Solution\n\n",fontsize=14)


    # Predicted solution plot
    x_test = np.linspace(-2, 6, 201)
    t_test = np.linspace(0, 1, 201)

    X_test, T_test = np.meshgrid(x_test, t_test)

    # Predicted solution for 8000
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.plot_surface(X_test, T_test, U_full_grid, cmap='plasma')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('t', fontsize=12)
    ax.set_zlabel('u(x,t)', fontsize=12)
    ax.set_zticks(np.linspace(0, 1, 6))
    ax.set_title("Predicted Solution For Grid Size: 10000\nError: " + str(np.round(0.1*errorfull_grid, 4)) + "\n", fontsize=14)


    # Predicted solution for 4000
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.plot_surface(X_test, T_test, U_half_grid, cmap='plasma')
    ax.set_xlabel('x',fontsize=12)
    ax.set_ylabel('t',fontsize=12)
    ax.set_zlabel('u(x,t)',fontsize=12)
    ax.set_zticks(np.linspace(0, 1, 6))
    ax.set_title("Predicted Solution For Grid Size: 5000\nError: " + str(np.round(errorhalf_grid, 4)) + "\n", fontsize=14)


    # Predicted solution for 2000
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.plot_surface(X_test, T_test, U_quarter_grid, cmap='plasma')
    ax.set_xlabel('x',fontsize=12)
    ax.set_ylabel('t',fontsize=12)
    ax.set_zlabel('u(x,t)',fontsize=12)
    ax.set_zticks(np.linspace(0, 1, 6))
    ax.set_title("Predicted Solution For Grid Size: 2500\nError: " + str(np.round(errorquarter_grid, 4)) + "\n", fontsize=14)

    plt.show()
