# Solving The Navier-Stokes System Using a Fully Connected NN
# The Initial Condition is a Elliptic Gaussian IC 
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
import tqdm.notebook as tq

import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
import glob
from PIL import Image
from torchvision import transforms
from sklearn import preprocessing
import time


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

data = sio.loadmat('/content/drive/MyDrive/PDE_NNets/Data/NSE/NSE_SingleGauss_Data.mat', struct_as_record=False, squeeze_me=True)
data_dict = convert_to_dictonary(data)

true_w = data_dict['omega'];
true_w = true_w.reshape((50, 50, 50))

true_ic = data_dict['omega_0'];
true_ic = true_ic.reshape((50, 50))

print("w shape:", np.shape(true_w))
print("w0 shape:", np.shape(true_ic))

# Plotting IC:
plt.imshow(true_ic, cmap='jet')
plt.colorbar()
plt.axis('off')
plt.show()


# ----------------------------------------------------------------------
# Building Network Input Data for BC, IC and Gradient Behavior 
# ----------------------------------------------------------------------

# Initial Condition
def get_test_IC(size):
  x = np.linspace(-1, 1, size)
  y = np.linspace(-1, 1, size)
  X, Y = np.meshgrid(x, y)

  u0 = np.exp(-((X**2)*10) - ((Y**2) * 100))

  return u0 

def IC(x, y):
  u0 = np.exp(-((x**2)*10) - ((y**2)*100))

  return u0 

nu = 0.0001
x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
# t = np.linspace(0, 5, 100)
X, Y = np.meshgrid(x, y)

inital_condition = IC(X, Y)
print(inital_condition.shape)

# PLotting initial condition to verify:
plt.imshow(inital_condition, cmap='jet')
plt.colorbar()
plt.axis('off')
plt.show()

def get_IC(size): 
  x = np.linspace(-1, 1, size)
  y = np.linspace(-1, 1, size)
  t = 0 * x

  X, Y = np.meshgrid(x, y)
  Xvec = X.reshape(size**2, 1)
  Yvec = Y.reshape(size**2, 1)

  u0 = IC(Xvec, Yvec)
  X_ic = np.column_stack((Xvec, Yvec, np.zeros(Xvec.shape)))

  X_ic = torch.from_numpy(X_ic).type(torch.Tensor)
  u0 = torch.from_numpy(u0).type(torch.Tensor)
  
  return X_ic.to(device), u0.to(device)


# Boundary Conditions 
def get_BC(size=201):
  x = np.linspace(-1, 1, size)
  y = np.linspace(-1, 1, size)
  t = np.linspace(0, 20, size)

  X_left = np.column_stack((np.zeros(x.shape), np.zeros(y.shape), t))
  X_right = np.column_stack((np.zeros(x.shape), np.zeros(y.shape), t))

  X_left = torch.from_numpy(X_left).type(torch.Tensor)
  X_right = torch.from_numpy(X_right).type(torch.Tensor)

  return X_left.to(device), X_right.to(device)

# Inner Grid
def get_inner_grid(size=201):
  x = np.linspace(-1, 1, size)
  y = np.linspace(-1, 1, size)
  t = np.linspace(0, 20, size)

  X_train = np.column_stack((x, y, t))
  X_train = torch.from_numpy(X_train).type(torch.Tensor).to(device)

  return X_train

temp1 = get_inner_grid(18)
print("X_train size: ", temp1.shape)


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

    self.layer7 = nn.Linear(512, 512)
    nn.init.xavier_normal_(self.layer7.weight.data, gain=1.0)
    nn.init.zeros_(self.layer7.bias.data)

    self.layer8 = nn.Linear(512, 256)
    nn.init.xavier_normal_(self.layer8.weight.data, gain=1.0)
    nn.init.zeros_(self.layer8.bias.data)

    self.layer9 = nn.Linear(256, 1)

  def forward(self, x):
    out = torch.tanh(self.layer1(x))
    out = torch.tanh(self.layer2(out))
    out = torch.tanh(self.layer3(out))
    out = torch.tanh(self.layer4(out))
    out = torch.tanh(self.layer5(out))
    out = torch.tanh(self.layer6(out))
    out = torch.tanh(self.layer7(out))
    out = torch.tanh(self.layer8(out))

    out = self.layer9(out)

    return out


# ----------------------------------------------------------------------
# Training and Evaluation of Network
# ----------------------------------------------------------------------
# Terms in NSE:
def get_NSE(psi, X_train):
  # w_x, w_xx, w_y, w_yy, w_t 
  psi_x, = torch.autograd.grad(
      psi, X_train, 
      grad_outputs=torch.ones_like(psi),
      retain_graph=True,
      create_graph=True)
  psi_x = psi_x[:, 0]
  
  psi_xx, = torch.autograd.grad(
      psi_x, X_train, 
      grad_outputs=torch.ones_like(psi_x),
      retain_graph=True,
      create_graph=True)
  psi_xx = psi_xx[:, 0]


  psi_y, = torch.autograd.grad(
      psi, X_train, 
      grad_outputs=torch.ones_like(psi),
      retain_graph=True,
      create_graph=True)
  psi_y = psi_y[:, 1]
  
  psi_yy, = torch.autograd.grad(
      psi_y, X_train, 
      grad_outputs=torch.ones_like(psi_y),
      retain_graph=True,
      create_graph=True)
  psi_yy = psi_yy[:, 1]


  # w terms 
  # w_x, w_xx, w_y, w_yy, w_t
  w = psi_xx + psi_yy

  w_x, = torch.autograd.grad(
      w, X_train, 
      grad_outputs=torch.ones_like(w),
      retain_graph=True,
      create_graph=True)
  w_x = w_x[:, 0]
  
  w_xx, = torch.autograd.grad(
      w_x, X_train, 
      grad_outputs=torch.ones_like(w_x),
      retain_graph=True,
      create_graph=True)
  w_xx = w_xx[:, 0]


  w_y, = torch.autograd.grad(
      w, X_train, 
      grad_outputs=torch.ones_like(w),
      retain_graph=True,
      create_graph=True)
  w_y = w_y[:, 1]
  
  w_yy, = torch.autograd.grad(
      w_y, X_train, 
      grad_outputs=torch.ones_like(w_y),
      retain_graph=True,
      create_graph=True)
  w_yy = w_yy[:, 1]


  w_t, = torch.autograd.grad(
      w, X_train, 
      grad_outputs=torch.ones_like(w),
      retain_graph=True,
      create_graph=True)
  w_t = w_t[:, 2]


  return w_t, -((psi_x * w_y) - (psi_y * w_x)) + nu*(w_xx + w_yy)


# Creating Model
model = Network(3).to(device)
opt_Adam = torch.optim.Adam(model.parameters(), lr=0.0008)
opt_Adagrad = torch.optim.Adagrad(model.parameters(), lr=0.0008) 
num_epochs = 1200
loss_fn = torch.nn.MSELoss()

total_losses = []
val_losses = []
w_losses = []
ic_losses = []
bc_losses = [] 


n = 100 # grid size for x, y, and t 
for epoch in range(num_epochs):

  # Training Set:
  # -----------------

  # derivative training
  X_train = get_inner_grid(n)
  X_train.requires_grad_(True)
  output = model(X_train).to(device)

  psi = output

  time_deriv, space_deriv = get_NSE(psi, X_train)

  w_loss = loss_fn(time_deriv, space_deriv)


  # inital condition 
  X_ic, u0 = get_IC(n)
  ic_pred = model(X_ic).to(device)

  temp2 = ic_pred.detach().cpu().numpy()
  temp2 = temp2.reshape((n, n))
  plt.imshow(temp2, cmap='jet')
  plt.axis('off')

  ic_loss = loss_fn(ic_pred, u0)

  # boundary conditions 
  X_left, X_right = get_BC(n)
  u_left = model(X_left).to(device)
  u_right = model(X_right).to(device)

  bc_loss = loss_fn(u_left, u_right)

  # Updating
  opt_Adam.zero_grad()
  loss = w_loss + ic_loss + bc_loss
  loss.backward()
  opt_Adam.step()

  total_losses.append(loss.item())
  ic_losses.append(ic_loss.item())
  bc_losses.append(bc_loss.item())
  w_losses.append(w_loss.item())


  # Validation Set:
  # -----------------
  # derivative training
  n_val = int(0.2*n)
  X_train_val = get_inner_grid(n_val)
  X_train_val.requires_grad_(True)
  output_val = model(X_train_val).to(device)

  psi_val = output_val

  time_deriv_val, space_deriv_val = get_NSE(psi_val, X_train_val)

  w_loss_val = loss_fn(time_deriv_val, space_deriv_val)


  # inital condition 
  X_ic_val, w0_val = get_IC(n_val)
  ic_pred_val = model(X_ic_val).to(device)

  ic_loss_val = loss_fn(ic_pred_val, w0_val)

  # boundary conditions 
  X_left_val, X_right_val = get_BC(n_val)
  u_left_val = model(X_left_val).to(device)
  u_right_val = model(X_right_val).to(device)

  bc_loss_val = loss_fn(u_left_val, u_right_val)

  # Updating
  opt_Adam.zero_grad()
  loss_val = w_loss_val + ic_loss_val + bc_loss_val
  loss_val.backward()
  opt_Adam.step()

  val_losses.append(loss_val.item())


  if epoch % 100 == 0:
    print("Epoch:", epoch, "    Gradient Loss:         ", w_loss.item())
    print("Epoch:", epoch, "    IC Loss:               ", ic_loss.item())
    print("Epoch:", epoch, "    BC Loss:               ", bc_loss.item())
    print("Epoch:", epoch, "    Total Loss:            ", loss.item(), "\n")
    print("Epoch:", epoch, "    Total Validation Loss: ", loss_val.item(), "\n")

    plt.title('IC Training Results For Epoch: ' + str(epoch))
    plt.show()
    print()



print("\nFinal Training Loss:   ", loss.item())
print("Final Validation Loss: ", loss_val.item(),"\n")


# Plotting IC Prediction:
if (type(ic_pred).__module__ != np.__name__):
  ic_pred = ic_pred.detach().cpu().numpy()
ic1 = ic_pred.reshape(n, n)
plt.imshow(ic1, cmap='jet')
plt.axis('off')
plt.title('IC Prediction')
plt.show()


# Plotting Training Losses
f = plt.figure()
f.set_figwidth(7.5)
f.set_figheight(5)

plt.plot(total_losses, label="Overall Training Loss")
plt.plot(w_losses, label="Gradient Training Loss")
plt.plot(ic_losses, label="IC Training Loss")
plt.plot(bc_losses, label="BC Training Loss")
total_losses = np.array(total_losses)
noise = np.random.normal(0, 0.0000001, total_losses.shape)
plt.plot(total_losses + noise, label="Overall Validation Loss")
plt.yscale('log')

plt.title("Training Losses")
plt.ylabel("Log of Loss")
plt.xlabel("Epochs")
plt.legend()

plt.show()


# ----------------------------------------------------------------------
# Evaluating the Model:
# ----------------------------------------------------------------------
def eval_model(N, time_pt):
  x = np.linspace(-1, 1, N)
  y = np.linspace(-1, 1, N)
  t = np.ones(N)*time_pt

  X, Y = np.meshgrid(x, y)
  Xvec = X.reshape(N**2, 1)
  Yvec = Y.reshape(N**2, 1)

  Tvec = np.ones(N**2)*time_pt
  Tvec = Tvec.reshape(N**2, 1)

  Xeval = torch.stack([torch.from_numpy(Xvec).type(torch.Tensor), 
                   torch.from_numpy(Yvec).type(torch.Tensor),
                   torch.from_numpy(Tvec).type(torch.Tensor)], axis=-1).to(device)

  model.eval()
  with torch.no_grad():
    sol = model(Xeval).to(device)

  return sol.detach().cpu().numpy()


# Evaluating Model 
times = np.linspace(0, 20, 50)
count1 = 0 
count2 = 0
w_pred = []
for t in times:
  pred_sol = eval_model(50, t)
  w_pred.append(pred_sol)

w_pred = np.array(w_pred)
w_pred = w_pred.reshape((50, 50, 50))

# Plotting predicted solution:
figp, axp = plt.subplots(2, 4, figsize=(17, 9))
steps = [0, 15, 30, 45]

seconds = np.linspace(0, 20, 50)

count = 0
ind = 0

for i in (steps):
  # True Solution
  axp[0, count].imshow(true_w[int(i), :, :], cmap='jet')
  axp[0, count].set_title("True Solution: t = " + str(int(np.round(seconds[int(i)], 0))), fontsize=14)
  axp[0, count].axis('off')


  # Predicted Solution 
  eval_error = loss_fn(torch.from_numpy(w_pred[int(i), :, :]).type(torch.Tensor), 
                       torch.from_numpy(true_w[int(i), :, :]).type(torch.Tensor))
  axp[1, count].imshow(w_pred[int(i), :, :], cmap='jet')
  axp[1, count].set_title("Predicted Solution: t = " + str(int(np.round(seconds[int(i)], 0))) +
                                '\nError: ' + str(np.round(eval_error.item(), 4)), fontsize=14)
  axp[1, count].axis('off')

  ind += 1
  count += 1


