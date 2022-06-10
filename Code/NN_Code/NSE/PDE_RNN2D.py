import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch.optim as optim
import tqdm.notebook as tq
from sklearn.preprocessing import StandardScaler
import h5py

# For work in Colab
#from google.colab import files
#from google.colab import drive
#drive.mount('/content/drive')

# Read in Data

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

def f_tform(u_0, modes, u_0_axes):
  #fft_test = np.fft.fft(u_0[1,0,:])

  # Take Fourier of X component
  u_0f = np.real(np.fft.fftn(u_0, axes=u_0_axes))
  #uf = np.real(np.fft.fftn(u, axis=u_axes))

  # FFT Shift X component
  u_0f = np.fft.fftshift(u_0f, axes=u_0_axes)

  # Truncate number of modes
  start_modes = int(np.floor(u_0.shape[2]/2) - modes/2)
  end_modes = int(np.floor(u_0.shape[2]/2) + modes/2)

  u_0f = u_0f[:,start_modes:end_modes,start_modes:end_modes,:];

  u_0f = np.reshape(u_0f, (u_0f.shape[0],u_0f.shape[1]**2, u_0f.shape[3]))

  return u_0f

import torch.nn.functional as F

def train_val_test(u_0, u):
  sz = u_0.shape[0]
  return u_0[0:int(sz*.75),:,:], u[:,0:int(sz*.75),:,:], u_0[int(sz*.75):int(sz) ,:,:], u[:,int(sz*.75):int(sz),:,:], u_0[int(sz*0):0 ,:,:], u[:,int(sz*.85):sz ,:,:]

class PDE_RNN(nn.Module)  :
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, spectral, biDir, dropout=0.0):
        super(PDE_RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.spectral = spectral
        self.num_layers = num_layers
        self.output_dim = output_dim
        if biDir:
          self.D = 2
        else:
          self.D = 1

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=biDir, batch_first = True)

        self.fc = nn.Linear(hidden_dim*self.D, output_dim)

    def forward(self, x, time_size):
        h_0 = torch.zeros(self.D*self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(cuda0)

        outputs = torch.empty((time_size, x.size(0), 1, self.output_dim)).to(cuda0)

        for t in range(time_size):
          #print('t = ' + str(t) + ', H_size: ' + str(h_0.shape))
          if t==0:
            out, h_0 = self.gru(x, h_0)
          else:
            if self.spectral:
              x = outf
            else:
              x = out
            out, h_0 = self.gru(x, h_0)


          out = self.fc(out)

          if self.spectral:

            out_sq = torch.reshape(out, (out.shape[0], 1, 50, 50))

            # Take Fourier of X and Y component
            outf = torch.real(torch.fft.fftn(out_sq, dim=[2, 3]))

            # FFT Shift X and Y component
            outf = torch.fft.fftshift(outf, dim=[2, 3])

            # Truncate number of modes
            start_modes = int(np.floor(out_sq.shape[3]/2) - 32/2)
            end_modes = int(np.floor(out_sq.shape[3]/2) + 32/2)
            outf = outf[:,:,start_modes:end_modes, start_modes:end_modes].float()
            outf = torch.reshape(outf, (outf.shape[0], 1, 32**2))

          out = out[:,:,:]
          outputs[t, :, :, :] = out;

        return outputs

f = h5py.File('.../.../RandomGaussNS_2000_50ts.mat')
list(f.keys())

omega = np.array(f['omega'])#.permute(0,3,1,2)
omega_0 = np.array(f['omega_0'])

data = sio.loadmat('.../.../SingleGaussNS_20s.mat', struct_as_record=True, squeeze_me=False)

data_dict = convert_to_dictonary(data)


omega_test = torch.Tensor(np.expand_dims(data_dict['omega'], axis=[1, 2]))
omega_0_test = torch.Tensor(np.expand_dims(data_dict['omega_0'], axis=[0]))

num_modes = 32
omega_0_sq = np.reshape(omega_0, (2000, 50, 50, 1))

omega_0_f = f_tform(omega_0_sq, num_modes, [1, 2])

omega = torch.Tensor(np.expand_dims(omega, axis=1)).permute(3, 0, 1, 2)
omega_0f = torch.Tensor(omega_0_f).permute(0, 2, 1)

omega_0_test_sq = np.reshape(omega_0_test, (1, 50, 50, 1))
omega_0_test_f = f_tform(omega_0_test_sq, num_modes, [1, 2])
omega_0_test_f = torch.Tensor(omega_0_test_f).permute(0, 2, 1)

train, train_targets, val, val_targets, test, test_targets = train_val_test(omega_0f, omega)

cuda0 = torch.device('cuda:0')


input_dim = train.shape[2];
output_dim = train_targets.shape[3]
hidden_dim = 1024;
num_layers = 3;
is_spectral = True
num_samples = 1500
biDir = False
time_steps = 49

model = PDE_RNN(input_dim, hidden_dim, num_layers, output_dim, is_spectral, biDir).to(cuda0)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
train_loss = []
val_loss = []
epochs = 800

batch_size = 500
num_batches = int(num_samples / batch_size)

for i in tq.tqdm(range(epochs)):
  batch_loss = []
  for n in range(num_batches):

    start_idx = (n)*batch_size
    end_idx = start_idx + batch_size
    input = torch.tensor(train[start_idx:end_idx, :,:]).to(cuda0)
    targets = torch.tensor(train_targets[1:time_steps + 1,start_idx:end_idx,:,:]).to(cuda0)

    # Predict
    optimizer.zero_grad()
    pred = model(input, time_steps)
    loss = criterion(pred, targets)

    loss.backward()
    optimizer.step()
    batch_loss.append(loss.item())

    if i % 100 == 0:
      print("Training loss: " + str(loss.item()))

  train_loss.append(np.mean(batch_loss))

  with torch.no_grad():
    inputs = torch.Tensor(val).to(cuda0)
    targets = torch.Tensor(val_targets[:time_steps,:,:,:]).to(cuda0)
    pred = model(inputs, time_steps)
    loss = criterion(pred, targets)
    val_loss.append(loss.item())


with torch.no_grad():
  inputs = torch.Tensor(omega_0_test_f).to(cuda0)
  targets = torch.Tensor(omega_test[:time_steps,:,:,:]).to(cuda0)
  pred = model(inputs, time_steps)
  loss = criterion(pred, targets)
  print("Test Loss: " + str(loss.item()))

test = np.concatenate((np.expand_dims(omega_0_test, axis = 0), pred.cpu().detach()),  axis=0)
pred_cpu = np.reshape(np.squeeze(test) , (time_steps + 1, 50, 50))
#np.save("omega_rnn.npy", pred_cpu)
#files.download('omega_rnn.npy')

fig, ax = plt.subplots(1, 4, figsize=(17, 9))
u = np.reshape(np.squeeze(omega_test[:time_steps,0,:,:].cpu().detach()), (time_steps, 50, 50))

full_pred = np.concatenate((np.expand_dims(omega_0_test, axis = 0), pred.cpu().detach()),  axis=0)
pred_cpu = np.reshape(np.squeeze(test) , (time_steps + 1, 50, 50))


ax[0, 0].imshow(np.array(u[0,:,:]), cmap='jet', extent=[-1, 1,1,-1])
ax[0, 0].set_title('True Solution: t= 0', fontsize=14)
ax[0, 0].set_xlabel('x')
ax[0, 0].set_ylabel('t')
ax[0, 0].axis('off')

ax[0, 1].imshow(np.array(u[14,:,:]), cmap='jet',  extent=[-1, 1,1,-1])
ax[0, 1].set_title('True Solution: t = 6', fontsize=14)
ax[0, 1].set_xlabel('x', fontsize=12)
ax[0, 1].set_ylabel('t', fontsize=12)
ax[0, 1].axis('off')

ax[0, 2].imshow(np.array(u[29,:,:]), cmap='jet', extent=[-1, 1,1,-1])
ax[0, 2].set_title('True Solution: t = 12', fontsize=14)
ax[0, 2].set_xlabel('x', fontsize=12)
ax[0, 2].set_ylabel('t', fontsize=12)
ax[0, 2].axis('off')

ax[0, 3].imshow(np.array(u[44,:,:]), cmap='jet',  extent=[-1, 1,1,-1])
ax[0, 3].set_title('True Solution: t = 18', fontsize=14)
ax[0, 3].set_xlabel('x', fontsize=12)
ax[0, 3].set_ylabel('t', fontsize=12)
ax[0, 3].axis('off')


loss = criterion(torch.Tensor(u[0,:,:]), torch.Tensor(pred_cpu[0,:,:]))
ax[0].imshow(np.array(pred_cpu[0,:,:]), cmap='jet', extent=[-1, 1,1,-1])
ax[0].set_title('Predicted Solution: t = 0', fontsize=14)
ax[0].set_xlabel('x', fontsize=12)
ax[0].set_ylabel('t', fontsize=12)
ax[0].axis('off')

loss = criterion(torch.Tensor(u[14,:,:]), torch.Tensor(pred_cpu[14,:,:]))
ax[1].imshow(np.array(pred_cpu[14,:,:]), cmap='jet', extent=[-1, 1,1,-1])
ax[1].set_title('Predicted Solution: t = 6\n Error: ' + str(round(loss.item(), 4)), fontsize=14)
ax[1].set_xlabel('x', fontsize=12)
ax[1].set_ylabel('t', fontsize=12)
ax[1].axis('off')

loss = criterion(torch.Tensor(u[29,:,:]), torch.Tensor(pred_cpu[29,:,:]))
ax[2].imshow(np.array(pred_cpu[29,:,:]), cmap='jet', extent=[-1, 1,1,-1])
ax[2].set_title('Predicted Solution: t = 12\n Error: ' + str(round(loss.item(), 4)), fontsize=14)
ax[2].set_xlabel('x', fontsize=12)
ax[2].set_ylabel('t', fontsize=12)
ax[2].axis('off')

loss = criterion(torch.Tensor(u[44,:,:]), torch.Tensor(pred_cpu[44,:,:]))
ax[3].imshow(np.array(pred_cpu[44,:,:]), cmap='jet', extent=[-1, 1,1,-1])
ax[3].set_title('Predicted Solution: t = 18\n Error: ' + str(round(loss.item(), 4)), fontsize=14)
ax[3].set_xlabel('x', fontsize=12)
ax[3].set_ylabel('t', fontsize=12)
ax[3].axis('off')

fig.subplots_adjust(wspace=0.5)
plt.show()
