import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch.optim as optim
import tqdm.notebook as tq
from sklearn.preprocessing import StandardScaler
#from tqdm import tqdm

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


def f_tform(u, u_0, modes, u_axes, u_0_axes):
  #fft_test = np.fft.fft(u_0[1,0,:])

  # Take Fourier of X component
  u_0f = np.real(torch.fft.fftn(u_0, dim=u_0_axes))
  uf = np.real(torch.fft.fftn(u, dim=u_axes))

  # FFT Shift X component
  u_0f = torch.fft.fftshift(u_0f, dim=u_0_axes)
  uf = torch.fft.fftshift(uf, dim=u_axes)

  # Truncate number of modes
  start_modes = int(np.floor(u_0.shape[2]/2) - modes/2)
  end_modes = int(np.floor(u_0.shape[2]/2) + modes/2)

  u_0f = u_0f[:,:,start_modes:end_modes]
  uf = uf[:,:,:,start_modes:end_modes]

  return u_0f, uf

  import torch.nn.functional as F
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

        self.lstm = nn.GRU(input_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=biDir, batch_first = True)

        self.fc = nn.Linear(hidden_dim*self.D, output_dim)

    def forward(self, x, time_size):
        h_0 = torch.zeros(self.D*self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(cuda0)
        outputs = torch.empty((time_size, x.size(0), 1, self.output_dim)).to(cuda0)

        for t in range(time_size):
          if t==0:
            out, h_0 = self.lstm(x, h_0)
          else:
            if self.spectral:
              x = outf
            else:
              x = out
            out, h_0 = self.lstm(x, h_0)


          out = self.fc(out)

          if self.spectral:
            # Take Fourier of X component
            outf = torch.real(torch.fft.fft(out, axis=2))

            # FFT Shift X component
            u_0f = torch.fft.fftshift(outf, dim=2)

            # Truncate number of modes
            start_modes = int(np.floor(self.output_dim/2) - 32/2)
            end_modes = int(np.floor(self.output_dim/2) + 32/2)

            outf = outf[:,:,start_modes:end_modes].float()

          out = out[:,:,:]
          outputs[t, :, :, :] = out;

        return outputs

data = sio.loadmat('/content/drive/My Drive/Colab Notebooks/Single_SLW.mat', struct_as_record=True, squeeze_me=False)

data_dict = convert_to_dictonary(data)
print(data_dict['u'].shape)
print(data_dict['u_0'].shape)

u_test = torch.Tensor(np.expand_dims(data_dict['u'], axis=[1, 2]))
u_0_test = torch.Tensor(np.expand_dims(data_dict['u_0'], axis=[0]))

print(u_test.shape)
print(u_0_test.shape)
plt.plot(range(u_0_test.shape[2]), np.squeeze(u_0_test[0,:,:]))

data = sio.loadmat('/content/drive/My Drive/Colab Notebooks/Random_SlantedWave_2000.mat', struct_as_record=True, squeeze_me=False)
data_dict = convert_to_dictonary(data)
u_t = data_dict['u']
u = torch.Tensor(np.expand_dims(data_dict['u'], axis=1)).permute(0,3,1,2)
u_0 = torch.Tensor(data_dict['u_0']).permute((2, 0, 1))
print(u.shape)
print(u_0.shape)
plt.plot(range(u_0.shape[2]), np.squeeze(u_0[1,:,:]))

cuda0 = torch.device('cuda:0')

def train_val_test(u_0, u):
  sz = u_0.shape[0]
  return u_0[0:int(sz*.70),:,:], u[:,0:int(sz*.70),:,:], u_0[int(sz*.70):int(sz*.85) ,:,:], u[:,int(sz*.70):int(sz*.85),:,:], u_0[int(sz*.85):sz ,:,:], u[:,int(sz*.85):sz ,:,:]

train, train_targets, val, val_targets, test, test_targets = train_val_test(u_0f, u)

#from traitlets.traitlets import HasTraits
input_dim = train.shape[2];
output_dim = u_0.shape[2]
print(input_dim)
hidden_dim = 1024;
num_layers = 2;
is_spectral = True
num_samples = 1000
biDir = False
time_steps = 200

model = PDE_RNN(input_dim, hidden_dim, num_layers, output_dim, is_spectral, biDir).to(cuda0)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
train_loss = []
val_loss = []
epochs = 800

for i in tq.tqdm(range(epochs)):
    input = torch.tensor(train[:num_samples, :,:]).to(cuda0)
    targets = torch.tensor(train_targets[:time_steps,:num_samples,:,:]).to(cuda0)

    #print(input)
    # Predict
    optimizer.zero_grad()
    pred = model(input, time_steps)

    loss = criterion(pred, targets)



    loss.backward()
    optimizer.step()
    train_loss.append(loss.item())

    if i % 100 == 0:
      print("Training loss: " + str(loss.item()))

    with torch.no_grad():
      inputs = torch.Tensor(val).to(cuda0)
      targets = torch.Tensor(val_targets[:time_steps,:,:,:]).to(cuda0)
      pred = model(inputs, time_steps)
      loss = criterion(pred, targets)
      val_loss.append(loss.item())


with torch.no_grad():
  inputs = torch.Tensor(test).to(cuda0)
  targets = torch.Tensor(val_targets[:time_steps,:,:,:]).to(cuda0)
  pred = model(inputs, time_steps)
  loss = criterion(pred, targets)
  print("Test Loss: " + str(loss.item()))

fig, ax = plt.subplots(1, 2, figsize=(20, 10), subplot_kw=dict(projection='3d'))

X, T = np.meshgrid(range(u_0.shape[2]), range(time_steps))

ax[0].plot_surface(X, T, np.array(np.squeeze(u[:time_steps,8,:,:].cpu().detach())), cmap='plasma')
ax[0].set_title('True')
ax[0].set_xlabel('x')
ax[0].set_ylabel('t')
ax[0].set_zlabel('u(x,t)')


ax[1].plot_surface(X, T, np.array(np.squeeze(pred[:,8,:,:].cpu().detach())), cmap='plasma')
ax[1].set_title('Predicted')
ax[1].set_xlabel('x')
ax[1].set_ylabel('t')
ax[1].set_zlabel('u(x,t)')
plt.show()


time_steps = 200
with torch.no_grad():
  inputs = torch.Tensor(u_0f_test).to(cuda0)
  targets = torch.Tensor(u_test[:time_steps,:,:,:]).to(cuda0)
  pred = model(inputs, time_steps)
  loss = criterion(pred, targets)

fig, ax = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection='3d'))


x = np.linspace(-2, 6, 201)
t = np.linspace(0, 1, time_steps + 1)

X, T = np.meshgrid(x, t)
X2, T2 = np.meshgrid(range(u_0.shape[2]), range(201))
ax[0].plot_surface(X, T, np.array(np.squeeze(u_test[:time_steps + 1,0,:,:].cpu().detach())), cmap='plasma')
ax[0].set_title('True Solution\n\n', fontsize=14)
ax[0].set_xlabel('x', fontsize=12)
ax[0].set_ylabel('t', fontsize=12)
ax[0].set_zlabel('u(x,t)', fontsize=12)




ax[1].plot_surface(X, T, np.array(np.squeeze(full_pred)), cmap='plasma')
ax[1].set_title('Predicted Solution\n Error: ' + str(round(loss.item(), 4)) + '\n', fontsize=14)
ax[1].set_zlim([0, 1])
ax[1].set_xlabel('x', fontsize=12)
ax[1].set_ylabel('t', fontsize=12)
ax[1].set_zlabel('u(x,t)', fontsize=12)
plt.show()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot()
ax.plot(range(epochs), train_loss,  label='Training Loss')
ax.plot(range(epochs), val_loss, label='Validation Loss')
ax.set_yscale('log')

ax.legend()
ax.set_title('Training and Validation Losses')
ax.set_xlabel('Epochs')
ax.set_ylabel('Log Loss')
