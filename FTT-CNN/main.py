import json
import torch
import argparse
import numpy as np
import torch.nn as nn

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from models import flip_invariant_FFT_CNN
from utils import list_arrays_to_tensor, count_trainable_parameters, split_data
from datasets import load_flow_dataset
from trainer import Trainer


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='seed for test/train split, default is 0')
args = parser.parse_args()

print('loading dataset...')
X, y = list_arrays_to_tensor(load_flow_dataset())
X = torch.permute(X, (0, 3, 1, 2))[:, :, :128, :]

indices = split_data(X.shape[0], seed=args.seed)
x_train, x_val, x_test = X[indices['train']], X[indices['val']], X[indices['test']]
y_train, y_val, y_test = y[indices['train']], y[indices['val']], y[indices['test']]

# standardize output
y_train, y_val, y_test = y_train.numpy(), y_val.numpy(), y_test.numpy()
y_scaler = StandardScaler()
y_scaler.fit(y_train)
y_train, y_val, y_test = torch.tensor(y_scaler.transform(y_train)), torch.tensor(y_scaler.transform(y_val)), torch.tensor(y_scaler.transform(y_test))


trainset = TensorDataset(x_train, y_train)
valset = TensorDataset(x_val, y_val)
testset = TensorDataset(x_test, y_test)


print('constructing model...')
input_shape = trainset[0][0].shape[-2:]
input_dim = trainset[0][0].shape[0]
model = flip_invariant_FFT_CNN(
    input_dim=1, 
    input_shape=(128, 384),
    conv_units=[64, 64, 128, 256, 512],
    dense_units=[512, 256, 32, 2],
    kernel_sizes=[8, 8, 5, 3, 2],
    bandwidths=[.8, .5, .3, .3, .5],
    )
print(f'Number of trainable parameters: {count_trainable_parameters(model)}')


train_config = {
    'name': f'fft_exp{args.seed}', 
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'loss': nn.MSELoss(), 
    'batch': 16,
    'epochs': 150,
    'lr': 0.0001,
    'scheduler_step': 10,
    'parallelize': False,
}

trainer = Trainer(
    model,
    train_config,
)

# Train model
print('training model...')
trainer.fit(
    train_data=trainset,
    val_data=valset,
)

## Load state_dict
if isinstance(trainer.model, nn.DataParallel):
    trainer.model = trainer.model.module
trainer.model.load_state_dict(torch.load(
    f"./models/{train_config['name']}_best.pt",
    weights_only=True,
))

# Evaluate model
print('evaluating model...')
test_loader = DataLoader(testset, batch_size=8, shuffle=False)

_, preds, targets = trainer.test(test_loader, shifts=False)
print(f'test accuracy: {r2_score(targets.detach().cpu().numpy(), preds.detach().cpu().numpy())}')

_, preds, targets = trainer.test(test_loader, flips=True, shifts=True)
print(f'flipped and shifted test accuracy: {r2_score(targets.detach().cpu().numpy(), preds.detach().cpu().numpy())}')

# compute R2, MAE, and MSE independently for each output 
preds, targets = preds.detach().cpu().numpy(), targets.detach().cpu().numpy()
preds, targets = y_scaler.inverse_transform(preds), y_scaler.inverse_transform(targets)
results = {'r2': [], 'MAE': [], 'MSE': []}
for i in range(preds.shape[-1]):
    predictions = preds[..., i].reshape(-1, 1)
    true_values = targets[..., i].reshape(-1, 1)
    results['r2'].append(r2_score(true_values, predictions))
    results['MAE'].append(mean_absolute_error(true_values, predictions))
    results['MSE'].append(mean_squared_error(true_values, predictions))

print(results)
