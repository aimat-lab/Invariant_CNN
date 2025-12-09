import os
import random
from typing import Tuple, List, Any, Union

import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from models import InvariantCNN
from utils import PATH
from utils import render_latex, latex_table

# == SOURCE PARAMETERS ==
# The path to the source dataset folder

# :param SOURCE_PATH:
#       The path to the source dataset folder. THis folder should contain specific .NPY files which 
#       contain the images and the labels of the dataset.

SOURCE_PATH = "./dataset/"
# :param NUM_VAL:
#       The number of validation samples to use for the model. This should be a number between 0 and 1.
#       If set to 0.1, 10% of the training samples will be used for validation. Could also be set to an integer
#       value to specify the number of samples to use.
NUM_VAL: Union[int, float] = 0.05

# == MODEL PARAMETERS ==
# Parameters related to the construction of the model

# :param CONV_UNITS:
#       The number of convolutional units to use for the model. This should be a list of integers where
#       each integer represents the number of filters to use for each convolutional layer. Each integer
#       in the list will represent/add one layer.
CONV_UNITS: List[int] = [16, 32, 64, 128, 256]
# :param DENSE_UNITS:
#       The number of dense units to use for the model. This should be a list of integers where each integer
#       represents the number of units to use for each dense layer. Each integer in the list will represent/add
#       one layer. The final number of units in this list should match the number of target values that 
#       the model should predict.
DENSE_UNITS: List[int] = [256, 128, 64, 2]
# :param KERNEL_SIZE:
#       The kernel size to use for the convolutional layers.
KERNEL_SIZE: int = 8
# :param USE_APS:
#       Whether to use the Adaptive Polyphase Sampling (APS) layer in the model. If set to True, the model will
#       use the APS layer after each convolutional layer to perform the strided downsampling.
USE_APS: bool = False

# == TRANING PARAMETERS ==
# Parameters related to the training process of the model

# :param EPOCHS:
#       The number of epochs to train the model for.
EPOCHS: int = 100
# :param BATCH_SIZE:
#       The batch size to use for training.
BATCH_SIZE: int = 16
# :param LEARNING_RATE:
#       The learning rate to use for training.
LEARNING_RATE: float = 1e-5
# :param DEVICE:
#       The device to use for training. This should be either 'cpu' or 'cuda'.
DEVICE: str = 'gpu'

# == EVALUATION PARAMETERS ==

# :param NUM_EXAMPLES:
#       The number of examples to use for the shift and flip invariance evaluation.
#       Selecting a high number here may incur a high runtime after the training of the model.
NUM_EXAMPLES: int = 3

NUM_CV = 5


__DEBUG__ = False

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

def load_dataset(e: Experiment
                 ) -> Tuple[list, list, list]:
    """
    Loads the dataset and returns it as a tuple (train_list, test_list) of lists where the 
    first element is a list containing tuples (x, y) of the training set samples and the second 
    element is a list containing elements (x, y) of the test set samples.
    """
    
    # ~ train set
    x_train_path = os.path.join(e.SOURCE_PATH, 'X_removed_island.npy')
    x_train = np.load(x_train_path)
    
    y_train_path = os.path.join(e.SOURCE_PATH, 'y_removed_island.npy')
    y_train = np.load(y_train_path)
    
    # ~ test set
    x_test_path = os.path.join(e.SOURCE_PATH, 'X_test_removed_island.npy')
    x_test = np.load(x_test_path)
    
    y_test_path = os.path.join(e.SOURCE_PATH, 'y_test_removed_island.npy')
    y_test = np.load(y_test_path)
    
    # ~ flat channels
    x_flat_path = os.path.join(e.SOURCE_PATH, 'x_flat.npy')
    x_flat = np.load(x_flat_path)
    
    y_flat_path = os.path.join(e.SOURCE_PATH, 'y_flat.npy')
    y_flat = np.load(y_flat_path)
    
    full = [(x, y) for x, y in zip(x_train, y_train)] + [(x, y) for x, y in zip(x_flat[5:], y_flat[5:])]
    test = [(x, y) for x, y in zip(x_test, y_test)] + [(x, y) for x, y in zip(x_flat[:5], y_flat[:5])]
    
    num_val: int = e.NUM_VAL if isinstance(e.NUM_VAL, int) else int(len(full) * e.NUM_VAL)
    print(f'using {num_val} validation samples...')
    indices = list(range(len(full) - 1))
    
    indices_val = random.sample(indices, k=num_val)
    indices_train = list(set(indices) - set(indices_val))
    
    train = [full[i] for i in indices_train]
    val = [full[i] for i in indices_val]
    
    return (
        train,
        val,
        test,
    )


@experiment.hook('evaluate_shift_invariance', replace=False, default=True)
def evaluate_shift_invariance(e: Experiment,
                              model: InvariantCNN,
                              x: np.array,
                              key: str = 'test',
                              ) -> np.ndarray:
    """
    Given a ``model`` an input element ``x`` and a unique string key, this function will evaluate the shift invariance
    of the model on that given element by applying all possible horizontal shifts of the image and comparing the 
    predictions of the original image with the predictions of the shifted images.
    
    This function will also plot the results and save them to the experiment archive folder.
    """
    # constructing the shifted versions
    width = x.shape[3]
    shifts = list(range(width))
    x_shifted = np.concatenate([np.roll(x, shift, axis=3) for shift in shifts], axis=0)

    # query the model
    out_example = model.forward_array(x) 
    out_shifted = model.forward_array(x_shifted)
    
    # calculate the differences as percentages of the original prediction
    diffs = []
    for shift, out_shift in zip(shifts, out_shifted):
        diff = np.abs(out_example - out_shift)
        diff = (diff / np.mean(out_example))*100
        diffs.append(diff)
        
    diffs = np.concatenate(diffs, axis=0)
    fig, (ax_img, ax_cf, ax_st) = plt.subplots(ncols=3, nrows=1, figsize=(20, 6))
    ax_img.imshow(x[0, 0], cmap='gray')
    ax_img.set_title(f'Original Image - {key}\n'
                     f'$C_f$: {out_example[0, 0]:.3f} - $S_t$: {out_example[0, 1]:.3f}')
    
    ax_st.plot(shifts, diffs[:, 0])
    ax_st.set_xlabel('Shift [px]')
    ax_st.set_ylabel('Prediction Difference [%]')
    ax_st.set_title('Shift Invariance Test $S_t$')
    y_lo, y_hi = ax_st.get_ylim()
    if y_hi < 0.1:
        ax_st.set_ylim(-0.02, 0.1)

    ax_cf.plot(shifts, diffs[:, 1])
    ax_cf.set_xlabel('Shift [px]')
    ax_cf.set_ylabel('Prediction Difference [%]')
    ax_cf.set_title('Shift Invariance Test $C_f$')
    y_lo, y_hi = ax_cf.get_ylim()
    if y_hi < 0.1:
        ax_cf.set_ylim(-0.02, 0.1)
    
    e.commit_fig(f'shift_invariance_{key}.png', fig)
    
    # At the end we want to return a numpy array that contains the differences of 
    return diffs


@experiment.hook('evaluate_flip_invariance', default=False, replace=False)
def evaluate_flip_invariance(e: Experiment,
                             model: InvariantCNN,
                             x: np.ndarray,
                             key: str = 'test'
                             ) -> np.ndarray:
    """
    Given a ``model`` and an input element ``x``, this function will evaluate the vertical flip invariance of the model
    on that given element by comparing the predictions of the original image with the predictions of the flipped image.
    """
    x_flip = np.flip(x, axis=2)
    
    # ~ query the model with both
    out = model.forward_array(x)
    out_flip = model.forward_array(x_flip)

    # calculating the percentage(!) of the deviation w.r.t. the original prediction
    diff = np.abs(out - out_flip)
    diff = (diff / np.mean(out))*100
    
    return diff


@experiment
def experiment(e: Experiment):
    
    e.log('starting experiment...')
    
    # ~ data loading
    e.log('loading flow channel dataset...')
    train, val, test = load_dataset(e)
    x_train = np.array([x.transpose(2, 0, 1) for x, _ in train])[:, :, :, :]
    
    # Combine existing train, val, and test sets
    #    (If 'val' is a separate subset, you can combine that too, or treat it separately.)
    x_train = np.array([x.transpose(2, 0, 1) for x, _ in train])  # shape: (N_train, 1, H, W)
    y_train = np.array([y for _, y in train])                     # shape: (N_train, 2) or similar

    x_val = np.array([x.transpose(2, 0, 1) for x, _ in val])      # shape: (N_val, 1, H, W)
    y_val = np.array([y for _, y in val])

    x_test = np.array([x.transpose(2, 0, 1) for x, _ in test])    # shape: (N_test, 1, H, W)
    y_test = np.array([y for _, y in test])

    X_all = np.concatenate([x_train, x_val, x_test], axis=0)
    Y_all = np.concatenate([y_train, y_val, y_test], axis=0)
    
    e.log(f'Combined dataset shape: {X_all.shape}, {Y_all.shape}')
    
    e.log('example shape:')
    e.log(x_train[0].shape)
    input_shape = (x_train[0].shape[1], x_train[0].shape[2])
    height, width = input_shape
    
    # ~ Keep track of metrics across the 5 splits
    r2_cf_list, mae_cf_list, mse_cf_list = [], [], []
    r2_st_list, mae_st_list, mse_st_list = [], [], []
    
    # ~ repeated (5 times) cross-validation with 90:10 splits
    
    for cv_idx in range(NUM_CV):
        e.log(f'===== CV iteration {cv_idx+1} / {NUM_CV} =====')
        
        # ~ 1) Train/test split (90:10) with shuffling
        X_train_cv, X_test_cv, Y_train_cv, Y_test_cv = train_test_split(
            X_all, Y_all, test_size=0.1, shuffle=True, random_state=cv_idx
        )
        
        # ~ 2) Scale the target values using only the training set
        scaler = StandardScaler()
        Y_train_cv_scaled = scaler.fit_transform(Y_train_cv)
        Y_test_cv_scaled  = scaler.transform(Y_test_cv)
    
        # ~ model training
        e.log('constructing model...')
        model = InvariantCNN(
        input_dim=1, 
        input_shape=input_shape,
        conv_units=e.CONV_UNITS,
        dense_units=e.DENSE_UNITS,
        learning_rate=e.LEARNING_RATE,
        kernel_size=e.KERNEL_SIZE,
        use_aps=e.USE_APS,
        )
        e.log('model summary:')
        e.log(str(model))
        
        # Convert to PyTorch tensors
        x_train_tensor = torch.tensor(X_train_cv, dtype=torch.float32)
        y_train_tensor = torch.tensor(Y_train_cv_scaled, dtype=torch.float32)
        x_test_tensor  = torch.tensor(X_test_cv,  dtype=torch.float32)
        y_test_tensor  = torch.tensor(Y_test_cv_scaled,  dtype=torch.float32)

        e.log('creating tensor datasets...')
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset  = TensorDataset(x_test_tensor,  y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=e.BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=e.BATCH_SIZE, shuffle=False)
            
        # Train the model
        trainer = pl.Trainer(max_epochs=e.EPOCHS, accelerator=e.DEVICE)
        try:
            trainer.fit(model, train_loader, test_loader)
        except KeyboardInterrupt:
            trainer.should_stop = True
        
        # Evaluate
        model.eval()
        model.set_scaler(scaler)
        
        e.log('saving the model...')
        model_path = os.path.join(e.path, f'model_{cv_idx+1}.ckpt')
        model.save(model_path)
        model = model.load(model_path)
        
        # ~ model evaluation
        e.log('evaluating model...')
        y_pred_cv = model.forward_array(X_test_cv, use_scaler=True)  # shape: (N_test, 2)
        #y_test = scaler.inverse_transform(y_test)
        
        # ~ Compute metrics for Cf
        r2_cf  = r2_score(Y_test_cv[:, 0], y_pred_cv[:, 0])
        mae_cf = mean_absolute_error(Y_test_cv[:, 0], y_pred_cv[:, 0])
        mse_cf = mean_squared_error(Y_test_cv[:, 0], y_pred_cv[:, 0])  # <-- MSE for Cf
        
        # ~ Compute metrics for St
        r2_st  = r2_score(Y_test_cv[:, 1], y_pred_cv[:, 1])
        mae_st = mean_absolute_error(Y_test_cv[:, 1], y_pred_cv[:, 1])
        mse_st = mean_squared_error(Y_test_cv[:, 1], y_pred_cv[:, 1])  # <-- MSE for St
        
        e.log(f'    * Fold {cv_idx+1}:C_f - R2: {r2_cf:.3f}, MAE: {mae_cf}, MSE: {mse_cf}')
        e.log(f'    * Fold {cv_idx+1}:S_t - R2: {r2_st:.3f}, MAE: {mae_st}, MSE: {mse_st}')
        
        # We also want to save all the resulting metrics to the experiment registry so that we can easily access 
        # them later on again.
        e['y/test/pred'] = y_pred_cv
        e['y/test/true'] = Y_test_cv
        
        e['metrics/cf/r2'] = r2_cf
        e['metrics/cf/mae'] = mae_cf
        e['metrics/cf/mse'] = mae_cf
        
        e['metrics/st/r2'] = r2_st
        e['metrics/st/mae'] = mae_st
        e['metrics/st/mse'] = mae_st

        
        # Append to lists
        r2_cf_list.append(r2_cf)
        mae_cf_list.append(mae_cf)
        mse_cf_list.append(mse_cf)
        
        r2_st_list.append(r2_st)
        mae_st_list.append(mae_st)
        mse_st_list.append(mse_st)
        
        # regression plot
        fig, (ax_cf, ax_st) = plt.subplots(ncols=2, nrows=1, figsize=(14, 6))
        sns.histplot(
            x=Y_test_cv[:, 0], y=y_pred_cv[:, 0], bins=50, pmax=0.9, ax=ax_cf, cmap="Blues", cbar=True
        )
        ax_cf.plot([Y_test_cv[:, 0].min(), Y_test_cv[:, 0].max()], [Y_test_cv[:, 0].min(), Y_test_cv[:, 0].max()], 'k--', lw=2)
        ax_cf.set_title(f'$C_f$ Regression\n'
                        f'R2: {r2_cf:.3f}, MAE: {mae_cf}')
        ax_cf.set_xlabel('True Values')
        ax_cf.set_ylabel('Predicted Values')

        sns.histplot(
            x=Y_test_cv[:, 1], y=y_pred_cv[:, 1], bins=50, pmax=0.9, ax=ax_st, cmap="Blues", cbar=True
        )
        ax_st.plot([Y_test_cv[:, 1].min(), Y_test_cv[:, 1].max()], [Y_test_cv[:, 1].min(), Y_test_cv[:, 1].max()], 'k--', lw=2)
        ax_st.set_title('$S_t$ Regression\n'
                        f'R2: {r2_st:.3f}, MAE: {mae_st}')
        ax_st.set_xlabel('True Values')
        ax_st.set_ylabel('Predicted Values')

        reg_plot_path = f'regression_plot_fold_{cv_idx+1}.png'
        e.commit_fig(reg_plot_path, fig)
    
    # ~ Post-processing: compute average metrics over the 5 runs
    e.log('===== Cross-Validation Results (averaged) =====')
    e.log(f'C_f R2:  mean={np.mean(r2_cf_list):.3f}  std={np.std(r2_cf_list):.3f}')
    e.log(f'C_f MAE: mean={np.mean(mae_cf_list)} std={np.std(mae_cf_list)}')
    e.log(f'C_f MSE: mean={np.mean(mse_cf_list)} std={np.std(mse_cf_list)}')
    
    e.log(f'S_t R2:  mean={np.mean(r2_st_list):.3f}  std={np.std(r2_st_list):.3f}')
    e.log(f'S_t MAE: mean={np.mean(mae_st_list)} std={np.std(mae_st_list)}')
    e.log(f'S_t MSE: mean={np.mean(mse_st_list)} std={np.std(mse_st_list)}')
    
    # Optionally, store final results in the experiment
    e['metrics_cv/cf/r2']  = r2_cf_list
    e['metrics_cv/cf/mae'] = mae_cf_list
    e['metrics_cv/cf/mse'] = mse_cf_list
    
    e['metrics_cv/st/r2']  = r2_st_list
    e['metrics_cv/st/mae'] = mae_st_list
    e['metrics_cv/st/mse'] = mse_st_list
        
    # evaluating the shift invariance
    
    # we are going to choose a random image from the test set here and then construct all of it's 
    # shifted versions. Then we pass the original image and all the shifts through the model and record 
    # the differences in the predictions w.r.t. to the original image.
    e.log(f'evaluating shift invariance on {e.NUM_EXAMPLES} examples...')
    
    x_path = os.path.join(e.PATH, 'figure_4_X.npy')
    X_1 = np.load(x_path)
    X_1 = np.transpose(X_1, (0, 3, 1, 2))
    print(f'x shape is :{X_1.shape}')
    input_shape = (X_1[0].shape[1], X_1[0].shape[2])
    
    shift_diffs: List[np.ndarray] = []
    flip_diffs: List[np.ndarray] = []
    for c in range(e.NUM_EXAMPLES):
        
        e.log(f' * {c}/{e.NUM_EXAMPLES}')
        # choose a random image
        num_tests = len(x_test)
        index = random.randint(0, num_tests)
        # x_example: (1, 1, height, width)
        x_example = x_test[index:index+1] # preserve dimensions
        shift_diff: np.ndarray = e.apply_hook(
            'evaluate_shift_invariance',
            model=model,
            x=x_example,
            key=c,
        )
        shift_diffs.append(shift_diff)
        shift_diffs_cf.append(shift_diff[:, 0])
        shift_diffs_st.append(shift_diff[:, 1])
        
        flip_diff: (1, 2)
        # :hook evaluate_flip_invariance:
        #       Given the model and one flow channel element, this hook will evaluate the vertical flip 
        #       invariance by comparing the model output of the shifted element with the output of the original 
        #       image.
        flip_diff: np.ndarray = e.apply_hook(
             'evaluate_flip_invariance',
             model=model,
             x=x_example,
             key=c,
         )
        flip_diffs.append(flip_diff)
        
    # ~ invariance result table
    # Here we automatically render a table with the results on the shift and flip invariance evaluation.
    
    e.log('aggregating invariance evaluations...')
    shift_diffs = np.stack(shift_diffs, axis=0)
    flip_diffs = np.stack(flip_diffs, axis=0)
    
    _, latex_string = latex_table(
        column_names=[
            '$C_t$ Flip Diff (\%)', 
            '$S_t$ Flip Diff (\%)', 
            '$C_t$ Shift Diff (\%)', 
            '$S_t$ Shift Diff (\%)'
        ],
        rows=[[
            flip_diffs[:, :, 0].flatten().tolist(),
            flip_diffs[:, :, 1].flatten().tolist(),    
            shift_diffs[:, :, 0].flatten().tolist(),
            shift_diffs[:, :, 1].flatten().tolist(),
        ]],
    )
    e.commit_raw('invariances.tex', latex_string)
    
    pdf_path = os.path.join(e.path, 'invariances.pdf')
    render_latex({'content': latex_string}, output_path=pdf_path)

experiment.run_if_main()
