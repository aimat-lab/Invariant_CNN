import argparse
import gc
import os
import sys

import tensorflow as tf
import numpy as np
import pandas as pd
import yaml
from pickle import dump
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score as sk_r2
from sklearn.metrics import mean_absolute_error as sk_mae
from sklearn.metrics import mean_squared_error as sk_mse
import tensorflow.keras.backend as K

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from architecture_cnn_periodic import LogLearningRateScheduler
from architecture_cnn_periodic import Cnnmodel_Shift_Flip
from architecture_cnn_periodic import tf_r2


def create_output_folder(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Output directory '{output_path}' created successfully.")
    else:
        print(f"Output directory '{output_path}' already exists.")

# Call the function to create the "output" directory directly
create_output_folder("./output")


METRIC_FUNCTIONS = dict(r2=sk_r2, mae=sk_mae, mse=sk_mse)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="train_config.yaml")
    parser.add_argument("--testing", type=str2bool, nargs='?', const=True, default=False, help="Activate testing mode.")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    print(config)
    print(args.testing)
    return config, args.testing


def _get_trainable_parameters(model):
    return np.sum([K.count_params(w) for w in model.trainable_weights])


def _get_columns():
    metric_columns = [f'{key}_Cf' for key in METRIC_FUNCTIONS.keys()]
    metric_columns.extend([f'{key}_St' for key in METRIC_FUNCTIONS.keys()])
    true_columns = [f'{key}_Cf_true' for key in METRIC_FUNCTIONS.keys()]
    true_columns.extend([f'{key}_St_true' for key in METRIC_FUNCTIONS.keys()])
    columns = metric_columns + true_columns
    return columns, metric_columns, true_columns



def evaluate(name, output_path, X_train, y_train, X_val, y_val, epochs, x_scaler, y_scaler, model_kwargs, results,
             batch_size=256, pooling="normal"):
    print(f"The shape of X_train is {X_train.shape}")
    if pooling=="normal":
        model = Cnnmodel_Shift_Flip(input_shape=X_train[0].shape, model_kwargs=model_kwargs)
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.MeanSquaredError()
    lr_scheduler = LogLearningRateScheduler(epochs=epochs, lr_stop=model_kwargs['final_lr'])
    callbacks = [lr_scheduler]
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[
                      tf.keras.metrics.MeanAbsoluteError(),
                      tf_r2]
                  )
    #print(model.summary())
    #print(f'evaluate x: {X_train.shape}')
    if model_kwargs["n_conv_steps"]==1:
        xx, yy = 65, 192
    elif model_kwargs["n_conv_steps"]==2:
        xx, yy = 33, 96
    elif model_kwargs["n_conv_steps"]==3:
        xx, yy = 17, 48
    elif model_kwargs["n_conv_steps"]==4:
        xx, yy = 9, 24
    elif model_kwargs["n_conv_steps"]==5:
        xx, yy = 5, 12
    
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks
        )
    
    print(model.summary())
    ##Model Prediction
    y_pred = model.predict(X_val)


    '''
    Checking shift and flip work ->  Difference predictions of y-prediction and y-true 
    1) Check flipping prediction
    2) Check shifting prediction
    '''

    ## 1)Check flipping prediction
    # batch_flipped = []
    # for counter, i in enumerate(np.random.randint(1, X_val.shape[2], X_val.shape[0])):
    #     batch_flipped.append(X_val[counter][::-1,:,:])#[::-1,:,:] <- flip and shift
    # batch_flipped_img = np.array(batch_flipped)
    y_pred_flipped = model.predict(X_val[:,::-1,:,:]) # this should flip the short axis of the images
    print(y_pred_flipped.shape)
    print(y_pred)
    print(y_pred_flipped)
    max_abs_pred_diff = np.max(np.abs(y_pred_flipped - y_pred))
    mean_abs_pred_diff = np.mean(np.abs(y_pred_flipped - y_pred))
    # Construct the output file path
    output_file = os.path.join(output_path, "max_mean_differences.txt")

    # Open the file for writing
    with open(output_file, 'a') as file:
        file.write(f"{name}: max abs difference in predictions of normal and flipped images: " + str(max_abs_pred_diff) + "\n")
        file.write(f"{name}: mean abs difference in predictions of normal and flipped images: " + str(mean_abs_pred_diff) + "\n")

    bins = np.linspace(-0.001, 0.001, 20)
    plt.figure()
    labels = ['Cf', 'St']
    plt.hist(y_pred_flipped - y_pred, bins=bins, color=['#FFA500', '#8A2BE2'], label=labels)
    plt.xlabel("Prediction difference flipped (a.u.)")
    plt.ylabel("Number of occurences")
    plt.tick_params(labelsize=8)
    plt.savefig(os.path.join(output_path, f"{name}_prediction_difference_flipped_hist.png"), dpi=200)
    #plt.show()
    #plt.close()


    ## 2)Check shifting prediction
    batch_shifted = []
    for counter, i in enumerate(np.random.randint(1, X_val.shape[2], X_val.shape[0])):
        batch_shifted.append(np.roll(X_val[counter], i, axis=1))
    batch_shifted_img = np.array(batch_shifted)

    y_pred_shifted = model.predict(batch_shifted_img)
    max_abs_pred_diff = np.max(np.abs(y_pred_shifted - y_pred))
    mean_abs_pred_diff = np.mean(np.abs(y_pred_shifted - y_pred))
    with open(output_file, 'a') as file:
        file.write(f"{name}: max abs difference in predictions of normal and shifted images: " + str(
            max_abs_pred_diff) + "\n")
        file.write(f"{name}: mean abs difference in predictions of normal and shifted images: " + str(
            mean_abs_pred_diff) + "\n")

    plt.figure()
    plt.hist(y_pred_shifted - y_pred, bins=bins, color=['#FFA500', '#8A2BE2'], label=labels)
    plt.xlabel("Prediction difference shift (a.u.)")
    plt.ylabel("Number of occurences")
    plt.savefig(os.path.join(output_path, f"{name}_prediction_difference_shifted_hist.png"), dpi=200)
    # plt.show()
    # plt.close()
    
    y_pred_true = y_scaler.inverse_transform(y_pred)
    y_test_true = y_scaler.inverse_transform(y_val)

    print(model.summary())

    for key, func in METRIC_FUNCTIONS.items():
        results.loc[name, f'{key}_Cf'] = func(y_val[:, 0], y_pred[:, 0])
        results.loc[name, f'{key}_St'] = func(y_val[:, 1], y_pred[:, 1])
        results.loc[name, f'{key}_Cf_true'] = func(y_test_true[:, 0], y_pred_true[:, 0])
        results.loc[name, f'{key}_St_true'] = func(y_test_true[:, 1], y_pred_true[:, 1])
    if 'trainable_parameters' in results.columns:
        results.loc[name, 'trainable_parameters'] = np.sum([K.count_params(w) for w in model.trainable_weights])
    results.to_csv(os.path.join(output_path, 'results.csv'))

    # Saving stuff
    with open(os.path.join(output_path, f'{name}_meta_data.npy'), 'bw') as f:
        dump(history.history, f)
    model.save(os.path.join(output_path, f'{name}_model'))

    hist = history.history
    print(hist)
    df = pd.DataFrame(index=list(range(epochs)), columns=hist.keys())
    for key in hist.keys():
        df.loc[:, key] = hist[key]
    df.to_csv(os.path.join(output_path, f'{name}_history.csv'))

    valid_keys = [key for key in hist.keys() if 'val_' not in key]
    for key in valid_keys:
        fig, ax = plt.subplots()
        ax.plot(range(epochs), hist[key])
        try:
            ax.plot(range(epochs), hist[f"val_{key}"])
        except KeyError:
            pass
        plt.savefig(os.path.join(output_path, f"{name}_{key}.png"), bbox_inches='tight')
        plt.close(fig)
    return results


def load_data(dataset_path):
    with open(os.path.join(dataset_path, 'X_removed_island.npy'), 'br') as f:
        X = np.load(f)
    with open(os.path.join(dataset_path, 'X_test_removed_island.npy'), 'br') as f:
        X_test = np.load(f)
    with open(os.path.join(dataset_path, 'y_removed_island.npy'), 'br') as f:
        y = np.load(f)
    with open(os.path.join(dataset_path, 'y_test_removed_island.npy'), 'br') as f:
        y_test = np.load(f)
    with open(os.path.join(dataset_path, 'x_flat.npy'), 'br') as f:
        x_flat = np.load(f)
    with open(os.path.join(dataset_path, 'y_flat.npy'), 'br') as f:
        y_flat = np.load(f)

    X = np.concatenate((X, x_flat[5:]))
    X_test = np.concatenate((X_test, x_flat[:5]))
    y = np.concatenate((y, y_flat[5:]))
    y_test = np.concatenate((y_test, y_flat[:5]))

    return X, X_test, y, y_test


def test(config):
    return train(config, testing=True)


def train(config, testing=False):
    dataset_path = config['dataset_path']
    output_path = config['output_path']
    model_kwargs = config['suggestion']
    epochs = 150
    batch_size = 74
    n_folds = 10
    X, X_test, y, y_test = load_data(dataset_path)

    if testing:
        print("Running in test mode")
        X = X[:50]
        X_test = X_test[:50]
        y = y[:50]
        y_test = y[:50]
        batch_size = 5
        n_folds = 3
        epochs = 1
        print("Test mode: Use only %i samples for training and %i samples for testing" % (len(X), len(y)))
        print(
            "Test mode: Use a batch size of %i, %i fold cross validation and %i epochs" % (batch_size, n_folds, epochs))

    print("Dataset size:")
    print("Training data:")
    print(X.shape, y.shape)
    print("Testing data:")
    print(X_test.shape, y_test.shape)

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)
    y_test_scaled = y_scaler.transform(y_test)

    with open(os.path.join(output_path, 'y_scaler.npy'), 'bw') as f:
        dump(y_scaler, f)

    cv = KFold(n_splits=n_folds, shuffle=True)

    columns, metric_columns, true_columns = _get_columns()
    columns.extend(['trainable_parameters', 'mean', 'std'])
    metric_columns.extend(['trainable_parameters', 'mean', 'std'])

    cv_index = [f'cv_{i}' for i in range(n_folds)]
    cv_index.extend(['mean', 'median', 'std', 'full_model'])
    results = pd.DataFrame(columns=columns, index=cv_index)

    n = 0
    difference_file_path = os.path.join(output_path, "difference_predict_true.txt")

    with open(difference_file_path, "a") as file:
        for train_index, val_index in cv.split(X):
            gc.collect()
            name = f'cv_{n}'
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y_scaled[train_index], y_scaled[val_index]
            print(f'shape of X_train (cross-validation): {X_train.shape}')
            print(f'shape of y_train (cross-validation): {y_train.shape}')
            results = evaluate(
                name=name, output_path=output_path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                epochs=epochs, x_scaler=None, y_scaler=y_scaler, results=results, batch_size=batch_size,
                model_kwargs=model_kwargs, pooling="normal"
            )
            with open(os.path.join(output_path, f'{name}_meta_data.npy'), 'bw') as f_meta:
                dump(train_index, f_meta)
                dump(val_index, f_meta)
            n += 1

        # Evaluate on the full model
        results = evaluate(
            name='full_model', output_path=output_path, X_train=X, y_train=y_scaled, X_val=X_test, y_val=y_test_scaled,
            epochs=epochs, x_scaler=None, y_scaler=y_scaler, results=results, batch_size=batch_size,
            model_kwargs=model_kwargs, pooling="normal"
        )

    results.loc['mean', :] = results.loc[cv_index[:-1], :].mean(axis=0)
    results.loc['median', :] = results.loc[cv_index[:-1], :].median(axis=0)
    results.loc['std', :] = results.loc[cv_index[:-1], :].std(axis=0)
    results.to_csv(os.path.join(output_path, 'results.csv'))

    sigopt_returns = list()
    for key in metric_columns:
        sigopt_returns.append({'name': key, 'value': results.at['mean', key]})

    return sigopt_returns, None


if __name__ == "__main__":
    config, testing = parse_config()
    results, meta_data = train(config, testing=testing)
    print(results)
    print('\n')
    print(meta_data)
    sys.exit(0)
