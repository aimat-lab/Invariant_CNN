import keras.layers
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from tensorflow.keras.layers import Add
from tensorflow.keras.backend import sum
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def tf_r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true-K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())


def periodic_padding_flexible(tensor, axis, padding=1):
    """
        add periodic padding to a tensor for specified axis
        tensor: input tensor
        axis: on or multiple axis to pad along, int or tuple
        padding: number of cells to pad, int or tuple

        return: padded tensor
    """
    if isinstance(axis,int):
        axis = (axis,)
    if isinstance(padding,int):
        padding = (padding,)

    ndim = len(tensor.shape)
    for ax,p in zip(axis,padding):
        # create a slice object that selects everything from all axes,
        # except only 0:p for the specified for right, and -p: for left

        ind_right = [slice(-p,None) if i == ax else slice(None) for i in range(ndim)]
        ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
        right = tensor[ind_right]
        left = tensor[ind_left]
        middle = tensor
        tensor = tf.concat([right,middle,left], axis=ax)

    return tensor
    
class LogLearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    """
    Make learning rate schedule function for log reduction.
    Args:
        lr_start (float, optional): Learning rate to start with. The default is 1e-3.
        lr_stop (float, optional): Final learning rate at the end of epo. The default is 1e-5.
        epochs (int, optional): Total number of epochs to reduce learning rate towards. The default is 100.
        epomin (int, optional): Minimum number of epochs at beginning to leave learning rate constant. The default is 10.
    Example:
        model.fit(callbacks=[LogLearningRateScheduler()])
    """
    def __init__(self, lr_start=1e-3, lr_stop=1e-5, epochs=100, epomin=10, verbose=0):
        self.lr_start = lr_start
        self.lr_stop = lr_stop
        self.epochs = epochs
        self.epomin = epomin
        super(LogLearningRateScheduler, self).__init__(schedule=self.schedule_epoch_lr, verbose=verbose)

    def schedule_epoch_lr(self, epoch, lr):
        if epoch < self.epomin:
            out = self.lr_start
        else:
            out = np.exp(
                float(
                    np.log(self.lr_start) - (np.log(self.lr_start) - np.log(self.lr_stop)) /
                    (self.epochs - self.epomin) * (epoch - self.epomin)
                )
            )
            print('lr scheduler', epoch, out)
        return float(out)

    def get_config(self):
        config = super(LogLearningRateScheduler, self).get_config()
        config.update({"lr_start": self.lr_start, "lr_stop": self.lr_stop, "epochs": self.epochs, "epomin": self.epomin})
        return config


class Conv(tf.keras.layers.Layer):

    def __init__(self, input_shape, pool_size=2, strides=2, **kwargs):
        super(Conv, self).__init__()

        self.layer_conv = []
        self.layer_conv += [

            tf.keras.layers.Conv2D(
                filters=kwargs['n_filters'],
                kernel_size=(kwargs['kernel_size'], kwargs['kernel_size']),
                padding='same',
                input_shape=input_shape,
                name="Conf2D_%i"%(0)),
            tf.keras.layers.ReLU(),

            tf.keras.layers.MaxPool2D(pool_size,
                                      strides,
                                      padding='same')]
        for i in range(kwargs['n_conv_steps']-1):

            self.layer_conv += [
                tf.keras.layers.Conv2D(
                    filters=kwargs['n_filters'],
                    kernel_size=(kwargs['kernel_size'], kwargs['kernel_size']),
                    padding='same',
                    name="Conf2D_%i"%(i+1)),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size,
                                          strides,
                                          padding='same')]

    def call(self, x):
        for i, l in enumerate(self.layer_conv):
            x = l(x)
            print(f'cnn layer: {l.name} {x.shape}')
        print("Final layer")
        return x


class ConvPeriodicPadding(tf.keras.layers.Layer):

    def __init__(self, input_shape, pool_size=2, strides=2, **kwargs):
        super(ConvPeriodicPadding, self).__init__()

        self.kernel_size = kwargs['kernel_size']
        self.layer_conv = []
        self.layer_conv += [
            tf.keras.layers.Conv2D(
                filters=kwargs['n_filters'],
                kernel_size=(kwargs['kernel_size'], kwargs['kernel_size']),
                padding='valid',
                input_shape=input_shape,
                name="Conf2D_%i"%(0)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=pool_size,
                                      strides=strides,
                                      padding='valid')
            # tf.keras.layers.AveragePooling2D(pool_size=pool_size,
            #                                  strides=strides,
            #                                  padding='valid')
            ]
        
        for i in range(kwargs['n_conv_steps']-1):

            self.layer_conv += [
                tf.keras.layers.Conv2D(
                    filters=kwargs['n_filters'],
                    kernel_size=(kwargs['kernel_size'], kwargs['kernel_size']),
                    padding='valid',
                    name="Conf2D_%i"%(i+1)),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=pool_size,
                                          strides=strides,
                                          padding='valid')
                # tf.keras.layers.AveragePooling2D(pool_size=pool_size,
                #                           strides=strides,
                #                           padding='valid')
                ]
                
    def call(self, x):
        print("\n\n   ###   Start NN")
        intermediate_sum = []
        for i, l in enumerate(self.layer_conv):  
            if i%3==0: # conv layer
                print("   ###   Conv layer")
                padding_required = (self.kernel_size - 1) // 2
                x = periodic_padding_flexible(x, axis=(1,2), padding=(padding_required,padding_required))
                print(f'cnn layer: periodic padding for conf {x.shape}, axis: (1,2), padding: ({padding_required},{padding_required})')
                x = l(x)
                print(f'cnn layer: {l.name} {x.shape}')
            elif i%3==1: # act layer
                print("   ###   Act layer")
                x = l(x)
                print(f'cnn layer: {l.name} {x.shape}')
            elif i%3==2: # pooling layer
                print("   ###   Pool layer")
                x = periodic_padding_flexible(x, axis=1, padding=1)
                print(f'cnn layer: periodic padding for pool {x.shape}, axis: (1), padding: (1)')
                x = l(x)
                print(f'cnn layer: {l.name} {x.shape}')
                intermediate_sum.append(x)
                #print(f'Sum list after pooling layer:{intermediate_sum.shape}')
        return(x), (intermediate_sum)
        #return intermediate_sum



class CustomSum(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomSum, self).__init__()
        self.customsum = tf.keras.layers.Conv1D(
            input_shape=np.zeros((1, 5, 12, 117)),
            kernel_size=kwargs['kernel_size_customsum'],
            filters=kwargs['n_filters_customsum'])
    def call(self, x):
        x = self.customsum(x)
        return x


class FC(tf.keras.layers.Layer):
    def __init__(self,  **kwargs):
        super(FC, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=kwargs['dense_size'], activation='relu')
        self.dropout = tf.keras.layers.Dropout(rate=kwargs['dropout'],)
        self.fc2 = tf.keras.layers.Dense(2, activation='linear')

    def call(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Cnnmodel_Shift_Flip(tf.keras.Model):
    def __init__(self, input_shape, model_kwargs):
        super(Cnnmodel_Shift_Flip, self).__init__()
        #print(f'input shape {input_shape}')
        self.conv = ConvPeriodicPadding(input_shape=input_shape, pool_size=2, strides=2, **model_kwargs)
        self.customsum = CustomSum(**model_kwargs)
        self.fc = FC(**model_kwargs)

    def call(self, x):
        x2, intermediate_states = self.conv(x)
        x2_flip, intermediate_states_flip = self.conv(tf.image.flip_up_down(x))
        print("shape before summation: ", x2.shape)
        x2 = self.customsum(sum(x2, axis=-2))
        print("shape after summation and Conv1D: ", x2.shape)
        x2_flip = self.customsum(sum(x2_flip, axis=-2))

        summed = []
        summed_flip = []
        for s in intermediate_states:
            print("shape of intermediate state:", s.shape)
            print("shape of state after sum:", (sum(s, axis=-2)).shape)
            summed.append(self.customsum(sum(s, axis=-2)))
            print("length of list after summation along y-axis: ", len(summed))
        for f in intermediate_states_flip:
            print("shape of flip's intermediate state:", f.shape)
            print("shape of flip's state after sum:", (sum(f, axis=-2)).shape)
            summed_flip.append(self.customsum(sum(f, axis=-2)))

        x3 = tf.concat(summed, axis=-2) #shape:(None,119,8)
        x3_flip = tf.concat(summed_flip, axis=-2)
        print("shape after concatenate / flip and non-flip: ", x3.shape, x3_flip.shape)
        x3_add = Add()([x3, x3_flip])
        print("shape after Add:", x3_add.shape)
        outputs = self.fc(x3_add)
        print("   ###   End NN\n\n")
        print(f'shape of outputs:{outputs.shape}')
        print(f'shape of x2:{x2.shape}')
        print(f'shape of x3:{x3.shape}')
        return outputs
