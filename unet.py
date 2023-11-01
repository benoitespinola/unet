import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dropout, Conv3DTranspose, concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

import xarray as xr
import numpy as np
import glob

from time import sleep

def tf_setup(use_CPU, use_GPU):
    if(use_CPU):
        # Set CPU and GPU device placement strategy
        physical_devices = tf.config.list_physical_devices('CPU')
        if len(physical_devices) >= 40:
            tf.config.set_logical_device_configuration(
                physical_devices[0],
                [tf.config.LogicalDeviceConfiguration(), tf.config.LogicalDeviceConfiguration()] * 20
            )
    if(use_GPU):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if len(gpus) >= 4:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration()] * 4
                )

# Define U-Net architecture for 3D regression with custom input shape
def build_unet_3d_regression(input_shape):
    inputs = Input(input_shape)

    # Encoder (Contracting Path)
    def conv_block(x, filters):
        x = Conv3D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = Conv3D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        return x

    conv1 = conv_block(inputs, 64)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = conv_block(pool1, 128)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = conv_block(pool2, 256)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = conv_block(pool3, 512)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    # Bottleneck
    conv5 = conv_block(pool4, 1024)
    drop5 = Dropout(0.5)(conv5)

    # Decoder (Expansive Path)
    def upsample_block(x, filters, merge_layer):
        up = Conv3DTranspose(filters, 2, strides=(2, 2, 2), padding='same', activation='relu', kernel_initializer='he_normal')(x)
        merge = concatenate([merge_layer, up], axis=4)
        x = conv_block(merge, filters)
        return x

    conv6 = upsample_block(drop5, 512, drop4)
    conv7 = upsample_block(conv6, 256, conv3)
    conv8 = upsample_block(conv7, 128, conv2)
    conv9 = upsample_block(conv8, 64, conv1)

    # Output layer with linear activation for regression
    outputs = Conv3D(1, 1, activation='linear')(conv9)

    for i in range(10):
        outputs = Conv3D(1, (58,1,1), activation='linear')(outputs)
    
    outputs = Conv3D(1, (11,1,1), activation='linear')(outputs)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model

def load_tensors(file_paths, label_variable_name, bathy_path, check=True):
    # Step 1: Load multiple NetCDF files into an xarray dataset
    file_paths = glob.glob(file_paths)
    dataset = xr.open_mfdataset(file_paths, combine='by_coords')
    
    # Step 2: preprocessing
    bathy = load_bathy(bathy_path)
    dataset['bathy'] = bathy.bathy
    del bathy
    dataset['bathy'] = dataset['bathy'].expand_dims(time = dataset['time'])
    
    vars_to_expand = [
        'is_winter_by_date',
        'is_spring_by_date',
        'is_summer_by_date',
        'is_autumn_by_date',
        'c_date_ML',
        's_date_ML',
        'c_time_ML',
        's_time_ML']
    
    for var_to_expand in vars_to_expand:
        dataset[var_to_expand] = dataset[var_to_expand].expand_dims(
            lat=dataset['lat'], lon=dataset['lon']
        )

    dataset = dataset.transpose('time', 'lat', 'lon')

    dataset = dataset.isel(lat = slice(0,128), 
                           lon = slice(0, 64))
    
    # Step 3: Extract data variables from the xarray dataset
    bathy = dataset.bathy.values
    dataset = dataset.drop_vars(['bathy'])

    labels = dataset[label_variable_name]
    data_vars = dataset.drop_vars([label_variable_name])

    data_variable_names = list(data_vars.keys())

    # Step 4: Convert xarray data variables to NumPy arrays
    data_arrays = [data_vars[var_name].values for var_name in data_variable_names]
    labels_array = labels.values

    # Step 5: Convert NumPy arrays to TensorFlow tensors
    data_tensors = [tf.convert_to_tensor(data_array, dtype=tf.float32) for data_array in data_arrays]
    labels_tensor = tf.convert_to_tensor(labels_array, dtype=tf.float32)
    bathy_tensor = tf.convert_to_tensor(bathy, dtype=tf.float32)

    if check:
        for i, data_tensor in enumerate(data_tensors):
            print(f"Data Tensor {i + 1} ({data_variable_names[i]}) Shape:", data_tensor.shape)
        print("Labels Tensor Shape:", labels_tensor.shape)

    data_tensors = tf.stack(data_tensors, axis=3)

    if check:
        print("Stacked Tensors Shape:", data_tensors.shape)

    return data_tensors, labels_tensor, bathy_tensor

def split_ts_tensors_2(data_tensors, labels_tensor, bathy_tensor, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2, dim=0):
    total_samples = data_tensors.shape[dim]
    train_split = 608
    validation_split = int((total_samples-train_split)/2)

    data_train = data_tensors[:train_split, ...]
    labels_train = labels_tensor[:train_split, ...]
    bathy_train = bathy_tensor[:train_split, ...]

    data_validation = data_tensors[train_split:validation_split, ...]
    labels_validation = labels_tensor[train_split:validation_split, ...]
    bathy_validation = bathy_tensor[train_split:validation_split, ...]

    data_test = data_tensors[validation_split:, ...]
    labels_test = labels_tensor[validation_split:, ...]
    bathy_test = bathy_tensor[validation_split:, ...]

    return data_train, labels_train, bathy_train, data_validation, labels_validation, bathy_validation, data_test, labels_test, bathy_test

def split_ts_tensors(data_tensors, labels_tensor, bathy_tensor, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2, dim=0):
    total_samples = data_tensors.shape[dim]
    train_split = int(train_ratio * total_samples)
    validation_split = int((train_ratio + validation_ratio) * total_samples)

    data_train = data_tensors[:train_split, ...]
    labels_train = labels_tensor[:train_split, ...]
    bathy_train = bathy_tensor[:train_split, ...]

    data_validation = data_tensors[train_split:validation_split, ...]
    labels_validation = labels_tensor[train_split:validation_split, ...]
    bathy_validation = bathy_tensor[train_split:validation_split, ...]

    data_test = data_tensors[validation_split:, ...]
    labels_test = labels_tensor[validation_split:, ...]
    bathy_test = bathy_tensor[validation_split:, ...]

    return data_train, labels_train, bathy_train, data_validation, labels_validation, bathy_validation, data_test, labels_test, bathy_test

def split_rd_tensors(data, labels):
    train_size = 0.8
    test_val_size = 1 - train_size
    test_size = 0.5
    random_state = 42

    data_train, data_temp, labels_train, labels_temp = train_test_split(
        data, labels,
        test_size=test_val_size,
        random_state=random_state)

    data_validation, data_test, labels_validation, labels_test = train_test_split(
        data_temp,
        labels_temp,
        test_size=test_size,
        random_state=random_state)

    return data_train, labels_train, data_validation, labels_validation, data_test, labels_test


def load_bathy(file_path):
    return xr.open_mfdataset(file_path)
    
def calculate_se(x, y, z):
    e_1 = tf.minimum(x, y) + tf.minimum(z - x, z - y)
    e_2 = tf.abs(x - y)

    se = tf.square(tf.minimum(e_1, e_2))

    return se

def MSE(X, Y, Z):
    # Expand Z to match the shape of X and Y
    Z_expanded = tf.expand_dims(Z, axis=-1)  # Expand the last dimension
    
    SE = calculate_se(X, Y, Z_expanded)
    MSE = tf.reduce_mean(SE)
    return MSE

def RMSE(X, Y, Z):
    return tf.sqrt(MSE(X,Y,Z))

def my_loss(bathy):
    def the_loss(y_true, y_pred):
        this_bathy = bathy[0:y_true.shape[0], 0:y_true.shape[1], 0:y_true.shape[2]]
        return RMSE(y_true, y_pred, this_bathy)
    return the_loss

#class MLD_RMSE(tf.keras.losses.Loss)

def main(file_paths, label_variable_name, bathy_path):
    if(True):
        print('Loading data')
        data, labels, bathy = load_tensors(
            file_paths = file_paths, 
            label_variable_name = label_variable_name, 
            bathy_path = bathy_path,
            check=True)

        print('----------------')
        print("Dara shape:", data.shape)
        print("Labels shape:", labels.shape)
        print("Bathy shape:", bathy.shape)
        print('----------------')

        data_train, labels_train, bathy_train, data_validation, labels_validation, bathy_validation, data_test, labels_test, bathy_test = split_ts_tensors_2(
            data_tensors=data,
            labels_tensor=labels,
            bathy_tensor=bathy,
            train_ratio=0.8,
            validation_ratio=0.15,
            test_ratio=0.05,
            dim=0
        )

        print('##########')
        print("Training data shape:", data_train.shape)
        print("Training labels shape:", labels_train.shape)
        print("Training bathy shape:", bathy_train.shape)
        print('-----')
        print("Validation data shape:", data_validation.shape)
        print("Validation labels shape:", labels_validation.shape)
        print("Validation bathy shape:", bathy_validation.shape)
        print('-----')
        print("Test data shape:", data_test.shape)
        print("Test labels shape:", labels_test.shape)
        print("Test bathy shape:", bathy_test.shape)
        print('##########')
        print('##########')
        print('')
        print('')
 
    #input_shape = (608, 128, 64, 126)
    input_shape = (608, 128, 64, 126)  # Custom input shape for 3D data
    model = build_unet_3d_regression(input_shape)
    loss_function = my_loss(bathy)#'mean_squared_error'
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=loss_function)
    model.summary()

    print('Start training')
    # Train the model using your training data
    epochs = 10  # Adjust the number of epochs as needed
    batch_size = 608  # Adjust the batch size as needed

    history = model.fit(
        data_train, labels_train,
        validation_data=(data_validation, labels_validation),
        epochs=epochs,
        batch_size=batch_size
    )
    print('Rraining done')

    print('######## History ##########')
    print(history)
    print('##################')

    # Evaluate the model on the test dataset
    print('Start evaluating')
    test_loss = model.evaluate(data_test, labels_test)
    print('Evaluation done')

    print('######## Test Loss ##########')
    print(test_loss)
    print('##################')

    # Optionally, make predictions using the trained model
    print('Start predicting')
    predictions = model.predict(data_test)
    print('Predictions done')

    print('######## Predictions ##########')
    print(predictions)

if __name__ == '__main__':

    tf_setup(use_CPU=True, use_GPU=False)

    file_paths = '/scratch/project_2007862/pp_files/training/training_2012_*.nc'
    bathy_path = '/scratch/project_2007862/pp_files/bathy.nc'
    label_variable_name = 'MLD'
    main(
        file_paths = file_paths,
        label_variable_name = label_variable_name,
        bathy_path = bathy_path)
