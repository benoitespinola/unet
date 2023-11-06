import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dropout, Conv3DTranspose, concatenate, ConvLSTM2D
from tensorflow.keras.models import Model
#from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import xarray as xr
import numpy as np
#import glob

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

        x = Conv3D(filters, 3, activation='relu', 
                   padding='same', 
                   kernel_initializer='he_normal')(x)
        
        x = Conv3D(filters, 3, activation='relu', 
                   padding='same', 
                   kernel_initializer='he_normal')(x)
        return x
    pool_size = (2, 2, 2)
    conv1 = conv_block(inputs, 64)
    
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)

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
        # Here dimensions are mismatched
        up = Conv3DTranspose(
                            filters, 2, 
                            strides=(2, 2, 2), 
                            padding='same', 
                            activation='relu', 
                            kernel_initializer='he_normal')(x)
        
        merge = concatenate([merge_layer, up], axis=4)
        x = conv_block(merge, filters)
        return x

    conv6 = upsample_block(drop5, 512, drop4)
    conv7 = upsample_block(conv6, 256, conv3)
    conv8 = upsample_block(conv7, 128, conv2)
    conv9 = upsample_block(conv8, 64, conv1)

    # Output layer with linear activation for regression
    outputs = Conv3D(1, 1, activation='linear')(conv9)

    for _ in range(7):
        outputs = Conv3D(1, (6,1,1), activation='linear')(outputs)
    
    outputs = Conv3D(1, (2,1,1), activation='linear')(outputs)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model

def load_tensors(
        file_paths, label_variable_name,
        bathy_path, batch_dim = 0, batch_size=64,
        check=True, load_bathy=False):
    # Step 1: Load multiple NetCDF files into an xarray dataset
    print('#'*20,' Loading files ', '#'*20)
    dataset = xr.open_mfdataset(file_paths, combine='by_coords')

    if load_bathy:
        bathy = xr.open_mfdataset(bathy_path)
        dataset['bathy'] = bathy.bathy
        dataset['bathy'] = dataset['bathy'].expand_dims(time = dataset['time'])
        del bathy
    
    # Step 2: preprocessing
    print('#'*20,' Preprocessing data ', '#'*20)
    
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
        try:
            dataset[var_to_expand] = dataset[var_to_expand].expand_dims(
            lat=dataset['lat'], lon = dataset['lon'])
        except:
            try:
                dataset[var_to_expand] = dataset[var_to_expand].expand_dims(
                lat=dataset['lat'])
            except:
                dataset[var_to_expand] = dataset[var_to_expand].expand_dims(
                lon = dataset['lon'])

    dataset = dataset.transpose('time', 'lat', 'lon')

    dataset = dataset.isel(lat = slice(0,128), 
                           lon = slice(0, 64))
    
    ds_lat = dataset.lat.values
    ds_lon = dataset.lon.values
    ds_time = dataset.time.values
    
    # Step 3: Extract data variables from the xarray dataset
    if load_bathy:
        bathy = dataset.bathy.values
        dataset = dataset.drop_vars(['bathy'])

    labels = dataset[label_variable_name]
    data_vars = dataset.drop_vars([label_variable_name])

    labels = labels.fillna(0)

    data_variable_names = list(data_vars.keys())

    # Step 4: Convert xarray data variables to NumPy arrays
    data_arrays = [data_vars[var_name].values for var_name in data_variable_names]
    labels_array = labels.values

    # Step 5: Convert NumPy arrays to TensorFlow tensors
    data_tensors = [tf.convert_to_tensor(data_array, dtype=tf.float32) for data_array in data_arrays]
    labels_tensor = tf.convert_to_tensor(labels_array, dtype=tf.float32)
    if load_bathy:
        bathy_tensor = tf.convert_to_tensor(bathy, dtype=tf.float32)

    if check:
        for i, data_tensor in enumerate(data_tensors):
            print(f"Data Tensor {i + 1} ({data_variable_names[i]}) Shape:", data_tensor.shape)
        print("Labels Tensor Shape:", labels_tensor.shape)

    data_tensors = tf.stack(data_tensors, axis=3)

    if check:
        print("Stacked Tensors Shape:", data_tensors.shape)
    if load_bathy:
        data_tensors, labels_tensor, bathy_tensor, ds_time = make_batches(
            data_tensors, labels_tensor, bathy_tensor, ds_time,
            dim=batch_dim, batch_size=batch_size, has_bathy = True)
        if check:
            print("Batched Data Shape:", data_tensors.shape)
            print("Batched Labels Shape:", labels_tensor.shape)
            print("Batched Bathy Shape:", bathy_tensor.shape)
    
        return data_tensors, labels_tensor, bathy_tensor, ds_time, ds_lat, ds_lon
    else:
        data_tensors, labels_tensor, ds_time = make_batches(
            data_tensors, labels_tensor, None, ds_time,
            dim=batch_dim, batch_size=batch_size, has_bathy = False)
        if check:
            print("Batched Data Shape:", data_tensors.shape)
            print("Batched Labels Shape:", labels_tensor.shape)
            print("Batched Bathy Shape:", bathy_tensor.shape)
    
        return data_tensors, labels_tensor, ds_time, ds_lat, ds_lon
    
    

def make_batch(X, dim, batch_size):

    X_len = X.get_shape().as_list()[dim]
    
    idx = np.arange(0, X_len, batch_size)
    idx += X_len - idx[-1] - 1 # take the last elements only
    id_start = idx[:-1]
    id_end = idx[1:]

    data_tensors = [
        X[i_s:i_e, ...]
        for i_s, i_e in zip(id_start, id_end)]
    
    return tf.stack(data_tensors, axis=0)

def make_batch_np(X, dim, batch_size):

    X_len = X.shape[dim]
    
    idx = np.arange(0, X_len, batch_size)
    idx += X_len - idx[-1] - 1 # take the last elements only
    id_start = idx[:-1]
    id_end = idx[1:]

    data_tensors = [
        np.copy(X[i_s:i_e, ...])
        for i_s, i_e in zip(id_start, id_end)]
    
    data_tensors = np.stack(data_tensors, axis=0)

    print( '*'*20 ,' Time batch stack: ', data_tensors.shape)
    return data_tensors


def make_batches(
        data_tensors, labels_tensor, bathy_tensor,  ds_time,
        dim, batch_size, has_bathy):
    
    data_tensors = make_batch(data_tensors, dim, batch_size)

    labels_tensor = make_batch(labels_tensor, dim, batch_size)
    labels_tensor = labels_tensor[:,36:64,:,:]

    ds_time = make_batch_np(ds_time, dim, batch_size)
    ds_time = ds_time[:, 36:64]

    if has_bathy:
        bathy_tensor = make_batch(bathy_tensor, dim, batch_size)
        return data_tensors, labels_tensor, bathy_tensor, ds_time
    else:
        return data_tensors, labels_tensor, ds_time


def split_ts_tensors(data_tensors, labels_tensor, bathy_tensor,
                     train_ratio=0.8, dim=0):
    total_samples = data_tensors.shape[dim]
    train_split = int(total_samples*train_ratio)

    data_train = data_tensors[:train_split, ...]
    labels_train = labels_tensor[:train_split, ...]
    bathy_train = bathy_tensor[:train_split, ...]

    data_test = data_tensors[train_split:, ...]
    labels_test = labels_tensor[train_split:, ...]
    bathy_test = bathy_tensor[train_split:, ...]

    return (data_train, labels_train, bathy_train,
    data_test, labels_test, bathy_test)

def split_v_ts_tensors(data_tensors, labels_tensor, bathy_tensor,
                     train_ratio=0.8, val_ratio=0.5, dim=0):
    
    (data_train, labels_train, bathy_train,
    data_test, labels_test, bathy_test) = split_ts_tensors(
                                            data_tensors,
                                            labels_tensor,
                                            bathy_tensor,
                                            train_ratio,
                                            dim)
    
    (data_val, labels_val, bathy_val,
    data_test, labels_test, bathy_test) = split_ts_tensors(
                                            data_tensors,
                                            labels_tensor,
                                            bathy_tensor,
                                            val_ratio,
                                            dim)

    return (
        data_train, labels_train, bathy_train,
        data_val, labels_val, bathy_val,
        data_test, labels_test, bathy_test)

def calculate_se(x, y, z):
    e_1 = tf.minimum(x, y) + tf.minimum(z - x, z - y)
    e_2 = tf.abs(x - y)

    se = tf.square(tf.minimum(e_1, e_2))

    return se

def pre_process_Z(Z, axis=0):
    Z_expanded = tf.expand_dims(Z, axis= axis)  # Expand the 1st dimension
    return tf.expand_dims(Z_expanded, axis= axis)  # Expand the 1st dimension
    
def MSE(X, Y, Z):
    # Expand Z to match the shape of X and Y
    Z_expanded = pre_process_Z(Z, axis=0)  # Expanddims
    
    SE = calculate_se(X, Y, Z_expanded)
    MSE = tf.reduce_mean(SE)
    return MSE

def MAE(X, Y, Z):
    # Expand Z to match the shape of X and Y
    Z_expanded = pre_process_Z(Z, axis=0)  # Expanddims

    SE = calculate_se(X, Y, Z_expanded)
    AE = tf.sqrt(SE)
    MAE = tf.reduce_mean(AE)
    return MAE


def RMSE(X, Y, Z):
    return tf.sqrt(MSE(X,Y,Z))

def my_RMSE(bathy):
    def the_RMSE(y_true, y_pred):
        this_bathy = bathy[0, 
                           0,
                           0:y_true.shape[2],
                           0:y_true.shape[3]]
        return RMSE(y_true, y_pred, this_bathy)
    the_RMSE.__name__ = 'custom_RMSE'
    return the_RMSE

def my_MAE(bathy):
    def the_MAE(y_true, y_pred):
        this_bathy = bathy[0, 
                           0,
                           0:y_true.shape[2],
                           0:y_true.shape[3]]
        return MAE(y_true, y_pred, this_bathy)
    the_MAE.__name__ = 'custom_MAE'
    return the_MAE

def compile_and_run_model(
        model,
        data_train, labels_train,
        data_val, labels_val,
        data_test, labels_test,
        optimizer,
        loss_fn,
        metrics,
        batch_size,
        ds_time, 
        ds_lat, 
        ds_lon,
        epochs = 7,
        VERBOSE = 1,
        save_preds = False,
        save_path = '',
        file_prefix = ''
        ):
    model.compile(optimizer = optimizer,
                  loss = loss_fn,
                  metrics = metrics)
    model.summary()

    ## Training/fitting the model
    print('Start training')
    # Train the model using your training data

    '''earlyStopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode = 'min',
        patience=3,
        verbose = VERBOSE)'''

    history = model.fit(
        data_train, labels_train,
        validation_data=(data_val, labels_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose = VERBOSE,
        #callbacks=[earlyStopping]
    )

    print('Training done')

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

    predictions = model.predict(data_test).squeeze()

    print(type(predictions))
    print(predictions.shape)

    print(type(ds_time), type(ds_lat), type(ds_lon))
    print(ds_time.shape, ds_lat.shape, ds_lon.shape)

    if save_preds:
        file_name_base = save_path + file_prefix
        for i in range(predictions.shape[0]):
            file_name = file_name_base + np.datetime_as_string(ds_time[i,0], unit ='h') + '.nc'
            print(file_name)
            ds = xr.Dataset(
                data_vars=dict(
                    MLD=(["time", "lat", "lon"], predictions[i,:,:,:].squeeze())
                ),
                coords=dict(
                    lon=ds_lon,
                    lat=ds_lat,
                    time=ds_time[i,:].squeeze()
                ),
                attrs=dict(description="Thermocline depth predictions."),
            )
            print(ds)
            print('-'*40)
            ds.to_netcdf(path=file_name, format = "NETCDF4")


def main(
        train_file_paths,
        val_file_paths,
        test_file_paths,
        label_variable_name, 
        bathy_path,
        save_path,
        batch_size = 64):

    ## Loading data
    print('Loading data')
    data_train, labels_train, bathy_train, time_train, lat_train, lon_train = load_tensors(
        file_paths = train_file_paths, 
        label_variable_name = label_variable_name, 
        bathy_path = bathy_path,
        batch_size = batch_size,
        check=True, load_bathy=True)
    
    data_val, labels_val, bathy_val, time_val, lat_val, lon_val = load_tensors(
        file_paths = val_file_paths, 
        label_variable_name = label_variable_name, 
        bathy_path = bathy_path,
        batch_size = batch_size,
        check=True, load_bathy=True)
    
    data_test, labels_test, bathy_test, time_test, lat_test, lon_test = load_tensors(
        file_paths = test_file_paths, 
        label_variable_name = label_variable_name, 
        bathy_path = bathy_path,
        batch_size = batch_size,
        check=True, load_bathy=True)
    
    print('##########')
    print("Training data shape:", data_train.shape)
    print("Training labels shape:", labels_train.shape)
    print("Training bathy shape:", bathy_train.shape)
    print('-----')
    print("Test data shape:", data_test.shape)
    print("Test labels shape:", labels_test.shape)
    print("Test bathy shape:", bathy_test.shape)
    print('-----')
    print("Time shape:", time_test.shape)
    print("Lat shape:", lat_test.shape)
    print("Lon shape:", lon_test.shape)
    print('##########')
    print('##########')
    print('')
    print('')

    ## Build and compile the model
    
    tf.keras.backend.clear_session()
          
    print('Compiling model')

    input_shape = (64, 128, 64, 126)

    my_rmse = my_RMSE(bathy_test) # 'mean_squared_error'
    my_mae = my_MAE(bathy_test)

    metrics = [my_rmse, my_mae, 'mean_squared_error', 'mean_absolute_error']
    optimizer = tf.keras.optimizers.Adam

    unet = build_unet_3d_regression(input_shape)

    compile_and_run_model(
        model = unet,
        data_train = data_train,
        labels_train = labels_train,
        data_val = data_val,
        labels_val = labels_val,
        data_test = data_test,
        labels_test = labels_test,
        optimizer=optimizer(),
        loss_fn=my_mae,
        metrics = metrics,
        batch_size=batch_size,
        ds_time = time_test, 
        ds_lat = lat_test, 
        ds_lon = lon_test,
        epochs = 7,
        VERBOSE = 1,
        save_preds = True,
        save_path = save_path,
        file_prefix = 'unet_results_'
        )

def get_train_file_paths():
    file_paths = np.array([])
    
    yr = 2012
    for mn in range(7,10):
        file_path = f'/pp_files/training/training_{yr}_{mn:02d}.nc'
        file_paths = np.append(file_paths, np.array(file_path))
    
    for yr in range(2013,2016):
        for mn in range(5,10):
            file_path = f'/pp_files/training/training_{yr}_{mn:02d}.nc'
            file_paths = np.append(file_paths, np.array(file_path))
    
    return file_paths

def get_val_file_paths(yr = 2017, start_month=5, end_month=7):
    file_paths = np.array([])
    
    for mn in range(start_month,end_month):
        file_path = f'/pp_files/training/training_{yr}_{mn:02d}.nc'
        file_paths = np.append(file_paths, np.array(file_path))

    return file_paths

def get_test_file_paths(yr = 2017, start_month=7, end_month=10):
    file_paths = np.array([])
    
    for mn in range(start_month,end_month):
        file_path = f'/pp_files/training/training_{yr}_{mn:02d}.nc'
        file_paths = np.append(file_paths, np.array(file_path))

    return file_paths

if __name__ == '__main__':

    tf_setup(use_CPU=True, use_GPU=False)

    train_file_paths = get_train_file_paths()
    print('#'*20,' Train file paths ', '#'*20)
    print(train_file_paths)

    val_file_paths = get_val_file_paths()
    print('#'*20,' Val file paths ', '#'*20)
    print(val_file_paths)

    test_file_paths = get_test_file_paths()
    print('#'*20,' Val file paths ', '#'*20)
    print(test_file_paths)
    
    bathy_path = '/pp_files/bathy.nc'
    save_path = '/ml_output/'
    label_variable_name = 'MLD'
  
    main(
        train_file_paths = train_file_paths,
        val_file_paths = val_file_paths,
        test_file_paths = test_file_paths,
        label_variable_name = label_variable_name,
        save_path = save_path,
        bathy_path = bathy_path)
