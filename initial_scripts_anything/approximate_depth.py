import os
import numpy as np
import imageio
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import joblib
#from keras_tuner import Hyperband
import tensorflow as tf
"""
def build_model(hp, input_dim, output_dim):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_input', min_value=32, max_value=512, step=32),
                    activation='relu', input_dim=input_dim))
    for i in range(hp.Int('n_layers', 1, 5)):
        model.add(Dense(units=hp.Int(f'units_layer_{i}', 32, 512, step=32), activation='relu'))
    model.add(Dense(output_dim, activation='linear'))
    model.compile(optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                  loss='mean_squared_error')
    return model
"""

def build_and_train_model(X_train_scaled, y_train_scaled, input_dim, units, n_layers, lr, output_dim=1):
    model = Sequential()
    model.add(Dense(units=units, activation='relu', input_dim=input_dim))
    for _ in range(n_layers - 1):
        model.add(Dense(units=units, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    
    # Train the model
    history = model.fit(X_train_scaled, y_train_scaled, epochs=10, validation_split=0.2, verbose=1)
    
    return model, min(history.history['val_loss'])


# Paths to directories
source_dir = "data/CVLab/hubble_output_resized_paper/dpt-to-use"
target_dir = "data/CVLab/hubble_output_resized/dpt"
output_dir = "data/CVLab/hubble_output_resized_paper/dpt"
image_output_dir = output_dir

# Ensure input directories exist
os.makedirs(source_dir, exist_ok=True)
os.makedirs(target_dir, exist_ok=True)

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Model path
model_path = 'depth_conversion_model_specific_pairs.h5'

# Check if the model exists to either load or train it
if os.path.exists(model_path) and os.path.exists('scaler_X.pkl') and os.path.exists('scaler_y.pkl'):
    print("Loading existing model...")
    model = load_model(model_path)
    # Load scalers as well, assuming they were saved previously
    # This step assumes scalers are available as 'scaler_X.pkl' and 'scaler_y.pkl'
    import joblib
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
else:
    # Specific datasets for training
    train_datasets = ['depth_0000.npz', 'depth_0015.npz', 'depth_0025.npz','depth_0040.npz', 'depth_0050.npz', 
                      'depth_0075.npz','depth_0085.npz', 'depth_0100.npz','depth_0115.npz', 'depth_0125.npz',
                      'depth_0140.npz', 'depth_0150.npz']

    # Aggregate data for training
    X_train_list, y_train_list = [], []
    for file_name in train_datasets:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)

        with np.load(source_path) as src_npz, np.load(target_path) as tgt_npz:
            X_train_list.append(src_npz['pred'].flatten())
            y_train_list.append(tgt_npz['pred'].flatten())

    # Preprocess training data
    X_train = np.concatenate(X_train_list).reshape(-1, 1)
    y_train = np.concatenate(y_train_list).reshape(-1, 1)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    input_dim = X_train.shape[1] 
    output_dim = 1

    layers_options = [3,4,5]
    units_options = [64, 128, 256]
    learning_rate_options = [1e-2, 1e-3, 1e-4]

    best_val_loss = float('inf')
    best_hyperparameters = {}

    for n_layers in layers_options:
        for units in units_options:
            for lr in learning_rate_options:
                print("Training new model...")
                model, val_loss = build_and_train_model(X_train_scaled, y_train_scaled, X_train_scaled.shape[1], units, n_layers, lr)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_hyperparameters = {'n_layers': n_layers, 'units': units, 'learning_rate': lr}
                    best_model = model
                    model.save('best_model.h5')

    # Log the best hyperparameters
    with open('best_hyperparameters.txt', 'w') as f:
        f.write(f"Best Hyperparameters:\nLayers: {best_hyperparameters['n_layers']}\n")
        f.write(f"Units: {best_hyperparameters['units']}\n")
        f.write(f"Learning Rate: {best_hyperparameters['learning_rate']}\n")
        f.write(f"Validation Loss: {best_val_loss}\n")
        
    """
    tuner = Hyperband(lambda hp: build_model(hp, input_dim=input_dim, output_dim=output_dim),
                  objective='val_loss',
                  max_epochs=10,
                  directory='keras_tuner_dir',
                  project_name='depth_map_optimization')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(X_train_scaled, y_train_scaled, epochs=50, validation_split=0.2, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Save the best hyperparameters to a file
    with open('best_hyperparameters.txt', 'w') as f:
        f.write(str(best_hps.values))

    model = tuner.hypermodel.build(best_hps)
    """

    best_model.fit(X_train_scaled, y_train_scaled, epochs=50, validation_split=0.2, verbose=1)

    best_model.save(model_path)
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')

# Generate outputs for all depth maps
for file_name in os.listdir(source_dir):
    if file_name.endswith('.npz'):
        source_path = os.path.join(source_dir, file_name)

        with np.load(source_path) as src_npz:
            depths_source = src_npz['pred'].flatten()

            # Preprocess source depth map
            X_source_scaled = scaler_X.transform(depths_source.reshape(-1, 1))
            
            # Predict using the trained model
            y_pred_scaled = model.predict(X_source_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
            converted_depths = y_pred.reshape(src_npz['pred'].shape)

            # Save the converted depths (.npz and .png)
            output_path = os.path.join(output_dir, file_name)
            np.savez(output_path, pred=converted_depths)

            img_name = file_name.replace('.npz', '')
            depth_image_path = os.path.join(image_output_dir, f"{img_name}.png")
            depth_array = converted_depths[0]

            depth_image = np.clip(255.0 / depth_array.max() * (depth_array - depth_array.min()), 0, 255).astype(np.uint8)
            imageio.imwrite(depth_image_path, depth_image)

print("Conversion and output generation completed for all depth maps.")
