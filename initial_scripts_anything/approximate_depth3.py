import os
import numpy as np
import imageio
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import joblib
import tensorflow as tf

def build_and_train_model(X_train_scaled, y_train_scaled, input_dim, units, n_layers, lr, output_dim=1):
    model = Sequential()
    model.add(Dense(units=units, activation='relu', input_dim=input_dim))
    for _ in range(n_layers - 1):
        model.add(Dense(units=units, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    
    # Train the model
    history = model.fit(X_train_scaled, y_train_scaled, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    
    return model, min(history.history['val_loss'])

# Paths to directories
source_dir = "data/CVLab/hubble_output_resized_paper/dpt-to-use"
target_dir = "data/CVLab/hubble_output_resized/dpt"
output_dir = "data/CVLab/hubble_output_resized_paper/dpt"
image_output_dir = output_dir

# Ensure input directories exist
os.makedirs(source_dir, exist_ok=True)
os.makedirs(target_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Specify the filename to process
file_name = 'depth_0000.npz'  # Replace 'specific_file_name.npz' with the actual file name

model_file_name = f'model_{file_name}.h5'
scaler_X_file_name = f'scaler_X_{file_name}.pkl'
scaler_y_file_name = f'scaler_y_{file_name}.pkl'

model_path = os.path.join(output_dir, model_file_name)
scaler_X_path = os.path.join(output_dir, scaler_X_file_name)
scaler_y_path = os.path.join(output_dir, scaler_y_file_name)

# Check if model and scalers already exist
if not (os.path.exists(model_path) and os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path)):
    print(f"Training model for {file_name}...")
    source_path = os.path.join(source_dir, file_name)
    target_path = os.path.join(target_dir, file_name)

    # Check if the corresponding target file exists
    if os.path.exists(target_path):
        with np.load(source_path) as src_npz, np.load(target_path) as tgt_npz:
            X_train = src_npz['pred'].flatten().reshape(-1, 1)
            y_train = tgt_npz['pred'].flatten().reshape(-1, 1)

            # Preprocess training data
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            y_train_scaled = scaler_y.fit_transform(y_train)
            
            input_dim = X_train_scaled.shape[1]
            output_dim = 1

            # Train the model with predefined hyperparameters
            model, _ = build_and_train_model(X_train_scaled, y_train_scaled, input_dim, 64, 2, 1e-3, output_dim)

            # Save the model and scalers
            model.save(model_path)
            joblib.dump(scaler_X, scaler_X_path)
            joblib.dump(scaler_y, scaler_y_path)
    else:
        print(f"Corresponding target file for {file_name} does not exist.")
else:
    print(f"Model and scalers for {file_name} already exist. Skipping training.")

# Generating output for the specified file
print(f"Generating output for {file_name}...")
source_path = os.path.join(source_dir, file_name)

# Ensure the model and scalers exist
if os.path.exists(model_path) and os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path):
    # Load the model and scalers for the current file
    model = load_model(model_path)
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

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

        depth_image_path = os.path.join(image_output_dir, file_name.replace('.npz', '.png'))
        depth_array = converted_depths[0]

        depth_image = np.clip(255.0 / depth_array.max() * (depth_array - depth_array.min()), 0, 255).astype(np.uint8)
        imageio.imwrite(depth_image_path, depth_image)
else:
    print(f"Model or scalers for {file_name} not found.")
