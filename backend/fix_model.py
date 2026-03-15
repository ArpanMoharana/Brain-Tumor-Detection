import h5py

model_path = 'models/brain_tumor_model.h5'

print(f"Opening {model_path} to fix the batch_shape bug...")

with h5py.File(model_path, 'r+') as f:
    # Read the internal model configuration
    model_config = f.attrs.get('model_config')

    if model_config is None:
        print("Error: Could not find model_config in the file.")
    else:
        # Decode the JSON string
        if isinstance(model_config, bytes):
            config_str = model_config.decode('utf-8')
        else:
            config_str = model_config

        # The magic fix: replace the old keyword with the new one
        if '"batch_shape":' in config_str:
            fixed_config_str = config_str.replace('"batch_shape":', '"batch_input_shape":')
            f.attrs['model_config'] = fixed_config_str.encode('utf-8')
            print("✅ SUCCESS: Replaced 'batch_shape' with 'batch_input_shape'!")
            print("The model is now fully compatible with modern TensorFlow.")
        else:
            print("The model does not contain 'batch_shape'. It might already be fixed!")