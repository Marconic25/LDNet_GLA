import numpy as np

# Assuming the successful load command from the previous steps
file_name = 'T20_20samples.npy'

try:
    # 1. Load the data using the security override (allow_pickle=True)
    data_loaded = np.load(file_name, allow_pickle=True) 
    
    # 2. Extract the single dictionary using .item()
    data_dict = data_loaded.item()

    print(f"🔬 **Inspecting the 'x' key from the dictionary**")
    print("-" * 40)
    
    if 'x' in data_dict:
        # 3. Access the NumPy array stored under the key 'x'
        input_data = data_dict['x']
        
        # 4. Print the shape and type of the array
        print(f"Key 'x' Data Type: {type(input_data)}")
        print(f"Final Shape of Input Data ('x'): {input_data.shape}")
        
        # 5. Access and preview a sample (e.g., the first sample, index 0)
        if len(input_data.shape) > 1:
            print(f"Shape of a single sample (input_data[0]): {input_data[0].shape}")
        else:
            print(f"Value Preview (First 5 elements): {input_data[:5]}")
            
    else:
        print("❌ Error: The key 'x' was not found in the dictionary.")

except FileNotFoundError:
    print(f"❌ Error: The file '{file_name}' was not found.")
except Exception as e:
    print(f"❌ An error occurred during final access: {e}")