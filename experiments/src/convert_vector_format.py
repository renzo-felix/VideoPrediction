import pickle
import sys
from pathlib import Path

def convert(input_path, output_path):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # train_steering.py outputs 'steering_vector_normalized' and 'training_correlation'
    layer = data['layer']
    dv = data['steering_vector_normalized']
    r = data['training_correlation']
    
    formatted_data = {
        'direction_results': {
            layer: {
                'direction_vector': dv,
                'pearson_r': r
            }
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(formatted_data, f)
    
    print(f"Converted {input_path} to {output_path} for layer {layer}")

if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2])
