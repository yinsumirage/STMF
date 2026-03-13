import torch
import numpy as np

def inspect_debug_data(path):
    print(f"Inspecting {path}...")
    try:
        data = torch.load(path, map_location='cpu')
        print("Keys in top level:", data.keys())
        
        data_batch = data['data_batch']
        print("\n--- Data Batch Inspection ---")
        if isinstance(data_batch, dict):
            print(f"Keys in data_batch: {data_batch.keys()}")
            
            for key in ['keypoints_2d', 'keypoints_3d', 'box_size', '_scale']:
                if key in data_batch:
                    val = data_batch[key]
                    print(f"\nKey: {key}")
                    if isinstance(val, torch.Tensor):
                        print(f"  Shape: {val.shape}")
                        print(f"  Mean: {val.mean().item():.6f}")
                    else:
                        print(f"  Value: {val}")
                    
                    # Check confidence (last channel)
                    if val.dim() >= 2:
                        conf = val[..., -1]
                        print(f"  Confidence Mean: {conf.mean().item():.6f}")
                        
                    if torch.all(val == 0):
                        print(f"  !!! CRITICAL: {key} is ALL ZEROS")
                else:
                    print(f"\nKey: {key} NOT FOUND in data_batch")
        else:
            print(f"data_batch is not a dict, it's a {type(data_batch)}")

        output = data['output']
        print("\n--- Model Output Inspection ---")
        if isinstance(output, dict):
            print(f"Keys in output: {output.keys()}")
            if 'pred_keypoints_2d' in output:
                pred_2d = output['pred_keypoints_2d']
                print(f"  pred_keypoints_2d Shape: {pred_2d.shape}")
                print(f"  pred_keypoints_2d Mean: {pred_2d.mean().item():.6f}")
        
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    inspect_debug_data('/home/mirage/STMF/tools/debug_training_data.pt')
