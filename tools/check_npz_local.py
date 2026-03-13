import numpy as np
import sys

def check_npz(path):
    print(f"Checking {path}...")
    try:
        data = np.load(path)
        print("Keys:", data.files)
        
        for k in ['keypoints_3d', 'keypoints_2d', 'sensor']:
            if k in data:
                val = data[k]
                if k != 'sensor':
                    conf = val[..., -1]
                    print(f"{k} shape: {val.shape}")
                    print(f"{k} confidence mean: {np.mean(conf)}")
                    print(f"{k} coord mean: {np.mean(val[..., :-1])}")
                    # Check if all zeros
                    if np.all(val == 0):
                        print(f"!!! {k} is ALL ZEROS")
                else:
                    print(f"{k} shape: {val.shape}")
                    print(f"{k} mean: {np.mean(val)}")
            else:
                print(f"{k} NOT FOUND")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_npz(sys.argv[1])
