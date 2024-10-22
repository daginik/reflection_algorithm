# Run this after running surfaces1 and movingsource0

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the reflectivity scores over time from the pickle file
with open('reflectivity_over_time.pkl', 'rb') as f:
    data = pickle.load(f)

all_scores = data['scores']
positions = data['positions']

# Find the indices of multiple triangles with non-zero reflectivity
triangle_indices = []
for i in range(all_scores.shape[1]):  # Iterate over all triangles
    if np.any(all_scores[:, i] > 0.8):  # Check if reflectivity > 0 at any time step
        triangle_indices.append(i)
    if len(triangle_indices) >= 5:  # Limit to the first 5 non-zero triangles
        break

if triangle_indices:
    # Plot the reflectivity scores of selected triangles over time
    for triangle_index in triangle_indices:
        plt.plot(range(all_scores.shape[0]), all_scores[:, triangle_index], label=f'Triangle {triangle_index}')
    
    plt.xlabel('Time Step')
    plt.ylabel('Reflectivity Score')
    plt.title(f'Reflectivity Scores for Multiple Triangles Over Time')
    plt.legend()
    plt.show()
else:
    print("No triangles with non-zero reflectivity were found.")
