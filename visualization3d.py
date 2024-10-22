import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import colors
import time

# Start runtime tracking
start_time = time.time()

# Load triangles and scores from pickle file
with open('triangles_and_scores.pkl', 'rb') as f:
    data = pickle.load(f)

triangles = data['triangles']
scores = data['scores']

# Normalize the scores for color mapping
min_score = min(scores)
max_score = max(scores)
norm_scores = (scores - min_score) / (max_score - min_score)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract the x, y, z coordinates from triangles
x = triangles[:, :, 0]
y = triangles[:, :, 1]
z = triangles[:, :, 2]

# Create a Poly3DCollection for plotting the triangles
polygons = Poly3DCollection(
    triangles, 
    array=norm_scores, 
    cmap='plasma',  # Color map for reflective surfaces
    edgecolor='k', 
    alpha=0.7  # Transparency for better view
)

# Add collection to the 3D plot
ax.add_collection3d(polygons)

# Set axis labels and title
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Elevation')
ax.set_title('Top Reflective Surfaces in 3D')

# Set axis limits based on the range of coordinates
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(y), np.max(y))
ax.set_zlim(np.min(z), np.max(z))

# Add color bar to indicate the reflectivity scores
plt.colorbar(polygons, ax=ax, label='Reflectivity Score')

# Show the plot
plt.show()

# Calculate and display the total runtime
end_time = time.time()
runtime = end_time - start_time
print(f"Total runtime: {runtime:.2f} seconds")
