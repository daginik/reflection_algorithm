import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import colors

# Start timer for measuring run time
start_time = time.time()

# Load triangles and scores from pickle file
with open('triangles_and_scores.pkl', 'rb') as f:
    data = pickle.load(f)

triangles = data['triangles']
scores = data['scores']

# Normalize the scores for color mapping
min_score = min(scores)
max_score = max(scores)

print(f"Min score: {min_score}, Max score: {max_score}")

# Handle cases where all scores are the same
if max_score - min_score > 0:
    norm_scores = (scores - min_score) / (max_score - min_score)
else:
    norm_scores = np.zeros_like(scores)  # Assign all scores to 0 if there's no variation

# Convert triangles to 2D (only x and y coordinates)
triangles_2d = triangles[:, :, :2]

# Create the plot
fig, ax = plt.subplots()

# Create a normalization object to scale the colors
norm = colors.Normalize(vmin=0, vmax=1)

# Create a PolyCollection to plot triangles
polygons = PolyCollection(
    triangles_2d, 
    array=norm_scores, 
    cmap='plasma',  # Choose a color map that works well with variations
    edgecolors='black', 
    alpha=0.7       # Add transparency to lighten the plot
)

# Add the collection to the plot
ax.add_collection(polygons)

# Set axis limits based on the triangles
ax.set_xlim(triangles_2d[:, :, 0].min(), triangles_2d[:, :, 0].max())
ax.set_ylim(triangles_2d[:, :, 1].min(), triangles_2d[:, :, 1].max())

# Add a color bar to show the score values
plt.colorbar(polygons, ax=ax, label='Reflectivity Score')

# Set labels and title
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('Top Reflective Surfaces')

plt.show()

# End timer
end_time = time.time()
runtime = end_time - start_time
print(f"Total runtime: {runtime:.2f} seconds")

"""
# Optionally, plot a histogram of the scores to visualize the distribution
plt.figure()
plt.hist(scores, bins=50, color='blue')
plt.title('Distribution of Reflectivity Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot of Reflectivity vs. Area
def triangle_area(triangle):
    a = np.linalg.norm(triangle[1] - triangle[0])
    b = np.linalg.norm(triangle[2] - triangle[1])
    c = np.linalg.norm(triangle[2] - triangle[0])
    s = (a + b + c) / 2
    return np.sqrt(s * (s - a) * (s - b) * (s - c))

areas = np.array([triangle_area(tri) for tri in triangles])
plt.scatter(areas, scores, c=scores, cmap='plasma', edgecolors='black')
plt.colorbar(label='Reflectivity Score')
plt.xlabel('Triangle Area')
plt.ylabel('Reflectivity Score')
plt.title('Scatter Plot of Reflectivity vs. Area')
plt.show()
"""