# Run this after running surfaces1 and triangulation2

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load triangles and scores from pickle file
with open('triangles_and_scores.pkl', 'rb') as f:
    data = pickle.load(f)

triangles = data['triangles']
scores = data['scores']

# Normalize the scores for color mapping
min_score = min(scores)
max_score = max(scores)
norm_scores = (scores - min_score) / (max_score - min_score) if max_score > min_score else np.zeros_like(scores)

# Convert triangles to 2D (x and y coordinates)
triangles_2d = triangles[:, :, :2]

"""
# 1. 2D Plot of Reflective Surfaces (Already Done)
fig, ax = plt.subplots()
polygons = PolyCollection(triangles_2d, array=norm_scores, cmap='plasma', edgecolors='black', alpha=0.7)
ax.add_collection(polygons)
ax.set_xlim(triangles_2d[:, :, 0].min(), triangles_2d[:, :, 0].max())
ax.set_ylim(triangles_2d[:, :, 1].min(), triangles_2d[:, :, 1].max())
plt.colorbar(polygons, ax=ax, label='Reflectivity Score')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('Top Reflective Surfaces (2D)')
plt.show()

# 2. 3D Surface Plot with Color Mapping
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
poly3d = Poly3DCollection(triangles, facecolors=plt.cm.plasma(norm_scores), edgecolors='k', alpha=0.7)
ax.add_collection3d(poly3d)
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z (Elevation)')
ax.set_title('Top Reflective Surfaces (3D)')
ax.set_xlim(triangles[:, :, 0].min(), triangles[:, :, 0].max())
ax.set_ylim(triangles[:, :, 1].min(), triangles[:, :, 1].max())
ax.set_zlim(triangles[:, :, 2].min(), triangles[:, :, 2].max())
plt.colorbar(polygons, ax=ax, label='Reflectivity Score')
plt.show()

# 3. Scatter Plot of Reflectivity vs. Area
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

# 4. Heatmap of Reflectivity Scores (Using X, Y grid positions)
grid_size = 100  # Define grid size for heatmap
heatmap, xedges, yedges = np.histogram2d(triangles[:, :, 0].flatten(), triangles[:, :, 1].flatten(), bins=grid_size, weights=scores)
plt.imshow(heatmap.T, origin='lower', cmap='plasma', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.colorbar(label='Average Reflectivity Score')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Heatmap of Reflectivity Scores')
plt.show()

# 5. Bar Graph of Top 10 Reflective Triangles  WORKS FINE
def triangle_area(triangle):
    a = np.linalg.norm(triangle[1] - triangle[0])
    b = np.linalg.norm(triangle[2] - triangle[1])
    c = np.linalg.norm(triangle[2] - triangle[0])
    s = (a + b + c) / 2
    return np.sqrt(s * (s - a) * (s - b) * (s - c))

areas = np.array([triangle_area(tri) for tri in triangles])

top_n = 10
sorted_indices = np.argsort(scores)[-top_n:]
top_scores = scores[sorted_indices]
top_areas = areas[sorted_indices]
plt.bar(np.arange(top_n), top_scores, tick_label=[f'Triangle {i}' for i in sorted_indices])
plt.xlabel('Triangle Index')
plt.ylabel('Reflectivity Score')
plt.title(f'Top {top_n} Reflective Triangles')
plt.show()

# 6. Contour Plot of Reflectivity Scores
plt.tricontourf(triangles_2d[:, :, 0].flatten(), triangles_2d[:, :, 1].flatten(), scores, cmap='plasma')
plt.colorbar(label='Reflectivity Score')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Contour Plot of Reflectivity Scores')
plt.show()

"""

# 7. Cumulative Distribution Function (CDF) of Scores
sorted_scores = np.sort(scores)
cdf = np.arange(len(sorted_scores)) / float(len(sorted_scores))
plt.plot(sorted_scores, cdf)
plt.xlabel('Reflectivity Score')
plt.ylabel('Cumulative Probability')
plt.title('CDF of Reflectivity Scores')
plt.show()

# 8. Violin Plot of Reflectivity Scores
plt.violinplot(scores, showmedians=True)
plt.xlabel('Reflective Surfaces')
plt.ylabel('Reflectivity Score')
plt.title('Violin Plot of Reflectivity Scores')
plt.show()

# 9. Histogram of Reflectivity Score Distribution
plt.hist(scores, bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel('Reflectivity Score')
plt.ylabel('Frequency')
plt.title('Distribution of Reflectivity Scores')
plt.show()

