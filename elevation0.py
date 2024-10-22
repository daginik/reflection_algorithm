import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load triangles and scores from pickle file
with open('triangles_and_scores.pkl', 'rb') as f:
    data = pickle.load(f)

triangles = data['triangles']  # Shape: (num_triangles, 3, 3) where each triangle has 3 points, and each point has x, y, z
scores = data['scores']

# Select triangles with high reflection scores
threshold = np.percentile(scores, 95)  # Adjust the threshold to consider top 5% most reflective surfaces
high_reflective_triangles = triangles[scores >= threshold]

# Extract the average elevation (z-coordinate) of each triangle
average_elevations = np.mean(high_reflective_triangles[:, :, 2], axis=1)

# Analyze elevation distribution
mean_elevation = np.mean(average_elevations)
median_elevation = np.median(average_elevations)
min_elevation = np.min(average_elevations)
max_elevation = np.max(average_elevations)

print(f"Mean Elevation: {mean_elevation}")
print(f"Median Elevation: {median_elevation}")
print(f"Min Elevation: {min_elevation}")
print(f"Max Elevation: {max_elevation}")

# Plot histogram of elevation values
plt.figure(figsize=(10, 6))
plt.hist(average_elevations, bins=50, color='green', alpha=0.7)
plt.title('Elevation Distribution of Highly Reflective Surfaces')
plt.xlabel('Elevation (z-coordinate)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot a scatter plot of average elevation vs. reflectivity score
high_reflective_scores = scores[scores >= threshold]

plt.figure(figsize=(10, 6))
plt.scatter(average_elevations, high_reflective_scores, color='blue', alpha=0.5)
plt.title('Elevation vs Reflectivity Scores for Highly Reflective Surfaces')
plt.xlabel('Average Elevation (z-coordinate)')
plt.ylabel('Reflectivity Score')
plt.grid(True)
plt.show()
