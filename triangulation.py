
# triangulation.py

import numpy as np

# Load triangles from the pickle file generated in surfaces.py
import pickle
with open('triangles.pkl', 'rb') as f:
    triangles = pickle.load(f)

# Function to calculate the surface normal of a triangle
def calculate_normal(triangle):
    A, B, C = triangle
    normal = np.cross(B - A, C - A)  # Cross product of two edges of the triangle
    normal = normal / np.linalg.norm(normal)  # Normalize the normal vector
    return normal

# Function to calculate the reflected signal direction
def reflect_signal(signal_dir, normal):
    reflected_dir = signal_dir - 2 * np.dot(signal_dir, normal) * normal
    return reflected_dir

# Function to check if the reflected signal hits the receiver
def hits_receiver(triangle, reflected_dir, receiver_pos):
    triangle_center = triangle.mean(axis=0)
    direction_to_receiver = receiver_pos - triangle_center
    direction_to_receiver = direction_to_receiver / np.linalg.norm(direction_to_receiver)  # Normalize
    
    alignment = np.dot(reflected_dir, direction_to_receiver)
    return alignment > 0.9  # Arbitrary threshold; tweak as needed

# Function to compute a score for each triangle based on reflection
def compute_reflection_score(triangle, reflected_dir, source_pos, receiver_pos):
    triangle_center = triangle.mean(axis=0)
    distance_to_source = np.linalg.norm(triangle_center - source_pos)
    distance_to_receiver = np.linalg.norm(triangle_center - receiver_pos)
    
    score = 1 / (distance_to_source + distance_to_receiver)  # Simple inverse distance weighting
    return score

# Function to rank the triangles based on their reflective ability
def rank_triangles(triangles, source_pos, receiver_pos):
    scores = []
    for triangle in triangles:
        normal = calculate_normal(triangle)
        signal_dir = triangle.mean(axis=0) - source_pos
        reflected_dir = reflect_signal(signal_dir, normal)
        
        if hits_receiver(triangle, reflected_dir, receiver_pos):
            score = compute_reflection_score(triangle, reflected_dir, source_pos, receiver_pos)
            scores.append((triangle, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Output the top 20 triangles
    top_triangles = scores[:20]
    for idx, (triangle, score) in enumerate(top_triangles):
        print(f"Rank {idx + 1}: Triangle {triangle} with score {score}")

    return top_triangles

if __name__ == "__main__":
    source_pos = np.array([10, 10, 10])  # Example source position
    receiver_pos = np.array([20, 20, 5])  # Example receiver position

    # Rank triangles and get the top 20 reflective surfaces
    top_triangles = rank_triangles(triangles, source_pos, receiver_pos)
 
    # Print the top 20 triangles
    for i, (triangle, score) in enumerate(top_triangles):
        print(f"Rank {i+1}: Triangle {triangle} with score {score}")