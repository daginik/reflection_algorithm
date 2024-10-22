# Reflection based scoring

import time
import pickle
import numpy as np

# Start time for measuring runtime
start_time = time.time()

def compute_reflectivity_score(triangle, source, receiver):
    # Extract the three vertices of the triangle
    p1, p2, p3 = triangle
    
    # Calculate the vectors along the edges of the triangle
    v1 = p2 - p1
    v2 = p3 - p1

    # Calculate the normal vector to the triangle (cross product of edge vectors)
    normal = np.cross(v1, v2)
    
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    
    # Calculate the vectors from the triangle to the source and receiver
    vec_to_source = source - p1
    vec_to_receiver = receiver - p1
    
    # Normalize these vectors
    vec_to_source = vec_to_source / np.linalg.norm(vec_to_source)
    vec_to_receiver = vec_to_receiver / np.linalg.norm(vec_to_receiver)
    
    # Dot products to measure alignment of the normal vector
    dot_source = np.dot(normal, vec_to_source)
    dot_receiver = np.dot(normal, vec_to_receiver)
    
    # The score is the product of the two dot products
    # We want both to be positive, meaning the triangle "faces" both the source and receiver
    score = max(0, dot_source) * max(0, dot_receiver)
    
    return score

if __name__ == "__main__":
    # Load triangles from the pickle file
    with open('triangles.pkl', 'rb') as f:
        triangles = pickle.load(f)
    
    print(f"Loaded {len(triangles)} triangles for scoring.")
    
    # Define the positions of the source and receiver
    source = np.array([250, 250, 200])  # Replace with actual source coordinates
    receiver = np.array([10, 20, 5])  # Replace with actual receiver coordinates

    # Compute reflectivity scores for the triangles
    scores = []
    for triangle in triangles:
        score = compute_reflectivity_score(triangle, source, receiver)
        scores.append(score)

    scores = np.array(scores)
    print(f"Computed reflectivity scores for {len(scores)} triangles.")

    # Save the triangles and their scores in a pickle file
    with open('triangles_and_scores.pkl', 'wb') as f:
        data = {'triangles': triangles, 'scores': scores}
        pickle.dump(data, f)
    
    print("Saved triangles and scores to 'triangles_and_scores.pkl'.")

    # End timer and print runtime
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total runtime: {runtime:.2f} seconds")