
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
    
    # Vector from triangle to source
    vec_to_source = source - p1
    vec_to_source = vec_to_source / np.linalg.norm(vec_to_source)
    
    # Compute the reflection of the source vector off the triangle
    reflection_dir = vec_to_source - 2 * np.dot(vec_to_source, normal) * normal
    
    # Vector from triangle to receiver
    vec_to_receiver = receiver - p1
    vec_to_receiver = vec_to_receiver / np.linalg.norm(vec_to_receiver)
    
    # Check alignment of reflection direction with receiver direction
    score = max(0, np.dot(reflection_dir, vec_to_receiver))
    
    

    
    return score

if __name__ == "__main__":
    # Load triangles from the pickle file
    with open('triangles.pkl', 'rb') as f:
        triangles = pickle.load(f)
    
    print(f"Loaded {len(triangles)} triangles for scoring.")
    
    # Define the positions of the source and receiver
    source = np.array([50, 50, 100])  # Replace with actual source coordinates
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