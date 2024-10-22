# Run this after running surfaces1

import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Start time for measuring runtime
start_time = time.time()

def compute_reflectivity_score(triangle, source, receiver):
    p1, p2, p3 = triangle
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal_magnitude = np.linalg.norm(normal)
    if normal_magnitude == 0:
        return 0  # Degenerate triangle, no reflectivity
    normal = normal / normal_magnitude
    
    vec_to_source = source - p1
    vec_to_receiver = receiver - p1
    vec_to_source = vec_to_source / np.linalg.norm(vec_to_source)
    vec_to_receiver = vec_to_receiver / np.linalg.norm(vec_to_receiver)
    
    dot_source = np.dot(normal, vec_to_source)
    dot_receiver = np.dot(normal, vec_to_receiver)
    
    score = max(0, dot_source) * max(0, dot_receiver)
    return score

def simulate_moving_source(triangles, receiver, start_position, end_position, steps):
    positions = np.linspace(start_position, end_position, steps)
    all_scores = []
    
    for i, source in enumerate(positions):
        print(f"Time step {i+1}: Source position = {source}")
        scores = []
        for triangle in triangles:
            score = compute_reflectivity_score(triangle, source, receiver)
            scores.append(score)
        all_scores.append(scores)
    
    return np.array(all_scores), positions

if __name__ == "__main__":
    # Load triangles from the pickle file
    with open('triangles.pkl', 'rb') as f:
        triangles = pickle.load(f)
    '''
    # Define the positions of the receiver and source movement
    receiver = np.array([10, 20, 5])  # Fixed receiver position
    start_position = np.array([0, 0, 100])  # Initial source position
    end_position = np.array([500, 500, 100])  # Final source position
    steps = 50  # Number of steps for the source to move
    '''
    # Define the positions of the receiver and source movement
    receiver = np.array([100, 200, 50])  # Fixed receiver position
    start_position = np.array([500, 500, 100])  # Initial source position
    end_position = np.array([50000, 50000, 100])  # Final source position
    steps = 10  # Number of steps for the source to move


    # Simulate the moving source
    all_scores, positions = simulate_moving_source(triangles, receiver, start_position, end_position, steps)

    # Save the results to a pickle file
    with open('reflectivity_over_time.pkl', 'wb') as f:
        data = {'scores': all_scores, 'positions': positions}
        pickle.dump(data, f)

    # End timer and print runtime
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total runtime: {runtime:.2f} seconds")

    # Optionally, visualize the reflectivity changes over time for a specific triangle
    triangle_index = 100  # Index of the triangle to track over time
    plt.plot(range(steps), all_scores[:, triangle_index])
    plt.xlabel('Time Step')
    plt.ylabel('Reflectivity Score')
    plt.title(f'Reflectivity Score for Triangle {triangle_index} Over Time')
    plt.show()

