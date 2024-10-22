import time
import numpy as np
from osgeo import gdal
from scipy.spatial import Delaunay

# Start the timer to measure run time at the end
start_time = time.time()

def load_surface_data_from_tif(tif_file):
    # Open the DEM file
    ds = gdal.Open(tif_file)
    if ds is None:
        raise FileNotFoundError(f"Unable to open {tif_file}")

    # Read raster data
    band = ds.GetRasterBand(1)  # Assuming the elevation data is in the first band
    elevation_data = band.ReadAsArray()

    # Get geo-transform (to map pixel coordinates to real-world coordinates)
    transform = ds.GetGeoTransform()

    # Generate X, Y coordinates based on the DEM's geo-transform
    x = np.arange(0, elevation_data.shape[1]) * transform[1] + transform[0]
    y = np.arange(0, elevation_data.shape[0]) * transform[5] + transform[3]
    X, Y = np.meshgrid(x, y)

    # Flatten X, Y, and Z (elevation) arrays into points
    Z = elevation_data.flatten()
    points = np.column_stack([X.flatten(), Y.flatten(), Z])

    return points

def tessellate_surface(points):
    # Perform Delaunay triangulation on the surface points (use X and Y coordinates only)
    tri = Delaunay(points[:, :2])
    triangles = points[tri.simplices]
    return triangles

if __name__ == "__main__":
    #tif_file = r"C:\Users\selim\source\repos\Reflection_Project\The project\data\bogota.tif"  # .tif file path
    tif_file = r"C:\Users\selim\source\repos\Reflection_Project\The project\data\e_tile.tif"
    points = load_surface_data_from_tif(tif_file)
    triangles = tessellate_surface(points)
    print(f"Tessellated {len(triangles)} triangles from the DEM data.")

    # Save 'triangles' using pickle for further use
    import pickle
    with open('triangles.pkl', 'wb') as f:
        pickle.dump(triangles, f)
    
    print("Triangles saved to 'triangles.pkl'.")

    # End the timer
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total runtime: {runtime:.2f} seconds")