# an implementation in python for determining the Wadell roundness based on the work 
    #W. Fei, G.A. Narsilio, M.M. Disfani, 
    # Impact of three-dimensional sphericity and roundness on heat transfer in granular materials, 
    # Powder Technol. 355 (2019) 770–781.
# Author: René Rekers
import numpy as np
import pyvista as pv
from scipy.ndimage import distance_transform_edt
from skimage import measure
def wadellroundness(voxel_mask, smoothiters = 100):
    verts, faces, _, _ = measure.marching_cubes(voxel_mask, level=0.5)
    faces_pv = np.hstack(np.c_[np.full(len(faces), 3), faces])
    mesh = pv.PolyData(verts, faces_pv)
    mesh = mesh.smooth_taubin(n_iter=smoothiters,pass_band=0.1)
    # 3. Taubin Smoothing (Non-shrinking)
    # Higher n_iter for small radii helps remove the heavy voxel blockiness
    dt = distance_transform_edt(voxel_mask)
    r_inscribed = dt.max()

    #print(f"Radius of Inscribed Sphere (R_i): {r_inscribed:.4f}")

    # 2. Calculate Principal Curvatures
    # We need 'Maximum' curvature to find the sharpest points (smallest radii)
    # k_max corresponds to the 1/r_min we need for Wadell
    k_max = mesh.curvature(curv_type="Maximum")

    # Avoid division by zero and handle flat surfaces (k=0)
    # If k is negative or zero, the radius is effectively infinite (not a corner)
    k_max_safe = np.where(k_max <= 0, 1e-10, k_max)
    r_min = 1.0 / k_max_safe

    # 3. Identify Wadell Corners
    # A vertex is a corner if its local radius of curvature < inscribed radius
    corner_mask = r_min < r_inscribed
    r_corners = r_min[corner_mask]

    # 4. Calculate Roundness
    # Average of radii of all corners divided by the inscribed radius
    if(len(r_corners)>1):
        return np.mean(r_corners) / r_inscribed
    else:
        return 1
