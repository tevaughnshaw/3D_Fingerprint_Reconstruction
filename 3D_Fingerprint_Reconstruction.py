import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
import open3d as o3d
import pyvista as pv
from vtkmodules.util.numpy_support import vtk_to_numpy
import pickle
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

def _to_xyz_array(x, name="points", trim_bad=False):
    """
    Convert x to an (N,3) numpy array of 3D points.

    If x is 1-D and len % 3 == 0 -> reshape(-1,3).
    If x is 1-D and len % 3 != 0 -> by default raise; if trim_bad True, drop trailing entries.
    If x is 2-D and shape[1]==3 -> assume OK.
    Otherwise try to ravel and reshape (with same divisibility rules).
    Returns np.empty((0,3)) if input is empty.
    """
    x = np.asarray(x)
    if x.size == 0:
        return np.empty((0, 3))

    # already in (N,3) form
    if x.ndim == 2 and x.shape[1] == 3:
        return x.copy()

    # 1-D flattened data
    if x.ndim == 1:
        if x.size % 3 == 0:
            return x.reshape(-1, 3)
        else:
            if trim_bad:
                keep = (x.size // 3) * 3
                print(f"[reshape] WARNING: 1D {name} size {x.size} not divisible by 3 — trimming to {keep}.")
                return x[:keep].reshape(-1, 3) if keep > 0 else np.empty((0, 3))
            else:
                raise ValueError(f"1D {name} length {x.size} not divisible by 3. Consider trim_bad=True to trim.")
    # higher dimensional or nested arrays -> try to ravel and reshape
    flat = x.ravel()
    if flat.size % 3 == 0:
        return flat.reshape(-1, 3)
    else:
        if trim_bad:
            keep = (flat.size // 3) * 3
            print(f"[reshape] WARNING: flattened {name} size {flat.size} not divisible by 3 — trimming to {keep}.")
            return flat[:keep].reshape(-1, 3) if keep > 0 else np.empty((0, 3))
        else:
            raise ValueError(f"Cannot convert {name} to (N,3) — flattened size {flat.size} not divisible by 3.")


# load data
SC_mesh = o3d.io.read_triangle_mesh("/Users/tevaughnshaw/3D_OCT_tools/topZip/SC meshes/LUK_023JH001M_SC_undistorted.ply")

contact_points = pickle.load(open("/Users/tevaughnshaw/3D_OCT_tools/topZip/tracked_points_undistorted.pkl", "rb"))

# diagnostic print to see the raw structure
print("contact_points type:", type(contact_points))
try:
    print("contact_points shape/len:", np.asarray(contact_points).shape)
except Exception:
    print("contact_points: could not show shape (non-array-like)")

# typical expected formats and how we handle them:
# 1) contact_points is a list/tuple of two entries: [entry, exit]
# 2) contact_points is an array of shape (2, N, 3) or (2, M) flattened etc.

entry_points = np.empty((0, 3))
exit_points = np.empty((0, 3))

# case A: list/tuple of length 2
if isinstance(contact_points, (list, tuple)) and len(contact_points) == 2:
    entry_raw, exit_raw = contact_points[0], contact_points[1]
    # try to convert; use trim_bad=True to avoid crashes when length % 3 != 0
    entry_points = _to_xyz_array(entry_raw, name="entry_points", trim_bad=True)
    exit_points = _to_xyz_array(exit_raw, name="exit_points", trim_bad=True)

else:
    arr = np.asarray(contact_points)
    # if shape is (2, N, 3) or (2, M), handle either entry/exit in first dimension
    if arr.ndim >= 2 and arr.shape[0] == 2:
        entry_points = _to_xyz_array(arr[0], name="entry_points", trim_bad=True)
        exit_points = _to_xyz_array(arr[1], name="exit_points", trim_bad=True)
    else:
        # fallback: attempt to interpret as flattened [entry..., exit...] but we can't know exact split
        # so we try to split in half (best-effort). If odd length, trim last element.
        flat = arr.ravel()
        total_pts = (flat.size // 3)
        if total_pts == 0:
            entry_points = np.empty((0, 3))
            exit_points = np.empty((0, 3))
        else:
            # split roughly in half
            half = total_pts // 2
            keep = half * 3
            entry_points = flat[:keep].reshape(-1, 3)
            exit_points = flat[keep:(half*3 + (total_pts-half)*3)].reshape(-1, 3) if flat.size > keep else np.empty((0, 3))
            print(f"[reshape] INFO: fallback flattened split — entry {entry_points.shape}, exit {exit_points.shape}")

print("entry_points.shape:", entry_points.shape)
print("exit_points.shape:", exit_points.shape)


# normalize
vertices = np.asarray(SC_mesh.vertices)
triangles = np.asarray(SC_mesh.triangles)
center = vertices.mean(axis=0)
scale = np.linalg.norm(vertices - center, axis=1).max()
vertices_norm = (vertices - center) / scale
entry_points_norm = (entry_points - center) / scale if entry_points.size else np.empty((0, 3))
exit_points_norm = (exit_points - center) / scale if exit_points.size else np.empty((0, 3))

# edge thresholding
edges = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
edges = np.unique(np.sort(edges, axis=1), axis=0)
edge_lengths = np.linalg.norm(vertices_norm[edges[:, 0]] - vertices_norm[edges[:, 1]], axis=1)
threshold = 2 * edge_lengths.mean()
tree = cKDTree(vertices_norm)

entry_dists, _ = (tree.query(entry_points_norm) if entry_points_norm.size else (np.array([]), np.array([])))
exit_dists, _ = (tree.query(exit_points_norm) if exit_points_norm.size else (np.array([]), np.array([])))

filtered_entry_points = entry_points_norm[entry_dists < threshold] if entry_points_norm.size else np.empty((0, 3))
filtered_exit_points = exit_points_norm[exit_dists < threshold] if exit_points_norm.size else np.empty((0, 3))

# compute minimum distance between entry points
entry_tree = cKDTree(filtered_entry_points)
nearest_dists, _ = entry_tree.query(filtered_entry_points, k=2)
min_dist = nearest_dists[:, 1].min()

# visualization setup
faces = np.hstack(np.c_[np.full(len(triangles), 3), triangles]).astype(np.int32)
SC_pv = pv.PolyData(vertices_norm, faces)
entry_pv = pv.PolyData(filtered_entry_points)
entry_pv['colors'] = np.tile([0, 0, 255], (len(filtered_entry_points), 1))
exit_pv = pv.PolyData(filtered_exit_points)
exit_pv['colors'] = np.tile([0, 0, 255], (len(filtered_exit_points), 1))

# box widget for cropping
box_bounds = {}

def capture_box(box):
    """
    Captures the bounds of the interactive box widget and stores them
    in a global dictionary `box_bounds` for later use in cropping the mesh and points.
    """
    box_bounds['bounds'] = box.bounds

def split_by_x_and_select_bottom(mesh, points):
    """
    Splits a mesh and a set of points along the median x-axis and
    selects the 'bottom' half. This version uses a boolean mask to avoid errors.
    """
    if mesh.n_points == 0 or len(points) == 0:
        return pv.PolyData(), np.empty((0, 3))

    x_center = np.median(mesh.points[:, 0])

    # create a boolean mask to split the points
    left_mask = mesh.points[:, 0] < x_center
    right_mask = mesh.points[:, 0] >= x_center

    # extract the left and right halves using the boolean mask
    left_mesh = mesh.extract_points(left_mask, adjacent_cells=True)
    right_mesh = mesh.extract_points(right_mask, adjacent_cells=True)

    # extract the surface to get a PolyData object
    left_mesh = left_mesh.extract_surface()
    right_mesh = right_mesh.extract_surface()

    left_points = points[points[:, 0] < x_center]
    right_points = points[points[:, 0] >= x_center]

    # heuristic to determine the 'bottom' mesh based on average Z-coordinate
    if left_mesh.n_points == 0 and right_mesh.n_points == 0:
        return pv.PolyData(), np.empty((0, 3))
    elif left_mesh.n_points == 0:
        return right_mesh, right_points
    elif right_mesh.n_points == 0:
        return left_mesh, left_points

    if left_mesh.center[2] < right_mesh.center[2]:
        return left_mesh, left_points
    else:
        return right_mesh, right_points


def reduce_clusters(points, eps=0.02, min_samples=3):
    """
    Robustly reduce clusters in `points`. Accepts inputs of shape:
      - (N, 3)
      - (3,) single point
      - 1-D length divisible by 3 -> reshape to (-1,3)

    Returns: array of cluster centroids shape (M,3) or empty (0,3).
    """
    points = np.asarray(points)

    # early exits
    if points.size == 0:
        return np.empty((0, 3))

    # normalize shapes into (N,3)
    if points.ndim == 1:
        if points.size == 3:
            points = points.reshape(1, 3)
        elif points.size % 3 == 0:
            points = points.reshape(-1, 3)
        else:
            # can't sensibly reshape: return empty and log
            print(f"[reduce_clusters] WARNING: 1D points length {points.size} not divisible by 3 -> returning empty.")
            return np.empty((0, 3))
    elif points.ndim == 2:
        if points.shape[1] != 3:
            # try to flatten then reshape if possible
            flat = points.ravel()
            if flat.size % 3 == 0:
                points = flat.reshape(-1, 3)
            else:
                print(f"[reduce_clusters] WARNING: points shape {points.shape} not compatible -> returning empty.")
                return np.empty((0, 3))
    else:
        # 3+ dims -> try to flatten sensibly
        flat = points.ravel()
        if flat.size % 3 == 0:
            points = flat.reshape(-1, 3)
        else:
            print(f"[reduce_clusters] WARNING: points ndim {points.ndim} not compatible -> returning empty.")
            return np.empty((0, 3))

    if len(points) == 0:
        return np.empty((0, 3))

    # DBSCAN clustering and centroid calculation
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_
    centroids = np.array([points[labels == lbl].mean(axis=0) for lbl in sorted(set(labels)) if lbl != -1])

    return centroids

# def visualize_entry_midpoint_web(entry_points, entry_mesh, k=3):
#     """
#     Visualizes the sweat duct entry points on the mesh and draws lines
#     connecting each point to its k-nearest neighbors. This helps in
#     understanding the local spatial arrangement of the sweat ducts and
#     the overall structure of the internal fingerprint ridge.
#     """
#
#     if len(entry_points) == 0:
#         return
#     tree = cKDTree(entry_points)
#     lines = [[i, j] for i, pt in enumerate(entry_points) for j in tree.query(pt, k=k + 1)[1][1:]]
#     line_cells = np.array([[2, *line] for line in lines]).reshape(-1, 3)
#     line_mesh = pv.PolyData(entry_points, lines=pv.CellArray(line_cells))
#     p = pv.Plotter()
#     p.add_mesh(entry_mesh, color='lightgray', opacity=0.4)
#     p.add_points(entry_points, color='green', point_size=12, render_points_as_spheres=True)
#     p.add_mesh(line_mesh, color='blue', line_width=1)
#     p.add_axes()
#     p.add_title(f"SD Midpoints with {k}-Nearest Neighbors")
#     p.show()

def cells_from_point_ids(mesh, point_ids):
    """
    Return unique cell indices that reference any of the given point indices.
    Works across different PyVista versions / PolyData representations.
    """
    point_ids = np.atleast_1d(point_ids).astype(int)
    if point_ids.size == 0:
        return np.array([], dtype=int)

    # 1) fast path: mesh.faces (common for PolyData; triangulated meshes -> groups of 4: [3, i,j,k])
    if hasattr(mesh, "faces") and mesh.faces is not None and len(mesh.faces) > 0:
        faces = np.asarray(mesh.faces)
        # if triangulated, faces can be reshaped to (-1,4) => [3, i,j,k]
        if faces.size % 4 == 0:
            try:
                tris = faces.reshape(-1, 4)[:, 1:4].astype(int)
                mask = np.isin(tris, point_ids).any(axis=1)
                return np.where(mask)[0].astype(int)
            except Exception:
                # fallback to general parser below
                pass

        # general parser for heterogeneous polygons (safe)
        cell_ids = []
        i = 0
        cid = 0
        n = faces.size
        while i < n:
            nv = int(faces[i])
            pts = faces[i + 1 : i + 1 + nv].astype(int)
            if np.intersect1d(pts, point_ids).size > 0:
                cell_ids.append(cid)
            i += nv + 1
            cid += 1
        return np.array(cell_ids, dtype=int)

    # 2) cell_connectivity + offset (some versions)
    if hasattr(mesh, "cell_connectivity") and hasattr(mesh, "offset"):
        cell_data = np.asarray(mesh.cell_connectivity)
        offsets = np.asarray(mesh.offset)
        n_cells = int(mesh.n_cells)
        cell_ids = []
        for cid in range(n_cells):
            start = int(offsets[cid])
            end = int(offsets[cid + 1]) if cid + 1 < n_cells else len(cell_data)
            pts = cell_data[start:end].astype(int)
            if np.intersect1d(pts, point_ids).size > 0:
                cell_ids.append(cid)
        return np.array(cell_ids, dtype=int)

    # 3) cells (older interface)
    if hasattr(mesh, "cells") and mesh.cells is not None and len(mesh.cells) > 0:
        cells = np.asarray(mesh.cells)
        cell_ids = []
        i = 0
        cid = 0
        n = cells.size
        while i < n:
            nv = int(cells[i])
            pts = cells[i + 1 : i + 1 + nv].astype(int)
            if np.intersect1d(pts, point_ids).size > 0:
                cell_ids.append(cid)
            i += nv + 1
            cid += 1
        return np.array(cell_ids, dtype=int)

    # 4) fallback using VTK API
    try:

        vtk_polys = mesh.GetPolys().GetData()
        arr = vtk_to_numpy(vtk_polys)
        # parse arr like faces
        cell_ids = []
        i = 0
        cid = 0
        n = arr.size
        while i < n:
            nv = int(arr[i])
            pts = arr[i + 1 : i + 1 + nv].astype(int)
            if np.intersect1d(pts, point_ids).size > 0:
                cell_ids.append(cid)
            i += nv + 1
            cid += 1
        return np.array(cell_ids, dtype=int)
    except Exception:
        # if nothing works, give a helpful error with introspection info
        available = [a for a in ("faces", "cells", "cell_connectivity", "offset") if hasattr(mesh, a)]
        raise RuntimeError(
            "Could not locate cell connectivity on mesh. "
            f"Available attributes: {available}. "
            "Please print `dir(mesh)` and your pyvista.__version__ and share them if this persists."
        )

def iterative_curvature_filter(
        mesh,
        iterations=2,
        percentile=99.9,
        smooth_passes=15,
        lambda_factor=0.1,
        curvature_type="mean"
):
    """
    Iteratively filters out high-curvature points and re-smooths the mesh.

    This function handles progressive mesh cleaning by repeatedly:
    1. Ensuring the mesh is a clean, triangulated PolyData object.
    2. Computing curvature.
    3. Identifying and removing cells connected to high-curvature points (top 5% by default).
    4. Post-smoothing the mesh to remove new irregularities.

    Returns the final cleaned mesh.
    """
    current_mesh = mesh.copy()

    for i in range(iterations):
        print(f"\n[ITERATION {i + 1}/{iterations}] Starting iterative filter...")

        # ensure the mesh is a clean, triangulated PolyData object at the start of each loop
        current_mesh = current_mesh.extract_surface().triangulate()
        if current_mesh.n_points == 0:
            print(f"[ITERATION {i + 1}] Mesh is empty. Exiting loop.")
            break

        # compute curvature on the current mesh
        curv = current_mesh.curvature(curv_type=curvature_type)
        cutoff = np.percentile(np.abs(curv), percentile)
        bad_curv_mask = np.abs(curv) > cutoff

        if not bad_curv_mask.any():
            print(f"[ITERATION {i + 1}] No more high-curvature points to remove.")
            break

        # remove high-curvature cells
        bad_point_ids = np.where(bad_curv_mask)[0]
        bad_cell_ids = cells_from_point_ids(current_mesh, bad_point_ids)
        bad_cell_ids = np.unique(bad_cell_ids)

        if bad_cell_ids.size > 0:
            keep_mask = np.ones(current_mesh.n_cells, dtype=bool)
            keep_mask[bad_cell_ids] = False
            current_mesh = current_mesh.extract_cells(np.where(keep_mask)[0])
            print(
                f"[ITERATION {i + 1}] Removed {len(bad_point_ids)} points, {len(bad_cell_ids)} cells. New triangles: {current_mesh.n_cells}")
        else:
            print(f"[ITERATION {i + 1}] No cells to remove.")
            break

        # post-smoothing the mesh after removal
        current_mesh = current_mesh.extract_surface().triangulate()
        current_mesh = current_mesh.smooth_taubin(
            n_iter=smooth_passes,
            pass_band=lambda_factor
        )

    return current_mesh

# leaving here for single point selection and local neighborhood analysis
# def run_statistical_analysis(mesh, center_point, neighbors):
#     """
#     Performs and visualizes a comprehensive statistical analysis of the local
#     geometry around a selected point and its neighbors. It calculates and
#     displays the curvature of the mesh, a heatmap of pairwise distances
#     between points, and a bar plot of normal vector alignments. It also
#     runs statistical tests (t-test, Shapiro-Wilk) to quantify these properties.
#     """
#
#     all_points = np.vstack([center_point, neighbors])
#     labels = ['Center'] + [f'Neighbor {i + 1}' for i in range(len(neighbors))]
#     mesh2 = iterative_curvature_filter(
#         mesh
#     )
#     # 2. Repair the small holes left by the filtering
#     if mesh2.n_points > 0:
#         # A reasonable hole size for a normalized mesh is a good starting point.
#         # You may need to adjust this value based on your specific data.
#         hole_size = 0.5
#         print(f"\n[INFO] Filling holes with max size: {hole_size}")
#         repaired_mesh = mesh2.fill_holes(hole_size=hole_size)
#     else:
#         repaired_mesh = mesh2.copy()  # In case the mesh is empty
#
#     curv = repaired_mesh.curvature(curv_type='mean')
#     low, high = np.percentile(curv, [2, 98])
#     cutoff_viz = np.percentile(np.abs(curv), 99.9)  # Example: use a very high percentile for visualization
#     bad_curv_mask = np.abs(curv) > cutoff_viz
#     curvature_plotter = pv.Plotter()
#     scalar_bar_args = {
#         'title': 'Mean Curvature',
#         'position_x': 0.05,
#         'position_y': 0.05,
#         'title_font_size': 16,
#         'label_font_size': 14
#     }
#     curvature_plotter.add_mesh(
#         repaired_mesh,
#         scalars=curv,
#         clim=[low, high],
#         cmap='viridis',
#         show_scalar_bar=True,
#         show_edges=False,
#         opacity=0.8,
#         scalar_bar_args=scalar_bar_args
#     )
#     scalar_bar_actor = curvature_plotter.scalar_bars['Mean Curvature']
#     scalar_bar_actor.GetBackgroundProperty().SetOpacity(0.0)
#     curvature_plotter.add_points(center_point, color='white', point_size=15, render_points_as_spheres=True,
#                                  label='Selected Point')
#     curvature_plotter.add_points(neighbors, color='black', point_size=10, render_points_as_spheres=True,
#                                  label='Neighbors')
#
#     if np.any(bad_curv_mask):
#         curvature_plotter.add_points(repaired_mesh.points[bad_curv_mask], color='red', point_size=10,
#                                      render_points_as_spheres=True, label='High Curvature')
#
#     curvature_plotter.add_title("Curvature Analysis")
#     curvature_plotter.add_legend()
#     curvature_plotter.show(auto_close=False)
#
#
#     # spatial distribution analysis (heatmap)
#     dist_matrix = cdist(all_points, all_points)
#     plt.figure(figsize=(8, 7))
#     sns.heatmap(dist_matrix, cmap="magma", annot=True, fmt=".2f", linewidths=1.5, linecolor='white',
#                 xticklabels=labels, yticklabels=labels,
#                 cbar_kws={'label': 'Distance Scale'})
#     plt.title("Pairwise Euclidean Distance Heatmap")
#     plt.xticks(rotation=45, ha='right')
#     plt.yticks(rotation=0)
#     plt.tight_layout()
#     plt.show(block=False)
#
#
#     # orientation analysis (dot product bar plot) ---
#     repaired_mesh.compute_normals(inplace=True)
#     normal_tree = cKDTree(repaired_mesh.points)
#     _, point_indices = normal_tree.query(all_points)
#     all_normals = repaired_mesh['Normals'][point_indices]
#     center_normal = all_normals[0]
#     neighbor_normals = all_normals[1:]
#
#     dot_products = [np.dot(center_normal, n) for n in neighbor_normals]
#
#     plt.figure(figsize=(8, 6))
#     bars = plt.bar(range(len(dot_products)), dot_products, color='skyblue')
#     plt.axhline(y=1.0, color='r', linestyle='--', label='Perfect Alignment')
#     plt.xlabel("Neighbor Index")
#     plt.ylabel("Dot Product of Normals")
#     plt.title("Normal Vector Alignment with Center Point")
#     xtick_labels = [f'Neighbor {i + 1}' for i in range(len(dot_products))]
#     plt.xticks(range(len(dot_products)), xtick_labels, rotation=45, ha='right')
#     # add values inside each bar
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=10)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.show(block=False)
#
#     # run statistical tests and summary plot
#     print("\n--- Statistical Test Results ---")
#     curvature_of_points = curv[point_indices]
#     curvature_std_dev = np.std(curvature_of_points)
#     print(f"Curvature Distribution: Standard Deviation = {curvature_std_dev:.4f}")
#     t_stat_curv, p_value_curv = ttest_1samp(curvature_of_points, 0)
#     print(f"Curvature vs. Zero: t-statistic={t_stat_curv:.4f}, p-value={p_value_curv:.4e}")
#
#     neighbor_dists = cdist([center_point], neighbors)[0]
#     shapiro_w_dist, p_value_dist = shapiro(neighbor_dists)
#     print(f"Distance Uniformity (Shapiro-Wilk): W-statistic={shapiro_w_dist:.4f}, p-value={p_value_dist:.4e}")
#
#     t_stat_orient, p_value_orient = ttest_1samp(dot_products, 1)
#     print(f"Orientation vs. Flat: t-statistic={t_stat_orient:.4f}, p-value={p_value_orient:.4e}")
#
#     p_values = [p_value_curv, p_value_dist, p_value_orient]
#     test_names = ["Curvature vs. Zero", "Distance Uniformity", "Orientation vs. Flat"]
#
#     plt.figure(figsize=(8, 6))
#     bars = plt.bar(test_names, p_values, color=['skyblue', 'lightgreen', 'salmon'])
#     plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Threshold (p=0.05)')
#     plt.xlabel("Statistical Test")
#     plt.ylabel("p-value")
#     plt.title("Summary of Local Geometry Statistical Tests")
#     plt.ylim(0, 1)
#     plt.xticks(rotation=15, ha='right')
#     plt.legend()
#     plt.tight_layout()
#     plt.show(block=True)


# def select_and_analyze_single_point(mesh, points, k=8):
#     """
#     Provides an interactive interface for the user to select a single sweat
#     duct point on the mesh. Once a point is selected, it identifies its
#     k-nearest neighbors and passes this neighborhood to the
#     `run_statistical_analysis` function for detailed examination.
#     """
#     plotter = pv.Plotter()
#     point_tree = cKDTree(points)
#     plotter.add_mesh(mesh, color='lightgray', opacity=0.5, pickable=False)
#     initial_color = (0.0, 1.0, 1.0)  # Cyan
#     point_actors = [plotter.add_mesh(pv.Sphere(radius=0.01, center=pt), color=initial_color) for pt in points]
#     plotter.add_text("Rotate mesh. Press 's' to select.", font_size=16, position='upper_left')
#     selection_mode = [False]
#     selected_info = {'point': None,
#                      'actor': None,
#                      'neighbors': [],
#                      'neighbor_actors': [],
#                      'neighbor_colors': []
#                      }
#     def on_click(event):
#         if not selection_mode[0]:
#             print("Press 's' to start selecting.")
#             return
#         if selected_info['actor']:
#             selected_info['actor'].GetProperty().SetColor(initial_color)
#         for actor in selected_info['neighbor_actors']:
#             actor.GetProperty().SetColor(initial_color)
#         click_pos = plotter.pick_mouse_position()
#         if click_pos is None:
#             return
#
#         dist, idx = point_tree.query(click_pos)
#         selected_point = points[idx]
#         neighbor_dists, neighbor_indices = point_tree.query(selected_point, k=k + 1)
#         neighbors = points[neighbor_indices[1:]]
#
#         # assign unique colors for each neighbor
#         cmap = plt.get_cmap("tab10")
#         neighbor_colors = []
#         for i in range(len(neighbors)):
#             rgb = cmap(i % cmap.N)[:3] # take only first 3 channels
#             if np.allclose(rgb, (0, 0, 1)): # avoid blue color for neighbor points
#                 rgb = cmap((i + 1) % cmap.N)[:3]
#             neighbor_colors.append(rgb)
#
#         selected_info.update({
#             'point': selected_point,
#             'actor': point_actors[idx],
#             'neighbors': neighbors,
#             'neighbor_actors': [point_actors[i] for i in neighbor_indices[1:]],
#             'neighbor_colors': neighbor_colors
#         })
#         selected_info['actor'].GetProperty().SetColor(0.0, 0.0, 1.0) # blue for selected point
#         # color each neighbor differently
#         for actor, col in zip(selected_info['neighbor_actors'], neighbor_colors):
#             actor.GetProperty().SetColor(col)
#
#         legend_actor = [None]
#         # update legend
#         legend_entries = [["Selected Point", "blue"]]
#         for i, col in enumerate(neighbor_colors):
#             legend_entries.append([f"Neighbor {i + 1}", col])
#         # Remove old legend if it exists
#         if legend_actor[0] is not None:
#             plotter.remove_actor(legend_actor[0])
#
#         # Add new legend and store its actor
#         legend_actor[0] = plotter.add_legend(
#             legend_entries,
#             bcolor=None,
#             size=(0.2, 0.25),
#             loc='lower left',
#             font_family='arial'
#         )
#
#         print(f"Selected: {selected_point}")
#
#     def activate_selection_mode():
#         selection_mode[0] = True
#         plotter.add_text("Selection ACTIVE.\n"
#                          "-Click a point.\n"
#                          "-ENTER to confirm.",
#                          font_size=16, position='upper_right'
#         )
#         print("Selection mode activated.")
#     def confirm_selection():
#         if not selection_mode[0] or selected_info['point'] is None:
#             print("Please select a point first.")
#             return
#         center_point = selected_info['point']
#         neighbors = selected_info['neighbors']
#         if len(neighbors) < 1:
#             print("No neighbors found.")
#             return
#         run_statistical_analysis(mesh, center_point, neighbors)
#         plotter.close()
#     plotter.track_click_position(callback=on_click, side='left')
#     plotter.add_key_event('s', activate_selection_mode)
#     plotter.add_key_event('Return', confirm_selection)
#     plotter.show()

def run_full_analysis_pipeline(mesh, reduced_duct_points, k=8):
    """
    Revised pipeline to perform a full analysis on all reduced sweat duct points.

    1. Pre-processes and cleans the mesh.
    2. Loops through each reduced duct point.
    3. Finds neighbors and computes all metrics for each point.
    4. Aggregates all data for final visualizations and statistical tests.
    """
    if reduced_duct_points.size == 0 or mesh.n_points == 0:
        print("Input points or mesh is empty. Cannot run analysis.")
        return

    print("\n--- Step 1: Pre-processing and cleaning the mesh ---")
    cleaned_mesh = iterative_curvature_filter(
        mesh,
        iterations=1,
        percentile=99.9,
        smooth_passes=30,
        lambda_factor=0.5
    )

    if cleaned_mesh.n_points > 0:
        hole_size = 0.05
        cleaned_mesh = cleaned_mesh.fill_holes(hole_size=hole_size)
    else:
        print("Cleaned mesh is empty. Exiting.")
        return

    # recompute curvature and normals on the final repaired mesh
    cleaned_mesh.compute_normals(inplace=True)
    curv = cleaned_mesh.curvature(curv_type='mean')
    normal_tree = cKDTree(cleaned_mesh.points)

    all_duct_curvatures = []
    all_neighbor_dists = []
    all_dot_products = []

    print("\n--- Step 2: Looping through each sweat duct point ---")
    duct_tree = cKDTree(reduced_duct_points)

    for i, center_point in enumerate(reduced_duct_points):
        # find neighbors
        neighbor_dists, neighbor_indices = duct_tree.query(center_point, k=k + 1)
        neighbors = reduced_duct_points[neighbor_indices[1:]]

        if neighbors.size > 0:
            # compute distance, curvature, and normal alignment
            all_points = np.vstack([center_point, neighbors])

            # curvature of the duct points
            _, point_indices = normal_tree.query(all_points)
            all_duct_curvatures.extend(curv[point_indices])

            # distances to neighbors
            neighbor_dists_for_point = cdist([center_point], neighbors)[0]
            all_neighbor_dists.extend(neighbor_dists_for_point)

            # normal alignment
            all_normals = cleaned_mesh['Normals'][point_indices]
            center_normal = all_normals[0]
            neighbor_normals = all_normals[1:]
            dot_products_for_point = [np.dot(center_normal, n) for n in neighbor_normals]
            all_dot_products.extend(dot_products_for_point)

        print(f"Processed duct point {i + 1}/{len(reduced_duct_points)}")


    print("\n--- Step 3: Aggregating metrics and running statistical analysis ---")
    avg_curvature = np.mean(all_duct_curvatures)
    std_curvature = np.std(all_duct_curvatures)
    avg_distance = np.mean(all_neighbor_dists)
    std_distance = np.std(all_neighbor_dists)
    avg_dot_product = np.mean(all_dot_products)
    std_dot_product = np.std(all_dot_products)

    print(f"Average Curvature: {avg_curvature:.4f} (STD: {std_curvature:.4f})")
    print(f"Average Neighbor Distance: {avg_distance:.4f} (STD: {std_distance:.4f})")
    print(f"Average Normal Dot Product: {avg_dot_product:.4f} (STD: {std_dot_product:.4f})")


    print("\n--- Step 4: Generating visualizations ---")

    # 3D mean curvature plot
    low, high = np.percentile(curv, [1, 99])
    curvature_plotter = pv.Plotter()
    scalar_bar_args = {
        'title': 'Mean Curvature',
        'position_x': 0.05,
        'position_y': 0.05,
        'title_font_size': 16,
        'label_font_size': 14
    }
    curvature_plotter.add_mesh(
        cleaned_mesh,
        scalars=curv,
        clim=[low, high],
        cmap='viridis',
        show_scalar_bar=True,
        show_edges=False,
        opacity=0.8,
        scalar_bar_args=scalar_bar_args
    )
    curvature_plotter.add_points(reduced_duct_points, color='black', point_size=10, render_points_as_spheres=True,
                                 label='Sweat Duct Points')
    curvature_plotter.add_title("3D Mean Curvature Analysis")
    curvature_plotter.show(auto_close=False)

    # heatmap of average distances all duct points
    dist_matrix = cdist(reduced_duct_points, reduced_duct_points)
    plt.figure(figsize=(10, 9))
    sns.heatmap(
        dist_matrix,
        cmap="magma",
        annot=False,  # Show values in cells
        fmt=".2f",  # Format numbers to 2 decimal places
        linewidths=0.5,
        linecolor='white',
        xticklabels=[f"P{i}" for i in range(len(reduced_duct_points))],
        yticklabels=[f"P{i}" for i in range(len(reduced_duct_points))],
        cbar_kws={'label': 'Distance Scale'}
    )
    plt.title("Pairwise Euclidean Distance Heatmap (All Duct Points)")
    plt.xlabel("Duct Points")
    plt.ylabel("Duct Points")
    plt.tight_layout()
    plt.show(block=False)

    # 2D Quiver plot
    # plt.figure(figsize=(10, 10))
    # # Project points to 2D plane (e.g., ignore Z-axis)
    # projected_points = reduced_duct_points[:, :2]
    # projected_normals = repaired_mesh['Normals'][normal_tree.query(reduced_duct_points)[1]][:, :2]
    # # Plot the points
    # plt.scatter(projected_points[:, 0], projected_points[:, 1], c='blue', s=50)
    # # Plot the vectors
    # plt.quiver(projected_points[:, 0], projected_points[:, 1],
    #            projected_normals[:, 0], projected_normals[:, 1],
    #            color='red', angles='xy', scale_units='xy', scale=1)
    # plt.title("2D Quiver Plot of Normal Vectors")
    # plt.xlabel("X-coordinate")
    # plt.ylabel("Y-coordinate")
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show(block=False)

    # normal vector alignment
    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(len(all_dot_products)), all_dot_products, color='skyblue')
    plt.axhline(y=avg_dot_product, color='red', linestyle='--', label=f'Average Dot Product ({avg_dot_product:.2f})')
    plt.xlabel("Duct Point-Neighbor Pair Index")
    plt.ylabel("Dot Product of Normals")
    plt.title("Normal Vector Alignment for All Ducts")
    plt.legend()
    plt.show(block=False)

    # statistical analysis summary plot
    print("\n--- Statistical Test Results ---")
    t_stat_curv, p_value_curv = ttest_1samp(all_duct_curvatures, 0)
    t_stat_orient, p_value_orient = ttest_1samp(all_dot_products, 1)

    print(f"Curvature vs. Zero: t-statistic={t_stat_curv:.4f}, p-value={p_value_curv:.4e}")
    print(f"Orientation vs. Flat: t-statistic={t_stat_orient:.4f}, p-value={p_value_orient:.4e}")

    p_values = [p_value_curv, p_value_orient]
    test_names = ["Curvature vs. Zero", "Orientation vs. Flat"]
    plt.figure(figsize=(6, 5))
    plt.bar(test_names, p_values, color=['skyblue', 'salmon'])
    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Threshold (p=0.05)')
    plt.ylabel("p-value")
    plt.title("Summary of Aggregated Statistical Tests")
    plt.ylim(0, 1)
    plt.xticks(rotation=15, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)

def split_and_visualize(mesh, points):
    """
    Opens a new interactive window with a plane widget. The user can
    position this plane to manually split the mesh and points, isolating
    a specific part of the internal fingerprint. Pressing 's' on the
    keyboard confirms the split and triggers the subsequent analysis
    and visualization steps.
    """
    plotter = pv.Plotter()

    plane_params = {}

    def plane_callback(normal, origin):
        # store the normal and origin from the plane widget
        plane_params['normal'] = np.array(normal)
        plane_params['origin'] = np.array(origin)

    # show instructions in the top left
    plotter.add_text(
        "Plane widget controls:\n"
        "- Drag on handle to move\n"
        "- Rotate on edge to reorient\n"
        "- Press 's' to confirm split",
        position='upper_left', font_size=12, color='black'
    )

    # add the mesh and points for visualization
    plotter.add_mesh(mesh, color='lightgray', opacity=0.5)
    plotter.add_points(points, color='blue', point_size=5, render_points_as_spheres=True)

    # add the plane widget
    plotter.add_plane_widget(
        callback=plane_callback,
        normal='x',  # initial normal
        origin=mesh.center,  # center of the mesh
        assign_to_axis=None,
        tubing=False,
        origin_translation=True
    )

    def perform_split_and_analyze():
        if 'origin' not in plane_params:
            print("Please adjust the plane and press 's' again.")
            return

        plane_normal = plane_params['normal']
        plane_origin = plane_params['origin']

        # compute signed distance of each point to the plane
        point_distances = np.dot(points - plane_origin, plane_normal)

        # flip normal if most points are on the positive side (i.e. below visually)
        if np.mean(point_distances) > 0:
            print("[INFO] Flipping plane normal to capture points below.")
            plane_normal = -plane_normal
            point_distances = -point_distances

        # filter points below the plane
        final_points = points[point_distances <= 0]

        if len(final_points) == 0:
            print("No points found on the bottom mesh. Please adjust the plane.")
            return

        # distance of mesh vertices to plane (for triangle filtering)
        # mesh_distances = np.dot(mesh.points - plane_origin, plane_normal)
        # cell_indices = mesh.faces.reshape(-1, 4)[:, 1:]
        # cell_points_distances = mesh_distances[cell_indices]
        # mask = np.all(cell_points_distances <= 0, axis=1)
        # final_mesh = mesh.extract_cells(mask)
        # Clip the mesh cleanly using the normal and origin
        final_mesh = mesh.clip(normal=-plane_normal, origin=plane_origin, invert=False)
        try:
            final_mesh = final_mesh.extract_surface().clean().triangulate()
        except Exception as e:
            print("[ERROR] Failed to extract surface or triangulate:", e)
            return

        # extract largest connected component mesh
        try:
            if final_mesh.n_cells > 0:
                final_mesh = final_mesh.connectivity(extraction_mode='largest')
                print("Largest connected component extracted.")
            else:
                print("[WARNING] Mesh is empty. Skipping connectivity filter.")
                return
        except Exception as e:
            print(f"[ERROR] Failed to extract largest connected component: {e}")
            return

        # remove floating points that were attached to top mesh
        # create a KDTree from the final mesh's vertices
        mesh_tree = cKDTree(final_mesh.points)

        # for each point, find the distance to the closest mesh vertex
        distances, _ = mesh_tree.query(final_points)

        # define a threshold to identify floating points
        # a point is considered "floating" if it's too far from the mesh.
        # a small threshold like 0.01 is a good starting point.
        distance_threshold = 0.01

        # create a mask to keep only the points close to the mesh
        points_mask = distances < distance_threshold

        # apply the mask to the final_points array
        filtered_points = final_points[points_mask]

        print(f"Removed {len(final_points) - len(filtered_points)} floating points.")

        plotter.close()
        # point cluster reduction
        entry_reduced = reduce_clusters(filtered_points, eps=0.05)

        if entry_reduced.size == 0:
            print("No clusters found to visualize after reduction.")
            return

        entry_reduced = np.asarray(entry_reduced)
        if entry_reduced.ndim == 1 and entry_reduced.size == 3:
            entry_reduced = entry_reduced.reshape(1, 3)

        if entry_reduced.ndim != 2 or entry_reduced.shape[1] != 3:
            print(f"Unexpected reduced points shape: {entry_reduced.shape}")
            return

        # visualize entry points and midpoints (leaving in case single point neighborhood analysis)
        # try:
        #     visualize_entry_midpoint_web(entry_reduced, final_mesh, k=4)
        # except Exception as e:
        #     print(f"[visualize_entry_midpoint_web] failed: {e}")
        #
        # # Select a point and perform analysis
        # try:
        #     select_and_analyze_single_point(final_mesh, entry_reduced, k=8)
        # except Exception as e:
        #     print(f"[select_and_analyze_single_point] failed: {e}")

        try:
            run_full_analysis_pipeline(final_mesh, entry_reduced, k=8)
        except Exception as e:
            print(f"[run_full_analysis_pipeline] failed: {e}")

    # press the 's' key to trigger split and analysis
    plotter.add_key_event('s', perform_split_and_analyze)

    # display interactive plane widget
    plotter.show()


# modify crop_and_display to orchestrate the process
def crop_and_display():
    """
    This is the main orchestration function. It is called when the user
    presses 'Return' on the initial plotter. It crops the mesh and points
    based on the bounds of the box widget, selects the "bottom" part of
    the split mesh, and then calls `split_and_visualize` to begin the
    next stage of interactive analysis with the plane widget.
    """
    if 'bounds' not in box_bounds:
        return
    bounds = box_bounds['bounds']

    cropped_mesh = SC_pv.clip_box(bounds, invert=False)
    if cropped_mesh.n_points == 0:
        print("Cropping resulted in an empty mesh. Please adjust the box.")
        return
    cropped_mesh = cropped_mesh.extract_surface()
    # test_mesh = cropped_mesh.copy()

    in_box = lambda pts: np.all([(pts[:, i] >= bounds[2 * i]) & (pts[:, i] <= bounds[2 * i + 1]) for i in range(3)], axis=0)
    cropped_points = filtered_entry_points[in_box(filtered_entry_points)]

    if len(cropped_points) == 0:
        print("No points found in crop.")
        return

    try:
        plotter.clear()   # clear current window / interactor
    except Exception as e:
        print(f"Warning: plotter.close() raised {e}")

    split_and_visualize(cropped_mesh, cropped_points)


# initial boundary box plotter setup
plotter = pv.Plotter()
plotter.enable_anti_aliasing()
plotter.add_mesh(SC_pv, color='gray', opacity=0.5, show_edges=False)
plotter.add_points(entry_pv, scalars='colors', rgb=True, point_size=10, render_points_as_spheres=True)
plotter.add_points(exit_pv, scalars='colors', rgb=True, point_size=10, render_points_as_spheres=True)
box_widget = plotter.add_box_widget(callback=capture_box, bounds=SC_pv.bounds, factor=0.6, color='purple',
                                    outline_translation=True)
plotter.add_key_event("Return", crop_and_display)
plotter.add_axes()
plotter.add_title("Adjust Box, then Press ENTER to Crop")
plotter.show(auto_close=False)