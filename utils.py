import numpy as np
import skimage.measure as measure
import shapely.geometry as shapely
import matplotlib.pyplot as plt


def sample_w(
    num_freq: int,
    w_range: tuple[float, float]
) -> np.ndarray:
    """
    Samples the weights
    """
    return np.random.uniform(*w_range, (1, 2 * num_freq))


def create_shape(
    num_freq: int|list[int],
    w_range: tuple[float, float],
    resolution: int,
    base_r: float,
    ellipsis_ratio: float,
    *,
    return_time: bool = False
) -> np.ndarray:
    """
    Creates the shape(s)

    If multiple frquencies are given, the shapes with lower number of frequencies are the lower frequencies (lf) of the higher frequency (hf) shapes.
    This is usefull for creating a paired dataset between hf and lf shapes

    Shapes are modeled as an elipsis with a bias to the redius when measured in polar coordinates.
    The bias is modelled as a random fourier series.
    """
    is_list = isinstance(num_freq, list)
    num_freq = [num_freq] if not is_list else num_freq
    max_num_freq = max(num_freq)
    assert max(map(abs, w_range)) * max_num_freq < base_r, "Function might self-intersect"

    t = 2 * np.pi * np.linspace(0, 1, resolution)[None, :]

    angles = t * np.arange(1, max_num_freq + 1)[:, None]
    sin = np.sin(angles)
    cos = np.cos(angles)

    ws = sample_w(max_num_freq, w_range)

    shapes = np.empty((len(num_freq), 2, resolution))
    for idx, freq in enumerate(num_freq):
        masked_ws = ws.copy()
        masked_ws[
            :,
            (np.arange(2 * max_num_freq) % max_num_freq) > freq
        ] = 0
        
        r = masked_ws @ np.concat((sin, cos), axis=0) + base_r

        x = r * np.cos(t)
        y = ellipsis_ratio * r * np.sin(t)

        shapes[idx] = np.concat((x, y), axis=0)

    out = shapes if is_list else shapes[0]

    if return_time:
        return out, t

    return out


def plot_shape(shape: np.ndarray) -> None:
    """
    Displays the shape
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.plot(*shape)
    plt.show()


def calculate_sdf(
    shape: np.ndarray,
    sdf_resolution: int,
    max_radius: float,    
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes an array of shape and calculates the signed distance field
    """
    
    # the the space of the sdf
    x, y = shape
    grid = np.linspace(-max_radius, max_radius, sdf_resolution)

    # we calculate the distance between each point in the grid and the shape
    x_dist_pairs = grid[:, None] - x[None, :]
    y_dist_pairs = grid[:, None] - y[None, :]

    x_mesh = np.repeat(x_dist_pairs[:, None, :], sdf_resolution, axis=1)
    y_mesh = np.repeat(y_dist_pairs[None, :, :], sdf_resolution, axis=0)

    # we get the distance between each point in the grid to the shape at any time t
    sq_dist_mesh = np.add(
        np.square(x_mesh),
        np.square(y_mesh))
    
    # we minimize the distance along the time axis to get the closest point to the shape for each grid
    idxs = sq_dist_mesh.argmin(axis=2)
    idxs0, idxs1 = np.indices(idxs.shape)
    udf = np.sqrt(sq_dist_mesh[idxs0, idxs1, idxs])

    # we calculate the sign of the sdf using shapely, as I couldn't find a way to vectorize this
    signs = np.empty((sdf_resolution, sdf_resolution), dtype=bool)
    for x_idx, x_coord in enumerate(grid):
        for y_idx, y_coord in enumerate(grid):
            signs[x_idx, y_idx] = shapely.Polygon(shape.T).contains(shapely.Point(x_coord, y_coord))

    # nore really sure where the rotation and flipping of the images occurs
    udf = udf.T[::-1]
    signs = signs.T[::-1]

    # we return the signed distance
    return ~signs * udf - signs * udf, udf, signs


def plot_sdf(
    sdf: np.ndarray|None=None,
    udf: np.ndarray|None=None,
    signs: np.ndarray|None=None,
    shape: np.ndarray|None=None,
    max_radius: float|None=None
):
    """
    Plots the sdf
    """

    if max_radius is None:
        max_radius = np.linalg.norm(shape, axis=0).max() if shape is not None else 1
    
    num_plots = sum(map(lambda x: x is not None, (sdf, udf, signs)))
    num_plots += (num_plots == 0) and (shape is not None)

    if num_plots == 0:
        return

    extent = (-max_radius, max_radius, -max_radius, max_radius)
    
    fig, axs = plt.subplots(1, num_plots)
    if num_plots == 1:
        axs = [axs]

    idx = 0
    if sdf is not None:
        axs[idx].imshow(sdf, extent=extent)
        axs[idx].set_title('SDF')
        idx += 1
    if udf is not None:
        axs[idx].imshow(udf, extent=extent)
        axs[idx].set_title('UDF')
        idx += 1
    if signs is not None:
        axs[idx].imshow(signs, extent=extent)
        axs[idx].set_title('Signs')
        idx += 1
    if shape is not None:
        for ax in axs:
            ax.plot(*shape, c='C1')
    
    return fig, axs


def get_prototype_shape(
    sdfs: np.ndarray,
    sdf_resolution: int,
    max_radius: float
) -> np.ndarray:
    """
    Calculates a prototype shape from a set of signed distance fields.
    The prototype is extracted by averaging the SDFs and returning the longest 0-isocontour
    """
    mean_sdf = np.sum(sdfs, axis=0)
    contours = measure.find_contours(mean_sdf, 0)
    prototype = 2 * max_radius * (max(contours, key=len).T / sdf_resolution - .5)
    return ((1,), (-1,)) * prototype[::-1]
