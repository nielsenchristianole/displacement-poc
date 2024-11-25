import tqdm
import numpy as np
import torch
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

    x_mesh = x_dist_pairs[:, None, :]
    y_mesh = y_dist_pairs[None, :, :]

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
    _shape = shapely.Polygon(shape.T)
    for x_idx, x_coord in enumerate(grid):
        for y_idx, y_coord in enumerate(grid):
            signs[x_idx, y_idx] = _shape.contains(shapely.Point(x_coord, y_coord))

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
    prototype = ((1,), (-1,)) * prototype[::-1]
    
    # align the shape
    if not np.all(np.sort(np.atan2(*prototype)) == np.atan2(*prototype)):
        taos = np.atan2(*np.concat((prototype[:, [-1]], prototype), axis=1))
        roll_val = prototype.shape[1] - np.diff(taos).argmax()
        prototype = np.roll(prototype, roll_val, axis=1,)
        taos = np.atan2(*prototype)

        if np.diff(taos).mean() < 0:
            prototype = np.flip(prototype, axis=1)
            taos = np.atan2(*prototype)
        assert (np.sort(taos) == taos).all(), 'The shape is not star-convex with s_0=(0,0), I think?'

    return prototype


def project_shape_torch(
    shape: torch.Tensor,
    *,
    reference_shape: torch.Tensor,
    reference_taos: torch.Tensor
) -> torch.Tensor:
    """
    this is used as part of the optimization, bind the references using functools.partial
    """
    
    # calculate a mapping between old and new coords
    new_taos = torch.atan2(*shape)
    upper_idxs = torch.searchsorted(reference_taos, new_taos, side='right')
    lower_idxs = upper_idxs - 1
    tao_lower = reference_taos[lower_idxs]
    tao_upper = reference_taos[upper_idxs]

    interps = (new_taos - tao_lower) / (tao_upper - tao_lower)

    # map the shape
    shape_proj = reference_shape[:, lower_idxs] * (1 - interps) + reference_shape[:, upper_idxs] * interps

    return shape_proj


def fit_shape(
    prototype_shape: np.ndarray,
    orig_shape: np.ndarray,
    *,
    num_max_steps: int = 10_000,
    return_energies: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:

    orig_shape = torch.from_numpy(orig_shape.copy())


    # pad the shape
    taos = np.atan2(*prototype_shape)
    assert np.all(np.sort(taos) == taos), 'The prototype shape is not star-convex with s_0=(0,0), I think?'
    offset = ((taos.min() == -np.pi) or (taos.max() == np.pi)).astype(int) + 1
    reference_shape = torch.from_numpy(np.concatenate((prototype_shape[:, -offset:], prototype_shape, prototype_shape[:, :offset]), axis=1))
    reference_taos = torch.from_numpy(np.concatenate((taos[-offset:] - 2 * np.pi, taos, taos[:offset] + 2 * np.pi)))
    assert (reference_taos.min() < -np.pi) and (reference_taos.max() > np.pi), 'Assumptions not met'

    orig_shape_proj = project_shape_torch(
        orig_shape,
        reference_shape=reference_shape,
        reference_taos=reference_taos)

    shape_proj = torch.nn.parameter.Parameter(orig_shape_proj.clone())

    energies = torch.full((num_max_steps+1,), np.nan)
    optimizer = torch.optim.Adam([shape_proj], lr=1e-3)
    for i in tqdm.trange(1, num_max_steps+1):
        optimizer.zero_grad()

        _shape_proj = project_shape_torch(
            shape_proj,
            reference_shape=reference_shape,
            reference_taos=reference_taos
        )
        energy = torch.square(
            _shape_proj - torch.roll(_shape_proj, 1, dims=1)).sum(dim=0).mean()
        energies[i] = energy.item()
        energy.backward()
        optimizer.step()

        if i % 100 == 0 and abs(energies[i] - energies[i - 99]) < 1e-7:
            break

    shape_proj = project_shape_torch(
        shape_proj.detach(),
        reference_shape=reference_shape,
        reference_taos=reference_taos).numpy()
    
    energies = energies[1:i].numpy()
    if return_energies:
        return shape_proj, energies
    return shape_proj


def project_shape(
    proj_shape: np.ndarray,
    *,
    reference_shape: np.ndarray,
    extrat_vals: np.ndarray|None = None,
) -> np.ndarray:
# ) -> tuple[np.ndarray, np.ndarray]:
    """
    this is used as part of the optimization, bind the references using functools.partial
    """
    extrat_vals = reference_shape if extrat_vals is None else extrat_vals

    # align the shape
    if not np.all(np.sort(np.atan2(*reference_shape)) == np.atan2(*reference_shape)):
        taos = np.atan2(*np.concat((reference_shape[:, [-1]], reference_shape), axis=1))
        roll_val = reference_shape.shape[1] - np.diff(taos).argmax()
        reference_shape = np.roll(reference_shape, roll_val, axis=1)
        extrat_vals = np.roll(extrat_vals, roll_val, axis=1)
        taos = np.atan2(*reference_shape)

        if np.diff(taos).mean() < 0:
            reference_shape = np.flip(reference_shape, axis=1)
            extrat_vals = np.flip(extrat_vals, axis=1)
            taos = np.atan2(*reference_shape)
        assert (np.sort(taos) == taos).all(), "The reference is not sorted"

    # calculate the padding
    offset = ((taos.min() == -np.pi) or (taos.max() == np.pi)).astype(int) + 1

    # pad the reference shape
    reference_shape = np.concatenate((reference_shape[:, -offset:], reference_shape, reference_shape[:, :offset]), axis=1)
    extrat_vals = np.concatenate((extrat_vals[:, -offset:], extrat_vals, extrat_vals[:, :offset]), axis=1)
    reference_taos = np.concatenate((taos[-offset:] - 2 * np.pi, taos, taos[:offset] + 2 * np.pi))
    assert (reference_taos.min() < -np.pi) or (reference_taos.max() > np.pi), 'Assumptions not met'
    
    # calculate a mapping between old and new coords
    new_taos = np.atan2(*proj_shape)
    upper_idxs = np.searchsorted(reference_taos, new_taos, side='right')
    lower_idxs = upper_idxs - 1
    tao_lower = reference_taos[lower_idxs]
    tao_upper = reference_taos[upper_idxs]

    interps = (new_taos - tao_lower) / (tao_upper - tao_lower)

    # project the shape
    shape_proj = extrat_vals[:, lower_idxs] * (1 - interps) + extrat_vals[:, upper_idxs] * interps

    return shape_proj
