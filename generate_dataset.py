from pathlib import Path

import tqdm
import numpy as np

from utils import create_shape, calculate_sdf


num_freq = 12
w_range = (-.5, .5)
resolution = 1000
base_r = 7
ellipsis_ratio = 2.
sdf_resolution = 500
num_samples = 1000
out_dir = './data/train'



max_radius = max(ellipsis_ratio, 1 / ellipsis_ratio) * (max(map(abs, w_range)) * num_freq + base_r)
out_dir = Path(out_dir)
for subdir in ('sdf', 'udf', 'signs', 'shape'):
    (out_dir / subdir).mkdir(parents=True, exist_ok=True)

for idx in tqdm.trange(num_samples):

    if (out_dir / 'sdf' / f'{idx:05d}.npy').exists():
        continue

    shape = create_shape(num_freq, w_range, resolution, base_r, ellipsis_ratio)
    sdf, udf, signs = calculate_sdf(shape, sdf_resolution, max_radius)
    
    np.save(out_dir / 'sdf' / f'{idx:05d}.npy', sdf)
    np.save(out_dir / 'udf' / f'{idx:05d}.npy', udf)
    np.save(out_dir / 'signs' / f'{idx:05d}.npy', signs)
    np.save(out_dir / 'shape' / f'{idx:05d}.npy', shape)
