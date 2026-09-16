"""Small end-to-end simulation used to validate the Docker image."""

import numpy as np

import fastdyn_fic_dmf as dmf


connectivity = np.array(
    [
        [0.0, 0.2],
        [0.2, 0.0],
    ]
)

params = dmf.default_params(
    C=connectivity,
    batch_size=16,
    return_bold=True,
    return_fic=True,
    return_rate=True,
    seed=1,
)

rates_e, rates_i, bold, fic = dmf.run(params, nb_steps=2_000)

assert rates_e.shape == (2, 2_000)
assert rates_i.shape == (2, 2_000)
assert bold.shape == (2, 1)
assert fic.shape == (2, 2_000)
assert all(np.isfinite(result).all() for result in (rates_e, rates_i, bold, fic))

print("fastdyn_fic_dmf simulation succeeded")
print(f"rates_e={rates_e.shape}, rates_i={rates_i.shape}, bold={bold.shape}, fic={fic.shape}")
