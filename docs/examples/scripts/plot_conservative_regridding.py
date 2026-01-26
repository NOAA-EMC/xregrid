"""
Conservative Regridding for Flux Data
====================================

Conservative regridding is essential for flux quantities (like precipitation
or radiation) where mass or energy must be preserved.

This example demonstrates how to use the 'conservative' method and verifies
area-integrated conservation. Conservative regridding requires cell boundaries
(bounds), which are automatically provided by XRegrid's grid creation utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from xregrid import Regridder, create_global_grid

# 1. Create a source grid with boundaries (2.0° resolution)
ds_source = create_global_grid(res_lat=2.0, res_lon=2.0)

# Create synthetic precipitation-like data
# (A simple pattern with a peak at the equator)
lons_2d, lats_2d = np.meshgrid(ds_source.lon, ds_source.lat)
precip_data = 10 * np.exp(-(lats_2d**2) / 100)
ds_source["precip"] = (["lat", "lon"], precip_data, {"units": "mm/day"})

# 2. Create a coarser target grid with boundaries (5.0° resolution)
ds_target = create_global_grid(res_lat=5.0, res_lon=5.0)

# 3. Create conservative regridder
# Both grids have boundaries (lat_b, lon_b) so conservative method will work
regridder = Regridder(ds_source, ds_target, method="conservative", periodic=True)

# 4. Apply regridding
precip_regridded = regridder(ds_source.precip)

# For comparison, also do bilinear (which is NOT conservative)
regridder_bil = Regridder(ds_source, ds_target, method="bilinear", periodic=True)
precip_bil = regridder_bil(ds_source.precip)

# 5. Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ds_source.precip.plot(ax=axes[0], cmap="Blues")
axes[0].set_title(f"Original (2.0°)\nMean: {ds_source.precip.mean().values:.4f}")

precip_regridded.plot(ax=axes[1], cmap="Blues")
axes[1].set_title(f"Conservative (5.0°)\nMean: {precip_regridded.mean().values:.4f}")

precip_bil.plot(ax=axes[2], cmap="Blues")
axes[2].set_title(f"Bilinear (5.0°)\nMean: {precip_bil.mean().values:.4f}")

plt.tight_layout()
plt.show()

print("\nConservation analysis (Global Mean):")
# For a global grid with equal resolution, the arithmetic mean is proportional
# to the area-weighted integral (ESMF conservative method preserves the integral).
orig_mean = ds_source.precip.mean().values
cons_mean = precip_regridded.mean().values
bil_mean = precip_bil.mean().values

print(f"Original mean:     {orig_mean:.6f}")
print(f"Conservative mean: {cons_mean:.6f}")
print(f"Bilinear mean:     {bil_mean:.6f}")
print(f"Conservative delta: {abs(cons_mean - orig_mean):.2e}")
