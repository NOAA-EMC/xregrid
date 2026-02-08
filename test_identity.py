import xarray as xr
import numpy as np
import xregrid

ds = xr.tutorial.open_dataset('air_temperature')
# Take first time step
data = ds.air.isel(time=0)

# Regrid to itself
regridder = xregrid.Regridder(data, data, method='bilinear')
result = regridder(data)

diff = np.abs(result.values - data.values)
print(f"Max difference: {np.max(diff)}")
print(f"Mean difference: {np.mean(diff)}")

import matplotlib.pyplot as plt
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
data.plot(ax=ax1)
ax1.set_title("Original")
result.plot(ax=ax2)
ax2.set_title("Regridded to self")
(result - data).plot(ax=ax3)
ax3.set_title("Difference")
plt.savefig("test_identity.png")
