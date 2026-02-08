import xarray as xr
import numpy as np
import dask.array as da
import dask.distributed

def my_func(data, weight):
    print(f"Type of data: {type(data)}")
    print(f"Type of weight: {type(weight)}")
    return data * weight

def test():
    client = dask.distributed.Client(n_workers=1)

    data = xr.DataArray(da.from_array(np.ones((10, 10)), chunks=(5, 10)), dims=('x', 'y'))
    weight_val = 2.0
    weight_future = client.scatter(weight_val)

    print("--- Using Future in kwargs ---")
    try:
        res = xr.apply_ufunc(my_func, data, kwargs={'weight': weight_future}, dask='parallelized', output_dtypes=[float])
        print(res.compute().values[0, 0])
    except Exception as e:
        print(f"Failed in kwargs: {e}")

    print("\n--- Using Future in args ---")
    try:
        # We need to tell xarray that weight_future is a scalar (no core dims)
        res = xr.apply_ufunc(my_func, data, weight_future, dask='parallelized', output_dtypes=[float], input_core_dims=[['y'], []], output_core_dims=[['y']])
        print(res.compute().values[0, 0])
    except Exception as e:
        print(f"Failed in args: {e}")
        import traceback
        traceback.print_exc()

    client.close()

test()
