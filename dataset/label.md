The label.py file is designed to perform dataset labeling as part of this [issue](https://github.com/Ai-ithub/iFDC---FDDDC/issues/19).

The following conditions must be met to run this script:
-The input dataset must contain only the specified columns listed below. (Note: This requirement may change soon, as the code is scheduled for an update.):

```python
# example of Input Features:
time_series_features = [
    'viscosity', 
    'temperature',
    'shear_rate', 
    'salinity',
    'pH'
]
```

To enhance performance and increase processing speed, this code leverages the Dask library for parallel execution.
Therefore, you can adjust the configuration based on your system's hardware to optimize runtime efficiency.

```python
client = Client(n_workers=1, threads_per_worker=4, memory_limit='6GB')
```

