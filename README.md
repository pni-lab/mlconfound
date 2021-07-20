# mlconfound

Tools for analyzing and quantifying effects of counfounder variables "
                "on machine learning model predictions.
## Install
````
pip install mlconfound
````

## Usage

````
# y   : prediction target
# yhat: prediction
# c   : confounder

from mlconfound.stats import confound_test
confound_test(y, H0_yhat, c)
````

See `notebooks/quickstart.ipynb` for more details.


## Documentation
*coming soon*
