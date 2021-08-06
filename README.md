# mlconfound

Tools for analyzing and quantifying effects of counfounder variables "
                "on machine learning model predictions.
## Install
````
pip install git+https://github.com/pni-lab/mlconfound
````
*pipy support coming soon*

## Usage

````
# y   : prediction target
# yhat: prediction
# c   : confounder

from mlconfound.stats import confound_test

test_partially_confounded(y, yhat, c)
````

See `notebooks/quickstart.ipynb` for more details.


## Documentation
*coming soon*
