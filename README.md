# mlconfound
[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/spisakt/pTFCE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/spisakt/pTFCE/graphs/commit-activity)
[![GitHub license](https://img.shields.io/github/license/spisakt/pTFCE.svg)](https://github.com/spisakt/pTFCE/blob/master/LICENSE)
[![GitHub release](https://img.shields.io/github/release/spisakt/pTFCE.svg)](https://github.com/spisakt/pTFCE/releases/)
[![GitHub issues](https://img.shields.io/github/issues/spisakt/pTFCE.svg)](https://GitHub.com/spisakt/pTFCE/issues/)
[![GitHub issues-closed](https://img.shields.io/github/issues-closed/spisakt/pTFCE.svg)](https://GitHub.com/spisakt/pTFCE/issues?q=is%3Aissue+is%3Aclosed)
[![Documentation Status](https://readthedocs.org/projects/mlconfound/badge/?version=latest)](https://mlconfound.readthedocs.io/en/latest/?badge=latest)

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

from mlconfound.stats import test_partially_confounded

test_partially_confounded(y, yhat, c)
````

See documentation for more details.

## Documentation [![Documentation Status](https://readthedocs.org/projects/mlconfound/badge/?version=latest)](https://mlconfound.readthedocs.io/en/latest/?badge=latest)
https://mlconfound.readthedocs.io/en/latest/ 
