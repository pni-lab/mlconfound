# mlconfound
[![GitHub license](https://img.shields.io/github/license/pni-lab/mlconfound.svg)](https://github.com/pni-lab/mlconfound/blob/master/LICENSE)
[![GitHub release](https://img.shields.io/github/release/pni-lab/mlconfound.svg)](https://github.com/pni-lab/mlconfound/releases/)
![GitHub CI](https://github.com/pni-lab/mlconfound/actions/workflows/ci.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/mlconfound/badge/?version=latest)](https://mlconfound.readthedocs.io/en/latest/?badge=latest)
[![GitHub issues](https://img.shields.io/github/issues/pni-lab/mlconfound.svg)](https://GitHub.com/pni-lab/mlconfound/issues/)
[![GitHub issues-closed](https://img.shields.io/github/issues-closed/pni-lab/mlconfound.svg)](https://GitHub.com/pni-lab/mlconfound/issues?q=is%3Aissue+is%3Aclosed)

Tools for analyzing and quantifying effects of counfounder variables on machine learning model predictions.
## Install
````
pip install mlconfound
````

## Usage

````
# y   : prediction target
# yhat: prediction
# c   : confounder

from mlconfound.stats import test_partially_confounded

test_partially_confounded(y, yhat, c)
````

Read the docs for more details.

## Documentation [![Documentation Status](https://readthedocs.org/projects/mlconfound/badge/?version=latest)](https://mlconfound.readthedocs.io/en/latest/?badge=latest)
https://mlconfound.readthedocs.io
