# mlconfound
[![GitHub license](https://img.shields.io/github/license/pni-lab/mlconfound.svg)](https://github.com/pni-lab/mlconfound/blob/master/LICENSE)
[![GitHub release](https://img.shields.io/github/release/pni-lab/mlconfound.svg)](https://github.com/pni-lab/mlconfound/releases/)
![GitHub CI](https://github.com/pni-lab/mlconfound/actions/workflows/ci.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/mlconfound/badge/?version=latest)](https://mlconfound.readthedocs.io/en/latest/?badge=latest)
[![arXiv](https://img.shields.io/badge/arXiv-2111.00814-<COLOR>.svg)](https://arxiv.org/abs/2111.00814)
[![GitHub issues](https://img.shields.io/github/issues/pni-lab/mlconfound.svg)](https://GitHub.com/pni-lab/mlconfound/issues/)
[![GitHub issues-closed](https://img.shields.io/github/issues-closed/pni-lab/mlconfound.svg)](https://GitHub.com/pni-lab/mlconfound/issues?q=is%3Aissue+is%3Aclosed)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pni-lab/mlconfound/master?labpath=notebooks%2Fquickstart.ipynb)

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

from mlconfound.stats import partial_confound_test

partial_confound_test(y, yhat, c)
````

Run the quickstart notebook in Binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pni-lab/mlconfound/master?labpath=notebooks%2Fquickstart.ipynb)

Read the docs for more details.

## Documentation [![Documentation Status](https://readthedocs.org/projects/mlconfound/badge/?version=latest)](https://mlconfound.readthedocs.io/en/latest/?badge=latest)
https://mlconfound.readthedocs.io

## Citation
T. Spisak, Statistical quantification of confounding bias in predictive modelling, preprint on [arXiv:2111.00814](http://arxiv-export-lb.library.cornell.edu/abs/2111.00814), 2021.
