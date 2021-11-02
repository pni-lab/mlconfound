# About mlconfound

The lack of rigorous non-parametric statistical tests of confoudner-effects significantly hampers the development of
robust, valid and generalizable predictive models in many fields of research.
The package `mlconfound` implements the *partial* and *full confounder tests* [1], that build on a recent theoretical framework of conditional 
independence testing [2] and test the null hypothesis of *no bias* and *fully biased model*, respectively.
The proposed tests set no assumptions about the distribution of the predictive model output that is often non-normal.
As shown by theory and simulations, the test are statistically valid, robust and display a high statistical power.

![usage](_static/schematic.png "Usage.")


#### References
[1] T. Spisak, Statistical quantification of confounding bias in predictive modelling, preprint on `arXiv:2111.00814 <http://arxiv-export-lb.library.cornell.edu/abs/2111.00814>`_, 2021.


[2] Berrett, T. B., Wang, Y., Barber, R. F., and Samworth, R. J. (2020). The conditional permutation test for
independencewhile controlling for confounders.Journal of the Royal Statistical Society: 
Series B (Statistical Methodology),82(1):175–197.

#### Contact / bug report
[GitHub Issues](https://github.com/pni-lab/mlconfound/issues): 
[![GitHub issues](https://img.shields.io/github/issues/pni-lab/mlconfound.svg)](https://GitHub.com/pni-lab/mlconfound/issues/)
[![GitHub issues-closed](https://img.shields.io/github/issues-closed/pni-lab/mlconfound.svg)](https://GitHub.com/pni-lab/mlconfound/issues?q=is%3Aissue+is%3Aclosed)


#### Author
Tamas Spisak

<tamas.spisak@uk-essen.de>

[PNI-Lab, University Hospital essen, Germany](https://pni-lab.github.io/)

#### See also:
* [Install](install.md)
* [Quickstart](quickstart.rst)
* [Documentation](docs.md)


[*Back to main page*](index.rst)  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; *Give feedback:* [![Star on Github](https://img.shields.io/github/stars/pni-lab/mlconfound.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/pni-lab/mlconfound/stargazers/)
