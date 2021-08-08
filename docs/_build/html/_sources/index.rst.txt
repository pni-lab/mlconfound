.. mlconfound documentation master file, created by
   sphinx-quickstart on Fri Aug  6 13:49:20 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the docs of 'mlconfound'!
==========
Tools for analyzing and quantifying effects of counfounder variables on machine learning model predictions.


Install
--------
.. code-block:: bash

   pip install mlconfound


Example
--------
.. code-block:: python

   from mlconfound.stats import partial_confound_test

   partial_confound_test(y, yhat, c)


.. image:: _static/biased-model-example.png
  :width: 500
  :alt: biased model example

Source
-------
`https://github.com/pni-lab/mlconfound <https://github.com/pni-lab/mlconfound>`_


More
--------

.. toctree::
   about.md
   install.md
   quickstart.md
   docs.md
   :maxdepth: 1

Documentation index
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

