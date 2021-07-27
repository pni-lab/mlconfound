import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='mlconfound',
    version='0.4',
    packages=setuptools.find_packages(),
    scripts=[],
    author="Tamas Spisak",
    author_email="tamas.spisak@uk-essen.de",
    description="Tools for analyzing and quantifying effects of counfounder variables "
                "on machine learning model predictions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pni-lab/mlconfound",
    install_requires=[
        'joblib>=1.0',
        'tqdm>=4.61',
        'numpy>=1.18',
        'scipy>=1.4',
        'statsmodels>=0.11',
        'pandas>=1.0',
        'matplotlib>=3.2',
        'seaborn>=0.11.1',
        'graphviz>=0.17',
        'dot2tex>=2.11'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
)
