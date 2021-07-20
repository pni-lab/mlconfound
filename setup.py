import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='mlconfound',
     version='0.2',
     packages=setuptools.find_packages(),
     scripts=[],
     author="Tamas Spisak",
     author_email="tamas.spisak@uk-essen.de",
     description="Tools for analyzing and quanifying effects of counfounder variables "
                 "on machine learning model predictions.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/pni-lab/mlconfound",
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
         "Operating System :: OS Independent",
     ],
 )