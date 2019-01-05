#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "cloudpickle",
    "blosc",
    "concise",
    "dask",
    "deepdish",
    "gin-config",
    "gin-train",
    "joblib",
    "kipoi",
    "lmdb",
    "openpyxl",
    "pyarrow",
    "pytest",
    "pyvcf",
    "scikit-learn",
    "torch",
    "tensorflow==1.9",
    "toolz",
]

test_requirements = [
    "pytest",
    "virtualenv",
]
# TODO - require conda to be installed? - to create custom environments


setup(
    name='kipoi-cadd',
    version='0.0.1',
    description="CADD extension for Kipoi: code command-line tool",
    author="Daniela Simancas",
    author_email='simancas@in.tum.de',
    url='...',
    long_description=readme,
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "develop": ["bumpversion",
                    "wheel",
                    "jedi",
                    "epc",
                    "pytest",
                    "pytest-pep8",
                    "pytest-cov"],
    },
    entry_points={
        'console_scripts': ['kipoi_cadd = kipoi_cadd.__main__:main']},
    license="...",
    zip_safe=False,
    keywords=["model zoo", "deep learning",
              "computational biology", "bioinformatics", "genomics", "cadd"],
    test_suite='tests',
    package_data={'kipoi_cadd': ['logging.conf']},
    tests_require=test_requirements
)