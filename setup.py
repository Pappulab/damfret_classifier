from setuptools import setup

setup(
    name='damfret_classifier',
    description='This is a Python package for classifying DAmFRET data based on supervised learning. The data can also be analyzed to allow mechanistic inferences regarding nucleation of ordered assemblies.',
    version='1.0',
    author='Jared Lalmansingh, Kiersten Ruff',
    author_email='jared.lalmansingh@wustl.edu, kiersten.ruff@wustl.edu',
    license='GPL v2',
    url='https://github.com/pappulab/damfret_classifier',
    packages=['damfret_classifier'],
    scripts=['bin/classify_damfret', 'bin/tsv-comparator'],
    zip_safe=True
)