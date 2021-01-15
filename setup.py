from setuptools import setup

setup(
    name='damfret_classifier',
    description='A Python package which implements a DAmFRET classifier using a phenomenological model.',
    version='1.0',
    author='Jared Lalmansingh, Kiersten Ruff',
    author_email='jared.lalmansingh@wustl.edu, kiersten.ruff@wustl.edu',
    license='GPL v2',
    url='https://github.com/pappulab/damfret_classifier',
    packages=['damfret_classifier'],
    scripts=['bin/classify_damfret'],
    zip_safe=True
)