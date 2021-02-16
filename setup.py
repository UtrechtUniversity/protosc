# based on https://github.com/pypa/sampleproject - MIT License

from setuptools import setup, find_packages

setup(
    name='protosc',
    version='0.0.1',
    author='Protosc Development Team',
    description='protosc',
    long_description='more protosc',
    packages=find_packages(exclude=['data', 'docs', 'tests', 'examples']),
    python_requires='~=3.6',
    install_requires=[
        'numpy',
        'opencv-python'
    ]
)
