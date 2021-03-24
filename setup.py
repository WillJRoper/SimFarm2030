import setuptools


setuptools.setup(
    name='simfarm',
    version='0.0.1',
    author='Anisa Aubin',
    author_email='a.aubin@sussex.ac.uk',
    description=(
        'A package containing tools for extracting weather data '
        'for the UK and combining it with a gaussian growth response '
        'function for specific cultivars and locations in the UK'),
    packages=setuptools.find_packages(),
    install_requires=[
        'click',
        'corner',
        'emcee',
        'h5py',
        'matplotlib',
        'netCDF4',
        'numba',
        'ordered_set',
        'pandas',
        'pystan',
        'seaborn',
        'tqdm',
    ],
    extras_require={
        'tests': [
            'pytest'
        ]
    }
)
