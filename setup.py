# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))


with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(

    name='YADDUM',  # Required
    version='0.1.0',  # Required
    description='Python library for dual-Doppler uncertainty assessment',  # Optional
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/niva83/YADDUM',  # Optional
    author='Nikola Vasiljevic',  # Optional
    author_email='niva@dtu.dk',  # Optional
    classifiers=[  # Optional
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.7',
    ],

    packages=['YADDYM'],  # Required
    python_requires='>=3.7',
    install_requires=[
                      'xarray', 
                      'netCDF4', 
                      'matplotlib', 
                      'pathlib',
                      'jupyter', 
                      'rasterio'
                      ]
)