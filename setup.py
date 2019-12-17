# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
here = path.abspath(path.dirname(__file__))


with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(

    name='yaddum',  # Required
    version='0.2.0',  # Required
    description='Yet Another Dual-Doppler Uncertainty Model (YADDUM): Python libpackage for dual-Doppler uncertainty assessment',  # Optional
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

    packages=['yaddum'],  # Required
    python_requires='==3.7.3',
    install_requires=[
                      'xarray', 
                      'netCDF4', 
                      'matplotlib', 
                      'jupyter',
                      'pylint'
                      ]
)