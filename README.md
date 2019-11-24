<h3 align="center">YADDUM</h3>
<h4 align="center">Yet Another Dual-Doppler Uncertainty Model</h4>

<div align="center">



  [![DOI](https://zenodo.org/badge/221973907.svg)](https://zenodo.org/badge/latestdoi/221973907) [![License](https://img.shields.io/badge/license-BSD-green)](/LICENSE) <a href="https://www.buymeacoffee.com/z57lyJbHo" rel="nofollow"><img alt="https://img.shields.io/badge/Donate-Buy%20me%20a%20coffee-yellowgreen.svg" src="https://warehouse-camo.cmh1.psfhosted.org/1c939ba1227996b87bb03cf029c14821eab9ad91/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4275792532306d6525323061253230636f666665652d79656c6c6f77677265656e2e737667"></a>

</div>

---

<p align="center"> A Python package for simple and fast uncertainty assessment of dual-Doppler retrievals of wind speed and wind direction.
    <br> 
</p>

## Table of Contents
- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Built Using](#built_using)
- [Contributing](#contributing)
- [Authors](#authors)
- [How to cite](#cite)
- [Acknowledgments](#acknowledgement)
<!-- - [TODO](../TODO.md) -->

## About <a name = "about"></a>
<!-- Write about 1-2 paragraphs describing the purpose of your project. -->

*YADDUM* is focused on delivering a simple yet effective dual-Doppler uncertainty model. This package is based on the [dual-Doppler uncertainty model](https://zenodo.org/record/1441178)  developed by [Nikola Vasiljevic](https://orcid.org/0000-0002-9381-9693) and [Michael Courtney](https://orcid.org/0000-0001-6286-5235). *YADDUM* is applicable for [wind lidars](https://www.mdpi.com/2072-4292/8/11/896) and [radars](https://www.mdpi.com/2072-4292/10/11/1701).

<br> 



## Getting Started <a name = "getting_started"></a>
These instructions will get you a copy of *YADDUM* up and running on your local machine. 

### Prerequisites
*YADDUM* is written in Python 3, therefore you will need to have [Python 3](https://realpython.com/installing-python/) installed on your local machine. For the simplicity of *YADDUM* installation having [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is advisable. 

### Installing
There are two ways how you can install *YADDUM*, either using [conda](https://docs.conda.io/en/latest/) or downloading the source code and manually installing the package.

In case you have [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your local machine you need to execute the following line in your (conda) terminal.

```
conda create -n UNCERTAINTY -c nikola_v yaddum
```

This command will create new [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) *UNCERTAINTY* and install *YADDUM* package with all the dependencies. In this way your existing conda environments will be left intact.  

Alternatively you can download the source code as [a zip file](https://github.com/niva83/YADDUM/archive/master.zip), unzip it in a desired folder, in terminal navigate to the folder and run the following command:

```
python setup.py install
```

This command will download all the dependencies and install them first before installing *YADDUM* package. Beware that, unless you activate your preferred [virtualenv](https://virtualenv.pypa.io/en/latest/), this command will overwrite any existing packaged. Therefore, if you are not confortable to setup [virtualenv](https://virtualenv.pypa.io/en/latest/) opt for the installation using conda.

## Running the test <a name = "tests"></a>
An example of Python script which uses *YADDUM* package is provided in [test folder](https://github.com/niva83/YADDUM/blob/master/test/test.py). Simply download the script and run the following command:
```
python test.py
```
The end result should be a plot that looks like this:

![Azimuth_Gain](../assets/Figure_1.png?raw=true)

## Usage <a name="usage"></a>
Add notes about how to use the system.


## Built Using <a name = "built_using"></a>
- [Python](https://www.python.org/) - Languange
- [xarray](http://xarray.pydata.org/en/stable/#) - Package
- [numpy](https://numpy.org/) - Package

## Authors <a name = "authors"></a>
- [@niva83](https://github.com/niva83/) - idea and work

## How to cite <a name = "cite"></a>
This package is persisted using [Zenodo](https://zenodo.org/) making it citable and preserved. Simply click on the banner bellow to find your preferred citation options (e.g., BibTex): 

[![DOI](https://zenodo.org/badge/221973907.svg)](https://zenodo.org/badge/latestdoi/221973907)

## Contributing <a name = "contributing"></a>
If you want to take an active part in the further development of *YADDUM* make a pull request or post an issue in this repository.

## Acknowledgements <a name = "acknowledgement"></a>
[Michael Courtney](https://orcid.org/0000-0001-6286-5235) for the support in developing [dual-Doppler uncertainty model](https://zenodo.org/record/1441178) which is the basis of this package.
