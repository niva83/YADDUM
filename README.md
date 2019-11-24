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

![Concept](../assets/concept.png?raw=true)
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
*YADDUM* contains five classes (see image below) which are: *Atmosphere*, *Measurements*, *Instruments*, *Lidars* and *Uncertainty*. However, users only interacts with the class *Uncertainty* as this class inherits the properties of the remaining four classes. 

![Classes_Relations](../assets/classes_relation.png?raw=true)

The worflow with *YADDUM* is relatively simple and essentially consists of the following steps:
1. Atmosphere parametrization using the method `add_atmosphere(atmosphere_id, model, model_parameters)`
2. Localization of measurement points using the method `add_measurements(measurements_id, category, **kwargs)`
3. Localization and description of lidars using the method `add_lidar(instrument_id, position, category, **kwargs)``
4. Calculation of the measurement uncertainty using the method `calculate_uncertainty(instrument_ids, measurements_id, atmosphere_id, model)`

The methods `add_atmosphere()`, `add_measurements()` and `add_lidar()` create three [Python dictionaries](https://www.w3schools.com/python/python_dictionaries.asp) `atmosphere`, `measurements` and `instruments`, while calling the method `calculate_uncertainty()` produces two [xarray datasets](http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html), namely `wind_field` and `uncertainty`. 

Let's use the following example to show how to interact with *YADDUM* package.
First, we import *YADDUM* package together with several additional packages and create a *YADDUM* object:
```
import yaddum as yaddum
import numpy as np  
import matplotlib.pyplot as plt
import xarray as xr

lidar_uncertainty = yaddum.Uncertainty()
```
Following the previous workflow we will parametrize atmosphere: 
```
model_pars={'wind_speed':10,
            'upward_air_velocity':0,
            'wind_from_direction':0,
            'reference_height':100,
            'shear_exponent':0.2}

lidar_uncertainty.add_atmosphere('pl_1', 'power_law', model_pars)
```
The above commands will add [power law model](https://en.wikipedia.org/wiki/Wind_profile_power_law) to the object (specifically dictionary `lidar_uncertanty.atmosphere`) together with the set of parameters. Currently *YADDUM* only supports this atmospheric model.

Next we will add measurement points to our object. We can either add an arbitrary array of points or create 2D horizontal mesh of points:
```
points = np.array([[500,-500,100]])
lidar_uncertainty.add_measurements('pts', 'points', positions = points)

lidar_uncertainty.add_measurements('mesh', 'horizontal_mesh', 
                                   resolution = 10, 
                                   mesh_center = np.array([0,0,100]), 
                                   extent = 5000)
```
In the first case we have added single measurement point with coordinates of (500, -500, 100) in Northing, Easting and Height, while in the second case we have provided position of the mesh center (0,0,100), set the mesh resolution (10) and mesh extent in Easting and Northing (5000). The unit for each of these values is meters. Both category of measurement points (i.e., single point and mesh points) now exist in our object and they are distinguishable by their ids ('*pts*' and '*mesh*'). They are stored in the measurement dictionary:
```
lidar_uncertainty.measurements['mesh’]
lidar_uncertainty.measurements['pts']

```

In our last step prior the uncertainty calculation we will lidars to our object:
```
uncertainty_pars = {'u_estimation':0.1,
                    'u_azimuth':0.1,
                    'u_elevation':0.1, 
                    'u_range':1}

lidar_pos_1 = np.array([0,0,0])
lidar_pos_2 = np.array([1000,1000,0])


lidar_uncertainty.add_lidar('koshava', lidar_pos_1, **uncertainty_pars)
lidar_uncertainty.add_lidar('whittle', lidar_pos_2, **uncertainty_pars)
```
With the code above we have added two lidars  '*koshava*' and '*whittle*' together with their positions and their intrinsic uncertainties which reflect their ability to:
- Estimate radial velocity from the backscatter signal (*u_estimation*)
- Resolve range (i.e., distance) from which the backscatter signal is coming from (*u_range*)
- Point laser beams towards measurement points (*u_azimuth* and *u_elevation*)

Similarly to the measurement points and atmospheric model users have access to the lidar information by executing `lidar_uncertainty.instruments` command.

The last step is to call the method `calculate_uncertainty()` and specify ids of lidars, measurement point, atmospheric and uncertainty model to be used for the calculations:
```
lidar_uncertainty.calculate_uncertainty(['koshava', 'whittle'], 'mesh', 'pl_1', 
                                        uncertainty_model='dual-Doppler')
```

This last step will create the two xarray datasets contaning the results of the uncertainty calculation, which can be viewed graphically and explore numerically:
```
lidar_uncertainty.uncertainty.azimuth_gain.sel(instrument_id = 'koshava').plot()
plt.show()

lidar_uncertainty.uncertainty.azimuth_gain
```


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

