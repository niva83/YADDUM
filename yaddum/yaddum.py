"""This is a Python package for dual-Doppler uncertainty assessment.
It includes several class which are used to perform the assessment.

"""

from math import cos, sin, pi, radians
import numpy as np
import xarray as xr

def wind_vector_to_los(u,v,w, azimuth, elevation):
        """
        Projects wind vector to the beam line-of-sight (LOS).

        Parameters
        ----------
        u : ndarray
            nD array of `float` or `int` corresponding to u component of wind.
            In cf convention v is eastward_wind.
            Units m/s.
        v : ndarray
            nD array `float` or `int` corresponding to v component of wind.
            In cf convention v is northward_wind.
            Units m/s.
        w : ndarray
            nD array `float` or `int` corresponding to w component of wind.
            In cf convention w is upward_air_velocity.
            Units m/s.
        azimuth : ndarray
            nD array `float` or `int` corresponding to LOS direction in azimuth.
            Units degree.								
        elevation : ndarray
            nD array `float` or `int` corresponding to LOS direction in elevation.
            Units degree.
        
        Returns
        -------
        los : ndarray
            nD array `float` or `int` corresponding to LOS component of wind.
            In cf convention w is radial_velocity_of_scatterers_toward_instrument.
            Units m/s.
            
        Notes
        -----
        LOS or radial wind speed, :math:`{V_{radial}}`, is calculated using the 
        following mathematical expression:
        
        .. math::

            V_{radial} = u \sin({\\theta})\cos({\\varphi}) + 
                      v \cos({\\theta})\cos({\\varphi}) + 
                      w\sin({\\varphi})
        
        where :math:`{\\theta}` and :math:`{\\varphi}` are the azimuth and 
        elevation angle of the beam, :math:`{u}` is the wind component toward East, 
        :math:`{v}` is the wind component toward North, and :math:`{w}` is the 
        upward air velocity. The sign of :math:`{V_{radial}}` is assumed to be 
        positive if wind aprroaches the instrument, otherwise it is negative.
        """
        # handles both single values as well arrays
        azimuth = np.radians(azimuth)
        elevation = np.radians(elevation)
        los = u * np.sin(azimuth) * np.cos(elevation) + \
              v * np.cos(azimuth) * np.cos(elevation) + \
              w * np.sin(elevation)	

        return los
    
def generate_mesh(center, map_extent, mesh_res):
    """
    Generate a horizontal mesh containing equally spaced (measurement) points.

    Parameters
    ----------
    center : ndarray
        1D array containing data with `float` or `int` type corresponding to 
        Northing, Easting and Height coordinates of the mesh center.
        1D array data are expressed in meters.
    map_extent : int
        map extent in Northing (y) and Easting (x) in meters.
    mesh_res : int
        mesh resolution for Northing (y) and Easting (x) in meters.
    
    Returns
    -------
    mesh : ndarray
        nD array containing a list of mesh points.
    x : ndarray
        nD shaped array for Easting (x) coordinate of mesh points.
    y : ndarray
        nD shaped array for Northing (y) coordinate of mesh points.
        
    Notes
    -----
    The generated mesh will be squared, i.e. having the same length in both dimensions.
    """
    map_corners = np.array([center[:2] - int(map_extent), 
                            center[:2] + int(map_extent)])

    x, y = np.meshgrid(
        np.arange(map_corners[0][0], map_corners[1][0]+ int(mesh_res), int(mesh_res)),
        np.arange(map_corners[0][1], map_corners[1][1]+ int(mesh_res), int(mesh_res))
            )
		
    H_asl = np.full(x.shape, center[2])		
    H_agl = np.full(x.shape, center[3])		
    mesh = np.array([x, y, H_asl, H_agl]).T.reshape(-1, 4)
    return x, y, mesh


def generate_beam_coords(lidar_pos, meas_pt_pos):
    """
    Generates beam steering coordinates in spherical coordinate system.

    Parameters
    ----------
    lidar_pos : ndarray
        1D array containing data with `float` or `int` type corresponding to 
        Northing, Easting and Height coordinates of a lidar.
        Coordinates unit is meter.
    meas_pt_pos : ndarray
        nD array containing data with `float` or `int` type corresponding to 
        Northing, Easting and Height coordinates of a measurement point(s).
        Coordinates unit is meter.
    
    Returns
    -------
    beam_coords : ndarray
        nD array containing beam steering coordinates for given measurement points.
        Coordinates have following structure [azimuth, elevation, range].
        Azimuth and elevation angles are given in degree.
        Range unit is meter.
    """
    # testing if  meas_pt has single or multiple measurement points
    if len(meas_pt_pos.shape) == 2:
        x_array = meas_pt_pos[:, 0]
        y_array = meas_pt_pos[:, 1]
        z_array = meas_pt_pos[:, 2]
    else:
        x_array = np.array([meas_pt_pos[0]])
        y_array = np.array([meas_pt_pos[1]])
        z_array = np.array([meas_pt_pos[2]])


    # calculating difference between lidar_pos and meas_pt_pos coordiantes
    dif_xyz = np.array([lidar_pos[0] - x_array, lidar_pos[1] - y_array, lidar_pos[2] - z_array])    

    # distance between lidar and measurement point in space
    distance_3D = np.sum(dif_xyz**2,axis=0)**(1./2)

    # distance between lidar and measurement point in a horizontal plane
    distance_2D = np.sum(np.abs([dif_xyz[0],dif_xyz[1]])**2,axis=0)**(1./2)

    # in radians
    azimuth = np.arctan2(x_array-lidar_pos[0], y_array-lidar_pos[1])
    # conversion to metrological convention
    azimuth = (360 + azimuth * (180 / pi)) % 360

    # in radians
    elevation = np.arccos(distance_2D / distance_3D)
    # conversion to metrological convention
    elevation = np.sign(z_array - lidar_pos[2]) * (elevation * (180 / pi))
    
    beam_coord = np.transpose(np.array([azimuth, elevation, distance_3D]))

    return beam_coord

class Atmosphere:
    """
    A class containing methods and attributes related to atmosphere.

    Methods
    -------
    add_atmosphere(atmosphere_id, model, model_parameters)
        Adds description of the atmosphere to the atmosphere dictionary.

    """        
    def __init__(self):
        self.atmosphere = {}
        self.wind_field = None

    def add_atmosphere(self, atmosphere_id, model, model_parameters):
        """
        Adds description of the atmosphere to the atmosphere dictionary.
        This description is used to calculate the lidar uncertainty.
        
        Parameters
        ----------
        atmosphere_id : str, required
            String which identifies atmosphere instance in the dictionary.
        model : str, required
            This is a string describing which atmospheric model is used.
        model_parameters : dict, required
            This is a dictionary which contains parameters which detail 
            the selected atmospheric model.
            
        Raises
        ------
        UnsupportedModel
            If the selected model is not supported by the package.
        
        Notes
        -----
        Currently method 'add_atmosphere' only supports power law model of the
        atmosphere. The power law model requires following inputs in a form of
        Python dictionary: horizontal speed, wind direction, shear exponent and
        reference height (height above ground level) for horizontal speed.

        TODO
        ----
        - Support other atmospheric models (e.g., log wind profile) 

        """

        if (model != 'power_law'):
            raise ValueError("UnsupportedModel")

        if ('wind_speed' in model_parameters
            and model_parameters['wind_speed'] is not None
            and model_parameters['wind_speed'] is not 0

            and 'wind_from_direction' in model_parameters
            and model_parameters['wind_from_direction'] is not None

            and 'shear_exponent' in model_parameters
            and model_parameters['shear_exponent'] is not None
            and model_parameters['shear_exponent'] is not 0

            and 'reference_height' in model_parameters
            and model_parameters['reference_height'] is not None
            and model_parameters['reference_height'] >= 0):

            wind_speed = model_parameters["wind_speed"]
            wind_from_direction = model_parameters["wind_from_direction"]

            u = wind_speed * sin(radians(wind_from_direction))
            v = wind_speed * cos(radians(wind_from_direction))
            w = model_parameters['upward_air_velocity'] if 'upward_air_velocity' in model_parameters else 0

            model_parameters.update({
                                        'eastward_wind' : u,
                                        'northward_wind' : v,
                                        'upward_air_velocity' : w
                                    })

            dict_input = {atmosphere_id: {
                                    "model" : model,
                                    "model_parameters": model_parameters}}
            self.atmosphere.update(dict_input)
            print('Atmosphere \'' + atmosphere_id
                                    + '\' added to the atmosphere dictionary,'
                                    + ' which now contains ' 
                                    + str(len(self.atmosphere)) 
                                    + ' atmosphere instance(s).')

        else:
            print('Incorrect parameters for power law model!')

class Measurements(Atmosphere):
    """
    A class containing methods and attributes related to measurements.

    Methods
    -------
    add_atmosphere(atmosphere_id, model, model_parameters)
        Adds description of the atmosphere to the atmosphere dictionary.

    """        
    def __init__(self):
        self.measurements = {}
        Atmosphere.__init__(self)

    @staticmethod
    def check_measurement_positions(measurement_positions):
        """
        Validates the measurement position
        
        Parameters
        ----------
        measurement_positions : ndarray
            nD array containing data with `float` or `int` type corresponding
            to Northing, Easting and Height coordinates of the instrument.
            nD array data are expressed in meters.
        
        Returns
        -------
            True / False

        See also
        --------
        add_measurements() : adds measurements to the measurement dictionary

        """        
        if(type(measurement_positions).__module__ == np.__name__):
                if (len(measurement_positions.shape) == 2 
                    and measurement_positions.shape[1] == 4):
                        return True
                else:
                    # print('Wrong dimensions! Must be == (n,3) where ')
                    # print('n == number of measurement points!')
                    # print('Measurement positions were not added!')
                    return False
        else:
            # print('Input is not numpy array!')
            # print('Measurement positions were not added!')
            return False

    def add_measurements(self, measurements_id, 
                         category='points', utm_zone = '', **kwargs):
        """
        Adds desired measurement positions to the measurements dictionary.
        The measurement points are used for the uncertainty calculation.
        
        Parameters
        ----------
        measurements_id : str, required
            String which identifies measurements instance in the dictionary.
        category : str, required
            Indicates category of measurements that are added to the dictionary.
            This paremeter can be either equal to 'points' or 'horizontal_mesh'.
            Default value is set to 'points'.
        utm_zone : str, optional
            Indicates UTM zone in which points are located.
            Default values is set to None.

        Other Parameters
        -----------------
        positions : ndarray
            nD array containing data with `float` or `int` type corresponding 
            to Northing, Easting, Height above ground level, and Height above
            sea level coordinates of the measurement pts.
            nD array data are expressed in meters.
            This kwarg is required if category=='points'
        mesh_center : ndarray
            nD array containing data with `float` or `int` type
            corresponding to Northing, Easting and Height above ground level, 
            and Height above sea level of the mesh center.
            nD array data are expressed in meters.
            This kwarg is required if category=='horizontal_mesh'.
        extent : int
            mesh extent in Northing and Easting in meters.
            This kwarg is required if category=='horizontal_mesh'.
        resolution : int
            mesh resolution in meters.
            This kwarg is required if category=='horizontal_mesh'.
        
        Raises
        ------
        UnsupportedCategory
            If the category of measurement points is not supported.
        PositionsMissing
            If category=='points' but the position of points is not provided.
        InappropriatePositions
            If the provided points positions are not properly provided.
        MissingKwargs
            If one or more kwargs are missing.
                        
        TODO
        ----
        - Accept other categories such as LOS, PPI, RHI, VAD and DBS
        """
        if category not in {'points', 'horizontal_mesh'}:
            raise ValueError("UnsupportedCategory")

        if category == 'points' and 'positions' not in kwargs:
            raise ValueError("PositionsMissing")

        if (category == 'points' and 
            'positions' in kwargs and 
            not(self.check_measurement_positions(kwargs['positions']))):
            raise ValueError("InappropriatePositions")     
        
        if category == 'horizontal_mesh' and set(kwargs) != {'resolution','mesh_center', 'extent'}:            
           raise ValueError("MissingKwargs")
            
        if category == 'points':
            measurements_dict = {measurements_id : 
                                    {'category': category,
                                     'positions' : kwargs['positions']
                                                }
                                    }
            self.measurements.update(measurements_dict)
            print('Measurements \'' + measurements_id
                                    + '\' added to the measurement dictionary,'
                                    + ' which now contains ' 
                                    + str(len(self.measurements)) 
                                    + ' measurement instance(s).')            
        elif category == 'horizontal_mesh':
            x, y, mesh_points = generate_mesh(kwargs['mesh_center'], 
                                              kwargs['extent'],
                                              kwargs['resolution'])
            nrows, ncols = x.shape
            
            measurements_dict = {measurements_id : 
                                    {'category': category,
                                     'nrows' : nrows,
                                     'ncols' : ncols,
                                     'positions' : mesh_points
                                                }
                                    }
            self.measurements.update(measurements_dict)
            print('Measurements \'' + measurements_id
                                    + '\' added to the measurement dictionary,'
                                    + ' which now contains ' 
                                    + str(len(self.measurements)) 
                                    + ' measurement instance(s).')
            
    def __create_wind_ds(self, atmosphere, measurements, 
                            u, v, w, wind_speed, wind_from_direction):
        """
        Creates wind field xarray object.
        
        Parameters
        ----------
        atmosphere : dict
            Dictionary containing information on atmosphere.
        measurements : dict
            Dictionary containing information on measurements.
        u : ndarray
            nD array of `float` or `int` corresponding to u component of wind.
            In cf convention v is eastward_wind.
            Units m/s.
        v : ndarray
            nD array `float` or `int` corresponding to v component of wind.
            In cf convention v is northward_wind.
            Units m/s.
        w : ndarray
            nD array `float` or `int` corresponding to w component of wind.
            In cf convention w is upward_air_velocity.
            Units m/s.            
        wind_speed : ndarray
            nD array `float` or `int` corresponding to the wind speed.
            Units m/s.
        wind_from_direction : ndarray
            nD array `float` or `int` corresponding to the wind direction.
            Units degree.            
                        
        Notes
        ----
        Currently this method only supports points and horizontal mesh data structures.
        The method is inline with the cf convention for variable naming.
        """
        
        
        positions = measurements['positions']
        category = measurements['category']
        
        if category == 'points':
            self.wind_field = xr.Dataset({'eastward_wind':(['point'], u),
                                        'northward_wind':(['point'], v),
                                        'upward_air_velocity':(['point'], w),
                                        'wind_speed':(['point'], wind_speed),
                                        'wind_from_direction':(['point'], wind_from_direction)},
                                        coords={'Easting':(['point'], positions[:,0]),
                                                'Northing':(['point'], positions[:,1]),
                                                'Height_asl': (['point'], positions[:,2]),
                                                'Height_agl': (['point'], positions[:,3])}
                                        )
        
        if category == 'horizontal_mesh':
            nrows = measurements['nrows']
            ncols = measurements['ncols']
            self.wind_field = xr.Dataset({'eastward_wind':(['Northing', 'Easting'], u.reshape(nrows, ncols).T),
                                        'northward_wind':(['Northing', 'Easting'], v.reshape(nrows, ncols).T),
                                        'upward_air_velocity':(['Northing', 'Easting'], w.reshape(nrows, ncols).T),
                                        'wind_speed':(['Northing', 'Easting'], wind_speed.reshape(nrows, ncols).T),
                                        'wind_from_direction':(['Northing', 'Easting'], wind_from_direction.reshape(nrows, ncols).T)},
                                        coords={'Easting': np.unique(positions[:,0]),
                                                'Northing': np.unique(positions[:,1]),
                                                'Height_asl':  positions[1,2],
                                                'Height_agl':  positions[1,3]}
                                        )
        self.wind_field.attrs['title'] = 'Wind characteristics at measurement points of interest'
        self.wind_field.attrs['convention'] = 'cf'
        self.wind_field.attrs['atmospheric_model'] = atmosphere['model']
        self.wind_field.attrs['atmospheric_model_parameters'] = atmosphere['model_parameters']
        self.wind_field.eastward_wind.attrs['units'] = 'm s-1'
        self.wind_field.northward_wind.attrs['units'] = 'm s-1'
        self.wind_field.upward_air_velocity.attrs['units'] = 'm s-1'
        self.wind_field.wind_speed.attrs['units'] = 'm s-1'
        self.wind_field.wind_from_direction.attrs['units'] = 'degree'
        self.wind_field.Easting.attrs['units'] = 'm'
        self.wind_field.Northing.attrs['units'] = 'm'
        self.wind_field.Height_asl.attrs['units'] = 'm' 
        self.wind_field.Height_agl.attrs['units'] = 'm' 

    def __calculate_wind(self, measurements_id, atmosphere_id):
        """
        Calculates wind characteristics at the selected measurement points. 

        
        Parameters
        ----------
        measurements_id : str, required
            String which identifies measurements instance in the dictionary.
        atmosphere_id : str, required
            String which identifies atmosphere instance in the dictionary which
            is used to calculate wind vector at measurement points    
        """        
        atmosphere = self.atmosphere[atmosphere_id]
        measurements = self.measurements[measurements_id]
        
        shear_exponent = atmosphere['model_parameters']['shear_exponent']
        reference_height = atmosphere['model_parameters']['reference_height']

        gain = (measurements['positions'][:,3] / reference_height)**shear_exponent

        u = atmosphere['model_parameters']['eastward_wind'] * gain
        v = atmosphere['model_parameters']['northward_wind'] * gain
        w = np.full(gain.shape, atmosphere['model_parameters']['upward_air_velocity'])
        wind_speed = atmosphere['model_parameters']['wind_speed'] * gain
        wind_from_direction = np.full(gain.shape, atmosphere['model_parameters']['wind_from_direction'])
        self.__create_wind_ds(atmosphere, measurements, 
                              u, v, w, wind_speed, wind_from_direction)                                

class Instruments:
    """
    A class containing basic methods to operate on instruments dictionary.
    """    
    __KWARGS = {'uncertainty_model',
                'u_estimation', 
                'u_range', 
                'u_azimuth', 
                'u_elevation', 
                'u_radial', 
                'range_gain', 
                'azimuth_gain', 
                'elevation_gain',
                'atmosphere_id',              
                'measurements_id', 
                'probing_coordinates', 
                'radial_velocity', 
                'coordinate_system',
                'coordinates',
                'category',
                'linked_instruments' }
        
    def __init__(self):
        self.instruments = {}
        
    @staticmethod
    def check_instrument_position(instrument_position):
        """
        Validates the position of instrument
        
        Parameters
        ----------
        instrument_position : ndarray
            nD array containing data with `float` or `int` type
            corresponding to x, y and z coordinates of a lidar.
            nD array data are expressed in meters.
        
        Returns
        -------
            True / False
        """        
        if(type(instrument_position).__module__ == np.__name__):
                if (len(instrument_position.shape) == 1 
                    and instrument_position.shape[0] == 3):
                        return True
                else:
                    # print('Wrong dimensions! Must be == 3 !')
                    return False
        else:
            # print('Input is not numpy array!')
            return False        

    def update_instrument(self, instrument_id, **kwargs):
        """
        Updates a instrument instance in dictionary with information in kwargs.
        
        Parameters
        ----------
        instrument_id : str, required
            String which identifies instrument in the instrument dictionary.

        Other Parameters
        -----------------
        u_estimation : float, optional
            Uncertainty in estimating radial velocity from Doppler spectra.
            Unless provided, (default) value is set to 0.1 m/s.
        u_range : float, optional
            Uncertainty in detecting range at which atmosphere is probed.
            Unless provided, (default) value is set to 1 meter.
        u_azimuth : float, optional
            Uncertainty in the beam steering for the azimuth angle.
            Unless provided, (default) value is set to 0.1 degree.
        u_elevation : float, optional
            Uncertainty in the beam steering for the elevation angle.
            Unless provided, (default) value is set to 0.1 degree.
            
        Raises
        ------
        WrongId
            If for the provided instrument_id there is no key in the dictionary.
        WrongKwargs
            If one or more kwargs are incorrect.
                  
        Notes
        -----
        If end-user manually updates keys essential for uncertainty calculation
        auto-update of the uncertainty values will not take place! 
        Therefore, to update uncertainty values end-user must re-execute 
        calculate_uncertainty method.

        TODO
        ----
        - If certain keys are changes/updated trigger the uncertainty re-calc.
        """
        if instrument_id not in self.instruments:
            raise ValueError("WrongId")

        if (len(kwargs) > 0 and not(set(kwargs).issubset(self.__KWARGS))):
            raise ValueError("WrongKwargs")

        if (len(kwargs) > 0 and set(kwargs).issubset(self.__KWARGS)):             
            for key in kwargs:
                if key in {'u_estimation', 'u_range', 'u_azimuth', 'u_elevation'}:
                    self.instruments[instrument_id]['intrinsic_uncertainty'][key] = kwargs[key]    
    
                
class Lidars(Instruments):
    """
    A class containing methods and attributes related to wind lidars.
    
    Methods
    -------
    add_lidar(instrument_id, position, category, **kwargs):
        Adds a lidar instance to the instrument dictionary.        
    """
    def __init__(self):
        super().__init__()        
        
    def add_lidar(self, instrument_id, position, **kwargs):
        """
        Adds a lidar instance to the instrument dictionary.
        
        Parameters
        ----------
        instrument_id : str, required
            String which identifies instrument in the instrument dictionary.
        position : ndarray, required
            nD array containing data with `float` or `int` type corresponding 
            to Northing, Easting and Height coordinates of the instrument.
            nD array data are expressed in meters.
            
        Other Parameters
        -----------------
        u_estimation : float, optional
            Uncertainty in estimating radial velocity from Doppler spectra.
            Unless provided, (default) value is set to 0.1 m/s.
        u_range : float, optional
            Uncertainty in detecting range at which atmosphere is probed.
            Unless provided, (default) value is set to 1 m.
        u_azimuth : float, optional
            Uncertainty in the beam steering for the azimuth angle.
            Unless provided, (default) value is set to 0.1 deg.
        u_elevation : float, optional
            Uncertainty in the beam steering for the elevation angle.
            Unless provided, (default) value is set to 0.1 deg.
        
        Raises
        ------
        InappropriatePosition
            If the provided position of instrument is not properly provided.

        Notes
        --------
        Instruments can be add one at time.
        Currently only the instrument position in UTM coordinate system is supported.

        TODO
        ----
        - Support the instrument position in coordinate systems other than UTM
        - Integrate e-WindLidar attributes and vocabulary for lidar type
        """
        if not(self.check_instrument_position(position)):
            raise ValueError("InappropriatePosition")
        
            
        category="wind_lidar"
        
        instrument_dict = {instrument_id:{
                                            'category': category,
                                            'position': position,
                                            'intrinsic_uncertainty':{
                                                'u_estimation' : 0.1,     # default
                                                'u_range' : 1,     # default
                                                'u_azimuth': 0.1,  # default
                                                'u_elevation': 0.1 # default
                                                }
                                        }
                        }
            
        self.instruments.update(instrument_dict)
        self.update_instrument(instrument_id, **kwargs)
        print('Instrument \'' + instrument_id + '\' of category \'' + 
            category +'\' added to the instrument dictionary, ' + 
            'which now contains ' + str(len(self.instruments)) + 
            ' instrument(s).')
    

                

class Uncertainty(Measurements, Lidars):
    """
    A class containing methods to calculate single- and dual- Doppler uncertainty.

    Methods
    -------
    add_atmosphere(atmosphere_id, model, model_parameters)
        Adds description of the atmosphere to the atmosphere dictionary.
    add_instrument(instrument_id, position, category, **kwargs):
        Adds an instrument to the instrument dictionary.
    add_measurements(measurements_id, category, **kwargs)
        Adds desired measurement positions to the measurements dictionary.
    calculate_uncertainty(instrument_ids, measurements_id, atmosphere_id, uncertainty_model)
        Calculates a measurement uncertainty for a given instrument(s).    
    """
    

    def __init__(self):
        self.uncertainty = None
        Instruments.__init__(self)
        Measurements.__init__(self)
                 

    def __create_rad_ds(self, instrument_id, measurements):
        """
        Creates radial wind speed uncertainty xarray object.
        
        Parameters
        ----------
        instrument_id : str
            String indicating the instrument in the dictionary to be considered.
        measurements : dict
            Dictionary containing information on measurements.
            
        
        Returns
        -------
        ds : xarray
            xarray dataset containing radial wind speed uncertainty.
                        
        Notes
        ----
        Currently this method only supports points and horizontal mesh data structures.
        The method can be called only when the radial uncertainty has been calculated.
        """
        positions = measurements['positions']
        category = measurements['category']
        intrinsic_uncertainty = self.instruments[instrument_id]['intrinsic_uncertainty']
        
        if category == 'points':
            prob_cord = self.__probing_dict[instrument_id]
            rad_speed = self.__radial_vel_dict[instrument_id]
            azimuth_gain = self.__radial_uncertainty[instrument_id]['azimuth_gain']
            elevation_gain = self.__radial_uncertainty[instrument_id]['elevation_gain']
            range_gain = self.__radial_uncertainty[instrument_id]['range_gain']
            u_radial = self.__radial_uncertainty[instrument_id]['u_radial']

            ds = xr.Dataset({'azimuth':(['instrument_id','point'], np.array([prob_cord[:,0]])),
                            'elevation':(['instrument_id','point'], np.array([prob_cord[:,1]])),
                            'range':(['instrument_id','point'], np.array([prob_cord[:,2]])),
                            'radial_speed':(['instrument_id','point'], np.array([rad_speed])),
                            'azimuth_gain':(['instrument_id','point'], np.array([azimuth_gain])),
                            'elevation_gain':(['instrument_id','point'], np.array([elevation_gain])),                            
                            'range_gain':(['instrument_id','point'], np.array([range_gain.T])),
                            'radial_speed_uncertainty':(['instrument_id','point'], np.array([u_radial])),
                            # 'instrument_uncertainty':(['instrument_id'], np.array([intrinsic_uncertainty]))
                            },
                            coords={'Easting':(['point'], positions[:,0]),
                                    'Northing':(['point'], positions[:,1]),
                                    'Height': (['point'], positions[:,2]),
                                    'instrument_id': np.array([instrument_id])}                             
                                        )             
            
        
        if category == 'horizontal_mesh':
            nrows = measurements['nrows']
            ncols = measurements['ncols']
            
            prob_cord = self.__probing_dict[instrument_id].reshape(nrows, ncols,3)
            rad_speed = self.__radial_vel_dict[instrument_id].reshape(nrows, ncols)
            azimuth_gain = self.__radial_uncertainty[instrument_id]['azimuth_gain'].reshape(nrows, ncols)
            elevation_gain = self.__radial_uncertainty[instrument_id]['elevation_gain'].reshape(nrows, ncols)
            range_gain = self.__radial_uncertainty[instrument_id]['range_gain'].reshape(nrows, ncols)
            u_radial = self.__radial_uncertainty[instrument_id]['u_radial'].reshape(nrows, ncols)


            ds = xr.Dataset({'azimuth':(['instrument_id', 'Northing', 'Easting'], 
                                        np.array([prob_cord[:,:, 0].T])),
                            'elevation':(['instrument_id', 'Northing', 'Easting'], 
                                         np.array([prob_cord[:,:, 1].T])),
                            'range':(['instrument_id', 'Northing', 'Easting'], 
                                     np.array([prob_cord[:,:, 2].T])),
                            'radial_speed':(['instrument_id', 'Northing', 'Easting'], 
                                            np.array([rad_speed.T])),
                            'azimuth_gain':(['instrument_id', 'Northing', 'Easting'], 
                                            np.array([azimuth_gain.T])),
                            'elevation_gain':(['instrument_id', 'Northing', 'Easting'], 
                                              np.array([elevation_gain.T])),                            
                            'range_gain':(['instrument_id', 'Northing', 'Easting'], 
                                          np.array([range_gain.T])),
                            'radial_speed_uncertainty':(['instrument_id', 'Northing', 'Easting'], 
                                                        np.array([u_radial.T])),
                            'intrinsic_uncertainty':(['instrument_id'], np.array([intrinsic_uncertainty]))                                                                                    
                            },
                            coords={'Easting': np.unique(positions[:,0]), 
                                    'Northing': np.unique(positions[:,1]),
                                    'instrument_id': np.array([instrument_id]),
                                    'Height':  positions[0,2]}
                                        ) 
            


            
        return ds
    

    def __create_dd_ds(self, measurements):
        """
        Creates dual-Doppler uncertainty xarray object.
        
        Parameters
        ----------
        measurements : dict
            Dictionary containing information on measurements.
            
        
        Returns
        -------
        ds : xarray
            xarray dataset containing dual-Doppler uncertainty.
                        
        Notes
        ----
        Currently this method only supports points and horizontal mesh data structures.
        The method can be called only when the dual-Doppler uncertainty has been calculated.
        """
        
        positions = measurements['positions']
        category = measurements['category']
        
        if category == 'points':
            ds = xr.Dataset({'wind_speed_uncertainty':(['point'], 
                                                       self.__wind_speed_uncertainty),
                             'wind_from_direction_uncertainty':(['point'], 
                                                                self.__wind_from_direction_uncertainty),
                            },
                            coords={'Easting':(['point'], positions[:,0]),
                                    'Northing':(['point'], positions[:,1]),
                                    'Height': (['point'], positions[:,2])}                         
                                        )             
            
        
        if category == 'horizontal_mesh':
            ds = xr.Dataset({'wind_speed_uncertainty':(['Northing', 'Easting'], self.__wind_speed_uncertainty),
                             'wind_from_direction_uncertainty':(['Northing', 'Easting'], self.__wind_from_direction_uncertainty),
                            },
                            coords={'Easting': np.unique(positions[:,0]), 
                                    'Northing': np.unique(positions[:,1]),
                                    'Height':  positions[0,2]})   
            
        return ds

    @staticmethod
    def __update_metadata(ds, uncertainty_model):
        """
        Updates xarray dataset with metadata.
        
        Parameters
        ----------
        ds : xarray
            xarray dataset containing radial and/or dual-Doppler uncertainty.        
        uncertainty_model : str
            String indicating which uncertainty model was used for the uncertainty calculation.
        
        Returns
        -------
        ds : xarray
            xarray dataset updated with metadata. 
        """        
        # Update of metadata here
        ds.attrs['title'] = 'Radial speed uncertainty'
        ds.attrs['convention'] = 'cf'
        ds.attrs['uncertainty_model'] = 'Vasiljevic-Courtney_' + uncertainty_model
        ds.azimuth.attrs['units'] = 'degree'
        ds.elevation.attrs['units'] = 'degree'
        ds.range.attrs['units'] = 'm'
        ds.radial_speed.attrs['units'] = 'm s-1'
        ds.radial_speed.attrs['standard_name'] = 'radial_velocity_of_scatterers_toward_instrument'
        ds.radial_speed_uncertainty.attrs['units'] = 'm s-1'
        ds.azimuth_gain.attrs['units'] =   'rad-1'
        ds.elevation_gain.attrs['units'] = 'rad-1'
        ds.range_gain.attrs['units'] = 'm-1'
        ds.Easting.attrs['units'] = 'm'
        ds.Northing.attrs['units'] = 'm'
        ds.Height.attrs['units'] = 'm'
        if uncertainty_model == 'dual-Doppler':
            ds.attrs['title'] = 'Dual-Doppler uncertainty'
            ds.attrs['uncertainty_model'] = 'Vasiljevic-Courtney_' + uncertainty_model
            ds.wind_from_direction_uncertainty.attrs['units'] = 'degree'
            ds.wind_speed_uncertainty.attrs['units'] = 'm s-1'
        return ds




    def __calculate_azimuth_gain(self, instrument_id):
        """
        Calculates the gain for the azimuth component of the radial uncertainty.
        
        Parameters
        ----------
        instrument_id : str
            String indicating the instrument in the dictionary to be considered.
            
        Returns
        -------
        gain : ndarray
            nD array of azimuth gains for each measurement point.
        
        Notes
        --------
        The azimuth gain, :math:`{A_{{\\theta}}}`, is calculated using the 
        following mathematical expression:
        
        .. math::
            A_{{\\theta}} = \sin({\\theta} -\Theta ) \cos({\\varphi})
        
        where :math:`{\\theta}` and :math:`{\\varphi}` are the azimuth and 
        elevation angle of the beam, while :math:`{\Theta}` is the wind direction.
       
        """
        # Pull wind direction from xarray object and 
        # unwrap it to secure it is a 1D array.
        wind_from_direction = self.wind_field.wind_from_direction.values.T.reshape(-1)
        coords = self.__probing_dict[instrument_id]
        

        gain = (np.sin(np.radians(coords[:,0]) - np.radians(wind_from_direction)) *
                np.cos(np.radians(coords[:,1])))
        
        return gain
    
    def __calculate_elevation_gain(self, instrument_id):
        """
        Calculates the gain for the elevation component of the radial uncertainty.
        
        Parameters
        ----------
        instrument_id : str
            String indicating the instrument in the dictionary to be considered.
        
        Returns
        -------
        gain : ndarray
            nD array of elevation gains for each measurement point.
                    
        Notes
        --------
        The elevation gain, :math:`{A_{{\\varphi}}}`, is calculated using the 
        following mathematical expression:
        
        .. math::
            A_{{\\varphi}} = ({\\alpha} \cot({\\varphi})^{2}-1)\cos({\\theta} -\Theta)\sin({\\varphi})
        
        where :math:`{\\theta}` and :math:`{\\varphi}` are the azimuth and 
        elevation angle of the beam, :math:`{\Theta}` is the wind direction, 
        while :math:`{\\alpha}` is the wind shear exponent.
       
        """

        coords = self.__probing_dict[instrument_id]

        # Pull wind direction and wind shear from xarray object 
        wind_from_direction = self.wind_field.wind_from_direction.values.reshape(-1)
        shear_exponent = self.wind_field.attrs['atmospheric_model_parameters']['shear_exponent']

        gain = ((shear_exponent * (1/np.tan(np.radians(coords[:,1])))**2 - 1) * 
                np.cos(np.radians(coords[:,0]) - np.radians(wind_from_direction)) * 
                np.sin(np.radians(coords[:,1])))
        
        return gain

    def __calculate_range_gain(self, instrument_id):
        """
        Calculates the gain for the range component of the radial uncertainty.
        
        Parameters
        ----------
        instrument_id : str
            String indicating the instrument in the dictionary to be considered.
        
        Returns
        -------
        gain : ndarray
            nD array of range gains for each measurement point.
                    
        Notes
        --------
        The range gain, :math:`{A_{R}}`, is calculated using the 
        following mathematical expression:
        
        .. math::
            A_{R} = \\frac{{\\alpha}}{R} \cos({\\theta} -\Theta)\cos({\\varphi})
        
        where :math:`{\\theta}` and :math:`{\\varphi}` are the azimuth and 
        elevation angle of the beam, :math:`{\Theta}` is the wind direction, 
        :math:`{\\alpha}` is the wind shear exponent, while :math:`{R}` is 
        the range at which a measurement point is located along the beam.
       
        """

        coords = self.__probing_dict[instrument_id]
        # Pull wind direction and wind shear from xarray object 
        wind_from_direction = self.wind_field.wind_from_direction.values.T.reshape(-1)
        shear_exponent = self.wind_field.attrs['atmospheric_model_parameters']['shear_exponent']
        
        
        gain = ((shear_exponent / coords[:,2]) * 
                np.cos(np.radians(coords[:,0]) - 
                       np.radians(wind_from_direction)) * 
                np.cos(np.radians(coords[:,1])))
        
        return gain


    def __calculate_radial_uncertainty(self, instrument_id):
        """
        Calculates the radial wind speed uncertainty.
        
        Parameters
        ----------
        instrument_id : str
            String indicating the instrument in the dictionary to be considered.
        
        Returns
        -------
        dict_out : dict
            Dictionary containing selected uncertainty model, calculated radial 
            uncertainty and gains for each individual uncertainty component.
            
        Notes
        --------
        The radial wind speed uncertainty, :math:`{u_{V_{radial}}}`, is calculated 
        using the following mathematical expression:
        
        .. math::
            u_{V_{radial}}^2 = u_{estimation}^2 +
                           (V_{h} u_{{\\theta}} A_{{\\theta}} )^2 +
                           (V_{h} u_{{\\varphi}} A_{{\\varphi}} )^2 +
                           (V_{h} u_{R} A_{R} )^2 
                        
        
        where :math:`{u_{estimation}}`, :math:`{u_{{\\theta}}}`, :math:`{u_{{\\varphi}}}`,
        and :math:`{u_{R}}` are uncertainties for radial speed estimation, azimuth angle, 
        elevation angle and range respectively, :math:`{A_{{\\theta}}}`, 
        :math:`{A_{{\\varphi}}}` and :math:`{A_{R}}` are the uncertainty components
        gains for azimuth, elevation and range respectively, while :math:`{V_{h}}` 
        is the horizontal wind speed.
        """

        azimuth_gain = self.__calculate_azimuth_gain(instrument_id)
        elevation_gain = self.__calculate_elevation_gain(instrument_id)
        range_gain = self.__calculate_range_gain(instrument_id)

        # Pulling intrinsic uncertainties from the lidar dictionary
        u_azimuth = self.instruments[instrument_id]['intrinsic_uncertainty']['u_azimuth']
        u_elevation = self.instruments[instrument_id]['intrinsic_uncertainty']['u_elevation']
        u_range = self.instruments[instrument_id]['intrinsic_uncertainty']['u_range']
        u_estimation = self.instruments[instrument_id]['intrinsic_uncertainty']['u_estimation']
        
        # Pulling horizontal wind speed from the measurements dictionary
        wind_speed = self.wind_field.wind_speed.values.T.reshape(-1)

        abs_uncertainty = np.sqrt(

            (u_estimation)**2 + \
            (wind_speed * u_azimuth * azimuth_gain * (pi/180) )**2 + \
            # (wind_speed * u_elevation * elevation_gain * (pi/180))**2 + \
            (wind_speed * u_range * range_gain)**2

                                 )

        dict_out = {'azimuth_gain' : azimuth_gain, 
                    'elevation_gain' : elevation_gain,
                    'range_gain': range_gain,
                    'u_radial' : abs_uncertainty,
                    'uncertainty_model' : 'radial_velocity'
                    }

        return dict_out

    def __calculate_DD_speed_uncertainty(self, instrument_ids):
        """
        Calculates the dual-Doppler wind speed uncertainty.
        
        Parameters
        ----------
        instrument_id : str
            String indicating the instrument in the dictionary to be considered.
        
        Returns
        -------
        uncertainty : ndarray
            nD array of calculated dual-Doppler wind speed uncertainty.
            
        Notes
        --------
        The dual-Doppler wind speed uncertainty, :math:`{u_{V_{h}}}`, is calculated 
        using the following mathematical expression:
        
        .. math::

            u_{V_{h}}=\\frac{1}{V_{h} \sin({\\theta}_{1}-{\\theta}_{2})^2} * 
                      \\biggl((V_{radial_{1}}-V_{radial_{2}}\cos({\\theta}_{1}-{\\theta}_{2}))^{2}u_{V_{radial_{1}}}^{2} + 
                          
            (V_{radial_{2}}-V_{radial_{1}}\cos({\\theta}_{1}-{\\theta}_{2}))^{2}u_{V_{radial_{2}}}^{2}\\biggl)^{\\frac{1}{2}}

        where :math:`u_{V_{radial_{1}}}` and :math:`u_{V_{radial_{2}}}` are radial 
        uncertainties for measurements of radial velocities :math:`{V_{radial_{1}}}`
        and :math:`{V_{radial_{2}}}` by a dual-Doppler system (e.g., two lidars), 
        :math:`{\\theta_{1}}` and  :math:`{\\theta_{2}}` are the azimuth angles 
        of the two intersecting beams at a point of interest, while :math:`{V_{h}}` 
        is the horizontal wind speed at that point.       
        """

        azimuth_1 = self.uncertainty.azimuth.sel(instrument_id =instrument_ids[0]).values
        azimuth_2 = self.uncertainty.azimuth.sel(instrument_id =instrument_ids[1]).values
        angle_dif = np.radians(azimuth_1 - azimuth_2) # in radians

        los_1 = self.uncertainty.radial_speed.sel(instrument_id=instrument_ids[0]).values
        U_rad1 = self.uncertainty.radial_speed_uncertainty.sel(instrument_id =instrument_ids[0]).values

        los_2 = self.uncertainty.radial_speed.sel(instrument_id =instrument_ids[1]).values
        U_rad2 = self.uncertainty.radial_speed_uncertainty.sel(instrument_id =instrument_ids[1]).values

        wind_speed = self.wind_field.wind_speed.values

        
        uncertainty =((wind_speed * (np.sin(angle_dif))**2)**-1 * 
                      np.sqrt((los_1 - los_2 * np.cos(angle_dif))**2 * U_rad1**2 + 
                              (los_2 - los_1 * np.cos(angle_dif))**2 * U_rad2**2))

        return uncertainty
    
    def __calculate_DD_direction_uncertainty(self, instrument_ids):
        """
        Calculates the dual-Doppler wind direction uncertainty.
        
        Parameters
        ----------
        instrument_id : str
            String indicating the instrument in the dictionary to be considered.
        
        Returns
        -------
        uncertainty : ndarray
            nD array of calculated dual-Doppler wind speed uncertainty.
            
        Notes
        --------
        The dual-Doppler wind speed uncertainty, :math:`{u_{\Theta}}`, is calculated 
        using the following mathematical expression:
        
        .. math::


            u_{\Theta}=\\biggl(\\frac{u_{V_{radial_{1}}}^{2}V_{radial_{2}}^{2}+u_{V_{radial_{2}}}^{2}V_{radial_{1}}^{2}}{V_{h}^{4}\sin ({\\theta}_{1}-{\\theta}_{2})^{2}}\\biggl)^{\\frac{1}{2}}

        where :math:`u_{V_{radial_{1}}}` and :math:`u_{V_{radial_{2}}}` are radial 
        uncertainties for measurements of radial velocities :math:`{V_{radial_{1}}}`
        and :math:`{V_{radial_{2}}}` by a dual-Doppler system (e.g., two lidars), 
        :math:`{\\theta_{1}}` and  :math:`{\\theta_{2}}` are the azimuth angles 
        of the two intersecting beams at a point of interest, while :math:`{V_{h}}` 
        is the horizontal wind speed at that point.   
       
        """

        azimuth_1 = self.uncertainty.azimuth.sel(instrument_id =instrument_ids[0]).values
        azimuth_2 = self.uncertainty.azimuth.sel(instrument_id =instrument_ids[1]).values
        angle_dif = np.radians(azimuth_1 - azimuth_2) # in radians

        los_1 = self.uncertainty.radial_speed.sel(instrument_id =instrument_ids[0]).values
        U_rad1 = self.uncertainty.radial_speed_uncertainty.sel(instrument_id =instrument_ids[0]).values

        los_2 = self.uncertainty.radial_speed.sel(instrument_id =instrument_ids[1]).values
        U_rad2 = self.uncertainty.radial_speed_uncertainty.sel(instrument_id =instrument_ids[1]).values

        wind_speed = self.wind_field.wind_speed.values

        uncertainty = np.sqrt(((los_1*U_rad2)**2 + (los_2*U_rad1)**2) * 
                               (wind_speed**4 * np.sin(angle_dif)**2)**-1
                                                    )*(180/pi)
        
        return uncertainty


    def calculate_uncertainty(self, instrument_ids, 
                                    measurements_id, 
                                    atmosphere_id, 
                                    uncertainty_model = 'radial_uncertainty'):
        
        """
        Calculates a measurement uncertainty for a given instrument(s).

        Parameters
        ----------
        instrument_ids : list, required
            List of strings which identifies instruments the dictionary.
        measurements_id : str, required
            String corresponding to the key in measurements dictionary.
        atmosphere_id : str, required
            String corresponding to the key in atmosphere dictionary.
        uncertainty_model : str, optional
            String defining uncertainty model used for uncertainty calculations.
            default value set to 'radial_uncertainty'
        
        Notes
        -----
        Currently, this method calculates radial and dual-Doppler
        uncertainty for single (radial uncertainty) or a pair of instruments 
        (radial + dual-Doppler uncertainty). 
        
        The radial wind speed uncertainty, :math:`{u_{V_{radial}}}`, is calculated 
        using the following mathematical expression:
        
        .. math::
            u_{V_{radial}}^2 = u_{estimation}^2 +
                           (V_{h} u_{{\\theta}} A_{{\\theta}} )^2 +
                           (V_{h} u_{{\\varphi}} A_{{\\varphi}} )^2 +
                           (V_{h} u_{R} A_{R} )^2 
                        
        
        where :math:`{u_{estimation}}`, :math:`{u_{{\\theta}}}`, :math:`{u_{{\\varphi}}}`,
        and :math:`{u_{R}}` are uncertainties for radial speed estimation, azimuth angle, 
        elevation angle and range respectively, :math:`{A_{{\\theta}}}`, 
        :math:`{A_{{\\varphi}}}` and :math:`{A_{R}}` are the uncertainty components
        gains for azimuth, elevation and range respectively, while :math:`{V_{h}}` 
        is the horizontal wind speed.
        
        The gains :math:`{A_{{\\theta}}}`, :math:`{A_{{\\varphi}}}` and :math:`{A_{R}}`
        are calculated using the following mathematical expression:
        
        .. math::
            A_{{\\theta}} = \sin({\\theta} -\Theta ) \cos({\\varphi})
                    
        .. math::
            A_{{\\varphi}} = ({\\alpha} \cot({\\varphi})^{2}-1)\cos({\\theta} -\Theta)\sin({\\varphi})

        .. math::
            A_{R} = \\frac{{\\alpha}}{R} \cos({\\theta} -\Theta)\cos({\\varphi})                            
        
        The dual-Doppler wind speed uncertainty, :math:`{u_{V_{h}}}`, is calculated 
        using the following mathematical expression:
        
        .. math::

            u_{V_{h}}=\\frac{1}{V_{h} \sin({\\theta}_{1}-{\\theta}_{2})^2} * 
                      \\biggl((V_{radial_{1}}-V_{radial_{2}}\cos({\\theta}_{1}-{\\theta}_{2}))^{2}u_{V_{radial_{1}}}^{2} + 
                          
            (V_{radial_{2}}-V_{radial_{1}}\cos({\\theta}_{1}-{\\theta}_{2}))^{2}u_{V_{radial_{2}}}^{2}\\biggl)^{\\frac{1}{2}}

        where :math:`u_{V_{radial_{1}}}` and :math:`u_{V_{radial_{2}}}` are radial 
        uncertainties for measurements of radial velocities :math:`{V_{radial_{1}}}`
        and :math:`{V_{radial_{2}}}` by a dual-Doppler system (e.g., two lidars), 
        :math:`{\\theta_{1}}` and  :math:`{\\theta_{2}}` are the azimuth angles 
        of the two intersecting beams at a point of interest, while :math:`{V_{h}}` 
        is the horizontal wind speed at that point.        
        
        The dual-Doppler wind speed uncertainty, :math:`{u_{\Theta}}`, is calculated 
        using the following mathematical expression:
        
        .. math::

            u_{\Theta}=\\biggl(\\frac{u_{V_{radial_{1}}}^{2}V_{radial_{2}}^{2}+u_{V_{radial_{2}}}^{2}V_{radial_{1}}^{2}}{V_{h}^{4}\sin ({\\theta}_{1}-{\\theta}_{2})^{2}}\\biggl)^{\\frac{1}{2}}

        where :math:`u_{V_{radial_{1}}}` and :math:`u_{V_{radial_{2}}}` are radial 
        uncertainties for measurements of radial velocities :math:`{V_{radial_{1}}}`
        and :math:`{V_{radial_{2}}}` by a dual-Doppler system (e.g., two lidars), 
        :math:`{\\theta_{1}}` and  :math:`{\\theta_{2}}` are the azimuth angles 
        of the two intersecting beams at a point of interest, while :math:`{V_{h}}` 
        is the horizontal wind speed at that point.          
        """
        # Check if lidar_ids are correct if not exit
        if not(isinstance(instrument_ids, list)):
            raise ValueError('Instrument ids not provided as a list of strings!')
        if not(all(isinstance(id, str) for id in instrument_ids)):
            raise ValueError('One or more items in instrument id list not strings!')
        if not(set(instrument_ids).issubset(set(self.instruments))):
            raise ValueError('One or more ids don\'t exist in the instrument dictionary!')

        # Check if measurements_id is correct if not exit
        if not(isinstance(measurements_id, str)):
            raise ValueError('measurements_id is not a string!')    
        if measurements_id not in set(self.measurements):
            raise ValueError('Measurements id does not exist in the measurement dictionary!')
        if len(self.measurements[measurements_id]['positions']) == 0:
            raise ValueError('Measurements dictionary empty for the given id!')

        # Check if atmosphere_id is correct if not exit
        if not(isinstance(atmosphere_id, str)):
            raise ValueError('atmosphere_id is not a string!')
        if atmosphere_id not in set(self.atmosphere):
            raise ValueError('atmosphere_id does not exist in self.atmosphere!')

        # Calculate wind field

        atmosphere = self.atmosphere[atmosphere_id]
        measurements = self.measurements[measurements_id]
        
        self._Measurements__calculate_wind(measurements_id, atmosphere_id)
        
        # Calculate probing angles
        self.__probing_dict = {}
        for id in instrument_ids:
            coords = generate_beam_coords(self.instruments[id]['position'],
                                          measurements['positions'])
            self.__probing_dict.update({id:coords})
        
        # Calculate radial velocities for each lidar
        self.__radial_vel_dict = {}
        for id in instrument_ids:
            
            los = wind_vector_to_los(self.wind_field.eastward_wind.values.reshape(-1),
                                     self.wind_field.northward_wind.values.reshape(-1),
                                     self.wind_field.upward_air_velocity.values.reshape(-1),
                                     self.__probing_dict[id][:,0],
                                     self.__probing_dict[id][:,1])
            self.__radial_vel_dict.update({id:los})   
            
        self.radial_vel_dict = self.__radial_vel_dict
        self.probing_dict = self.__probing_dict
        # Calculate radial velocity uncertainty for each lidar
        self.__radial_uncertainty = {}
        for id in instrument_ids:
            self.__radial_uncertainty.update({id: self.__calculate_radial_uncertainty(id)})
            
        # Make radial uncertainty xarray object
        
        for i,id in enumerate(instrument_ids):
            ds_temp = self.__create_rad_ds(id, measurements)
            if i == 0:
                self.uncertainty = ds_temp
            else:
                self.uncertainty = xr.merge([self.uncertainty, ds_temp])
        self.uncertainty = self.__update_metadata(self.uncertainty, 'radial_uncertainty')
        
        if uncertainty_model == 'dual-Doppler':
            if len(instrument_ids) != 2:
                raise ValueError('instrument_ids must contain exactly two ids!')
            
            self.__wind_speed_uncertainty = self.__calculate_DD_speed_uncertainty(instrument_ids)
            self.__wind_from_direction_uncertainty = self.__calculate_DD_direction_uncertainty(instrument_ids)
            ds_temp = self.__create_dd_ds(measurements)
            self.uncertainty = xr.merge([self.uncertainty, ds_temp])
            self.uncertainty = self.__update_metadata(self.uncertainty, 'dual-Doppler')