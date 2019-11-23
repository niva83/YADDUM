import yaddum as yaddum
import numpy as np  
import matplotlib.pyplot as plt
import xarray as xr

lidar_uncertainty = yaddum.Uncertainty()

model_pars={'wind_speed':10,
            'upward_air_velocity':0,
            'wind_from_direction':0,
            'reference_height':100,
            'shear_exponent':0.2}
lidar_uncertainty.add_atmosphere('pl_1', 'power_law', model_pars)


lidar_uncertainty.add_measurements('mesh', 'horizontal_mesh', 
                                   resolution = 10, 
                                   mesh_center = np.array([0,0,100]), 
                                   extent = 5000)

points = np.array([[500,-500,100], [1000,2,300]])

lidar_uncertainty.add_measurements('pts', 'points', positions = points)


uncertainty_pars = {'u_estimation':0.1,
                    'u_azimuth':0.1,
                    'u_elevation':0.1, 
                    'u_range':1}

lidar_pos_1 = np.array([0,0,0])
lidar_pos_2 = np.array([1000,1000,0])


lidar_uncertainty.add_lidar('koshava', lidar_pos_1, **uncertainty_pars)
lidar_uncertainty.add_lidar('whittle', lidar_pos_2, **uncertainty_pars)

lidar_uncertainty.calculate_uncertainty(['koshava', 'whittle'], 'mesh', 'pl_1', 
                                        uncertainty_model='dual-Doppler')

lidar_uncertainty.uncertainty.azimuth_gain.sel(instrument_id = 'koshava').plot()
plt.show()