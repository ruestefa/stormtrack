
import numpy as np

clon, clat = 0.0, 30.0
rad_km = 800.0
area_km2 = np.pi*rad_km**2

nlat, nlon = 31, 35
lat1d = np.linspace(22.5, 37.5, nlat)
lon1d = np.linspace(-8.5, 8.5, nlon)
lat2d, lon2d = np.meshgrid(lat1d, lon1d)

_, X = 0, 1
mask = np.array([
[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
[_,_,_,_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_,_,_,_,_],
[_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_,_],
[_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_],
[_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_],
[_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_],
[_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_],
[_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_],
[_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_],
[_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
[_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
[_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
[_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
[_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
[_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
[_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
[_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
[_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
[_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
[_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
[_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_],
[_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_],
[_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_],
[_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_],
[_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_],
[_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_],
[_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_],
[_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_],
[_,_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_,_,_],
[_,_,_,_,_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_,_,_,_,_],
[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
], np.bool).T[:, ::-1]

