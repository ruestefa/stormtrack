import numpy as np

# fmt: off

clon, clat = 0.0, 40.0
rad_km = 800.0
area_km2 = np.pi*rad_km**2

nlat, nlon = 17, 21
lat1d = np.linspace(32.0, 48.0, nlat)
lon1d = np.linspace(-10.0, 10.0, nlon)
lat2d, lon2d = np.meshgrid(lat1d, lon1d)

_, X = 0, 1
mask = np.array([
[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
[_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_],
[_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_],
[_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_],
[_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
[_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
[_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
[_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
[_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
[_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
[_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
[_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
[_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_],
[_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_],
[_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_],
[_,_,_,_,_,_,_,X,X,X,X,X,X,X,_,_,_,_,_,_,_],
[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
], np.bool).T[:, ::-1]