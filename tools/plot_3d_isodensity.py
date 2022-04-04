# To run this file, install 'plotly'
# https://plotly.com/python/getting-started

import plotly.graph_objects as go
import numpy as np
import scipy.io as sio

mdic = sio.loadmat("data_simulation_chin18.4/fields_200000.mat", squeeze_me=True)
nx = mdic['nx']
lx = mdic['lx']
phi_a = mdic['phi_a']

X, Y, Z = np.mgrid[0:nx[0], 0:nx[1], 0:nx[2]]
values = np.reshape(phi_a, nx)

fig = go.Figure(data=go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    isomin=0.0,
    isomax=1.0,
    opacity=0.2, 
    surface_count=9, # number of isosurface. 2 by default:only min and max
    #caps=dict(x_show=False, y_show=False, z_show=False),
    colorscale='RdBu', #RdBu, jet, Plotly3, OrRd
    colorbar=dict(thickness=30,tickfont=dict(size=30,color="black")),
    reversescale=True,
    ))
fig.update_scenes(
    camera_projection_type="orthographic",
    xaxis_visible=False,
    yaxis_visible=False,
    zaxis_visible=False)
fig.write_html('3d_isosurface.html', auto_open=True)
#fig.write_image('3d_volume.svg')
