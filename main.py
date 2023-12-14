from cube import cube
from plane import plane
from sphere import sphere
from utils import intersect_color, normalize
import numpy as np

w, h = 1200, 900
O = np.array([.5, 0.35, -1.])  # camera origin
Q = np.array([0., 0., 0.])  # camera pointing to
img = np.zeros((h, w, 3))  # image array
rate = float(w) / h  # aspect ratio
S = (-1., -1. / rate + .25, 1., 1. /  rate + .25)  # screen coordinates

# scene objects
scene = [
    sphere([0.75, .1, 1.], .6, [0.8, 0.3, 0]),
    cube([-0.3, .01, .2], .3, [0, 0, 0.9]),
    sphere([-2.75, 0.1, 3.5], 0.6, [0.1, 0.572, 0.184]),
    plane([0., -.5, 0.], [0., 1., 0.])
]
light_point = np.array([5, 5., -10.])  # light source
light_color = np.ones(3)  # light color
ambient = 0.05  # 

for i, x in enumerate(np.linspace(S[0], S[2], w)):
    print("%.2f" % (i / float(w) * 100), "%")
    for j, y in enumerate(np.linspace(S[1], S[3], h)):
        Q[:2] = (x, y)
        D = normalize(Q - O)
        res_col = intersect_color(O, D, 1., scene, light_point, light_color, ambient)
        # print(res_col)
        img[h - j - 1, i, :] = res_col

from matplotlib import pyplot as plt

plt.imsave('test.png', img)