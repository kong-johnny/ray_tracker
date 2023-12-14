import numpy as np

def plane(
        position, 
        normal, 
        color=np.array([1., 1., 1.]), 
        reflection=0.15,
        diffuse=0.75,
        specular_c=0.3,
        specular_k=50,
        ):
    return dict(
        type='plane',
        position=np.array(position),
        normal=np.array(normal),
        # 默认为棋盘格
        color=lambda P: (np.array([1.,1.,1.]) if (int(P[0]*2)%2) == (int(P[2]*2)%2) else (np.array([0.,0.,0.]))),
        reflection=reflection, 
        diffuse=diffuse, 
        specular_c=specular_c, 
        specular_k=specular_k
    )