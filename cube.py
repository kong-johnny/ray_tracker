import numpy as np

def cube(
        position,
        length,
        color,
        reflection=0.6,
        diffuse=1,
        specular_c=0.6,
        specular_k=50,
):
    
    return dict(
        type='cube',
        position=np.array(position),
        length=np.array(length),
        color=np.array(color),
        reflection=reflection,
        diffuse=diffuse,
        specular_c=specular_c,
        specular_k=specular_k
    )