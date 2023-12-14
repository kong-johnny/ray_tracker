import numpy as np
def sphere(
        position,
        radius,
        color,
        reflection=0.85,
        diffuse=1,
        specular_c=0.6,
        specular_k=50,
):
    return dict(
        type='sphere',
        position=np.array(position),
        radius=np.array(radius),
        color=np.array(color),
        reflection=reflection,
        diffuse=diffuse,
        specular_c=specular_c,
        specular_k=specular_k
    )