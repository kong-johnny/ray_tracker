import numpy as np

def normalize(x):
    x /= np.linalg.norm(x)
    return x

def get_normal(obj, M):
    # coordination def: [0] is x, - left, + right [1] is y - down + up
    # obj: object
    # M: point on the surface of obj
    if obj['type'] == 'sphere':
        N = normalize(M - obj['position'])
    elif obj['type'] == 'plane':
        N = obj['normal']
    elif obj['type'] == 'cube':
        # up coordinate: 0, 1, 0
        # down coordinate: 0, -1, 0
        # left coordinate: -1, 0, 0
        # right coordinate: 1, 0, 0
        # front coordinate: 0, 0, -1
        # back coordinate: 0, 0, 1
        P = obj['position']
        L = obj['length']
        
        if abs(M[1] - P[1] - L/2) < 1e-9:
            return np.array([0, 1, 0])
        if abs(M[0] - P[0] - L/2) < 1e-9:
            return np.array([1, 0, 0])
        if abs(P[0] - M[0] - L/2) < 1e-9:
            return np.array([-1, 0, 0])
        if abs(P[1] - M[1] - L/2) < 1e-9:
            return np.array([0, -1, 0])
        if abs(P[2] - M[2] - L/2) < 1e-9:
            return np.array([0, 0, -1])
        if abs(M[2] - P[2] - L/2) < 1e-9:
            return np.array([0, 0, 1])
        N = np.zeros(3)

    return N

def intersect_plane(O, D, P, N):
    # O: ray origin
    # D: ray direction
    # P: any point on the plane
    # N: normal of the plane
    # return: distance from O to the intersection point
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-9:
        return np.inf  # parallel to the plane
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    return d

def intersect_sphere(O, D, C, R):
    # O: ray origin
    # D: ray direction
    # C: center of the sphere
    # R: radius of the sphere
    # return: distance from O to the intersection point
    a = np.dot(D, D)
    OC = C - O
    if (np.linalg.norm(OC) < R) or (np.dot(OC, D) < 0):
        return np.inf
    l = np.linalg.norm(np.dot(OC, D))
    m = np.linalg.norm(OC) * np.linalg.norm(OC) - l * l
    q = R * R - m

    return (l - np.sqrt(q)) if q >= 0 else np.inf

def intersect_cube(O, D, P, L):
    # O: ray origin
    # D: ray direction
    # P: position of the cube
    # L: length of the cube
    # return: distance from O to the intersection point
    assert(abs(np.linalg.norm(D) - 1) < 1e-9)
    t = np.inf
    # intersect with z=P[2] - L/2 and z=P[2] + L/2
    if abs(D[2]) < 1e-9:
        pass # parallel to the plane
    else:
        rate = (P[2] - L/2 - O[2]) / D[2]
        if rate > 0:
            intersect_point = O + rate * D
            if abs(intersect_point[0] - P[0]) < L/2 and abs(intersect_point[1] - P[1]) < L/2:
                t = rate
        rate = (P[2] + L/2 - O[2]) / D[2]
        if rate > 0:
            intersect_point = O + rate * D
            if abs(intersect_point[0] - P[0]) < L/2 and abs(intersect_point[1] - P[1]) < L/2:
                t = min(t, rate)
    # intersect with y=P[1] - L/2 and y=P[1] + L/2
    if abs(D[1]) < 1e-9:
        pass
    else:
        rate = (P[1] - L/2 - O[1]) / D[1]
        if rate > 0:
            intersect_point = O + rate * D
            if abs(intersect_point[0] - P[0]) < L/2 and abs(intersect_point[2] - P[2]) < L/2:
                t = min(t, rate)
        rate = (P[1] + L/2 - O[1]) / D[1]
        if rate > 0:
            intersect_point = O + rate * D
            if abs(intersect_point[0] - P[0]) < L/2 and abs(intersect_point[2] - P[2]) < L/2:
                t = min(t, rate)
    # intersect with x=P[0] - L/2 and x=P[0] + L/2
    if abs(D[0]) < 1e-9:
        pass
    else:
        rate = (P[0] - L/2 - O[0]) / D[0]
        if rate > 0:
            intersect_point = O + rate * D
            if abs(intersect_point[1] - P[1]) < L/2 and abs(intersect_point[2] - P[2]) < L/2:
                t = min(t, rate)
        rate = (P[0] + L/2 - O[0]) / D[0]
        if rate > 0:
            intersect_point = O + rate * D
            if abs(intersect_point[1] - P[1]) < L/2 and abs(intersect_point[2] - P[2]) < L/2:
                t = min(t, rate)
    return t

def intersect(O, D, obj):
    # O: ray origin
    # D: ray direction
    # obj: object
    # return: distance from O to the intersection point
    if obj['type'] == 'plane':
        return intersect_plane(O, D, obj['position'], obj['normal'])
    elif obj['type'] == 'sphere':
        return intersect_sphere(O, D, obj['position'], obj['radius'])
    elif obj['type'] == 'cube':
        return intersect_cube(O, D, obj['position'], obj['length'])
    

def get_color(O, P):
    color = O['color']
    if not hasattr(color, '__len__'):
        color = color(P)
    return color

def intersect_color(O, D, I, scene, light_point, light_color, ambient):
    # O: ray origin
    # D: ray direction
    # I: intensity
    # return: color of the intersection point

    min_dist = np.inf
    for i, obj in enumerate(scene):
        cur_dist = intersect(O, D, obj)
        if cur_dist < min_dist:
            min_dist = cur_dist
            obj_idx = i
    # print(min_dist)
    # exit()
    if min_dist == np.inf or I < 0.01:
        return np.zeros(3)
    
    obj = scene[obj_idx]
    P = O + min_dist * D  # intersection
    
    color = get_color(obj, P)
    N = get_normal(obj, P)
    toL = normalize(light_point - P)
    toO = normalize(O - P)

    c = ambient * color

    l = [intersect(P + N * .0001, toL, obj_sh) for k, obj_sh in enumerate(scene) if k != obj_idx]

    if not (l and min(l) < np.linalg.norm(light_point - P)):
        c += color * obj['diffuse'] * max(np.dot(N, toL), 0) * light_color
        c += obj['specular_c'] * max(np.dot(N, normalize(toL + toO)), 0) ** obj['specular_k'] * light_color

    reflect_ray = D - 2 * np.dot(D, N) * N
    c += obj['reflection'] * intersect_color(P + N * .0001, reflect_ray, I * obj['reflection'], scene, light_point, light_color, ambient)
    
    return np.clip(c, 0, 1)