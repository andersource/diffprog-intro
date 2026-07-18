import open3d as o3d
import numpy as np
# PARAMS = [
#     (8, 0, 0, 0),
#     (5.5, 0, 8, 0),
#     (.8, 0, 9, 5)
# ]

# PARAMS = [
#     (3.5, 0, 0, 0),
#     (3, -4.5, 0, 0),
#     (3, 4.5, 0, 0),
#     (3, 0, -4.5, 0),
#     (3, 0, 4.5, 0),
# ]

# PARAMS = []
# for t in np.linspace(0, 1, 11):
#     r = 1.1 + (1.3 * (1 - 2 * np.abs((.5 - t) ** 1))) ** 2
#     r_big = 2 + 5.5 * t
#     PARAMS.append((
#         r, 20 * t - 10,
#         r_big * np.sin(2.4 * np.pi * t),
#         r_big * np.cos(2.4 * np.pi * t)
#     ))

import json

POSE = json.loads("""
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"front" : [ 0.46474114250751791, 0.3444876400245302, -0.81568617515018405 ],
			"lookat" : [ -0.75, 0.0, 0.0 ],
			"up" : [ -0.61579677474250438, -0.53620121298641632, -0.57730632372131963 ],
			"zoom" : 0.69999999999999996
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
""")['trajectory'][0]

PARAMS = [
    (3, -10, 8, 0),
    (1.5, -6, 3.6, .2),
    (3.8, -5.6, -2.8, 6),
    (2.6, 5.4, -2, -5.6),
    (1.5, 6.4, -11.2, 1.2),
]

# SIDE_PARAMS = [
#     (4, -5, -3, 5),
#     (2.8, -8, 3.5, 3),
#     (2.4, -4, 12, 0)
# ]

SIDE_PARAMS = []


def get_all_params():
    res = list(PARAMS)
    for r, x, y, z in SIDE_PARAMS:
        for sign in [-1, 1]:
            res.append((r, sign * x, y, z))

    return res


def main():
    spheres = []
    for r, x, y, z in get_all_params():
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=100)
        sphere.translate((x, y, z))
        sphere.compute_vertex_normals()
        spheres.append(sphere)

    def orient(vis):
        ctl = vis.get_view_control()
        ctl.set_lookat(POSE['lookat'])
        ctl.set_up(POSE['up'])
        ctl.set_front(POSE['front'])
        ctl.set_zoom(POSE['zoom'])

    idx = 4
    def makefunc(axis, sign):
        def foo(vis):
            t = np.zeros(3)
            t[axis] = sign * .4
            spheres[idx].translate(t)
            vis.update_geometry(spheres[idx])

        return foo

    def p(vis):
        print(np.asarray(spheres[idx].vertices).mean(axis=0))

    callbacks = {
        ord('L'): orient,
        ord('A'): makefunc(0, -1),
        ord('D'): makefunc(0, 1),
        ord('Q'): makefunc(1, -1),
        ord('W'): makefunc(1, 1),
        ord('Z'): makefunc(2, -1),
        ord('X'): makefunc(2, 1),
        ord('P'): p
    }

    o3d.visualization.draw_geometries_with_key_callbacks(spheres, callbacks)

    # o3d.visualization.draw_geometries(spheres,
    #                                   lookat=POSE['lookat'],
    #                                   up=POSE['up'],
    #                                   front=POSE['front'],
    #                                   zoom=POSE['zoom'])


if __name__ == '__main__':
    main()
