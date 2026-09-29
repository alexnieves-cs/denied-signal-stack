"""MuJoCo render-only scene: textured terrain mesh + a mocap camera body (no physics, PLAN F6).

Settings: shadows off, headlight ambient=1 (unlit albedo; photometrics happen in numpy),
inertia="shell" for the open terrain mesh, camera K exact: fx=fy, centred principal point.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "cgl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from dss.geo.aoi import Raster  # noqa: E402
from dss.sim.world.tiles import ground_mesh  # noqa: E402


def build_model(img: Raster, dem: Raster, bounds, width: int, height: int, fy: float, mesh_step: float = 5.0,
                extra_geoms=None) -> mujoco.MjModel:
    xmin, xmax, ymin, ymax = bounds
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.visual.headlight.ambient = [1.0, 1.0, 1.0]
    spec.visual.headlight.diffuse = [0.0, 0.0, 0.0]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.quality.shadowsize = 0
    extent = max(xmax - xmin, ymax - ymin)
    spec.stat.extent = extent
    spec.visual.map.znear = 0.3 / extent
    spec.visual.map.zfar = 3.0

    # texture: image rows run north->south; MuJoCo's v=0 is the first row, so flip rows
    # so that uv (0,0) = (xmin, ymin)
    i0, j0 = img.xy_to_ij(xmin, ymax)
    i1, j1 = img.xy_to_ij(xmax, ymin)
    sub = img.data[int(round(i0 + 0.5)):int(round(i1 + 0.5)), int(round(j0 + 0.5)):int(round(j1 + 0.5)), :3]
    sub = np.ascontiguousarray(sub[::-1])
    tex = spec.add_texture(name="ortho", type=mujoco.mjtTexture.mjTEXTURE_2D, width=sub.shape[1], height=sub.shape[0], nchannel=3)
    tex.data = sub.tobytes()
    mat = spec.add_material(name="ground")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "ortho"
    mat.texuniform = False
    mat.texrepeat = [1, 1]
    mat.specular = 0.0
    mat.shininess = 0.0
    verts, uv, faces = ground_mesh(dem, xmin, xmax, ymin, ymax, mesh_step)
    mesh = spec.add_mesh(name="terrain")
    mesh.uservert = verts.astype(np.float32).ravel().tolist()
    mesh.usertexcoord = uv.astype(np.float32).ravel().tolist()
    mesh.userface = faces.astype(np.int32).ravel().tolist()
    mesh.userfacetexcoord = faces.astype(np.int32).ravel().tolist()
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    g = spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="terrain", material="ground")
    g.contype = 0
    g.conaffinity = 0
    for eg in extra_geoms or []:
        spec.worldbody.add_geom(**eg)
    body = spec.worldbody.add_body(name="cam_body", mocap=True)
    fovy = 2 * np.arctan(0.5 * height / fy)
    body.add_camera(name="cam0", fovy=float(np.rad2deg(fovy)))
    return spec.compile()


# MuJoCo camera frame: x right, y up, z backward. Our camera: RDF (x right, y down, z forward).
R_RDF_MJ = np.diag([1.0, -1.0, -1.0])
