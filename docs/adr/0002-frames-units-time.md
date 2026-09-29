# ADR-0002: frames, units, time

Status: accepted (Phase 0).

- World: ENU at the AOI origin. Body: FLU (IMU frame). Camera: RDF (x right, y down, z optical axis).
- Quaternions: Hamilton, scalar-first `[w, x, y, z]`, describing child orientation in parent (`R_parent_child`).
- OpenVINS JPL `q_GtoI` `[x, y, z, w]` is converted only at the ovserver protocol boundary (`dss.core.rotations.jpl_to_hamilton`): the same four numbers reorder to the Hamilton `q_ItoG`.
- NED/FRD appears only at the MAVLink boundary (`dss.core.frames`).
- Time: int64 nanoseconds in Python; float64 seconds only inside ovserver.
- Transforms: `T_a_b` maps points in b into a.
