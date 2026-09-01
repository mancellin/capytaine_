import struct
from pathlib import Path

import numpy as np
from numpy import pi

import capytaine as cpt
from capytaine.bem.airy_waves import airy_waves_free_surface_elevation
from capytaine.ui.vedo_animations import Animation


bem_solver = cpt.BEMSolver()


def generate_boat():
    boat_mesh = cpt.load_mesh("docs/examples/src/boat_200.mar", file_format="mar")
    boat = cpt.FloatingBody(
            mesh=boat_mesh,
            dofs=cpt.rigid_body_dofs(rotation_center=boat_mesh.center_of_buoyancy),
            center_of_mass = boat_mesh.center_of_buoyancy,
            )
    boat.inertia_matrix = boat.compute_rigid_body_inertia() / 40 # Artificially lower to have a more appealing animation
    return boat


def compute_animation_data(body, fs, omega, wave_amplitude, wave_direction):
    """Solve the BEM problems and compute the harmonic motion of the boat and
    of the free surface for a single incoming monochromatic wave.

    Returns a list of ``(mesh, vertices_motion)`` components — one for the
    boat, one for the free surface — plus the angular frequency ``omega``.
    ``vertices_motion`` is a complex array of shape ``(n_vertices, 3)``: for
    a given vertex, its instantaneous displacement at time ``t`` is
    ``Re[vertices_motion * exp(-1j*omega*t)]`` (see
    :meth:`capytaine.ui.vedo_animations.Animation.update`).

    This data can then be turned into a `vedo` animation (see
    :func:`setup_animation`) or exported to a binary file for the
    interactive web viewer on the documentation homepage (see
    :func:`export_animation_data`).
    """
    # SOLVE BEM PROBLEMS
    radiation_problems = [cpt.RadiationProblem(omega=omega, body=body.immersed_part(), radiating_dof=dof) for dof in body.dofs]
    radiation_results = bem_solver.solve_all(radiation_problems)
    diffraction_problem = cpt.DiffractionProblem(omega=omega, body=body.immersed_part(), wave_direction=wave_direction)
    diffraction_result = bem_solver.solve(diffraction_problem)

    dataset = cpt.assemble_dataset(radiation_results + [diffraction_result])
    rao = cpt.post_pro.rao(dataset, wave_direction=wave_direction)

    # COMPUTE FREE SURFACE ELEVATION
    # Compute the diffracted wave pattern
    incoming_waves_elevation = airy_waves_free_surface_elevation(fs.vertices, diffraction_result)
    diffraction_elevation = bem_solver.compute_free_surface_elevation(fs.vertices, diffraction_result)

    # Compute the wave pattern radiated by the RAO
    radiation_elevations_per_dof = {res.radiating_dof: bem_solver.compute_free_surface_elevation(fs.vertices, res) for res in radiation_results}
    radiation_elevation = sum(rao.sel(omega=omega, radiating_dof=dof).data * radiation_elevations_per_dof[dof] for dof in body.dofs)
    fs_elevation = wave_amplitude * (incoming_waves_elevation + diffraction_elevation + radiation_elevation)

    # COMPUTE BOAT MOTION
    # Compute the motion of each vertex of the mesh for the animation
    dofs_vertices_motions = {dof: body.dofs[dof].evaluate_motion_at_points(body.mesh.vertices) for dof in body.dofs}
    boat_vertices_motion = wave_amplitude * sum(rao.sel(omega=omega, radiating_dof=dof).data * dofs_vertices_motions[dof] for dof in body.dofs)

    # A free surface elevation is just a motion purely along z, with the same
    # shape (n_vertices, 3) as the boat's motion above.
    fs_vertices_motion = np.vstack([np.zeros_like(fs_elevation), np.zeros_like(fs_elevation), fs_elevation]).T

    components = [(body.mesh, boat_vertices_motion), (fs, fs_vertices_motion)]
    return components, omega


def setup_animation(body, fs, omega, wave_amplitude, wave_direction):
    components, omega = compute_animation_data(body, fs, omega, wave_amplitude, wave_direction)
    (_, boat_motion), (_, fs_motion) = components

    # Set up scene
    animation = Animation(loop_duration=2*pi/omega)
    animation.add_body(body, vertices_motion=boat_motion)
    animation.add_free_surface(fs, fs_motion[:, 2])  # add_free_surface expects the scalar elevation
    return animation


def export_animation_data(components, omega, path, camera_position=None, camera_target=None):
    """Write mesh + harmonic motion data to a small binary format that the
    generic web viewer (``docs/_static/harmonic_mesh_viewer.js``) can load
    with no knowledge of what the components physically represent — any
    mesh with a complex per-vertex motion works, not just this boat/wave
    scene.

    If `camera_position`/`camera_target` are not given, they are computed
    automatically from the bounding sphere of the *first* component
    (assumed to be the main visual subject, e.g. the floating body): the
    initial view looks at its centroid from a fixed direction, at a
    distance proportional to its size. This way the same generic viewer
    picks a sensible default camera for any other mesh exported the same
    way, regardless of that scene's actual scale.

    Binary format (little-endian):
        uint32 num_components
        float32 omega
        float32[3] camera_position
        float32[3] camera_target
        for each component:
            uint32 n_vertices, uint32 n_faces
            float32[n_vertices*3]  mean_vertices
            uint32[n_faces*3]      triangle_indices  (quads split into 2 triangles)
            float32[n_vertices*3]  motion_real
            float32[n_vertices*3]  motion_imag
    """
    if camera_position is None or camera_target is None:
        subject_vertices = np.asarray(components[0][0].vertices, dtype=np.float64)
        center = subject_vertices.mean(axis=0)
        radius = np.linalg.norm(subject_vertices - center, axis=1).max()
        if camera_target is None:
            camera_target = center
        if camera_position is None:
            direction = np.array([70.0, 70.0, 100.0])
            direction /= np.linalg.norm(direction)
            camera_position = center + direction * radius * 4.0

    with open(Path(path), "wb") as f:
        f.write(struct.pack("<I", len(components)))
        f.write(struct.pack("<f", omega))
        f.write(struct.pack("<3f", *np.asarray(camera_position, dtype=np.float32)))
        f.write(struct.pack("<3f", *np.asarray(camera_target, dtype=np.float32)))
        for mesh, motion in components:
            vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
            motion = np.asarray(motion, dtype=np.complex64)

            quads = np.asarray(mesh.faces, dtype=np.uint32)  # shape (n_faces, 4)
            triangles = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
            triangles = np.ascontiguousarray(triangles, dtype=np.uint32)

            f.write(struct.pack("<II", len(vertices), len(triangles)))
            f.write(vertices.tobytes())
            f.write(triangles.tobytes())
            f.write(np.ascontiguousarray(motion.real, dtype=np.float32).tobytes())
            f.write(np.ascontiguousarray(motion.imag, dtype=np.float32).tobytes())


if __name__ == '__main__':
    from vedo import Light

    body = generate_boat()
    omega = 1.5
    wave_amplitude = 1.0
    wave_direction = pi

    # Full-resolution version, rendered offline to a video (unchanged from before).
    fs = cpt.mesh_rectangle(size=(200.0, 200.0), resolution=(100, 100))
    anim = setup_animation(body, fs, omega=omega, wave_amplitude=wave_amplitude, wave_direction=wave_direction)
    params = dict(
        camera={"pos": (70, 70, 100)},
        lights=[Light([0, 0, 100], intensity=0.8)],
        resolution=(800, 600)
    )
    # anim.run(**params)
    anim.save("animated_boat.mp4", **params)

    # Lighter-weight version of the same scene, exported for the interactive
    # three.js animation on the documentation homepage.
    web_fs = cpt.mesh_rectangle(size=(200.0, 200.0), resolution=(56, 56))
    web_components, web_omega = compute_animation_data(body, web_fs, omega=omega, wave_amplitude=wave_amplitude, wave_direction=wave_direction)
    export_animation_data(web_components, web_omega, "docs/_static/boat_animation_data.bin")
