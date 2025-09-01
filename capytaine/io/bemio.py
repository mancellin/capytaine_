"""This module contains functions interacting with BEMIO file format and the Python BEMIO library."""
import logging

import numpy as np
import pandas as pd
from scipy.optimize import newton

LOG = logging.getLogger(__name__)


def _all_freqs_from_omega(omega, water_depth=np.inf, g=9.81):
    if water_depth == np.inf:
        k = omega**2/g
    else:
        k = newton(lambda x: x*np.tanh(x) - omega**2*water_depth/g, x0=1.0)/water_depth
    return {
        'omega': omega,
        'period': 2*np.pi/omega,
        'freq': omega/2*np.pi,
        'wavenumber': k,
        'wavelength': 2*np.pi/k,
    }

def dataframe_from_bemio(bemio_obj, *, wavenumber=True, wavelength=True):
    """Transform a :class:`bemio.data_structures.bem.HydrodynamicData` into a
        :class:`pandas.DataFrame`.

        Parameters
        ----------
        bemio_obj: Bemio data_stuctures.bem.HydrodynamicData class
            Loaded NEMOH, AQWA, or WAMIT data created using `bemio.io.nemoh.read`,
            `bemio.io.aqwa.read`, or `bemio.io.wamit.read` functions, respectively.
        wavenumber: bool
            If True, the coordinate 'wavenumber' will be added to the output dataset.
        wavelength: bool
            If True, the coordinate 'wavelength' will be added to the output dataset.
        """

    n_bodies = len(bemio_obj.body)

    dofs_of_a_body = np.array(['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw'])
    if n_bodies == 1:
        all_dofs = np.array([dofs_of_a_body])
    else:
        all_dofs = np.array([[bemio_obj.body[i].name + "__" + d for d in dofs_of_a_body] for i in range(n_bodies)])

    # The parameter below are expected to be the same for all bodies in the dataset.
    # (Capytaine could not really handle anyway if they weren't.)
    rho = bemio_obj.body[0].rho
    g = bemio_obj.body[0].g
    if bemio_obj.body[0].water_depth == 'infinite':
        water_depth = np.inf
    else:
        water_depth = bemio_obj.body[0].water_depth

    full_body_name = '+'.join([bemio_obj.body[i].name for i in range(n_bodies)])

    params = {
            "body_name": full_body_name,
            "water_depth": water_depth,
            "rho": rho,
            "g": g,
            "forward_speed": 0.0,
            "free_surface": 0.0,
            }

    difr_dict = []
    rad_dict = []

    for i in range(n_bodies):
        from_wamit = (bemio_obj.body[i].bem_code == 'WAMIT') # WAMIT coefficients need to be dimensionalized

        for omega_idx, omega in enumerate(bemio_obj.body[i].w):

            # DIFFRACTION DATA
            for dir_idx, dir in enumerate(bemio_obj.body[i].wave_dir):

                Fexc = np.empty(shape=bemio_obj.body[i].ex.re[:, dir_idx, omega_idx].shape, dtype=np.complex128)
                if from_wamit:
                    Fexc.real = bemio_obj.body[i].ex.re[:, dir_idx, omega_idx] * rho * g
                    Fexc.imag = bemio_obj.body[i].ex.im[:, dir_idx, omega_idx] * rho * g
                else:
                    Fexc.real = bemio_obj.body[i].ex.re[:, dir_idx, omega_idx]
                    Fexc.imag = bemio_obj.body[i].ex.im[:, dir_idx, omega_idx]

                try:
                    Fexc_fk = np.empty(shape=bemio_obj.body[i].ex.fk.re[:, dir_idx, omega_idx].shape, dtype=np.complex128)
                    if from_wamit:
                        Fexc_fk.real = bemio_obj.body[i].ex.fk.re[:, dir_idx, omega_idx] * rho * g
                        Fexc_fk.imag = bemio_obj.body[i].ex.fk.im[:, dir_idx, omega_idx] * rho * g
                    else:
                        Fexc_fk.real = bemio_obj.body[i].ex.fk.re[:, dir_idx, omega_idx]
                        Fexc_fk.imag = bemio_obj.body[i].ex.fk.im[:, dir_idx, omega_idx]

                except AttributeError:
                        # LOG.warning('\tNo Froude-Krylov forces found for ' + bemio_obj.body[i].name + ' at ' + str(dir) + \
                        #       ' degrees (omega = ' + str(omega) + '), replacing with zeros.')
                        Fexc_fk = np.full((bemio_obj.body[i].ex.re[:, dir_idx, omega_idx].size,), np.nan, dtype=np.complex128)

                difr_dict.append({
                        **params,
                        **_all_freqs_from_omega(omega, water_depth, g),
                        'kind': "DiffractionResult",
                        'wave_direction': np.radians(dir),
                        'influenced_dof': all_dofs[i],
                        'diffraction_force': Fexc,
                        'Froude_Krylov_force': Fexc_fk,
                        })

            for radiating_dof_idx, radiating_dof in enumerate(all_dofs[i]):

                A = bemio_obj.body[i].am.all[radiating_dof_idx, :, omega_idx]
                B = bemio_obj.body[i].rd.all[radiating_dof_idx, :, omega_idx]
                if from_wamit:
                    A = A * rho
                    B = B * rho * omega

                rad_dict.append({
                        **params,
                        **_all_freqs_from_omega(omega, water_depth, g),
                        'kind': "RadiationResult",
                        'omega':  omega,
                        'radiating_dof': radiating_dof,
                        'influenced_dof': all_dofs.ravel(),
                        'wave_direction': 0.0,
                        'added_mass': A,
                        'radiation_damping': B
                        })

    df = pd.concat([
        pd.DataFrame.from_dict(difr_dict).explode(['influenced_dof', 'diffraction_force', 'Froude_Krylov_force']),
        pd.DataFrame.from_dict(rad_dict).explode(['influenced_dof', 'added_mass', 'radiation_damping'])
        ])
    df = df.astype({'added_mass': np.float64, 'radiation_damping': np.float64, 'diffraction_force': np.complex128, 'Froude_Krylov_force': np.complex128})

    inf_dof_cat = pd.CategoricalDtype(categories=all_dofs.ravel())
    df["influenced_dof"] = df["influenced_dof"].astype(inf_dof_cat)
    if 'added_mass' in df.columns:
        rad_dof_cat = pd.CategoricalDtype(categories=all_dofs.ravel())
        df["radiating_dof"] = df["radiating_dof"].astype(rad_dof_cat)

    return df
