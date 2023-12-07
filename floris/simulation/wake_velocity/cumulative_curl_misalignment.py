# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from typing import Any, Dict

import numpy as np
from attrs import define, field
from scipy.special import gamma

from floris.simulation import (
    BaseModel,
    Farm,
    FlowField,
    Grid,
    Turbine,
)
from floris.utilities import (
    cosd,
    sind,
    tand,
)


@define
class CumulativeCurlMisalignmentVelocityDeficit(BaseModel):
    """
    The cumulative curl model is an implementation of the model described in
    :cite:`gdm-bay_2022`, which itself is based on the cumulative model of
    :cite:`bastankhah_2021`

    References:
    .. bibliography:: /references.bib
        :style: unsrt
        :filter: docname in docnames
        :keyprefix: gdm-
    """

    a_s: float = field(default=0.179367259)
    b_s: float = field(default=0.0118889215)
    c_s1: float = field(default=0.0563691592)
    c_s2: float = field(default=0.13290157)
    a_f: float = field(default=3.11)
    b_f: float = field(default=-0.68)
    c_f: float = field(default=2.41)
    alpha_mod: float = field(default=1.0)

    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> Dict[str, Any]:

        kwargs = {
            "x": grid.x_sorted,
            "y": grid.y_sorted,
            "z": grid.z_sorted,
            "u_initial": flow_field.u_initial_sorted,
        }
        return kwargs
    

    def function(
        self,
        i: int,
        x_i: np.ndarray,
        y_i: np.ndarray,
        z_i: np.ndarray,
        u_i: np.ndarray,
        misalignment_angle_i: np.ndarray,
        y_deflection_i: np.ndarray,
        z_deflection_i: np.ndarray,
        turbulence_intensity: np.ndarray,
        Ct_i: np.ndarray,
        rotor_diameter: np.ndarray,
        turb_u_wake: np.ndarray,
        C_n: np.ndarray,
        sigma_n: np.ndarray,
        y_deflections_n: np.ndarray,
        z_deflections_n: np.ndarray,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        u_initial: np.ndarray,
    ) -> None:
        # NOTE: In literature, index n is used for the current turbine and i for
        # for the upwind turbines. For consistency, here i is used for the current 
        # turbine and n for the upwind turbines.

        turbine_Ct = Ct_i
        turbine_ti = turbulence_intensity

        # TODO Should this be cbrt? This is done to match v2
        turb_avg_vels = np.cbrt(np.mean(u_i ** 3, axis=(3,4)))
        turb_avg_vels = turb_avg_vels[:,:,:,None,None]

        x_i_loc = np.mean(x_i, axis=(3,4))
        x_i_loc = x_i_loc[:,:,:,None,None]

        y_i_loc = np.mean(y_i, axis=(3,4))
        y_i_loc = y_i_loc[:,:,:,None,None]

        z_i_loc = np.mean(z_i, axis=(3,4))
        z_i_loc = z_i_loc[:,:,:,None,None]

        x_coord = np.mean(x, axis=(3,4))[:,:,:,None,None]

        y_loc = y
        y_coord = np.mean(y, axis=(3,4))[:,:,:,None,None]

        z_loc = z # np.mean(z, axis=(3,4))
        z_coord = np.mean(z, axis=(3,4))[:,:,:,None,None]

        y_deflections_n = y_deflections_n[:,:,:,:,None,None]
        z_deflections_n = z_deflections_n[:,:,:,:,None,None]
        
        sigma_n = sigma_n[:,:,:,:,None,None]

        delta_x = x - x_i

        sigma_i = wake_expansion(
            delta_x,
            turbine_Ct[:,:,i:i+1],
            turbine_ti[:,:,i:i+1],
            rotor_diameter[:,:,i:i+1],
            self.a_s,
            self.b_s,
            self.c_s1,
            self.c_s2,
        )

        sigma_n[i] = sigma_i[:,:,:,0:1,0:1]

        sum_lbda = np.zeros_like(u_initial)

        for n in range(0, i - 1):
            x_coord_n = x_coord[:,:,n:n+1]
            y_coord_n = y_coord[:,:,n:n+1]
            z_coord_n = z_coord[:,:,n:n+1]

            # For computing crossplanes, we don't need to compute downstream
            # turbines from out crossplane position.
            if x_coord[:,:,n:n+1].size == 0:
                break

            S_n = sigma_i ** 2 + sigma_n[n] ** 2

            Y_n = ((y_i_loc - y_deflection_i) - (y_coord_n - y_deflections_n[n])) ** 2 / (2 * S_n)
            Z_n = ((z_i_loc - z_deflection_i) - (z_coord_n - z_deflections_n[n])) ** 2 / (2 * S_n)

            lbda = 1.0 * sigma_n[n] ** 2 / S_n * np.exp(-Y_n) * np.exp(-Z_n)

            sum_lbda = sum_lbda + lbda * (C_n[n] / u_initial)

        # Vectorized version of sum_lbda calc; has issues with y_coord (needs to be
        # down-selected appropriately. Prelim. timings show vectorized form takes
        # longer than for loop.)
        # if ii >= 2:
        #     S = sigma_n ** 2 + sigma_i[0:ii-1, :, :, :, :, :] ** 2
        #     Y = (y_i_loc - y_coord - deflection_field) ** 2 / (2 * S)
        #     Z = (z_i_loc - z_coord) ** 2 / (2 * S)

        #     lbda = self.alpha_mod * sigma_i[0:ii-1, :, :, :, :, :] ** 2
        #     lbda /= S * np.exp(-Y) * np.exp(-Z)
        #     sum_lbda = np.sum(lbda * (Ctmp[0:ii-1, :, :, :, :, :] / u_initial), axis=0)
        # else:
        #     sum_lbda = 0.0

        # sigma_i[ii] = sigma_n

        # blondel
        # super gaussian
        # b_f = self.b_f1 * np.exp(self.b_f2 * TI) + self.b_f3
        x_tilde = np.abs(delta_x) / rotor_diameter[:,:,i:i+1]
        r_tilde = np.sqrt( (y_loc - y_i_loc - y_deflection_i) ** 2 \
                          + (z_loc - z_i_loc - z_deflection_i) ** 2 )
        r_tilde /= rotor_diameter[:,:,i:i+1]

        m = self.a_f * np.exp(self.b_f * x_tilde) + self.c_f
        a1 = 2 ** (2 / m - 1)
        a2 = 2 ** (4 / m - 2)

        # based on Blondel model, modified to include cumulative effects
        tmp = a2 - (
            (m * turbine_Ct[:,:,i:i+1])
            * cosd(misalignment_angle_i)
            / (
                16.0
                * gamma(2 / m)
                * np.sign(sigma_i)
                * (np.abs(sigma_i) ** (4 / m))
                * (1 - sum_lbda) ** 2
            )
        )

        # for some low wind speeds, tmp can become slightly negative, which causes NANs,
        # so replace the slightly negative values with zeros
        tmp = tmp * np.array(tmp >= 0)

        C_i = a1 - np.sqrt(tmp)

        C_i = C_i * (1 - sum_lbda)

        C_n[i] = C_i

        yR = y_loc - y_i_loc
        xR = x_i # + yR * tand(misalignment_angle_i)

        # add turbines together
        velDef = C_i * np.exp((-1 * r_tilde ** m) / (2 * sigma_i ** 2))

        
        velDef = velDef * np.array(x - xR >= 0.1)

        turb_u_wake = turb_u_wake + turb_avg_vels * velDef
        return (
            turb_u_wake,
            C_n,
            sigma_n,
        )
    

    def function_v2(
        self,
        i: int,
        x_i: np.ndarray,
        y_i: np.ndarray,
        z_i: np.ndarray,
        u_i: np.ndarray,
        misalignment_angle_i: np.ndarray,
        deflection_angle_i: np.ndarray, 
        y_deflection_i: np.ndarray,
        z_deflection_i: np.ndarray,
        turbulence_intensity: np.ndarray,
        Ct_i: np.ndarray,
        rotor_diameter: np.ndarray,
        turb_u_wake: np.ndarray,
        C_n: np.ndarray,
        sigma_n: np.ndarray,
        sigma_y_i: np.ndarray,
        sigma_z_i: np.ndarray,
        sigma_y_n: np.ndarray,
        sigma_z_n: np.ndarray,
        y_deflections_n: np.ndarray,
        z_deflections_n: np.ndarray,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        u_initial: np.ndarray,
    ) -> None:
        # NOTE: In literature, index n is used for the current turbine and i for
        # for the upwind turbines. For consistency, here i is used for the current 
        # turbine and n for the upwind turbines.

        turbine_Ct = Ct_i
        turbine_ti = turbulence_intensity

        # TODO Should this be cbrt? This is done to match v2
        turb_avg_vels = np.cbrt(np.mean(u_i ** 3, axis=(3,4)))
        turb_avg_vels = turb_avg_vels[:,:,:,None,None]

        x_i_loc = np.mean(x_i, axis=(3,4))
        x_i_loc = x_i_loc[:,:,:,None,None]

        y_i_loc = np.mean(y_i, axis=(3,4))
        y_i_loc = y_i_loc[:,:,:,None,None]

        z_i_loc = np.mean(z_i, axis=(3,4))
        z_i_loc = z_i_loc[:,:,:,None,None]

        x_coord = np.mean(x, axis=(3,4))[:,:,:,None,None]

        y_loc = y
        y_coord = np.mean(y, axis=(3,4))[:,:,:,None,None]

        z_loc = z # np.mean(z, axis=(3,4))
        z_coord = np.mean(z, axis=(3,4))[:,:,:,None,None]

        y_deflections_n = y_deflections_n[:,:,:,:,None,None]
        z_deflections_n = z_deflections_n[:,:,:,:,None,None]
        
        sigma_n = sigma_n[:,:,:,:,None,None]

        delta_x = x - x_i

        sigma_i = wake_expansion(
            delta_x,
            turbine_Ct[:,:,i:i+1],
            turbine_ti[:,:,i:i+1],
            rotor_diameter[:,:,i:i+1],
            self.a_s,
            self.b_s,
            self.c_s1,
            self.c_s2,
        )

        sigma_n[i] = sigma_i[:,:,:,0:1,0:1]

        sum_lbda = np.zeros_like(u_initial)


        sigma_y_n = sigma_y_n[:,:,:,:,None,None] / rotor_diameter[:,:,i:i+1]
        sigma_z_n = sigma_z_n[:,:,:,:,None,None] / rotor_diameter[:,:,i:i+1]

        sigma_y_i = sigma_y_i * np.ones_like(x) / rotor_diameter[:,:,i:i+1]
        sigma_z_i = sigma_z_i * np.ones_like(x) / rotor_diameter[:,:,i:i+1]


        # print(f'Shape sigma_i: {np.shape(sigma_i)}')
        # print(f'Shape sigma_y_i: {np.shape(sigma_y_i)}')
        # print(f'Shape sigma_n: {np.shape(sigma_n)}')
        # print(f'Shape sigma_n[i]: {np.shape(sigma_n[i])}')
        # print(f'Shape sigma_y_n: {np.shape(sigma_y_n)}')
        # print(f'Shape sigma_y_n[i]: {np.shape(sigma_y_n[i])}')
        # print(f'Sigma_i: {sigma_i}')
        # print(f'Sigma_y_i: {sigma_y_i}')
        # print(f'Sigma_n[i]: {sigma_n[i]}')
        # print(f'Sigma_y_n[i]: {sigma_y_n[i]}')


        for n in range(0, i - 1):
            x_coord_n = x_coord[:,:,n:n+1]
            y_coord_n = y_coord[:,:,n:n+1]
            z_coord_n = z_coord[:,:,n:n+1]

            # For computing crossplanes, we don't need to compute downstream
            # turbines from out crossplane position.
            if x_coord[:,:,n:n+1].size == 0:
                break

            S_n = sigma_y_i * sigma_z_i + sigma_y_n[n] * sigma_z_n[n]

            Y_n = ((y_i_loc - y_deflection_i) - (y_coord_n - y_deflections_n[n])) ** 2 / (2 * S_n)
            Z_n = ((z_i_loc - z_deflection_i) - (z_coord_n - z_deflections_n[n])) ** 2 / (2 * S_n)

            lbda = 1.0 * sigma_y_n[n] * sigma_z_n[n] ** 2 / S_n * np.exp(-Y_n) * np.exp(-Z_n)

            sum_lbda = sum_lbda + lbda * (C_n[n] / u_initial)

        # Vectorized version of sum_lbda calc; has issues with y_coord (needs to be
        # down-selected appropriately. Prelim. timings show vectorized form takes
        # longer than for loop.)
        # if ii >= 2:
        #     S = sigma_n ** 2 + sigma_i[0:ii-1, :, :, :, :, :] ** 2
        #     Y = (y_i_loc - y_coord - deflection_field) ** 2 / (2 * S)
        #     Z = (z_i_loc - z_coord) ** 2 / (2 * S)

        #     lbda = self.alpha_mod * sigma_i[0:ii-1, :, :, :, :, :] ** 2
        #     lbda /= S * np.exp(-Y) * np.exp(-Z)
        #     sum_lbda = np.sum(lbda * (Ctmp[0:ii-1, :, :, :, :, :] / u_initial), axis=0)
        # else:
        #     sum_lbda = 0.0

        # sigma_i[ii] = sigma_n

        # blondel
        # super gaussian
        # b_f = self.b_f1 * np.exp(self.b_f2 * TI) + self.b_f3
        x_tilde = np.abs(delta_x) / rotor_diameter[:,:,i:i+1]
        r_tilde = np.sqrt( (y_loc - y_i_loc - y_deflection_i) ** 2 \
                          + (z_loc - z_i_loc - z_deflection_i) ** 2 )
        r_tilde /= rotor_diameter[:,:,i:i+1]

        m = self.a_f * np.exp(self.b_f * x_tilde) + self.c_f
        a1 = 2 ** (2 / m - 1)
        a2 = 2 ** (4 / m - 2)

        # based on Blondel model, modified to include cumulative effects
        tmp = a2 - (
            (m * turbine_Ct[:,:,i:i+1])
            * cosd(misalignment_angle_i)
            / (
                16.0
                * gamma(2 / m)
                * np.sign(np.sqrt(sigma_y_i * sigma_z_i))
                * (np.abs(np.sqrt(sigma_y_i * sigma_z_i)) ** (4 / m))
                * (1 - sum_lbda) ** 2
            )
        )

        # for some low wind speeds, tmp can become slightly negative, which causes NANs,
        # so replace the slightly negative values with zeros
        tmp = tmp * np.array(tmp >= 0)

        C_i = a1 - np.sqrt(tmp)

        C_i = C_i * (1 - sum_lbda)

        C_n[i] = C_i

        yR = y_loc - y_i_loc
        xR = x_i # + yR * tand(misalignment_angle_i)

        # add turbines together
        velDef = C_i * np.exp((-1 * r_tilde ** m) / (2 * sigma_y_i * sigma_z_i))

        Y_tilde = (y_loc - y_i_loc - y_deflection_i) / rotor_diameter[:,:,i:i+1]
        Z_tilde = (z_loc - z_i_loc - z_deflection_i) / rotor_diameter[:,:,i:i+1]

        Y_tilde_rotated = cosd(deflection_angle_i) * Y_tilde - sind(deflection_angle_i) * Z_tilde
        Z_tilde_rotated = sind(deflection_angle_i) * Y_tilde + cosd(deflection_angle_i) * Z_tilde

        velDef = C_i * np.exp(-(
            np.abs(Y_tilde_rotated)**m / (2 * sigma_y_i**2)
          + np.abs(Z_tilde_rotated)**m / (2 * sigma_z_i**2)
        ))
        
        velDef = velDef * np.array(x - xR >= 0.1)

        turb_u_wake = turb_u_wake + turb_avg_vels * velDef
        return (
            turb_u_wake,
            C_n,
            sigma_n,
        )


def wake_expansion(
    delta_x,
    ct_i,
    turbulence_intensity_i,
    rotor_diameter,
    a_s,
    b_s,
    c_s1,
    c_s2,
):
    # Calculate Beta (Eq 10, pp 5 of ref. [1] and table 4 of ref. [2] in docstring)
    beta = 0.5 * (1.0 + np.sqrt(1.0 - ct_i)) / np.sqrt(1.0 - ct_i)
    k = a_s * turbulence_intensity_i + b_s
    eps = (c_s1 * ct_i + c_s2) * np.sqrt(beta)

    # Calculate sigma_tilde (Eq 9, pp 5 of ref. [1] and table 4 of ref. [2] in docstring)
    x_tilde = np.abs(delta_x) / rotor_diameter
    sigma_y = k * x_tilde + eps

    # [added dimension to get upstream values, empty, wd, ws, x, y, z  ]
    # return sigma_y[na, :, :, :, :, :, :]
    # Do this ^^ in the main function

    return sigma_y


# # TODO: Make seperate model for wake width
# def wake_expansion(
#     delta_x,
#     Ct_i,
#     turbulence_intensity_i,
#     rotor_diameter_i,
#     misalignment_angle_i,
#     a_s=0.179367259,
#     b_s=0.0118889215,
#     c_s1=0.0563691592,
#     c_s2=0.13290157,
# ):
#     k_y = a_s * turbulence_intensity_i + b_s
#     k_z = k_y

#     beta = 0.5 * (1 + np.sqrt(1 - Ct_i * cosd*misalignment_angle_i)) / np.sqrt(1 - Ct_i)
    
#     sigma_z0 = (c_s1 * Ct_i + c_s2) * np.sqrt(beta)
#     sigma_y0 = sigma_z0 * cosd(misalignment_angle_i)

#     x_tilde = np.abs(delta_x) / rotor_diameter_i
    
#     sigma_y = k_y * x_tilde + sigma_y0
#     sigma_z = k_z * x_tilde + sigma_z0

#     return sigma_y, sigma_z