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

from floris.simulation import (
    BaseModel,
    Farm,
    FlowField,
    Grid,
    Turbine,
)
from floris.utilities import cosd, sind, arccosd, arctan2d


@define
class GaussMisalignmentVelocityDeflection(BaseModel):
    """
    The Gauss deflection model is a blend of the models described in
    :cite:`gdm-bastankhah2016experimental` and :cite:`gdm-King2019Controls` for
    calculating the deflection field in turbine wakes.

    parameter_dictionary (dict): Model-specific parameters.
        Default values are used when a parameter is not included
        in `parameter_dictionary`. Possible key-value pairs include:

            -   **ka** (*float*): Parameter used to determine the linear
                relationship between the turbulence intensity and the
                width of the Gaussian wake shape.
            -   **kb** (*float*): Parameter used to determine the linear
                relationship between the turbulence intensity and the
                width of the Gaussian wake shape.
            -   **alpha** (*float*): Parameter that determines the
                dependence of the downstream boundary between the near
                wake and far wake region on the turbulence intensity.
            -   **beta** (*float*): Parameter that determines the
                dependence of the downstream boundary between the near
                wake and far wake region on the turbine's induction
                factor.
            -   **ad** (*float*): Additional tuning parameter to modify
                the wake deflection with a lateral offset.
                Defaults to 0.
            -   **bd** (*float*): Additional tuning parameter to modify
                the wake deflection with a lateral offset.
                Defaults to 0.
            -   **dm** (*float*): Additional tuning parameter to scale
                the amount of wake deflection. Defaults to 1.0
            -   **use_secondary_steering** (*bool*): Flag to use
                secondary steering on the wake velocity using methods
                developed in [2].
            -   **eps_gain** (*float*): Tuning value for calculating
                the V- and W-component velocities using methods
                developed in [7].
                TODO: Believe this should be removed, need to verify.
                See property on super-class for more details.

    References:
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: gdm-
    """
    ad: float = field(converter=float, default=0.0)
    bd: float = field(converter=float, default=0.0)
    alpha: float = field(converter=float, default=0.58)
    beta: float = field(converter=float, default=0.077)
    ka: float = field(converter=float, default=0.38)
    kb: float = field(converter=float, default=0.004)
    dm: float = field(converter=float, default=1.0)
    eps_gain: float = field(converter=float, default=0.2)
    use_secondary_steering: bool = field(converter=bool, default=True)

    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> Dict[str, Any]:

        kwargs = {
            "x": grid.x_sorted,
            "y": grid.y_sorted,
            "z": grid.z_sorted,
            "freestream_velocity": flow_field.u_initial_sorted,
            "wind_veer": flow_field.wind_veer,
        }
        return kwargs

    # @profile
    def function(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        misalignment_angle_i: np.ndarray,
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        rotor_diameter_i: float,
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        freestream_velocity: np.ndarray,
        wind_veer: float,
    ):
        """
        Calculates the deflection field of the wake. See
        :cite:`gdm-bastankhah2016experimental` and :cite:`gdm-King2019Controls`
        for details on the methods used.

        Args:
            x_locations (np.array): An array of floats that contains the
                streamwise direction grid coordinates of the flow field
                domain (m).
            y_locations (np.array): An array of floats that contains the grid
                coordinates of the flow field domain in the direction normal to
                x and parallel to the ground (m).
            z_locations (np.array): An array of floats that contains the grid
                coordinates of the flow field domain in the vertical
                direction (m).
            turbine (:py:obj:`floris.simulation.turbine`): Object that
                represents the turbine creating the wake.
            coord (:py:obj:`floris.utilities.Vec3`): Object containing
                the coordinate of the turbine creating the wake (m).
            flow_field (:py:class:`floris.simulation.flow_field`): Object
                containing the flow field information for the wind farm.

        Returns:
            np.array: Deflection field for the wake.
        """
        # ==============================================================

        # TODO: connect support for tilt
        tilt = 0.0 #turbine.tilt_angle

        # initial velocity deficits
        uR = (
            freestream_velocity
          * ct_i
          * cosd(misalignment_angle_i)
          / (2.0 * (1 - np.sqrt(1 - (ct_i * cosd(misalignment_angle_i)))))
        )
        u0 = freestream_velocity * np.sqrt(1 - ct_i)

        # length of near wake
        x0 = (
            rotor_diameter_i
            * (cosd(misalignment_angle_i) * (1 + np.sqrt(1 - ct_i * cosd(misalignment_angle_i))))
            / (np.sqrt(2) * (
                4 * self.alpha * turbulence_intensity_i + 2 * self.beta * (1 - np.sqrt(1 - ct_i))
            )) + x_i
        )

        # wake expansion parameters
        ky = self.ka * turbulence_intensity_i + self.kb
        kz = self.ka * turbulence_intensity_i + self.kb

        C0 = 1 - u0 / freestream_velocity
        M0 = C0 * (2 - C0)
        E0 = C0 ** 2 - 3 * np.exp(1.0 / 12.0) * C0 + 3 * np.exp(1.0 / 3.0)

        # TODO add wind veer and wake expansion dependent on angles
        # initial Gaussian wake expansion
        sigma_z0 = rotor_diameter_i * 0.5 * np.sqrt(uR / (freestream_velocity + u0))
        sigma_y0 = sigma_z0 # * cosd(yaw_i) * cosd(wind_veer)

        # yR = y - y_i
        xR = x_i # yR * tand(yaw) + x_i

        # yaw parameters (skew angle and distance from centerline)
        # skew angle in radians
        theta_c0 = self.dm * (0.3 * np.radians(misalignment_angle_i) / cosd(misalignment_angle_i))
        theta_c0 *= (1 - np.sqrt(1 - ct_i * cosd(misalignment_angle_i)))
        delta0 = np.tan(theta_c0) * (x0 - x_i)  # initial wake deflection;
        # NOTE: use np.tan here since theta_c0 is radians

        # deflection in the near wake
        delta_near_wake = ((x - xR) / (x0 - xR)) * delta0 + (self.ad + self.bd * (x - x_i))
        delta_near_wake = delta_near_wake * np.array(x >= xR)
        delta_near_wake = delta_near_wake * np.array(x <= x0)

        # TODO use same wake width trhoughout model
        # deflection in the far wake
        sigma_y = ky * (x - x0) + sigma_y0
        sigma_z = kz * (x - x0) + sigma_z0
        sigma_y = sigma_y * np.array(x >= x0) + sigma_y0 * np.array(x < x0)
        sigma_z = sigma_z * np.array(x >= x0) + sigma_z0 * np.array(x < x0)

        ln_deltaNum = (1.6 + np.sqrt(M0)) * (
            1.6 * np.sqrt(sigma_y * sigma_z / (sigma_y0 * sigma_z0)) - np.sqrt(M0)
        )
        ln_deltaDen = (1.6 - np.sqrt(M0)) * (
            1.6 * np.sqrt(sigma_y * sigma_z / (sigma_y0 * sigma_z0)) + np.sqrt(M0)
        )

        delta_far_wake = (
            delta0
          + theta_c0 * E0 / 5.2
          * np.sqrt(sigma_y0 * sigma_z0 / (ky * kz * M0))
          * np.log(ln_deltaNum / ln_deltaDen)
          + (self.ad + self.bd * (x - x_i))
        )

        delta_far_wake = delta_far_wake * np.array(x > x0)
        deflection = delta_near_wake + delta_far_wake

        return deflection

## GCH components

def misalignment_angles(
    yaw,
    tilt,
):
    misalignment_angle = arccosd(cosd(yaw) * cosd(tilt))
    deflection_angle = arctan2d(sind(yaw) * cosd(tilt), sind(tilt))

    return misalignment_angle, deflection_angle


def deflection_components(
    deflection_angle,
):
    y_component = sind(deflection_angle)
    z_component = cosd(deflection_angle)

    return y_component, z_component


def coordinates_top_bottom(
    deflection_angle,
    D,      
):
    y_top = 0.5 * sind(deflection_angle - 90) * D
    z_top = 0.5 * cosd(deflection_angle - 90) * D
    y_bottom = -0.5 * sind(deflection_angle - 90) * D
    z_bottom = -0.5 * cosd(deflection_angle - 90) * D
    
    return y_top, z_top, y_bottom, z_bottom


def gamma(
    D,
    velocity,
    Uinf,
    Ct,
    scale=1.0,
):
    """
    Vortex circulation strength. Units of XXX TODO

    Args:
        D (float): Rotor diameter of the current turbine
        velocity (np.array(float)): Velocities at the current turbine
        Uinf (float): Free-stream velocity
        Ct (float): Thrust coefficient at the current turbine

    Returns:
        [type]: [description]
    """
    # NOTE the cos commented below is included in Ct, U_inf because included in vel
    return scale * (np.pi / 8) * D * velocity * Ct # * cosd(yaw) * Uinf 


def vortex_velocities(
    Gamma,
    r0, 
    Y, 
    Z,
    decay,
):
    """
    Calculate transverse velocity components of a vortex
    with strength Gamma, core radius r0 and center Y, Z, 
    and decays with decay.
    """

    core_shape = 1 - np.exp(-(Y**2 + Z**2) / r0**2) 
    radius = (2 * np.pi * (Y**2 + Z**2)) + BaseModel.NUM_EPS

    V = Gamma * Z / radius * core_shape * decay
    W = -Gamma * Y / radius * core_shape * decay

    return V, W


# def calculate_effective_yaw(
def wake_added_misalignment(
    u_i,
    v_i,
    u_initial,
    delta_y,
    z_i,
    rotor_diameter,
    hub_height,
    ct_i,
    tip_speed_ratio,
    axial_induction_i,
    scale=1.0,
):
    """
    what yaw angle would have produced that same average spanwise velocity

    These calculations focus around the current turbine. The formulation could
    remove the dimension for n-turbines, but for consistency with other
    similar equations it is left. However, the turbine dimension should
    always have length 1.
    """

    # turbine parameters
    D = rotor_diameter              # scalar
    HH = hub_height                 # scalar
    Ct = ct_i                       # (wd, ws, 1, 1, 1) for the current turbine
    TSR = tip_speed_ratio           # scalar
    aI = axial_induction_i          # (wd, ws, 1, 1, 1) for the current turbine
    avg_v = np.mean(v_i, axis=(3,4))  # (wd, ws, 1, grid, grid)

    # flow parameters
    Uinf = np.mean(u_initial, axis=(2,3,4))
    Uinf = Uinf[:,:,None,None,None]

    # TODO: Allow user input for eps gain
    eps_gain = 0.2
    eps = eps_gain * D  # Use set value

    vel_top = ((HH + D / 2) / HH) ** 0.12 * np.ones((1, 1, 1, 1, 1))
    Gamma_top = gamma(
        D,
        vel_top,
        Uinf,
        Ct,
        scale,
    )

    vel_bottom = ((HH - D / 2) / HH) ** 0.12 * np.ones((1, 1, 1, 1, 1))
    Gamma_bottom = -1 * gamma(
        D,
        vel_bottom,
        Uinf,
        Ct,
        scale,
    )

    turbine_average_velocity = np.cbrt(np.mean(u_i ** 3, axis=(3,4)))
    turbine_average_velocity = turbine_average_velocity[:,:,:,None,None]
    Gamma_wake_rotation = 0.25 * 2 * np.pi * D * (aI - aI ** 2) * turbine_average_velocity / TSR

    ### compute the spanwise and vertical velocities induced by yaw

    # decay = eps ** 2 / (4 * nu * delta_x / Uinf + eps ** 2)   # This is the decay downstream
    yLocs = delta_y + BaseModel.NUM_EPS

    # top vortex
    # NOTE: this is the top of the grid, not the top of the rotor
    zT = z_i - (HH + D / 2) + BaseModel.NUM_EPS  # distance from the top of the grid
    rT = yLocs ** 2 + zT ** 2  # TODO: This is - in the paper
    # This looks like spanwise decay;
    # it defines the vortex profile in the spanwise directions
    core_shape = 1 - np.exp(-rT / (eps ** 2))
    v_top = (Gamma_top * zT) / (2 * np.pi * rT) * core_shape
    v_top = np.mean( v_top, axis=(3,4) )
    # w_top = (-1 * Gamma_top * yLocs) / (2 * np.pi * rT) * core_shape * decay

    # bottom vortex
    zB = z_i - (HH - D / 2) + BaseModel.NUM_EPS
    rB = yLocs ** 2 + zB ** 2
    core_shape = 1 - np.exp(-rB / (eps ** 2))
    v_bottom = (Gamma_bottom * zB) / (2 * np.pi * rB) * core_shape
    v_bottom = np.mean( v_bottom, axis=(3,4) )
    # w_bottom = (-1 * Gamma_bottom * yLocs) / (2 * np.pi * rB) * core_shape * decay

    # wake rotation vortex
    zC = z_i - HH + BaseModel.NUM_EPS
    rC = yLocs ** 2 + zC ** 2
    core_shape = 1 - np.exp(-rC / (eps ** 2))
    v_core = (Gamma_wake_rotation * zC) / (2 * np.pi * rC) * core_shape
    v_core = np.mean( v_core, axis=(3,4) )
    # w_core = (-1 * Gamma_wake_rotation * yLocs) / (2 * np.pi * rC) * core_shape * decay

    # Cap the effective yaw values between -45 and 45 degrees
    val = 2 * (avg_v - v_core) / (v_top + v_bottom)
    val = np.where(val < -1.0, -1.0, val)
    val = np.where(val > 1.0, 1.0, val)
    y = np.degrees( 0.5 * np.arcsin( val ) )

    return y[:,:,:,None,None]


def calculate_transverse_velocity_misalignment(
    u_i,
    u_initial,
    dudz_initial,
    delta_x,
    delta_y,
    z,
    rotor_diameter,
    hub_height,
    deflection,
    misalignment_angle,
    deflection_angle,
    effective_deflection_angle,
    ct_i,
    tsr_i,
    axial_induction_i,
    scale=1.0
):
    """
    Calculate transverse velocity components for all downstream turbines
    given the vortices at the current turbine.
    """

    # turbine parameters
    D = rotor_diameter
    HH = hub_height
    Ct = ct_i
    TSR = tsr_i
    aI = axial_induction_i

    # Top' and Bottom' coordinates from middle point
    y_top, z_top, y_bot, z_bot = coordinates_top_bottom(deflection_angle, D)

    # TODO: Make deflection of top and bottom point dependent on wake width
    # Deflection in y and z direction
    y_component, z_component = deflection_components(deflection_angle)
    y_deflection = y_component * deflection
    z_deflection = z_component * deflection

    # flow parameters
    # TODO: wind sheer is hard-coded here but should be connected to the input
    U_inf = np.mean(u_initial, axis=(2,3,4))
    U_inf = U_inf[:,:,None,None,None]
    U_top = U_inf * ((HH + z_top) / HH) ** 0.12 * np.ones((1, 1, 1, 1, 1))
    U_bot = U_inf * ((HH - z_bot) / HH) ** 0.12 * np.ones((1, 1, 1, 1, 1))

    U_turbine_avg = np.cbrt(np.mean(u_i ** 3, axis=(3,4)))
    U_turbine_avg = U_turbine_avg[:,:,:,None,None]

    Gamma_wr = -0.25 * 2 * np.pi * D * (aI - aI**2) * U_turbine_avg / TSR

    Gamma_top = -sind(misalignment_angle) * cosd(misalignment_angle) * gamma(
        D,
        U_top,
        U_inf,
        Ct,
        scale=1,
    )

    Gamma_bot = sind(misalignment_angle) * cosd(misalignment_angle) * gamma(
        D,
        U_bot,
        U_inf,
        Ct,
        scale=1,
    )

    # Core radius
    r0_wr = 0.2 * D
    r0_ma = 0.2 * D

    # Decay the vortices as they move downstream - using mixing length
    lmda = D / 8
    kappa = 0.41
    lm = kappa * z / (1 + kappa * z / lmda)
    nu = lm ** 2 * np.abs(dudz_initial)
    decay_wr = r0_wr ** 2 / (4 * nu * delta_x / U_inf + r0_wr ** 2)
    decay_ma = r0_ma ** 2 / (4 * nu * delta_x / U_inf + r0_ma ** 2)

    # Normalized coordinates 
    Y_wr = delta_y - y_deflection
    Z_wr = z - HH - z_deflection
    Y_top = Y_wr - y_top
    Z_top = Z_wr - z_top
    Y_bot = Y_wr - y_bot
    Z_bot = Z_wr - z_bot

    V_top, W_top = vortex_velocities(
        Gamma_top,
        r0_ma, 
        Y_top, 
        Z_top,
        decay_ma,
    )

    V_bot, W_bot = vortex_velocities(
        Gamma_bot,
        r0_ma, 
        Y_bot, 
        Z_bot,
        decay_ma,
    )

    V_wr, W_wr = vortex_velocities(
        Gamma_wr,
        r0_wr, 
        Y_wr, 
        Z_wr,
        decay_wr,
    )

    # Normalized coordinates for ground effect
    Y_wrg = delta_y - y_deflection
    Z_wrg = z + HH + z_deflection
    Y_topg = Y_wrg - y_top
    Z_topg = Z_wrg + z_top
    Y_botg = Y_wrg - y_bot
    Z_botg = Z_wrg + z_bot

    V_topg, W_topg = vortex_velocities(
        -Gamma_top,
        r0_ma, 
        Y_topg, 
        Z_topg,
        decay_ma,
    )

    V_botg, W_botg = vortex_velocities(
        -Gamma_bot,
        r0_ma, 
        Y_botg, 
        Z_botg,
        decay_ma,
    )

    V_wrg, W_wrg = vortex_velocities(
        -Gamma_wr,
        r0_wr, 
        Y_wrg, 
        Z_wrg,
        decay_wr,
    )

    V = V_wr + V_wrg + V_top + V_topg + V_bot + V_botg
    W = W_wr + W_wrg + W_top + W_topg + W_bot + W_botg

    # no spanwise and vertical velocity upstream of the turbine
    # V[delta_x < -1] = 0.0  # Subtract by 1 to avoid numerical issues on rotation
    # W[delta_x < -1] = 0.0  # Subtract by 1 to avoid numerical issues on rotation
    # TODO Should this be <= ? Shouldn't be adding V and W on the current turbine?
    V[delta_x <= 0.0] = 0.0  # Subtract by 1 to avoid numerical issues on rotation
    W[delta_x <= 0.0] = 0.0  # Subtract by 1 to avoid numerical issues on rotation

    return V, W


def misalignment_added_turbulence_mixing(
    u_i,
    I_i,
    v_i,
    w_i,
    turb_v_i,
    turb_w_i
):
    # Since turbulence mixing is constant for the turbine,
    # use the left two dimensions only here and expand
    # before returning. Dimensions are (wd, ws).

    I_i = I_i[:,:,0,0,0]

    average_u_i = np.cbrt(np.mean(u_i ** 3, axis=(2,3,4)))

    # Convert ambient turbulence intensity to TKE (eq 24)
    k = (average_u_i * I_i) ** 2 / (2 / 3)

    u_term = np.sqrt(2 * k)
    v_term = np.mean(v_i + turb_v_i, axis=(2,3,4))
    w_term = np.mean(w_i + turb_w_i, axis=(2,3,4))

    # Compute the new TKE (eq 23)
    k_total = 0.5 * ( u_term ** 2 + v_term ** 2 + w_term ** 2 )

    # Convert TKE back to TI
    I_total = np.sqrt( (2 / 3) * k_total ) / average_u_i

    # Remove ambient from total TI leaving only the TI due to mixing
    I_mixing = I_total - I_i

    return I_mixing[:,:,None,None,None]

# def yaw_added_recovery_correction(
#     self, U_local, U, W, x_locations, y_locations, turbine, turbine_coord
# ):
#         """
#         This method corrects the U-component velocities when yaw added recovery
#         is enabled. For more details on how the velocities are changed, see [1].
#         # TODO add reference to 1

#         Args:
#             U_local (np.array): U-component velocities across the flow field.
#             U (np.array): U-component velocity deficits across the flow field.
#             W (np.array): W-component velocity deficits across the flow field.
#             x_locations (np.array): Streamwise locations in wake.
#             y_locations (np.array): Spanwise locations in wake.
#             turbine (:py:class:`floris.simulation.turbine.Turbine`):
#                 Turbine object.
#             turbine_coord (:py:obj:`floris.simulation.turbine_map.TurbineMap.coords`):
#                 Spatial coordinates of wind turbine.

#         Returns:
#             np.array: U-component velocity deficits across the flow field.
#         """
#         # compute the velocity without modification
#         U1 = U_local - U

#         # set dimensions
#         D = turbine.rotor_diameter
#         xLocs = x_locations - turbine_coord.x1
#         ky = self.ka * turbine.turbulence_intensity + self.kb
#         U2 = (np.mean(W) * xLocs) / ((ky * xLocs + D / 2))
#         U_total = U1 + np.nan_to_num(U2)

#         # turn it back into a deficit
#         U = U_local - U_total

#         # zero out anything before the turbine
#         U[x_locations < turbine_coord.x1] = 0

#         return U
