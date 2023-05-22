import re
import math
from enum import Enum
from collections import namedtuple
from numpy.linalg import norm
from gym.spaces import Box, Discrete
from ..utils.utils import in_range_deg

"""
A class to wrap and extend the Property object implemented in JSBSim
"""
Property = namedtuple("Property", "name_jsbsim description min max access spaces clipped update")
Property.__new__.__defaults__ = (None, None, float("-inf"), float("+inf"), "RW", Box, True, None)


class JsbsimCatalog(Property, Enum):
    """
    A class to store and customize jsbsim properties
    """

    # position and attitude

    position_h_sl_ft = Property("position/h-sl-ft", "altitude above mean sea level [ft]", -1400, 85000)
    position_h_agl_ft = Property(
        "position/h-agl-ft", "altitude above ground level [ft]", position_h_sl_ft.min, position_h_sl_ft.max
    )
    attitude_pitch_rad = Property("attitude/pitch-rad", "pitch [rad]", -0.5 * math.pi, 0.5 * math.pi, access="R")
    attitude_theta_rad = Property("attitude/theta-rad", "rad", access="R")
    attitude_theta_deg = Property("attitude/theta-deg", "deg", access="R")
    attitude_roll_rad = Property("attitude/roll-rad", "roll [rad]", -math.pi, math.pi, access="R")
    attitude_phi_rad = Property("attitude/phi-rad", "rad", access="R")
    attitude_phi_deg = Property("attitude/phi-deg", "deg", access="R")
    attitude_heading_true_rad = Property("attitude/heading-true-rad", "rad", access="R")
    attitude_psi_deg = Property("attitude/psi-deg", "heading [deg]", 0, 360, access="R")
    attitude_psi_rad = Property("attitude/psi-rad", "rad", access="R")
    aero_beta_deg = Property("aero/beta-deg", "sideslip [deg]", -180, +180, access="R")
    position_lat_geod_deg = Property("position/lat-geod-deg", "geocentric latitude [deg]", -90, 90, access="R")
    position_lat_geod_rad = Property("position/lat-geod-rad", "rad", access="R")
    position_lat_gc_deg = Property("position/lat-gc-deg", "deg")
    position_lat_gc_rad = Property("position/lat-gc-rad", "rad")
    position_long_gc_deg = Property("position/long-gc-deg", "geodesic longitude [deg]", -180, 180)
    position_long_gc_rad = Property("position/long-gc-rad", "rad")
    position_distance_from_start_mag_mt = Property(
        "position/distance-from-start-mag-mt", "distance travelled from starting position [m]", access="R"
    )
    position_distance_from_start_lat_mt = Property("position/distance-from-start-lat-mt", "mt", access="R")
    position_distance_from_start_lon_mt = Property("position/distance-from-start-lon-mt", "mt", access="R")
    position_epa_rad = Property("position/epa-rad", "rad", access="R")
    position_geod_alt_ft = Property("position/geod-alt-ft", "ft", access="R")
    position_radius_to_vehicle_ft = Property("position/radius-to-vehicle-ft", "ft", access="R")
    position_terrain_elevation_asl_ft = Property("position/terrain-elevation-asl-ft", "ft")

    # velocities

    velocities_u_fps = Property("velocities/u-fps", "body frame x-axis velocity [ft/s]", -2200, 2200, access="R")
    velocities_v_fps = Property("velocities/v-fps", "body frame y-axis velocity [ft/s]", -2200, 2200, access="R")
    velocities_w_fps = Property("velocities/w-fps", "body frame z-axis velocity [ft/s]", -2200, 2200, access="R")
    velocities_v_north_fps = Property("velocities/v-north-fps", "velocity true north [ft/s]", -2200, 2200, access="R")
    velocities_v_east_fps = Property("velocities/v-east-fps", "velocity east [ft/s]", -2200, 2200, access="R")
    velocities_v_down_fps = Property("velocities/v-down-fps", "velocity downwards [ft/s]", -2200, 2200, access="R")
    velocities_vc_fps = Property("velocities/vc-fps", "airspeed in knots", 0, 4400, access="R")
    velocities_h_dot_fps = Property("velocities/h-dot-fps", "rate of altitude change [ft/s]", access="R")
    velocities_u_aero_fps = Property("velocities/u-aero-fps", "fps", access="R")
    velocities_v_aero_fps = Property("velocities/v-aero-fps", "fps", access="R")
    velocities_w_aero_fps = Property("velocities/w-aero-fps", "fps", access="R")
    velocities_mach = Property("velocities/mach", "", access="R")
    velocities_machU = Property("velocities/machU", "", access="R")
    velocities_eci_velocity_mag_fps = Property("velocities/eci-velocity-mag-fps", "fps", access="R")
    velocities_vc_kts = Property("velocities/vc-kts", "kts", access="R")
    velocities_ve_fps = Property("velocities/ve-fps", "fps", access="R")
    velocities_ve_kts = Property("velocities/ve-kts", "kts", access="R")
    velocities_vg_fps = Property("velocities/vg-fps", "fps", access="R")
    velocities_vt_fps = Property("velocities/vt-fps", "fps", access="R")
    velocities_p_rad_sec = Property("velocities/p-rad_sec", "roll rate [rad/s]", -2 * math.pi, 2 * math.pi, access="R")
    velocities_q_rad_sec = Property(
        "velocities/q-rad_sec", "pitch rate [rad/s]", -2 * math.pi, 2 * math.pi, access="R"
    )
    velocities_r_rad_sec = Property("velocities/r-rad_sec", "yaw rate [rad/s]", -2 * math.pi, 2 * math.pi, access="R")
    velocities_p_aero_rad_sec = Property("velocities/p-aero-rad_sec", "rad/sec", access="R")
    velocities_q_aero_rad_sec = Property("velocities/q-aero-rad_sec", "rad/sec", access="R")
    velocities_r_aero_rad_sec = Property("velocities/r-aero-rad_sec", "rad/sec", access="R")
    velocities_phidot_rad_sec = Property("velocities/phidot-rad_sec", "rad/s", -2 * math.pi, 2 * math.pi, access="R")
    velocities_thetadot_rad_sec = Property(
        "velocities/thetadot-rad_sec", "rad/s", -2 * math.pi, 2 * math.pi, access="R"
    )
    velocities_psidot_rad_sec = Property("velocities/psidot-rad_sec", "rad/sec", -2 * math.pi, 2 * math.pi, access="R")

    # Acceleration

    accelerations_pdot_rad_sec2 = Property(
        "accelerations/pdot-rad_sec2", "rad/sÂ²", -(8 / 180) * math.pi, (8 / 180) * math.pi, access="R"
    )
    accelerations_qdot_rad_sec2 = Property(
        "accelerations/qdot-rad_sec2", "rad/sÂ²", -(8 / 180) * math.pi, (8 / 180) * math.pi, access="R"
    )
    accelerations_rdot_rad_sec2 = Property(
        "accelerations/rdot-rad_sec2", "rad/sÂ²", -(8 / 180) * math.pi, (8 / 180) * math.pi, access="R"
    )
    accelerations_vdot_ft_sec2 = Property("accelerations/vdot-ft_sec2", "ft/sÂ²", -4.0, 4.0, access="R")
    accelerations_wdot_ft_sec2 = Property("accelerations/wdot-ft_sec2", "ft/sÂ²", -4.0, 4.0, access="R")
    accelerations_udot_ft_sec2 = Property("accelerations/udot-ft_sec2", "ft/sÂ²", -4.0, 4.0, access="R")
    accelerations_a_pilot_x_ft_sec2 = Property(
        "accelerations/a-pilot-x-ft_sec2", "pilot body x-axis acceleration [ft/sÂ²]", access="R"
    )
    accelerations_a_pilot_y_ft_sec2 = Property(
        "accelerations/a-pilot-y-ft_sec2", "pilot body y-axis acceleration [ft/sÂ²]", access="R"
    )
    accelerations_a_pilot_z_ft_sec2 = Property(
        "accelerations/a-pilot-z-ft_sec2", "pilot body z-axis acceleration [ft/sÂ²]", access="R"
    )
    accelerations_n_pilot_x_norm = Property(
        "accelerations/n-pilot-x-norm", "pilot body x-axis acceleration, normalised", access="R"
    )
    accelerations_n_pilot_y_norm = Property(
        "accelerations/n-pilot-y-norm", "pilot body y-axis acceleration, normalised", access="R"
    )
    accelerations_n_pilot_z_norm = Property(
        "accelerations/n-pilot-z-norm", "pilot body z-axis acceleration, normalised", access="R"
    )

    # aero

    aero_alpha_deg = Property("aero/alpha-deg", "deg", access="R")
    aero_beta_rad = Property("aero/beta-rad", "rad", access="R")

    # controls state

    @staticmethod
    def update_equal_engine_props(sim, prop):
        """
        Update the given property for all engines
        :param sim: simulation to use
        :param prop: property to update
        """
        value = sim.get_property_value(prop)
        n = sim.jsbsim_exec.get_propulsion().get_num_engines()
        for i in range(1, n):
            sim.jsbsim_exec.set_property_value(prop.name_jsbsim + "[" + str(i) + "]", value)

    def update_equal_throttle_pos(sim):
        JsbsimCatalog.update_equal_engine_props(sim, JsbsimCatalog.fcs_throttle_pos_norm)

    def update_equal_mixture_pos(sim):
        JsbsimCatalog.update_equal_engine_props(sim, JsbsimCatalog.fcs_mixture_pos_norm)

    def update_equal_feather_pos(sim):
        JsbsimCatalog.update_equal_engine_props(sim, JsbsimCatalog.fcs_feather_pos_norm)

    def update_equal_advance_pos(sim):
        JsbsimCatalog.update_equal_engine_props(sim, JsbsimCatalog.fcs_advance_pos_norm)

    fcs_left_aileron_pos_norm = Property("fcs/left-aileron-pos-norm", "left aileron position, normalised", -1, 1)
    fcs_right_aileron_pos_norm = Property("fcs/right-aileron-pos-norm", "right aileron position, normalised", -1, 1)
    fcs_elevator_pos_norm = Property("fcs/elevator-pos-norm", "elevator position, normalised", -1, 1)
    fcs_rudder_pos_norm = Property("fcs/rudder-pos-norm", "rudder position, normalised", -1, 1)
    fcs_flap_pos_norm = Property("fcs/flap-pos-norm", "flap position, normalised", 0, 1)
    fcs_speedbrake_pos_norm = Property("fcs/speedbrake-pos-norm", "speedbrake position, normalised", 0, 1)
    fcs_spoiler_pos_norm = Property("fcs/spoiler-pos-norm", "normalised")
    fcs_steer_pos_deg = Property("fcs/steer-pos-deg", "deg")
    fcs_throttle_pos_norm = Property(
        "fcs/throttle-pos-norm", "throttle position, normalised", 0, 1, update=update_equal_throttle_pos
    )
    fcs_mixture_pos_norm = Property("fcs/mixture-pos-norm", "normalised", update=update_equal_mixture_pos)
    gear_gear_pos_norm = Property("gear/gear-pos-norm", "landing gear position, normalised", 0, 1)
    gear_num_units = Property("gear/num-units", "number of gears", access="R")
    fcs_feather_pos_norm = Property("fcs/feather-pos-norm", "normalised", update=update_equal_feather_pos)
    fcs_advance_pos_norm = Property("fcs/advance-pos-norm", "normalised", update=update_equal_advance_pos)

    # controls command

    def update_equal_throttle_cmd(sim):
        JsbsimCatalog.update_equal_engine_props(sim, JsbsimCatalog.fcs_throttle_cmd_norm)

    def update_equal_mixture_cmd(sim):
        JsbsimCatalog.update_equal_engine_props(sim, JsbsimCatalog.fcs_mixture_cmd_norm)

    def update_equal_advance_cmd(sim):
        JsbsimCatalog.update_equal_engine_props(sim, JsbsimCatalog.fcs_advance_cmd_norm)

    def update_equal_feather_cmd(sim):
        JsbsimCatalog.update_equal_engine_props(sim, JsbsimCatalog.fcs_feather_cmd_norm)

    @staticmethod
    def update_equal_brake_props(sim):
        value = sim.get_property_value(JsbsimCatalog.fcs_center_brake_cmd_norm)
        sim.jsbsim_exec.set_property_value(JsbsimCatalog.fcs_left_brake_cmd_norm.name_jsbsim, value)
        sim.jsbsim_exec.set_property_value(JsbsimCatalog.fcs_right_brake_cmd_norm.name_jsbsim, value)

    def update_equal_brake_cmd(sim):
        JsbsimCatalog.update_equal_brake_props(sim)

    fcs_aileron_cmd_norm = Property("fcs/aileron-cmd-norm", "aileron commanded position, normalised", -1.0, 1.0)
    fcs_elevator_cmd_norm = Property("fcs/elevator-cmd-norm", "elevator commanded position, normalised", -1.0, 1.0)
    fcs_rudder_cmd_norm = Property("fcs/rudder-cmd-norm", "rudder commanded position, normalised", -1.0, 1.0)
    fcs_throttle_cmd_norm = Property(
        "fcs/throttle-cmd-norm", "throttle commanded position, normalised", 0.0, 0.9, update=update_equal_throttle_cmd
    )
    fcs_mixture_cmd_norm = Property(
        "fcs/mixture-cmd-norm", "engine mixture setting, normalised", 0.0, 1.0, update=update_equal_mixture_cmd
    )
    gear_gear_cmd_norm = Property("gear/gear-cmd-norm", "all landing gear commanded position, normalised", 0.0, 1.0)
    fcs_speedbrake_cmd_norm = Property("fcs/speedbrake-cmd-norm", "normalised")
    fcs_left_brake_cmd_norm = Property("fcs/left-brake-cmd-norm", "Left brake command(normalized)", 0.0, 1.0)
    fcs_center_brake_cmd_norm = Property(
        "fcs/center-brake-cmd-norm", "normalised", 0.0, 1.0, update=update_equal_brake_cmd
    )
    fcs_right_brake_cmd_norm = Property("fcs/right-brake-cmd-norm", "Right brake command(normalized)", 0.0, 1.0)
    fcs_spoiler_cmd_norm = Property("fcs/spoiler-cmd-norm", "normalised")
    fcs_flap_cmd_norm = Property("fcs/flap-cmd-norm", "normalised")
    fcs_steer_cmd_norm = Property("fcs/steer-cmd-norm", "Steer command(normalized)", -1.0, 1.0)
    fcs_advance_cmd_norm = Property("fcs/advance-cmd-norm", "normalised", update=update_equal_advance_cmd)
    fcs_feather_cmd_norm = Property("fcs/feather-cmd-norm", "normalised", update=update_equal_feather_cmd)

    # initial conditions

    ic_h_sl_ft = Property("ic/h-sl-ft", "initial altitude MSL [ft]", position_h_sl_ft.min, position_h_sl_ft.max)
    ic_h_agl_ft = Property("ic/h-agl-ft", "", position_h_sl_ft.min, position_h_sl_ft.max)
    ic_geod_alt_ft = Property("ic/geod-alt-ft", "ft")
    ic_sea_level_radius_ft = Property("ic/sea-level-radius-ft", "ft")
    ic_terrain_elevation_ft = Property("ic/terrain-elevation-ft", "ft")
    ic_long_gc_deg = Property("ic/long-gc-deg", "initial geocentric longitude [deg]")
    ic_long_gc_rad = Property("ic/long-gc-rad", "rad")
    ic_lat_gc_deg = Property("ic/lat-gc-deg", "deg")
    ic_lat_gc_rad = Property("ic/lat-gc-rad", "rad")
    ic_lat_geod_deg = Property("ic/lat-geod-deg", "initial geodesic latitude [deg]")
    ic_lat_geod_rad = Property("ic/lat-geod-rad", "rad")
    ic_psi_true_deg = Property(
        "ic/psi-true-deg", "initial (true) heading [deg]", attitude_psi_deg.min, attitude_psi_deg.max
    )
    ic_psi_true_rad = Property("ic/psi-true-rad", "rad")
    ic_theta_deg = Property("ic/theta-deg", "deg")
    ic_theta_rad = Property("ic/theta-rad", "rad")
    ic_phi_deg = Property("ic/phi-deg", "deg")
    ic_phi_rad = Property("ic/phi-rad", "rad")
    ic_alpha_deg = Property("ic/alpha-deg", "deg")
    ic_alpha_rad = Property("ic/alpha-rad", "rad")
    ic_beta_deg = Property("ic/beta-deg", "deg")
    ic_beta_rad = Property("ic/beta-rad", "rad")
    ic_gamma_deg = Property("ic/gamma-deg", "deg")
    ic_gamma_rad = Property("ic/gamma-rad", "rad")
    ic_mach = Property("ic/mach", "")
    ic_u_fps = Property("ic/u-fps", "body frame x-axis velocity; positive forward [ft/s]")
    ic_v_fps = Property("ic/v-fps", "body frame y-axis velocity; positive right [ft/s]")
    ic_w_fps = Property("ic/w-fps", "body frame z-axis velocity; positive down [ft/s]")
    ic_p_rad_sec = Property("ic/p-rad_sec", "roll rate [rad/s]")
    ic_q_rad_sec = Property("ic/q-rad_sec", "pitch rate [rad/s]")
    ic_r_rad_sec = Property("ic/r-rad_sec", "yaw rate [rad/s]")
    ic_roc_fpm = Property("ic/roc-fpm", "initial rate of climb [ft/min]")
    ic_roc_fps = Property("ic/roc-fps", "fps")
    ic_vc_kts = Property("ic/vc-kts", "kts")
    ic_vd_fps = Property("ic/vd-fps", "fps")
    ic_ve_fps = Property("ic/ve-fps", "fps")
    ic_ve_kts = Property("ic/ve-kts", "kts")
    ic_vg_fps = Property("ic/vg-fps", "fps")
    ic_vg_kts = Property("ic/vg-kts", "kts")
    ic_vn_fps = Property("ic/vn-fps", "fps")
    ic_vt_fps = Property("ic/vt-fps", "fps")
    ic_vt_kts = Property("ic/vt-kts", "kts")
    ic_vw_bx_fps = Property("ic/vw-bx-fps", "fps")
    ic_vw_by_fps = Property("ic/vw-by-fps", "fps")
    ic_vw_bz_fps = Property("ic/vw-bz-fps", "fps")
    ic_vw_dir_deg = Property("ic/vw-dir-deg", "deg")
    ic_vw_down_fps = Property("ic/vw-down-fps", "fps")
    ic_vw_east_fps = Property("ic/vw-east-fps", "fps")
    ic_vw_mag_fps = Property("ic/vw-mag-fps", "fps")
    ic_vw_north_fps = Property("ic/vw-north-fps", "fps")
    ic_targetNlf = Property("ic/targetNlf", "")

    # engines

    propulsion_engine_set_running = Property("propulsion/engine/set-running", "engine running (0/1 bool)")
    propulsion_set_running = Property("propulsion/set-running", "set engine running (-1 for all engines)", access="W")
    propulsion_tank_contents_lbs = Property("propulsion/tank/contents-lbs", "")

    # simulation

    simulation_dt = Property("simulation/dt", "JSBSim simulation timestep [s]", access="R")
    simulation_sim_time_sec = Property("simulation/sim-time-sec", "Simulation time [s]", access="R")
    simulation_do_simple_trim = Property("simulation/do_simple_trim", "", access="W")

    # Auto Pilot
    ap_vg_hold = Property("ap/vg-hold", "Auto Pilot ON OFF")


class ExtraCatalog(Property, Enum):
    """
    A class to define and store new properties not implemented in JSBSim
    """

    # state in other formats

    position_h_sl_m = Property(
        "position/h-sl-m", "altitude above mean sea level [m]", -500, 26000, access="R",
        update=lambda sim: sim.set_property_value(
            ExtraCatalog.position_h_sl_m,
            sim.get_property_value(JsbsimCatalog.position_h_sl_ft) * 0.3048))

    velocities_v_north_mps = Property(
        "velocities/v-north-mps", "velocity true north [m/s]", -700, 700, access="R",
        update=lambda sim: sim.set_property_value(
            ExtraCatalog.velocities_v_north_mps,
            sim.get_property_value(JsbsimCatalog.velocities_v_north_fps) * 0.3048))

    velocities_v_east_mps = Property(
        "velocities/v-east-mps", "velocity east [m/s]", -700, 700, access="R",
        update=lambda sim: sim.set_property_value(
            ExtraCatalog.velocities_v_east_mps,
            sim.get_property_value(JsbsimCatalog.velocities_v_east_fps) * 0.3048))

    velocities_v_down_mps = Property(
        "velocities/v-down-mps", "velocity downwards [m/s]", -700, 700, access="R",
        update=lambda sim: sim.set_property_value(
            ExtraCatalog.velocities_v_down_mps,
            sim.get_property_value(JsbsimCatalog.velocities_v_down_fps) * 0.3048))

    velocities_vc_mps = Property(
        "velocities/vc-mps", "airspeed in knots [m/s]", 0, 1400, access="R",
        update=lambda sim: sim.set_property_value(
            ExtraCatalog.velocities_vc_mps,
            sim.get_property_value(JsbsimCatalog.velocities_vc_fps) * 0.3048))

    velocities_u_mps = Property(
        "velocities/u-mps", "body frame x-axis velocity [m/s]", -700, 700, access="R",
        update=lambda sim: sim.set_property_value(
            ExtraCatalog.velocities_u_mps,
            sim.get_property_value(JsbsimCatalog.velocities_u_fps) * 0.3048))

    velocities_v_mps = Property(
        "velocities/v-mps", "body frame y-axis velocity [m/s]", -700, 700, access="R",
        update=lambda sim: sim.set_property_value(
            ExtraCatalog.velocities_v_mps,
            sim.get_property_value(JsbsimCatalog.velocities_v_fps) * 0.3048))

    velocities_w_mps = Property(
        "velocities/w-mps", "body frame z-axis velocity [m/s]", -700, 700, access="R",
        update=lambda sim: sim.set_property_value(
            ExtraCatalog.velocities_w_mps,
            sim.get_property_value(JsbsimCatalog.velocities_w_fps) * 0.3048))

    def update_delta_altitude(sim):
        value = (sim.get_property_value(ExtraCatalog.target_altitude_ft) - sim.get_property_value(JsbsimCatalog.position_h_sl_ft)) * 0.3048
        sim.set_property_value(ExtraCatalog.delta_altitude, value)

    def update_delta_heading(sim):
        value = in_range_deg(
            sim.get_property_value(ExtraCatalog.target_heading_deg) - sim.get_property_value(JsbsimCatalog.attitude_psi_deg)
        )
        sim.set_property_value(ExtraCatalog.delta_heading, value)

    def update_delta_velocities(sim):
        value = (sim.get_property_value(ExtraCatalog.target_velocities_u_mps) - sim.get_property_value(ExtraCatalog.velocities_u_mps))
        sim.set_property_value(ExtraCatalog.delta_velocities_u, value)

    @staticmethod
    def update_property_incr(sim, discrete_prop, prop, incr_prop):
        value = sim.get_property_value(discrete_prop)
        if value == 0:
            pass
        else:
            if value == 1:
                sim.set_property_value(prop, sim.get_property_value(prop) - sim.get_property_value(incr_prop))
            elif value == 2:
                sim.set_property_value(prop, sim.get_property_value(prop) + sim.get_property_value(incr_prop))
            sim.set_property_value(discrete_prop, 0)

    def update_throttle_cmd_dir(sim):
        ExtraCatalog.update_property_incr(
            sim, ExtraCatalog.throttle_cmd_dir, JsbsimCatalog.fcs_throttle_cmd_norm, ExtraCatalog.incr_throttle
        )

    def update_aileron_cmd_dir(sim):
        ExtraCatalog.update_property_incr(
            sim, ExtraCatalog.aileron_cmd_dir, JsbsimCatalog.fcs_aileron_cmd_norm, ExtraCatalog.incr_aileron
        )

    def update_elevator_cmd_dir(sim):
        ExtraCatalog.update_property_incr(
            sim, ExtraCatalog.elevator_cmd_dir, JsbsimCatalog.fcs_elevator_cmd_norm, ExtraCatalog.incr_elevator
        )

    def update_rudder_cmd_dir(sim):
        ExtraCatalog.update_property_incr(
            sim, ExtraCatalog.rudder_cmd_dir, JsbsimCatalog.fcs_rudder_cmd_norm, ExtraCatalog.incr_rudder
        )

    def update_detect_extreme_state(sim):
        """
        Check whether the simulation is going through excessive values before it returns NaN values.
        Store the result in detect_extreme_state property.
        """
        extreme_velocity = sim.get_property_value(JsbsimCatalog.velocities_eci_velocity_mag_fps) >= 1e10
        extreme_rotation = (
            norm(
                sim.get_property_values(
                    [
                        JsbsimCatalog.velocities_p_rad_sec,
                        JsbsimCatalog.velocities_q_rad_sec,
                        JsbsimCatalog.velocities_r_rad_sec,
                    ]
                )
            ) >= 1000
        )
        extreme_altitude = sim.get_property_value(JsbsimCatalog.position_h_sl_ft) >= 1e10
        extreme_acceleration = (
            max(
                [
                    abs(sim.get_property_value(JsbsimCatalog.accelerations_n_pilot_x_norm)),
                    abs(sim.get_property_value(JsbsimCatalog.accelerations_n_pilot_y_norm)),
                    abs(sim.get_property_value(JsbsimCatalog.accelerations_n_pilot_z_norm)),
                ]
            ) > 1e1
        )  # acceleration larger than 10G
        sim.set_property_value(
            ExtraCatalog.detect_extreme_state,
            extreme_altitude or extreme_rotation or extreme_velocity or extreme_acceleration,
        )

    # position and attitude

    delta_altitude = Property(
        "position/delta-altitude-to-target-m",
        "delta altitude to target [m]",
        -40000,
        40000,
        access="R",
        update=update_delta_altitude,
    )
    delta_heading = Property(
        "position/delta-heading-to-target-deg",
        "delta heading to target [deg]",
        -180,
        180,
        access="R",
        update=update_delta_heading,
    )
    delta_velocities_u = Property(
        "position/delta-velocities_u-to-target-mps",
        "delta velocities_u to target",
        -1400,
        1400,
        access="R",
        update=update_delta_velocities,
    )
    # controls command

    throttle_cmd_dir = Property(
        "fcs/throttle-cmd-dir",
        "direction to move the throttle",
        0,
        2,
        spaces=Discrete,
        access="W",
        update=update_throttle_cmd_dir,
    )
    incr_throttle = Property("fcs/incr-throttle", "incrementation throttle", 0, 1)
    aileron_cmd_dir = Property(
        "fcs/aileron-cmd-dir",
        "direction to move the aileron",
        0,
        2,
        spaces=Discrete,
        access="W",
        update=update_aileron_cmd_dir,
    )
    incr_aileron = Property("fcs/incr-aileron", "incrementation aileron", 0, 1)
    elevator_cmd_dir = Property(
        "fcs/elevator-cmd-dir",
        "direction to move the elevator",
        0,
        2,
        spaces=Discrete,
        access="W",
        update=update_elevator_cmd_dir,
    )
    incr_elevator = Property("fcs/incr-elevator", "incrementation elevator", 0, 1)
    rudder_cmd_dir = Property(
        "fcs/rudder-cmd-dir",
        "direction to move the rudder",
        0,
        2,
        spaces=Discrete,
        access="W",
        update=update_rudder_cmd_dir,
    )
    incr_rudder = Property("fcs/incr-rudder", "incrementation rudder", 0, 1)

    # detect functions

    detect_extreme_state = Property(
        "detect/extreme-state",
        "detect extreme rotation, velocity and altitude",
        0,
        1,
        spaces=Discrete,
        access="R",
        update=update_detect_extreme_state,
    )

    # target conditions

    target_altitude_ft = Property(
        "tc/h-sl-ft",
        "target altitude MSL [ft]",
        JsbsimCatalog.position_h_sl_ft.min,
        JsbsimCatalog.position_h_sl_ft.max,
    )
    target_heading_deg = Property(
        "tc/target-heading-deg",
        "target heading [deg]",
        JsbsimCatalog.attitude_psi_deg.min,
        JsbsimCatalog.attitude_psi_deg.max,
    )
    target_velocities_u_mps = Property(
        "tc/target-velocity-u-mps",
        "target heading [mps]",
        -700,
        700
    )
    target_vg = Property("tc/target-vg", "target ground velocity [ft/s]")
    target_time = Property("tc/target-time-sec", "target time [sec]", 0)
    target_latitude_geod_deg = Property("tc/target-latitude-geod-deg", "target geocentric latitude [deg]", -90, 90)
    target_longitude_geod_deg = Property(
        "tc/target-longitude-geod-deg", "target geocentric longitude [deg]", -180, 180
    )
    heading_check_time = Property("heading_check_time", "time to check whether current time reaches heading time", 0, 1000000)


class MixedCatalog(dict):
    """
    A class to store both jsbsim & extra properties initiated and used during jsbsim simulation.
    """

    def __getitem__(self, name):
        try:
            return super().__getitem__(name)
        except KeyError:  # look for the property in ExtraCatalog and JsbsimCatalog
            try:
                self[name] = ExtraCatalog[name].value
            except KeyError:
                self[name] = JsbsimCatalog[name].value
        return super().__getitem__(name)

    def __getattr__(self, name):
        return self[name]

    def add_jsbsim_props(self, jsbsim_props):
        """Add to Catalog jsbsim properties from jbsbsim_props

        Args:
            jsbsim_props (list): list of 'name_jsbsim (access)' of jsbsim properties
        """
        for jsbsim_prop in jsbsim_props:
            [name_jsbsim, access] = jsbsim_prop.split(" ")
            access = re.sub(r"[\(\)]", "", access)  # remove parenthesis from the flag
            name = re.sub(r"_$", "", re.sub(r"[\-/\]\[]+", "_", name_jsbsim))  # get property name from jsbsim name
            if name not in self:
                assert name not in ExtraCatalog.__members__, \
                    f"{name} has been defined in JSBSim, use another name in ExtraCatalog"
                try:
                    self[name] = JsbsimCatalog[name].value
                except KeyError:
                    self[name] = Property(name_jsbsim=name_jsbsim, access=access)


# an instantiation of MixedCatalog used for simulation
Catalog = MixedCatalog()
