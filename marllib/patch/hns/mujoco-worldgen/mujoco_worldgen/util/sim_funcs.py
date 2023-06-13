import logging
import numpy as np
import itertools

logger = logging.getLogger(__name__)

# #######################################
# ############ set_action ###############
# #######################################


def ctrl_set_action(sim, action):
    """
    For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if sim.model.nmocap > 0:
        _, action = np.split(action, (sim.model.nmocap * 7, ))
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                sim.data.ctrl[i] = action[i]
            else:
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]


# #######################################
# ############ get_reward ###############
# #######################################


def zero_get_reward(sim):
    return 0.0


def gps_dist(sim, obj0, obj1):
    obj0 = sim.data.get_site_xpos(obj0)
    obj1 = sim.data.get_site_xpos(obj1)
    diff = np.sum(np.square(obj0 - obj1))
    return diff + 0.3 * np.log(diff + 1e-4)


def l2_dist(sim, obj0, obj1):
    obj0 = sim.data.get_site_xpos(obj0)
    obj1 = sim.data.get_site_xpos(obj1)
    return np.sqrt(np.mean(np.square(obj0 - obj1)))

# #######################################
# ########### get_diverged ##############
# #######################################


def false_get_diverged(sim):
    return False, 0.0


def simple_get_diverged(sim):

    if sim.data.qpos is not None and \
         (np.max(np.abs(sim.data.qpos)) > 1000.0 or
          np.max(np.abs(sim.data.qvel)) > 100.0):
        return True, -20.0
    return False, 0.0

# #######################################
# ########### get_info ##############
# #######################################


def empty_get_info(sim):
    return {}

# #######################################
# ############## get_obs ################
# #######################################


def flatten_get_obs(sim):
    if sim.data.qpos is None:
        return np.zeros(0)
    return np.concatenate([sim.data.qpos, sim.data.qvel])


def image_get_obs(sim):
    return sim.render(100, 100, camera_name="rgb")

# Helpers


def get_body_geom_ids(model, body_name):
    """ Returns geom_ids in the body. """
    body_id = model.body_name2id(body_name)
    geom_ids = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == body_id:
            geom_ids.append(geom_id)
    return geom_ids


def change_geom_alpha(model, body_name_prefix, new_alpha):
    ''' Changes the visual transparency (alpha) of an object'''
    for body_name in model.body_names:
        if body_name.startswith(body_name_prefix):
            for geom_id in get_body_geom_ids(model, body_name):
                model.geom_rgba[geom_id, 3] = new_alpha


def joint_qpos_idxs(sim, joint_name):
    ''' Gets indexes for the specified joint's qpos values'''
    addr = sim.model.get_joint_qpos_addr(joint_name)
    if isinstance(addr, tuple):
        return list(range(addr[0], addr[1]))
    else:
        return [addr]


def qpos_idxs_from_joint_prefix(sim, prefix):
    ''' Gets indexes for the qpos values of all joints matching the prefix'''
    qpos_idxs_list = [joint_qpos_idxs(sim, name)
                      for name in sim.model.joint_names
                      if name.startswith(prefix)]
    return list(itertools.chain.from_iterable(qpos_idxs_list))


def joint_qvel_idxs(sim, joint_name):
    ''' Gets indexes for the specified joint's qvel values'''
    addr = sim.model.get_joint_qvel_addr(joint_name)
    if isinstance(addr, tuple):
        return list(range(addr[0], addr[1]))
    else:
        return [addr]


def qvel_idxs_from_joint_prefix(sim, prefix):
    ''' Gets indexes for the qvel values of all joints matching the prefix'''
    qvel_idxs_list = [joint_qvel_idxs(sim, name)
                      for name in sim.model.joint_names
                      if name.startswith(prefix)]
    return list(itertools.chain.from_iterable(qvel_idxs_list))


def body_names_from_joint_prefix(sim, prefix):
    ''' Returns a list of body names that contain joints matching the given prefix'''
    return [sim.model.body_id2name(sim.model.jnt_bodyid[sim.model.joint_name2id(name)])
            for name in sim.model.joint_names
            if name.startswith(prefix)]
