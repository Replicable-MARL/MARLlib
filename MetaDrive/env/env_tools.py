from metadrive.utils import norm


def update_neighbours_map(distance_map, vehicles, reward, info, config):
    distance_map.clear()
    keys = list(vehicles.keys())
    for c1 in range(0, len(keys) - 1):
        for c2 in range(c1 + 1, len(keys)):
            k1 = keys[c1]
            k2 = keys[c2]
            p1 = vehicles[k1].position
            p2 = vehicles[k2].position
            distance = norm(p1[0] - p2[0], p1[1] - p2[1])
            distance_map[k1][k2] = distance
            distance_map[k2][k1] = distance

    for kkk in info.keys():
        neighbours, nei_distances = find_in_range(kkk, config["neighbours_distance"], distance_map)
        info[kkk]["neighbours"] = neighbours
        info[kkk]["neighbours_distance"] = nei_distances
        nei_rewards = [reward[kkkkk] for kkkkk in neighbours]
        if nei_rewards:
            info[kkk]["nei_rewards"] = sum(nei_rewards) / len(nei_rewards)
        else:
            # i[kkk]["nei_rewards"] = r[kkk]
            info[kkk]["nei_rewards"] = 0.0  # Do not provides neighbour rewards
        info[kkk]["global_rewards"] = sum(reward.values()) / len(reward.values())


def find_in_range(v_id, distance, distance_map):
    if distance <= 0:
        return []
    max_distance = distance
    dist_to_others = distance_map[v_id]
    dist_to_others_list = sorted(dist_to_others, key=lambda k: dist_to_others[k])
    ret = [
        dist_to_others_list[i] for i in range(len(dist_to_others_list))
        if dist_to_others[dist_to_others_list[i]] < max_distance
    ]
    ret2 = [
        dist_to_others[dist_to_others_list[i]] for i in range(len(dist_to_others_list))
        if dist_to_others[dist_to_others_list[i]] < max_distance
    ]
    return ret, ret2
