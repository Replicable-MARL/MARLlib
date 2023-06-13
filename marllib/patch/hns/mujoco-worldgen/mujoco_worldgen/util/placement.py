import numpy as np
import scipy.optimize


def place_boxes(random_state, boxes, width, height, placement_margin=0.0):
    '''
    Tries to randomly place rectangular boxes between (0,0) and (width, height)
    boxes - list of dictionaries, each with keys 'size' and 'placement_xy'
            'size' must be (x,y)
            'placement_xy' is either None or (x_frac, y_frac) between 0 and 1
                (0, 0) is left lower corner, (.5, .5) is the center, etc
    width, height - dimensions of the outer bound to place boxes in
    placement_margin - minimum distance (orthogonal) between placed boxes
    Returns list of box positions (x,y) from left lower corner, or None if
        no solution found.
    '''
    width = float(width)
    height = float(height)
    assert width > 0 and height > 0, "invalid width, height"
    area = 0
    sizes = [box["size"] for box in boxes]
    for s in sizes:
        if s[0] > width - 2 * placement_margin or s[0] <= 0.0 or \
           s[1] > height - 2 * placement_margin or s[1] <= 0.0:
            return None
        area += (s[0] + placement_margin // 2) * (s[1] + placement_margin // 2)
    if area > 0.6 * (width - placement_margin) * (height - placement_margin) and len(sizes) > 1:
        return None
    if area > width * height:
        return None
    a_edge, b_edge = _get_edge_constraints(sizes, width, height, placement_margin)

    for _ in range(10):
        def get_matrices(xy):
            a_eq = []
            b_eq = []
            for idx, box in enumerate(boxes):
                if box["placement_xy"] is not None:
                    xy[idx] = box["placement_xy"][0] * (width - box["size"][0])
                    xy[idx + len(sizes)] = box["placement_xy"][1] * \
                        (height - box["size"][1])
                    for i in [idx, idx + len(sizes)]:
                        a = np.zeros(2 * len(sizes))
                        a[i] = 1.0
                        a_eq.append(a)
                        b_eq.append(xy[i])
            a_pairwise, b_pairwise, violated = _get_pairwise_constraints(xy, boxes, placement_margin)
            for idx, v in enumerate(violated):
                if not v:
                    for i in [idx, idx + len(sizes)]:
                        a = np.zeros(2 * len(sizes))
                        a[i] = 1.0
                        a_eq.append(a)
                        b_eq.append(xy[i])
            if len(a_eq) > 0:
                a_eq = np.stack(a_eq)
                b_eq = np.stack(b_eq)
            else:
                a_eq = None
                b_eq = None
            return a_eq, b_eq, a_pairwise, b_pairwise, violated

        best_xy = None
        best_violated = [True for _ in range(100)]
        for _ in range(10):
            xy = _get_random_xy(random_state, boxes, width, height, placement_margin)
            a_eq, b_eq, a_pairwise, b_pairwise, violated = get_matrices(xy)
            if sum(violated) < sum(best_violated):
                best_xy = xy
                best_violated = violated

        xy = best_xy
        a_eq, b_eq, a_pairwise, b_pairwise, violated = get_matrices(xy)

        # If it's not violated, than further optimization is not needed.
        if sum(violated) > -1e-4:
            return [(a, b) for a, b in zip(xy[:len(sizes)], xy[len(sizes):])]
        if len(a_pairwise) == 0:
            a = a_edge
            b = b_edge
        else:
            a = np.concatenate([a_edge, np.stack(a_pairwise)], 0)
            b = np.concatenate([b_edge, np.array(b_pairwise)], 0)

        random = random_state.uniform(-1, 1, 2 * len(sizes))
        sol = scipy.optimize.linprog(random, A_ub=-a, b_ub=-b, A_eq=a_eq, b_eq=b_eq)
        if sol.success:
            return _further_randomize(random_state, boxes, a, b, sol.x)
    return None


# Constrainsts ensuring that boxes are within 0-width, 0-height.
def _get_edge_constraints(sizes, width, height, placement_margin):
    # a xy >= b
    a_edge = []
    b_edge = []
    for idx, s in enumerate(sizes):
        # x_idx >= 0
        a = np.zeros(2 * len(sizes))
        a[idx] = 1.0
        a_edge.append(a)
        b_edge.append(placement_margin)

        # x_idx <= width - s_idx[0]
        # -x_idx >= s_idx[0] - width
        a = np.zeros(2 * len(sizes))
        a[idx] = -1.0
        a_edge.append(a)
        b_edge.append(s[0] - width + placement_margin)

        # y_idx >= 0
        a = np.zeros(2 * len(sizes))
        a[idx + len(sizes)] = 1.0
        a_edge.append(a)
        b_edge.append(placement_margin)

        # y_idx <= height - s_idx[1]
        # -y_idx >= s_idx[0] - height
        a = np.zeros(2 * len(sizes))
        a[idx + len(sizes)] = -1.0
        a_edge.append(a)
        b_edge.append(s[1] - height + placement_margin)
    a_edge = np.stack(a_edge)
    b_edge = np.array(b_edge)
    return a_edge, b_edge


# simplex algorithm locates all objects next to edges.
# We compute slack of the solution and we move objects
# within the slack randomly.
def _further_randomize(random_state, boxes, a, b, xy):
    # Determines how much positions can be shifted.
    slack = np.zeros((2, a.shape[1]))
    slack[0, :] = -np.inf
    slack[1, :] = np.inf
    # a * xy - b > sol_slack
    # if a[i][j] == 1 then xy[j] can be as small as -slack / np.sum(abs(a[i]))
    # if a[i][j] == -1 then xy[j] can be as big as slack / np.sum(abs(a[i]))
    # np.sum(abs(a[i]))
    sol_slack = np.matmul(a, xy) - b
    for i in range(a.shape[0]):
        row = np.sum(np.abs(a[i]))
        for j in range(a.shape[1]):
            if np.abs(a[i][j] - 1.) < 1e-4:
                slack[0][j] = min(
                    max(sol_slack[i], -slack[0][j]) / row, 0)
            elif np.abs(a[i][j] + 1.) < 1e-4:
                slack[1][j] = max(
                    min(sol_slack[i], slack[1][j]) / row, 0)
    assert((slack[0][:] <= slack[1][:]).all())
    dim = xy.shape[0] // 2
    for i in range(xy.shape[0]):
        if boxes[i % dim]["placement_xy"] is None:
            xy[i] += random_state.uniform(slack[0][i], slack[1][i])
    return [(a, b) for a, b in zip(xy[:dim], xy[dim:])]


# Generates random initial xy position of boxes. attempts to place
# them far from each other (with respect to each coordinate).
def _get_random_xy(random_state, boxes, width, height, placement_margin):
    xy = np.zeros(len(boxes) * 2)
    for t in range(3):
        if t > 0:
            dist = np.abs(np.expand_dims(xy, 1) - np.expand_dims(xy, 0))
            dist /= max(width, height)
            for i in range(dist.shape[0]):
                dist[i, i] = 1.0
        for i, box in enumerate(boxes):
            if t == 0 or np.min(dist[i, :]) < 0.1:
                xy[i] = random_state.uniform(placement_margin,
                                             width - box["size"][0] - placement_margin)
            if t == 0 or np.min(dist[i + len(boxes), :]) < 0.1:
                xy[i + len(boxes)] = random_state.uniform(placement_margin,
                                                          height - box["size"][1] - placement_margin)
    return xy


# determines all constrains between pairs of boxes.
def _get_pairwise_constraints(xy, boxes, placement_margin):
    a_pairwise = []
    b_pairwise = []
    violated = [0.0 for _ in range(len(boxes))]
    sizes = [box["size"] for box in boxes]
    for idx0, s0 in enumerate(sizes):
        for idx1, s1 in enumerate(sizes):
            if idx1 <= idx0:
                continue
            a_small = []
            b_small = []

            for axis in [0, 1]:
                # x0 >= x1 + s1[0] or y0 >= y1 + s1[1]
                a = np.zeros(2 * len(boxes))
                a[idx0 + axis * len(boxes)] = 1.0
                a[idx1 + axis * len(boxes)] = -1.0
                a_small.append(a)
                b_small.append(s1[axis] + placement_margin)

                # x1 >= x0 + s0[0] or y1 >= y0 + s0[1]
                a = np.zeros(2 * len(boxes))
                a[idx0 + axis * len(boxes)] = -1.0
                a[idx1 + axis * len(boxes)] = 1.0
                a_small.append(a)
                b_small.append(s0[axis] + placement_margin)

            a_small = np.stack(a_small)
            b_small = np.array(b_small)
            ret = np.matmul(a_small, xy) - b_small
            idx = np.argmax(ret)
            a_pairwise.append(a_small[idx])
            b_pairwise.append(b_small[idx])
            if ret[idx] <= -1e-4:
                violated[idx0] += ret[idx]
                violated[idx1] += ret[idx]
    return a_pairwise, b_pairwise, violated
