import math
import numpy as np
# metrics provided by websaliency

def discretize_gt(gt):
    out = np.zeros_like(gt, dtype=np.float32)
    out[gt > 0] = 1.0
    return out


def auc_judd(s_map, gt):
    gt = discretize_gt(gt)
    thresholds = []
    fix_pts = np.where(gt > 0)
    for (r, c) in zip(*fix_pts):
        thresholds.append(s_map[r, c])
    num_fix = np.sum(gt)
    thresholds = sorted(set(thresholds))
    area = [(0.0, 0.0)]
    for thresh in thresholds:
        temp = np.zeros_like(s_map)
        temp[s_map >= thresh] = 1.0
        num_overlap = np.where((temp + gt) == 2)[0].shape[0]
        tp = num_overlap / (num_fix + 1e-8)
        fp = (temp.sum() - num_overlap) / ((gt.shape[0] * gt.shape[1]) - num_fix + 1e-8)
        area.append((round(tp, 4), round(fp, 4)))
    area.append((1.0, 1.0))
    area.sort(key=lambda x: x[0])
    tp_list = [x[0] for x in area]
    fp_list = [x[1] for x in area]
    return float(np.trapz(np.array(tp_list), np.array(fp_list)))


def auc_shuff(s_map, gt, other_map, splits=100, stepsize=0.1):
    gt = discretize_gt(gt)
    other_map = discretize_gt(other_map)
    num_fix = int(np.sum(gt))
    x, y = np.where(other_map == 1)
    other_map_fixs = []
    for j in zip(x, y):
        other_map_fixs.append(j[0] * other_map.shape[1] + j[1])
    ind = len(other_map_fixs)
    assert ind == np.sum(other_map), 'something is wrong in auc shuffle'
    num_fix_other = min(ind, num_fix)
    if num_fix_other < 1:
        return 0.0

    random_numbers = []
    for _ in range(splits):
        perm = np.random.permutation(ind)
        random_numbers.append(perm)

    aucs = []
    for arr in random_numbers:
        subset = arr[:num_fix_other]
        r_sal = []
        H, W = s_map.shape
        for k in subset:
            rr = (k // W) % H
            cc = (k % W)
            r_sal.append(s_map[rr, cc])
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        thresholds = sorted(set(thresholds))
        area = [(0.0, 0.0)]
        for thresh in thresholds:
            temp = np.zeros_like(s_map)
            temp[s_map >= thresh] = 1.0
            num_overlap = np.where((temp + gt) == 2)[0].shape[0]
            tp = num_overlap / (num_fix + 1e-8)

            r_arr = np.array(r_sal)
            fp = np.count_nonzero(r_arr > thresh) / (num_fix + 1e-8)
            area.append((round(tp, 4), round(fp, 4)))
        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]
        auc_val = np.trapz(np.array(tp_list), np.array(fp_list))
        aucs.append(auc_val)
    return float(np.mean(aucs))


def cc(s_map, gt):
    s_map_norm = (s_map - np.mean(s_map)) / (np.std(s_map) + 1e-8)
    gt_norm = (gt - np.mean(gt)) / (np.std(gt) + 1e-8)
    r = (s_map_norm * gt_norm).sum() / math.sqrt((s_map_norm ** 2).sum() * (gt_norm ** 2).sum() + 1e-8)
    return float(r)


def kldiv(s_map, gt):
    eps = 1e-16
    s_sum = s_map.sum()
    g_sum = gt.sum()
    if s_sum < eps or g_sum < eps:
        return 0.0
    p = s_map / (s_sum + eps)
    q = gt / (g_sum + eps)
    ratio = (q + eps) / (p + eps)
    klv = (q * np.log(ratio)).sum()
    return float(klv)


def nss(s_map, gt):
    gt_ = discretize_gt(gt)
    s_map_norm = (s_map - np.mean(s_map)) / (np.std(s_map) + 1e-8)
    x, y = np.where(gt_ == 1)
    if len(x) == 0:
        return 0.0
    temp = []
    for (xx, yy) in zip(x, y):
        temp.append(s_map_norm[xx, yy])
    return float(np.mean(temp))
