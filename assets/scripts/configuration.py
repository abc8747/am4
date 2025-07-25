# --8<-- [start:fyj-conf]
def calc_fjy_conf(yd, jd, fd, capacity):
    fs = min(fd, capacity // 3)
    capacity -= fs * 3  # (1)!
    js = min(jd, capacity // 2)
    capacity -= js * 2  # (2)!
    ys = capacity # (3)!
    valid = ys < yd
    
    return (ys, js, fs, valid)
# --8<-- [end:fyj-conf]

# --8<-- [start:l-conf]
def calc_l_conf(ld, hd, capacity, l_training = 0, h_training = 0):
    l_cap = capacity * 0.7 * (1 + l_training / 100.0)
    if ld > l_cap:  # (1)!
        return (1, 0, True)

    l = ld / l_cap  # (2)!
    h = 1 - l  # (3)!
    valid = hd >= capacity * h * (1 + h_training / 100.0)
    return (l, h, valid)
# --8<-- [end:l-conf]
