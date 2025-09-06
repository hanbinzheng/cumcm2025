import numpy as np
import math

def is_blocked(missile: np.ndarray, smoke: np.ndarray) -> bool:
    tgt = np.array([0.0, 200.0, 5.0])
    r_tgt = math.sqrt(7.0 ** 2 + 5.0 ** 2)
    r_smoke = 10.0

    missile_smoke = smoke - missile
    dist_missile_smoke = np.linalg.norm(missile_smoke)

    missile_tgt = tgt - missile
    dist_missile_tgt = np.linalg.norm(missile_tgt)

    # calculate angles
    angle_missile_smoke = np.arcsin(
        np.clip(r_smoke / dist_missile_smoke, -1.0, 1.0)
    )
    angle_missile_tgt = np.arcsin(
        np.clip(r_tgt / dist_missile_tgt, -1.0, 1.0)
    )
    cos_two_axis = np.dot(missile_smoke, missile_tgt) / (
        dist_missile_smoke * dist_missile_tgt
    )
    angle_two_axis = np.arccos(
        np.clip(cos_two_axis, -1.0, 1.0)
    )


    # quick check when the smoke envelope the target or the missile
    smoke_wrap_missile = dist_missile_smoke <= r_smoke
    dist_smoke_tgt = np.linalg.norm(tgt - smoke)
    smoke_wrap_tgt = (dist_smoke_tgt <= r_smoke - r_tgt)
    wrap = smoke_wrap_missile or smoke_wrap_tgt
    if wrap:
        return wrap

    # check whether the smoke is between missile and target
    unit_missile_tgt = missile_tgt / dist_missile_tgt
    smoke_proj = np.dot(missile_smoke, unit_missile_tgt)
    in_between = (
        (smoke_proj >= 0) and (smoke_proj <= dist_missile_tgt)
    )

    # check the angle to determine whether blocked
    angle_within = angle_missile_smoke >= angle_missile_tgt + angle_two_axis

    return in_between and angle_within

if __name__ == '__main__':

    missile_init_posi = np.array([20000.0, 0.0, 2000.0])
    unit_v_missile = - missile_init_posi / np.linalg.norm(missile_init_posi)
    v_missile = 300.0 * unit_v_missile

    smoke_init_posi = np.array([17800.0, 0.0, 1800.0])
    v_smoke = 3.0 * np.array([0.0, 0.0, -1.0])
    v_drone = 120 * np.array([-1.0, 0.0, 0.0])
    g = 9.81

    # the time when the drone release the smoke bomb
    missile = missile_init_posi + v_missile * 1.5
    smoke = smoke_init_posi + v_drone * 1.5

    # the time when the smoke bomb bombed
    missile = missile + v_missile * 3.6
    smoke = smoke + v_drone * 3.6  # still fly with drone velocity
    smoke[2] -= 0.5 * 9.81 * 3.6 * 3.6

    list = []

    for t in np.linspace(0, 20.0, 20001):
        missile_t = missile + v_missile * t
        smoke_t = smoke + v_smoke * t
        if(is_blocked(missile_t, smoke_t)):
            list.append(t + 1.6 + 3.5)

    print(f"The time blcoked is form {list[0]}s to {list[-1]}s, for {list[-1] - list[0]} s.")
