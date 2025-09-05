import numpy as np
from question1 import is_blocked

drone_init_posi = np.array([17800.0, 0.0, 1800.0])
missile_init_posi = np.array([20000.0, 0.0, 2000.0])
v_missile = -300.0 * missile_init_posi / np.linalg.norm(missile_init_posi)
v_smoke = np.array([0.0, 0.0, -3.0])
gravity = 9.81

# define the objective function
def obj_func(posi_vector):
    """
    the metrics to optimize
    Args:
    - posi_vector: np.array of list, [vel_dir, vel_val, t_release, t_wait]
    """
    # get the velocity of drone
    v_dir_angle = posi_vector[0]
    v_drone = np.array([
        np.cos(v_dir_angle), np.sin(v_dir_angle), 0.0
    ]) * posi_vector[1]

    # get the time of release and wait for explode
    t_release = posi_vector[2]
    t_wait = posi_vector[3]

    # get release position, smoke bomb explode position, and missile position
    release_posi = drone_init_posi + v_drone * t_release
    explode_posi = release_posi + v_drone * t_wait
    explode_posi[2] -= 0.5 * gravity * t_wait ** 2
    missile_posi = missile_init_posi + v_missile * (t_release + t_wait)

    time_list = []
    for t in np.linspace(0, 20, 2001):
        # get the position of missile and smoke at time t
        missile_posi_t = missile_posi + t * v_missile
        smoke_posi_t = explode_posi + t * v_smoke

        if is_blocked(missile_posi_t, smoke_posi_t):
            time_list.append(t)

    # return the time intervals
    return (time_list[-1] - time_list[0]) if len(time_list) >= 1 else 0.0


# --------------------------------
# Particle and PSO
# --------------------------------
class Particle:
    def __init__(self, posi, vel):
        """
        Args:
        - posi: np.array([vel_dir, vel_val, t_release, t_wait])
        - vel: np.array([v_dir, v_val, v_release, v_wait])
        """
        self.posi = np.array(posi, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.best_posi = self.posi.copy()
        self.best_val = None  # calculated by PSO

    @staticmethod
    def create_random_particle():
        # static method to create random particle
        pos = np.array([
            np.random.uniform(0, 2 * np.pi),  # vel_dir
            np.random.uniform(70, 140),       # vel_val
            np.random.uniform(0.8, 5),        # t_release
            np.random.uniform(2, 4.5)         # t_wait
        ])
        vel = np.random.uniform(0, 1, size=4)
        return Particle(pos, vel)

class PSO:
    def __init__(
            self,
            objective_function = obj_func,
            num_particles = 50,
            num_iterations = 50,
            w = 0.7,   # 惯性权重, 控制例子速度的惯性
            c1 = 1.5,  # 认知系数
            c2 = 1.5   # 社会系数
    ):
        self.obj_func = objective_function
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.history = []

        # initialize global parpams
        self.global_best_posi = None
        self.global_best_val = -np.inf

        # initialize particles
        self.particles = [
            Particle.create_random_particle() for _ in range(self.num_particles)
        ]
        for particle in self.particles:
            # calculate the initial personal best value
            particle.best_val = self.obj_func(particle.posi)
            # calculate the initial global best value
            if particle.best_val > self.global_best_val:
                self.global_best_val = particle.best_val
                self.global_best_posi = particle.posi.copy()

    def update(self):
        # method to update a generation
        for particle in self.particles:
            # get the random scale
            r1, r2 = np.random.rand(2)

            # update the velocity
            particle.vel = (
                self.w * particle.vel +
                self.c1 * r1 * (particle.best_posi - particle.posi) +
                self.c2 * r2 * (self.global_best_posi - particle.posi)
            )
            # update the position
            particle.posi += particle.vel

            # get current value and update the personal best
            curr_val = self.obj_func(particle.posi)
            if curr_val >= particle.best_val:
                particle.best_val = curr_val
                particle.best_posi = particle.posi.copy()

            # update the global best
            if curr_val > self.global_best_val:
                self.global_best_val = curr_val
                self.global_best_posi = particle.posi.copy()

    def optimize(self):
        # method to optimize
        for i in range(self.num_iterations):
            self.update()
            self.history.append(self.global_best_posi.copy())
            print(
                f"Iteration {i + 1}/{self.num_iterations}, " +
                f"Global Best Position: {self.global_best_posi}, " +
                f"Global Best Value: {self.global_best_val}"
            )

if __name__ == '__main__':
    pso = PSO()
    pso.optimize()
    print("最优值 =", pso.global_best_val)
    print("最优位置 =", pso.global_best_posi)
