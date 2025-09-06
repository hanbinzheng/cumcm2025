import numpy as np
from typing import List, Tuple
from GAFrame import SingleDronePlan, Evaluator
from GAFrame import Genome, Individual, GeneticAlgorithm

"""
            init_posi: np.array([17800.0, 0.0, 1800.0])
            vel_dir: float,  # direction in angles
            vel_val: float,  # velocity value
            t_release: np.array([release1, release2, release3])
            t_wait: np.array([wait1, wait2, wait3])

"""

class Genome3(Genome):
    def __init__(self,
                 drone_plans: List[SingleDronePlan]):
        super().__init__(drone_plans)

    def mutate(self, mutation_rate: float):
        # get the plan
        plan = self.drone_plans[0]

        # mute the vel_dir
        if np.random.rand() < mutation_rate:
             plan.vel_dir += np.random.uniform(-np.pi/12, np.pi/12)

        # mute the vel_val
        if np.random.rand() < mutation_rate:
            new_vel_val = plan.vel_val + np.random.uniform(-10.0, 10.0)
            plan.vel_val = np.clip(new_vel_val, 70.0, 140.0)  # check boundary

        # mute the t_release
        if np.random.rand() < mutation_rate:
            new_t_release = plan.t_release + np.random.uniform(-1.0, 1.0, 3)
            plan.t_release = np.clip(new_t_release, 0.0, 1e2)

        # mute the t_wait
        if np.random.rand() < mutation_rate:
            new_t_wait = plan.t_wait + np.random.uniform(-1.0, 1.0, 3)
            plan.t_wait = np.clip(new_t_wait, 0.0, 1e2)

    def crossover(self, pair_genome: "Genome3") -> Tuple["Genome3", "Genome3"]:
        # get the copy
        copy1, copy2 = self.get_copy(), pair_genome.get_copy()
        plan1, plan2 = copy1.drone_plans[0], copy2.drone_plans[0]
        magnitude = 0.5

        # mutate the velocity
        child1_vel_dir = plan1.vel_dir * magnitude + (1 - magnitude) * plan2.vel_dir
        child1_vel_val = np.clip(
            plan1.vel_val * magnitude + (1 - magnitude) * plan2.vel_val, 70.0, 140.0
        )

        child2_vel_dir = plan2.vel_dir * magnitude + (1 - magnitude) * plan1.vel_dir
        child2_vel_val = np.clip(
            plan2.vel_val * magnitude + (1 - magnitude) * plan1.vel_val, 70.0, 140.0
        )
        
        plan1.vel_dir, plan2.vel_dir = child1_vel_dir, child2_vel_dir
        plan1.vel_val, plan2.vel_val = child1_vel_val, child2_vel_val

        # mutate the time
        par_t_release1, par_t_release2 = plan1.t_release, plan2.t_release
        par_t_wait1, par_t_wait2 = plan1.t_wait, plan2.t_wait

        child_t_release1 = np.clip(
            magnitude * par_t_release1 + (1 - magnitude) * par_t_release2,
            0.0,
            1e2
        )
        child_t_release2 = np.clip(
            magnitude * par_t_release2 + (1 - magnitude) * par_t_release1,
            0.0,
            1e2
        )

        child_t_wait1 = np.clip(
            magnitude * par_t_wait1 + (1 - magnitude) * par_t_wait2,
            0.0,
            1e2
        )
        child_t_wait2 = np.clip(
            magnitude * par_t_wait2 + (1 - magnitude) * par_t_wait1,
            0.0,
            1e2
        )

        plan1.t_release, plan2.t_release = child_t_release1, child_t_release2
        plan1.t_wait, plan2.t_wait = child_t_wait1, child_t_wait2

        return copy1, copy2


class Evaluator3(Evaluator):
    def __init__(self):
        missile_posi = np.array([20000.0, 0.0, 2000.0])
        super().__init__([missile_posi])

    def evaluate(self, genome: Genome3) -> float:
        plan = genome.drone_plans[0]
        t_release = 0.0
        v_missile = self.v_missile[0]
        missile_posi = self.missile_init_posi[0]
        time_list = []

        for idx in range(3):
            t_list = []
            t_release += plan.t_release[idx]
            t_exp = t_release + plan.t_wait[idx]
            missile_exp = missile_posi + t_exp * v_missile

            for t in np.linspace(0.0, 20.0, 201):
                smoke_posi_now = plan.get_smoke_posi(t + t_exp)[idx]
                if smoke_posi_now is None:
                    continue

                missile_now = missile_exp + t * v_missile

                if self.whether_blocked_simple(missile_now, smoke_posi_now):
                    time_list.append(t + t_exp)

        return self.__class__.calculate_total_time(time_list, 0.19)


class GeneticAlgorithm3(GeneticAlgorithm):
    def __init__(
            self,
            pop_size: int,
            evaluator: Evaluator3,
            crossover_rate: float,
            mutation_rate: float,
            num_gen: int
    ):
        super().__init__(
            pop_size = pop_size,
            evaluator = evaluator,
            crossover_rate = crossover_rate,
            mutation_rate = mutation_rate,
            num_gen = num_gen
        )

    def initialize_pop(self, pop_size: int) -> List[Individual]:
        pop = []

        # add a valid individual 4.69s
        plan = SingleDronePlan(
            init_posi = np.array([17800.0, 0.0, 1800.0]),
            vel_dir = 3.138734594663325,
            vel_val = 95.53315144880676,
            t_release = np.array([
                5.83314342e-04, 1.09184040e+00, 1.83565661e+00
            ]),
            t_wait = np.array([
                0.0, 3.73068478, 3.96467676
            ])
        )
        genome = Genome3([plan])
        pop.append(Individual(genome))

        # add a valid individual 4.50s
        plan = SingleDronePlan(
            init_posi = np.array([17800.0, 0.0, 1800.0]),
            vel_dir = 0.10570374302020343,
            vel_val = 123.288008617581,
            t_release = np.array([
                0.02500057, 0.02902798, 2.39741573
            ]),
            t_wait = np.array([
                2.6459009,  0.48762429, 4.73639645
            ])
        )
        genome = Genome3([plan])
        pop.append(Individual(genome))

        # add a valid individual 6.19s
        plan = SingleDronePlan(
            init_posi = np.array([17800.0, 0.0, 1800.0]),
            vel_dir = 0.1460053606438416,
            vel_val = 127.90179868301934,
            t_release = np.array([
                0.0, 0.12179974, 2.65560477
            ]),
            t_wait = np.array([
                1.00038769, 0.17474348, 4.92598046
            ])
        )
        genome = Genome3([plan])
        pop.append(Individual(genome))

        # 5.024735255645854
        plan = SingleDronePlan(
            init_posi = np.array([17800.0, 0.0, 1800.0]),
            vel_dir = 0.1460053606438416,
            vel_val = 124.57654040545049,
            t_release = np.array([
                0.0, 0.70084213, 3.55608462
            ]),
            t_wait = np.array([
                0.52557739, 0.0, 4.24751896
            ])
        )
        genome = Genome3([plan])
        pop.append(Individual(genome))

        for i in range(pop_size - 4):
            plan = SingleDronePlan(
                init_posi = np.array([17800.0, 0.0, 1800.0]),
                vel_dir = np.random.rand() * 2 * np.pi,
                vel_val = np.random.uniform(70.0, 140.0),
                t_release = np.random.uniform(0.0, 5.0, 3),
                t_wait = np.random.uniform(0.0, 5.0, 3)
            )
            genome = Genome3([plan])
            pop.append(Individual(genome))

        return pop


if __name__ == '__main__':
    GA = GeneticAlgorithm3(
        pop_size = 200,
        evaluator = Evaluator3(),
        crossover_rate = 0.5,
        mutation_rate = 0.5,
        num_gen = 1000
    )
    GA.evolve()
