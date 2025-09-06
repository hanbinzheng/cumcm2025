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
             plan.vel_dir += np.random.uniform(-np.pi/18, np.pi/18)

        # mute the vel_val
        if np.random.rand() < mutation_rate:
            new_vel_val = plan.vel_val + np.random.uniform(-5, 5)
            plan.vel_val = np.clip(new_vel_val, 70.0, 140.0)  # check boundary

        # mute the t_release
        if np.random.rand() < mutation_rate:
            new_t_release = plan.t_release + np.random.uniform(-0.2, 0.2, 3)
            plan.t_release = np.clip(new_t_release, 0.0, 1e2)

        # mute the t_wait
        if np.random.rand() < mutation_rate:
            new_t_wait = plan.t_wait + np.random.uniform(-0.2, 0.2, 3)
            plan.t_wait = np.clip(new_t_wait, 0.0, 1e2)

    def crossover(self, pair_genome: "Genome3") -> Tuple["Genome3", "Genome3"]:
        # get the copy
        copy1, copy2 = self.get_copy(), pair_genome.get_copy()
        plan1, plan2 = copy1.drone_plans[0], copy2.drone_plans[0]
        magnitude = 0.5

        # mutate the velocity
        par_vel1 = np.array([plan1.vel_dir, plan1.vel_val])
        par_vel2 = np.array([plan2.vel_dir, plan2.vel_val])

        child_vel1 = np.clip(
            magnitude * par_vel1 + (1 - magnitude) * par_vel2,
            70.0,
            140.0
        )
        child_vel2 = np.clip(
            magnitude * par_vel2 + (1 - magnitude) * par_vel1,
            70.0,
            140.0
        )

        plan1.vel_dir, plan2.vel_dir = child_vel1[0], child_vel2[0]
        plan1.vel_val, plan2.vel_val = child_vel1[1], child_vel2[1]

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

            for t in np.linspace(0.0, 20.0, 2001):
                smoke_posi_now = plan.get_smoke_posi(t + t_exp)[idx]
                if smoke_posi_now is None:
                    continue

                missile_now = missile_exp + t * v_missile

                if self.whether_blocked(missile_now, smoke_posi_now):
                    time_list.append(t + t_exp)

        return self.__class__.calculate_total_time(time_list, 1e-2)


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

        # add a valid individual 4.70s
        plan = SingleDronePlan(
            init_posi = np.array([17800.0, 0.0, 1800.0]),
            vel_dir = np.pi - 0.0058484,
            vel_val = 119.04132869183755,
            t_release = np.array([
                0.27225166041894844,
                1.4141872352269902 - 0.27225166041894844,
                7.8196489621119225 - 1.4141872352269902
            ]),
            t_wait = np.array([
                0.230217011218242,
                3.941950924383737,
                3.9580433878443215
            ])
        )
        genome = Genome3([plan])
        pop.append(Individual(genome))
        # add a valid individual 5.80s
        plan = SingleDronePlan(
            init_posi = np.array([17800.0, 0.0, 1800.0]),
            vel_dir = np.pi - 0.0094405844,
            vel_val = 79.67570687978107,
            t_release = np.array([
                0.13498821717161102,
                1.3646443474201302 - 0.13498821717161102,
                2.8358090092435333 - 1.3646443474201302
            ]),
            t_wait = np.array([
                1.8569791241597366, 3.3046120832452317, 3.0118646297956873
            ])
        )
        genome = Genome3([plan])
        pop.append(Individual(genome))
        # add a valid individual 6.60s
        plan = SingleDronePlan(
            init_posi = np.array([17800.0, 0.0, 1800.0]),
            vel_dir = np.pi - 0.00238633496,
            vel_val = 89.99678020054995,
            t_release = np.array([
                0.00058331434179154,
                1.0924237145022797 - 0.00058331434179154,
                2.9280803241571847 - 1.0924237145022797,
            ]),
            t_wait = np.array([
                0.12601158987295058, 2.749657885351035, 3.9112060643897655
            ])
        )
        genome = Genome3([plan])
        pop.append(Individual(genome))

        for i in range(pop_size - 3):
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
        pop_size = 50,
        evaluator = Evaluator3(),
        crossover_rate = 0.5,
        mutation_rate = 0.5,
        num_gen = 50
    )
    GA.evolve()
