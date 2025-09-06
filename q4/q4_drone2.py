import numpy as np
from typing import List, Tuple
from GAFrame import SingleDronePlan, Evaluator
from GAFrame import Genome, Individual, GeneticAlgorithm


class Genome2(Genome):
    def __init__(self,
                 drone_plans: List[SingleDronePlan]):
        super().__init__(drone_plans)

    def mutate(self, mutation_rate: float):
        # get the plan
        plan = self.drone_plans[0]

        # mute in the vel_dir
        if np.random.rand() < mutation_rate:
            plan.vel_dir += np.random.uniform(-np.pi/18, np.pi/18)
            
        # mute the vel_val
        if np.random.rand() < mutation_rate:
            new_vel_val = plan.vel_val + np.random.uniform(-5, 5)
            plan.vel_val = np.clip(new_vel_val, 70.0, 140.0)  # check boundary

        # mute t_release
        if np.random.rand() < mutation_rate:
            new_t_release = plan.t_release[0] + np.random.uniform(-1.0, 1.0)
            plan.t_release = [max(0, new_t_release)]

        # mute t_wait
        if np.random.rand() < mutation_rate:
            new_t_wait = plan.t_wait[0] + np.random.uniform(-1.0, 1.0)
            plan.t_wait = [max(0, new_t_wait)]

    def crossover(self, pair_genome: "Genome2") -> Tuple["Genome2", "Genome2"]:
        # do the crossover here
        copy1, copy2 = self.get_copy(), pair_genome.get_copy()
        plan1 = copy1.drone_plans[0]
        plan2 = copy2.drone_plans[0]

        # get the copy
        parent1 = np.array([
            plan1.vel_dir,
            plan1.vel_val,
            plan1.t_release[0],
            plan1.t_wait[0]
        ])
        parent2 = np.array([
            plan2.vel_dir,
            plan2.vel_val,
            plan2.t_release[0],
            plan2.t_wait[0]
        ])

        magnitude = 0.5

        child1 = magnitude * parent1 + (1 - magnitude) * parent2
        child2 = magnitude * parent2 + (1 - magnitude) * parent1

        plan1.vel_dir, plan2.vel_dir = child1[0], child2[0]
        plan1.vel_val, plan2.vel_val = child1[1], child2[1]
        plan1.t_release, plan2.t_release = np.array([child1[2]]), np.array([child2[2]])
        plan1.t_wait, plan2.t_wait = np.array([child1[3]]), np.array([child2[3]])

        return copy1, copy2

class Evaluator2(Evaluator):
    def __init__(self):
        missile_posi = np.array([20000.0, 0.0, 2000.0])
        super().__init__([missile_posi])

    def evaluate(self, genome: Genome2) -> float:
        plan = genome.drone_plans[0]
        t_explode = plan.t_release[0] + plan.t_wait[0]
        missile_posi = (
            self.missile_init_posi[0] + self.v_missile[0] * t_explode
        )

        time_list = []
        for t in np.linspace(0.0, 20.0, 401):
            smoke_posi_now = plan.get_smoke_posi(t + t_explode)[0]
            if smoke_posi_now is None:
                continue

            missile_posi_now = missile_posi + t * self.v_missile[0]
            if self.whether_blocked_simple(missile_posi_now, smoke_posi_now):
                time_list.append(t)

        if len(time_list) < 2:
            return 0.0
        else:
            return time_list[-1] - time_list[1]


class GeneticAlgorithm2(GeneticAlgorithm):
    def __init__(
            self,
            pop_size: int,
            evaluator: Evaluator2,
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

        # add a valid individual 2.85 s
        plan = SingleDronePlan(
            init_posi = np.array([12000, 1400, 1400]),
            vel_dir = 4.56499530436835,
            vel_val = 112.10169526591747,
            t_release = np.array([5.65915341]),
            t_wait = np.array([6.583785])
        )
        genome = Genome2([plan])
        pop.append(Individual(genome))

        # add a valid individual 3.2 s
        plan = SingleDronePlan(
            init_posi = np.array([12000, 1400, 1400]),
            vel_dir = 4.116188645333355,
            vel_val = 117.39817970222879,
            t_release = np.array([7.065268864301219]),
            t_wait = np.array([7.23193016])
        )
        genome = Genome2([plan])
        pop.append(Individual(genome))

        # add a valid individual 3.35 s
        plan = SingleDronePlan(
            init_posi = np.array([12000, 1400, 1400]),
            vel_dir = 4.150482423701107,
            vel_val = 111.23295808713637,
            t_release = np.array([6.861501531305942]),
            t_wait = np.array([7.51343526104483])
        )
        genome = Genome2([plan])
        pop.append(Individual(genome))

        # random initialize the rest one
        for i in range(pop_size - 3):
            plan = SingleDronePlan(
                init_posi = np.array([12000, 1400, 1400]),
                vel_dir = np.random.uniform(np.pi, 1.5 * np.pi),
                vel_val = np.random.uniform(90.0, 140.0),
                t_release = np.array([np.random.uniform(4.0, 8.0)]),
                t_wait = np.array([np.random.uniform(4.0, 8.0)])
            )
            genome = Genome2([plan])
            pop.append(Individual(genome))

        return pop


if __name__ == '__main__':
    GA = GeneticAlgorithm2(
        pop_size = 50,
        evaluator = Evaluator2(),
        crossover_rate = 0.75,
        mutation_rate = 0.4,
        num_gen = 500
    )
    GA.evolve()
