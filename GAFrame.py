import copy
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from question1 import is_blocked


# the constant used
gravity = 9.81
v_smoke = np.array([0.0, 0.0, -3.0])

# it's human duty that we guarantee
# SingleDronePlan, Genome, Evaluator matches

# class to denote the plan for one drone
class SingleDronePlan:
    def __init__(
            self,
            init_posi: np.ndarray,
            vel_dir: float,  # direction in angles
            vel_val: float,  # velocity value
            t_release: List[float],
            t_wait: List[float]
    ):
        # here, t_wait means the wait time after release
        # t_wait[i] is the time after the i^th release
        # the time of the i^th release = sum_{j <= i} t_release[j]
        self.init_posi = init_posi
        self.vel_dir = vel_dir
        self.vel_val = vel_val
        self.t_release = t_release
        self.t_wait = t_wait

    def get_smoke_posi(self, t: float) -> list[Optional[np.ndarray]]:
        """
        method to return the center of the smoke
        Args:
        - t: the current time
        Return:
        - smoke_posi: the list of the position of smoke.
          len(smoke_posi) = len(t_wait)
        """
        # the time of the release
        t_release = 0.0
        # initialize the smoke position
        smoke_posi = [None] * len(self.t_wait)

        vel_drone = self.vel_val * np.array([
            np.cos(self.vel_dir),
            np.sin(self.vel_dir),
            0.0
        ])  # velocity of drone

        for i in range(len(self.t_wait)):  # for each smoke bomb
            # calculate the different bomb
            t_release += self.t_release[i]  # time at when release
            t_drop = self.t_wait[i]  # time when drop with gravity
            t_smoke = t - t_release - t_drop  # time become somke

            # check whether the smoke is valid
            if (t_smoke > 20 or t_smoke < 0):
                # the smoke bomb fails, set smoke_posi to be None
                smoke_posi[i] = None
            else:
                # add the horizontal displacement
                smoke_posi_t = self.init_posi + vel_drone * (t_release + t_drop)
                # add the vertical displacement
                smoke_posi_t[2] -= 0.5 * gravity * t_drop ** 2
                smoke_posi_t += t_smoke * v_smoke
                smoke_posi[i] = smoke_posi_t
        return smoke_posi


# class to denote the genome of an individual
class Genome(ABC):
    def __init__(self,
                 drone_plans: List[SingleDronePlan]):
        self.drone_plans = drone_plans

    def get_copy(self) -> "Genome":
        return self.__class__(copy.deepcopy(self.drone_plans))

    @abstractmethod
    def get_info(self) -> str:
        pass

    @abstractmethod
    def mutate(self, mutation_rate: float):
        pass

    @abstractmethod
    def crossover(self, pair_genome: "Genome") -> Tuple["Genome", "Genome"]:
        # use get_copy() here if needed
        pass


# class to do the evlauation for a certain type Genome
class Evaluator(ABC):
    def __init__(self, missile_init_posi: List[np.ndarray]):
        self.missile_init_posi = missile_init_posi
        self.v_missile: List[np.ndarray] = [
            -300 * v / np.linalg.norm(v) for v in missile_init_posi
        ]

    @staticmethod
    def whether_blocked(
            missile_posi: np.ndarray, smoke_posi: np.ndarray
    ) -> bool:
        """
        static method to determine whether a single smoke at smoke_posi
        successfully blocked a single missile at missile_posi from the target
        Args:
        - missile_posi: the position of the missile
        - smoke_posi: the position of the smoke
        """
        return is_blocked(missile_posi, smoke_posi)

    @abstractmethod
    def evaluate(self, genome: Genome) -> float:
        pass


# class of attributes for an individual
class Individual:
    def __init__(
            self,
            genome: Genome,
    ):
        self.genome = genome
        self.evaluation = None  # to be set
        self.fitness = None

    def clear_evaluation_and_fitness(self):
        self.evaluation = None
        self.fitness = None

    def get_info(self) -> str:
        genome_info = self.genome.get_info()
        return f"Evaluation: {self.evaluation}\n" + genome_info

    def mutate(self, mutation_rate: float):
        # mutate and reset the evaluation and fitness
        self.genome.mutate(mutation_rate)
        self.clear_evaluation_and_fitness()

    def crossover(self, spouse: "Individual") -> List["Individual"]:
        # get new genome, and Genome.crossover() here guarantees the new genome
        new_genome1, new_genome2 = self.genome.crossover(spouse.genome)
        # the constructor guarantee that evaluation and fitness are None
        child1 = self.__class__(new_genome1)
        child2 = self.__class__(new_genome2)
        return [child1, child2]


# the ultimate genetic algorithm
class GeneticAlgorithm(ABC):
    def __init__(
            self,
            pop_size: int,
            evaluator: Evaluator,
            crossover_rate: float,
            mutation_rate: float,
            num_gen: int
    ):
        self.pop_size = pop_size
        self.evaluator = evaluator
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_gen = num_gen
        self.pop: List[Individual] = self.initialize_pop(self.pop_size)
        self.best_indv = None

    @abstractmethod
    def initialize_pop(self, pop_size: int) -> List[Individual]:
        # initalize a population of pop_size individuals
        pass

    @staticmethod
    def fitness_function(evaluation: float) -> float:
        # by default, return the evaluation itself
        return evaluation

    def evaluate_pop(self,
                     pop: List[Individual]) -> Tuple[List[float], List[float]]:
        evaluation_list = []
        fitness_list = []
        for indv in pop:
            # add evaluation and fitness if needed
            if indv.evaluation is None:
                indv.evaluation = self.evaluator.evaluate(indv.genome)
                indv.fitness = self.fitness_function(indv.evaluation)

            evaluation_list.append(indv.evaluation)
            fitness_list.append(indv.fitness)
        return evaluation_list, fitness_list

    def crossover(self, pop: List[Individual],
                  pop_size: Optional[int] = None) -> List[Individual]:
        # implement the cross over here
        if pop_size is None:
            pop_size = len(pop)

        indices = np.random.choice(pop_size, size=pop_size, replace=False)
        shuffled_pop = [pop[i] for i in indices]
        child_pop: List[Individual] = []

        for i in range(0, pop_size, 2):
            parent1 = shuffled_pop[i]
            if i + 1 < len(shuffled_pop):
                parent2 = shuffled_pop[i + 1]
                if np.random.rand() < self.crossover_rate:
                    children = parent1.crossover(parent2)
                    child_pop.extend(children)
            else:
                child_pop.append(
                    parent1.__class__(parent1.genome.get_copy())
                )

        if len(child_pop) > pop_size:
            child_pop = child_pop[:pop_size]
        return child_pop

    def mutate(self, pop: List[Individual]) -> List[Individual]:
        for indv in pop:
            indv.mutate(mutation_rate=self.mutation_rate)
        return pop

    def select(self, num: int, pop: List[Individual],
               fitness_list: Optional[List[float]] = None) -> List[Individual]:
        """
        elitism + roulette-wheel selection: keep the best individual,
        and select the rest using roulette-wheel
        Args:
        - num: select num from pop
        - fitness_list: the fitness of the pop
        """
        if fitness_list is None:
            _, fitness_list = self.evaluate_pop(pop)

        fitness_array = np.array(fitness_list)

        elite = pop[np.argmax(fitness_array)]

        if np.all(fitness_array == 0):
            rest = list(np.random.choice(pop, size=num-1, replace=True))
        else:
            probs = fitness_array / fitness_array.sum()
            rest = list(np.random.choice(pop, size=num-1, replace=True, p=probs))

        return [elite] + rest

    def update_best(self, pop: List[Individual],
                    evaluation_list: Optional[List[float]] = None) -> Individual:
        # return the best individual among the list.
        if evaluation_list is None:
            evaluation_list, _ = self.evaluate_pop(pop)

        the_best = pop[np.argmax(np.array(evaluation_list))]

        if (self.best_indv is None) or (the_best.evaluation > self.best_indv.evaluation):
            self.best_indv = copy.deepcopy(the_best)
        return self.best_indv

    def evolve(self):
        # implement the selection and evoluation here here
        self.pop = self.initialize_pop(self.pop_size)

        # start the loop here
        for generation in range(self.num_gen):
            # cross over
            child_pop = self.crossover(self.pop, pop_size=self.pop_size)
            # mutate
            mutated_pop = self.mutate(self.pop)
            # the new population
            new_pop = child_pop + mutated_pop

            # set the best indv
            eval_list, fit_list = self.evaluate_pop(new_pop)
            self.update_best(new_pop, evaluation_list=eval_list)

            # select
            self.pop = self.select(self.pop_size,
                                   new_pop, fitness_list=fit_list)
            info = (f"At generation {generation}, " +
                    f"the best: {self.best_indv.get_info()}\n")
            print(info)
