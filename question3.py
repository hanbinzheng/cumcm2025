import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import random
import copy
from scipy.optimize import minimize_scalar
import time
import math

class NSGAII_drone_plan(object):
    
    def __init__(self, max_iter):
        self.maxiter = max_iter

    def decide_if_cover(self, missile: np.ndarray, smoke: np.ndarray) -> bool:
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

    def calculate_one_cover_time(self, plan, p):
        missile_init_posi = np.array([20000.0, 0.0, 2000.0])
        unit_v_missile = - missile_init_posi / np.linalg.norm(missile_init_posi)
        v_missile = 300.0 * unit_v_missile

        smoke_init_posi = np.array([17800.0, 0.0, 1800.0])
        v_smoke = 3.0 * np.array([0.0, 0.0, -1.0])
        
        # 修复这里：确保plan[0]是长度为3的列表，plan[1]是包含一个值的列表
        v_drone = plan[1][0] * np.array(plan[0])
        g = 9.81

        # the time when the drone release the smoke bomb
        missile = missile_init_posi + v_missile * plan[2][p]
        smoke = smoke_init_posi + v_drone * plan[2][p]

        # the time when the smoke bomb bombed
        missile = missile + v_missile * plan[3][p]
        smoke = smoke + v_drone * plan[3][p]  # still fly with drone velocity
        smoke[2] -= 0.5 * 9.81 * plan[3][p] * plan[3][p]

        list = []

        for t in np.linspace(0, 20.0, 201):
            missile_t = missile + v_missile * t
            smoke_t = smoke + v_smoke * t
            if(self.decide_if_cover(missile_t, smoke_t)):
                list.append(t + 1.6 + 3.5)
        if len(list)>=2:
            return list[0], list[-1]
        else:
            return -1.0, -1.0
        
    def calculate_plan_cover_time(self, plan):
        list=[]
        for i in range(3):
            a,b=self.calculate_one_cover_time(plan, i)
            if a!=-1.0:
                list.append([a,b])
        
        if len(list)!=0:
            intervals=list    
            # 按区间起点排序
            intervals.sort(key=lambda x: x[0])
            # 合并重叠区间
            merged = []
            current_start, current_end = intervals[0]
            for i in range(1, len(intervals)):
                if intervals[i][0] <= current_end:
                    # 区间重叠，合并
                    current_end = max(current_end, intervals[i][1])
                else:
                    # 不重叠，保存当前区间
                    merged.append([current_start, current_end])
                    current_start, current_end = intervals[i]
            merged.append([current_start, current_end])
            # 计算总长度
            total_length = 0
            for interval in merged:
                total_length += interval[1] - interval[0]
            return (total_length+0.0)
        else:
            return 0.0

    def Initialization(self, n):
        total_time=sqrt(20000*20000+0+2000*2000)/300.0
        pop = []
        max_time=0.0
        for i in range(n):
            while True:
                # 创建每一个染色体 - 确保数据结构一致
                plan=[[0.0,0.0,0.0], [0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]]
                
                # 航向 - 确保生成长度为3的向量
                random_vec = np.array([1.0+random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01), 0.0])
                norm = np.linalg.norm(random_vec)
                unit_vector = random_vec / norm
                plan[0][0] = -abs(unit_vector[0])
                plan[0][1] = abs(unit_vector[1])
                plan[0][2] = unit_vector[2]  # 确保z分量也有值
                
                # 速度
                plan[1][0]=random.uniform(70, 140)
                
                # 投放的时间
                while True:
                    random_time_array=[random.uniform(0,10.0) for _ in range(3)]
                    random_time_array.sort()  # 直接排序
                    if (random_time_array[1]-random_time_array[0]>1.0) and (random_time_array[2]-random_time_array[1]>1.0):
                        plan[2]=random_time_array
                        break

                # 投出到起爆的时间间隔
                for bomb_num in range(3):  # 改为0-2的索引
                    # 修复计算逻辑
                    limit_t = sqrt(abs(1800.0 + plan[0][2] * plan[1][0] * plan[2][bomb_num])) / (0.5 * 9.81)
                    plan[3][bomb_num] = random.uniform(0.0, min(4.0, limit_t))
                                                       
                # 加入种群
                a = self.calculate_plan_cover_time(plan)

                if a > 0.0:
                    max_time = max(max_time, a)
                    pop.append(plan)
                    break
                
        print(f"Max time in initialization: {max_time}")
        return pop

    def Crossover(self, pop):
        newpop = []
        # Select 20 parents for crossover
        parents_index = random.sample(range(len(pop)), 20)
        father_index = parents_index[0:10]
        mother_index = parents_index[10:20]
        
        for i in range(10):
            new_plan = [[0.0,0.0,0.0], [0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]]
            father_plan = pop[father_index[i]]
            mother_plan = pop[mother_index[i]]

            # 航向 - 深度复制
            if random.randint(0,1) == 0:
                new_plan[0] = father_plan[0].copy()  # 使用copy()确保深度复制
            else:
                new_plan[0] = mother_plan[0].copy()

            # 速度 - 使用随机插值
            new_plan[1][0] = random.uniform(
                min(father_plan[1][0], mother_plan[1][0]), 
                max(father_plan[1][0], mother_plan[1][0])
            )

            # 投放时间 - 修复逻辑
            if random.randint(0,1) == 0:
                base_times = father_plan[2]
            else:
                base_times = mother_plan[2]
                
            while True:
                random_time_array = [abs(t + random.uniform(-0.5, 0.5)) for t in base_times]
                random_time_array.sort()
                if (random_time_array[1]-random_time_array[0]>1.0) and (random_time_array[2]-random_time_array[1]>1.0):
                    new_plan[2] = random_time_array
                    break

            # 投出到爆炸的时间间隔
            for bomb_num in range(3):
                new_plan[3][bomb_num] = random.uniform(0.0, 4.0)
                
            newpop.append(new_plan)
            
        return newpop

    def Mutate(self, pop):
        newpop = []
        # 选择20个进行变异操作
        mutate_index = random.sample(range(len(pop)), min(20, len(pop)))
        
        for i in range(len(mutate_index)):
            base_plan = copy.deepcopy(pop[mutate_index[i]])  # 深度复制
            new_plan = [[0.0,0.0,0.0], [0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]]
        
            # 航向变异
            base_vector = np.array(base_plan[0])
            max_angle_deg = 5
            max_angle_rad = np.radians(max_angle_deg)
            
            # 生成随机扰动
            perturbation = np.random.uniform(-max_angle_rad, max_angle_rad, 3)
            new_vector = base_vector + perturbation
            new_vector /= np.linalg.norm(new_vector)
            new_plan[0] = new_vector.tolist()
            
            # 速度变异
            speed_change = random.uniform(-10, 10)
            new_plan[1][0] = max(70, min(140, base_plan[1][0] + speed_change))

            # 投放时间变异
            if random.randint(0,1) == 0:
                # 小变异
                while True:
                    random_time_array = [t + random.uniform(-0.5, 0.5) for t in base_plan[2]]
                    random_time_array.sort()
                    if (random_time_array[1]-random_time_array[0]>1.0) and (random_time_array[2]-random_time_array[1]>1.0):
                        new_plan[2] = random_time_array
                        break
            else:
                # 大变异
                while True:
                    random_time_array = [random.uniform(0, 4.0) for _ in range(3)]
                    random_time_array.sort()
                    if (random_time_array[1]-random_time_array[0]>1.0) and (random_time_array[2]-random_time_array[1]>1.0):
                        new_plan[2] = random_time_array
                        break

            # 投出到爆炸的间隔时间
            for bomb_num in range(3):
                new_plan[3][bomb_num] = random.uniform(0.0, 6.0)
                
            newpop.append(new_plan)
            
        return newpop

    def Combine(self, pop, newpop1, newpop2):
        next_pop = []
        all_pop = pop + newpop1 + newpop2
        
        # 计算所有个体的适应度
        all_pop_time = []
        valid_plans = []
        
        for plan in all_pop:
            try:
                cover_time = self.calculate_plan_cover_time(plan)
                all_pop_time.append(cover_time)
                valid_plans.append(plan)
            except Exception as e:
                print(f"Error calculating cover time: {e}")
                continue
        
        if not valid_plans:
            return pop, 0.0
            
        # 排序并选择前100个
        sorted_indices = np.argsort(all_pop_time)[::-1]  # 从大到小排序
        for i in range(min(100, len(valid_plans))):
            next_pop.append(valid_plans[sorted_indices[i]])
        
        best_time = all_pop_time[sorted_indices[0]] if all_pop_time else 0.0
        return next_pop, best_time

    def do(self):
        pop = self.Initialization(100)
        for current_iter in range(self.maxiter):
            print(f"Generation {current_iter + 1}/{self.maxiter}")
            
            pop1 = self.Crossover(pop)
            pop2 = self.Mutate(pop)
            pop, best_time = self.Combine(pop, pop1, pop2)
            
            print(f"Best cover time: {best_time:.2f}s")
            
        return pop[0] if pop else None
        
if __name__ == "__main__":
    a = NSGAII_drone_plan(1000)
    best_plan = a.do()
    
