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
        #"""


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


    def calculate_plan_cover_time(self, plan):
        total_time=sqrt(20000^2+0+2000^2)/300.0
        time_space=total_time/20000.0
        cover_time=0.0
        for t in np.linspace(0, total_time, 20000):
            missile_t_posi=[20000-(t+0.0)/(total_time+0.0)*20000.0, 0.0, 2000-(t+0.0)/(total_time+0.0)*2000.0]
            if_cover=0
            for bomb_num in range(1,4):
                if if_cover!=1:
                    if t>(plan[2][bomb_num-1]+plan[3][bomb_num-1]) and t<(plan[2][bomb_num-1]+plan[3][bomb_num-1]+20):
                        bomb_t_posi=[17800.0+plan[0][0]*(plan[1][0]+0.0)*plan[2][bomb_num-1], 0.0+plan[0][1]*(plan[1][0]+0.0)*plan[2][bomb_num-1], 1800.0+plan[0][2]*(plan[1][0]+0.0)*plan[2][bomb_num-1]-0.5*9.81*plan[3][bomb_num-1]*plan[3][bomb_num-1]-3*(t-(plan[2][bomb_num-1]+plan[3][bomb_num-1]))]
                        if bomb_t_posi[2]>0.0:
                            if self.decide_if_cover(np.array(missile_t_posi), np.array(bomb_t_posi)):
                                if_cover=1
            if if_cover==1:
                cover_time+=time_space



    def Initialization(self, n):
        total_time=sqrt(20000^2+0+2000^2)/300.0
        pop = []
        for i in range(n):
            #创建每一个染色体
            plan=[[0.0,0.0,0.0],[0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
            #航向
            random_vec = np.random.randn(3)
            norm = np.linalg.norm(random_vec)
            unit_vector = random_vec / norm
            plan[0][0]=unit_vector[0]
            plan[0][1]=unit_vector[1]
            plan[0][2]=unit_vector[2]
            #速度
            plan[1][0]=random.uniform(70, 140)
            #投放的时间
            while True:
                random_time_array=[random.uniform(0.0, total_time), random.uniform(0.0, total_time), random.uniform(0.0, total_time)]
                if random_time_array[0] > random_time_array[1]:
                    random_time_array[0], random_time_array[1] = random_time_array[1], random_time_array[0]
                if random_time_array[0] > random_time_array[2]:
                    random_time_array[0], random_time_array[2] = random_time_array[2], random_time_array[0]
                if random_time_array[1] > random_time_array[2]:
                    random_time_array[1], random_time_array[2] = random_time_array[2], random_time_array[1]
                if (random_time_array[1]-random_time_array[0]>1.0) and (random_time_array[2]-random_time_array[1]>1.0):
                    plan[2][0]=random_time_array[0]
                    plan[2][1]=random_time_array[1]
                    plan[2][2]=random_time_array[2]
                    break
            #投出到起爆的时间间隔
            for bomb_num in range(1,4):
                limit_t=sqrt(abs(1800.0+plan[0][2]*(plan[1][0]+0.0)*plan[2][bomb_num-1])/(0.5*9.81))
                plan[3][bomb_num-1]=random.uniform(0.0, limit_t)
            #加入种群
            pop.append(plan)
            print(plan)
        return pop



    def Crossover(self, pop):
        newpop = []
        # Select 20 parents for crossover
        parents_index = random.sample(range(len(pop)), 20)
        father_index = parents_index[0:10]
        mother_index = parents_index[10:20]
        for i in range(10):
            new_plan=[[0.0,0.0,0.0],[0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
            father_plan=pop[father_index]
            mother_plan=pop[mother_index]

            #航向
            if random.randint(0,1)==0:
                base_vector=np.array(father_plan[0])
                max_angle_deg=5
                # 转换为弧度
                max_angle_rad = np.radians(max_angle_deg)
                # 生成随机扰动角度
                theta = 2 * np.pi * np.random.random()  # 方位角 [0, 2π]
                phi = np.random.random() * max_angle_rad  # 极角 [0, max_angle_rad]
                # 生成扰动向量（在切线空间）
                perturbation = np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi) - 1  # 使得扰动向量垂直于基向量
                ])
                # 将扰动向量旋转到基向量方向
                # 这里使用简单的加法近似（对于小角度有效）
                new_vector = base_vector + perturbation
                new_vector /= np.linalg.norm(new_vector)
                for j in range(3):
                    new_plan[0][j]=new_vector[j]+0.0
            else:
                base_vector=np.array(mother_plan[0])
                max_angle_deg=5
                # 转换为弧度
                max_angle_rad = np.radians(max_angle_deg)
                # 生成随机扰动角度
                theta = 2 * np.pi * np.random.random()  # 方位角 [0, 2π]
                phi = np.random.random() * max_angle_rad  # 极角 [0, max_angle_rad]
                # 生成扰动向量（在切线空间）
                perturbation = np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi) - 1  # 使得扰动向量垂直于基向量
                ])
                # 将扰动向量旋转到基向量方向
                # 这里使用简单的加法近似（对于小角度有效）
                new_vector = base_vector + perturbation
                new_vector /= np.linalg.norm(new_vector)
                for j in range(3):
                    new_plan[0][j]=new_vector[j]+0.0       

            #速度
            new_plan[1][0]=random.uniform(min(father_plan[1][0],mother_plan[1][0]), max(father_plan[1][0],mother_plan[1][0])+0.01)

            #投放时间
            random_time_array=[0.0, 0.0, 0.0]
            if random.randint(0,1)==0:
                while True:
                    for j in range(0,3):
                        random_time_array[j]=father_plan[2][j]+random.uniform(0.0, 0.2)
                    if random_time_array[0] > random_time_array[1]:
                        random_time_array[0], random_time_array[1] = random_time_array[1], random_time_array[0]
                    if random_time_array[0] > random_time_array[2]:
                        random_time_array[0], random_time_array[2] = random_time_array[2], random_time_array[0]
                    if random_time_array[1] > random_time_array[2]:
                        random_time_array[1], random_time_array[2] = random_time_array[2], random_time_array[1]
                    if (random_time_array[1]-random_time_array[0]>1.0) and (random_time_array[2]-random_time_array[1]>1.0):
                        new_plan[2][0]=random_time_array[0]
                        new_plan[2][1]=random_time_array[1]
                        new_plan[2][2]=random_time_array[2]                
                        break
            else:
                while True:
                    for j in range(0,3):
                        random_time_array[j]=mother_plan[2][j]+random.uniform(0.0, 0.2)
                    if random_time_array[0] > random_time_array[1]:
                        random_time_array[0], random_time_array[1] = random_time_array[1], random_time_array[0]
                    if random_time_array[0] > random_time_array[2]:
                        random_time_array[0], random_time_array[2] = random_time_array[2], random_time_array[0]
                    if random_time_array[1] > random_time_array[2]:
                        random_time_array[1], random_time_array[2] = random_time_array[2], random_time_array[1]
                    if (random_time_array[1]-random_time_array[0]>1.0) and (random_time_array[2]-random_time_array[1]>1.0):
                        new_plan[2][0]=random_time_array[0]
                        new_plan[2][1]=random_time_array[1]
                        new_plan[2][2]=random_time_array[2]                
                        break
                    
            #投出到爆炸的时间间隔
            for bomb_num in range(1,4):
                limit_t=sqrt(abs(1800.0+new_plan[0][2]*(new_plan[1][0]+0.0)*new_plan[2][bomb_num-1])/(0.5*9.81))
                new_plan[3][bomb_num-1]=random.uniform(0.0, limit_t)        

            newpop.append(new_plan)    
        return newpop
        


    def Mutate(self, pop):
        newpop=[]
        #选择20个进行变异操作
        mutate_index=random.sample(range(len(pop)), 20)
        for i in range(20):
            base_plan=pop[mutate_index]
            new_plan=[[0.0,0.0,0.0],[0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
        
            #航向
            base_vector=np.array(base_plan[0])
            max_angle_deg=5
            # 转换为弧度
            max_angle_rad = np.radians(max_angle_deg)
            # 生成随机扰动角度
            theta = 2 * np.pi * np.random.random()  # 方位角 [0, 2π]
            phi = np.random.random() * max_angle_rad  # 极角 [0, max_angle_rad]
            # 生成扰动向量（在切线空间）
            perturbation = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi) - 1  # 使得扰动向量垂直于基向量
            ])
            # 将扰动向量旋转到基向量方向
            # 这里使用简单的加法近似（对于小角度有效）
            new_vector = base_vector + perturbation
            new_vector /= np.linalg.norm(new_vector)
            for j in range(3):
                new_plan[0][j]=new_vector[j]+0.0
            
            #速度
            new_plan[1][0]=max(base_plan[1][0]+random.uniform(0, 0.2), 0.1)

            #投的时间
            random_time_array=[0.0, 0.0, 0.0]
            if random.randint(0,1)==0:
                while True:
                    for j in range(0,3):
                        random_time_array[j]=base_plan[2][j]+random.uniform(0.0, 0.2)
                    if random_time_array[0] > random_time_array[1]:
                        random_time_array[0], random_time_array[1] = random_time_array[1], random_time_array[0]
                    if random_time_array[0] > random_time_array[2]:
                        random_time_array[0], random_time_array[2] = random_time_array[2], random_time_array[0]
                    if random_time_array[1] > random_time_array[2]:
                        random_time_array[1], random_time_array[2] = random_time_array[2], random_time_array[1]
                    if (random_time_array[1]-random_time_array[0]>1.0) and (random_time_array[2]-random_time_array[1]>1.0):
                        new_plan[2][0]=random_time_array[0]
                        new_plan[2][1]=random_time_array[1]
                        new_plan[2][2]=random_time_array[2]                
                        break
            
            #投出到爆炸的间隔时间
            for bomb_num in range(1,4):
                limit_t=sqrt(abs(1800.0+new_plan[0][2]*(new_plan[1][0]+0.0)*new_plan[2][bomb_num-1])/(0.5*9.81))
                new_plan[3][bomb_num-1]=random.uniform(0.0, limit_t)    
            newpop.append(new_plan)
        return newpop

        
    def Combine(self, pop, newpop1, newpop2):
        next_pop=[]
        all_pop=pop+newpop1+newpop2
        all_pop_time=[0.0 for _ in range(len(all_pop))]
        for i in range(len(all_pop)):
            all_pop_time[i]=self.calculate_plan_cover_time(all_pop[i])
        arr = np.array(all_pop_time)
        rank_to_plan=np.argsort(arr)[::-1]
        for i in range(40):
            next_pop.append(all_pop[rank_to_plan[i]])
        return next_pop, self.calculate_plan_cover_time(next_pop[0])


    def do(self):
        pop=self.Initialization(40)
        for current_iter in range (self.maxiter):
            pop1=self.Crossover(pop)
            pop2=self.Mutate(pop)
            pop, b=self.Combine(pop,pop1,pop2)
            print(b)
        return 0
        
if __name__== "__main__":
    a=NSGAII_drone_plan(50)
    a.do()
