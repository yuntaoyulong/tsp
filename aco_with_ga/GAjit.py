import numpy as np
from numba import njit
from delete import delete_workaround
dist_matrix1_path = "distance_matrix.csv"
dist_matrix1 = np.genfromtxt(dist_matrix1_path, delimiter=',', dtype=int)
cities = len(dist_matrix1)
from distance import calculateDistance
from all3 import adaptedLargeLocalSearch
@njit(cache=True)
def initpop(population):
    fit = np.array([calculateDistance(chrom, dist_matrix1) for chrom in population])  # 适应度初始化,此处复杂度为pop_size
    NFE = len(population)  # 更新函数评价次数
    best = population[np.argmin(fit)]  # 最优个体初始化
    return fit, NFE, best

@njit(cache=True)
def constraints(x):
        return 0

@njit(cache=True)
def mut(population, best, labels, reachable, mut_way, dim):
    mut_population = np.empty((len(population), dim), dtype=np.int64)
    temp = calculateDistance(best, dist_matrix1)
    
    if temp > 10000000:
        prob = 0.5
    elif temp > 7000000:
        prob = 0.6
    elif temp > 5000000:
        prob = 0.7
    elif temp > 3700000:
        prob = 0.8
    elif temp > 2600000:
        prob = 0.8
    else:
        prob = 0.9
    for i in range(len(population)):
        v = np.copy(population[i])
        mut_choice = np.random.rand()
        if mut_choice < mut_way:
            idx1 = np.random.randint(0, dim)
            if np.random.rand() < prob:
                reachable_points = v[np.where(labels == labels[idx1])]  
            else:
                randomNodeSeednum = np.int64(np.random.randint(0,len(reachable[labels[idx1]])))
                reachable_points = v[np.where(labels == randomNodeSeednum)]
            randomNodeSeednum = np.int64(np.random.randint(0,len(reachable_points)))
            idx2 = np.int64(reachable_points[randomNodeSeednum])
            idx2 = np.where(v == idx2)[0][0]
            mut_choice = np.random.rand()
            if mut_choice < 0.3:
                # move
                del1 = delete_workaround(v, v[idx1])
                if idx1 < idx2:
                    v = np.concatenate((del1[:idx2], np.array([v[idx1]]), del1[idx2:]))
                else:
                    if idx2 + 2 >= len(v):
                        v = np.concatenate((del1[:idx2 - 150], np.array([v[idx1]]), del1[idx2-150:]))
                    else:
                        v = np.concatenate((del1[:idx2 + 1], np.array([v[idx1]]), del1[idx2 + 1:]))
                        
            elif mut_choice < 0.7:
                # reverse
                if idx1 < idx2:
                    v[idx1:idx2 + 1] = v[idx1:idx2 + 1][::-1]
                else:
                    v[idx2:idx1 + 1] = v[idx2:idx1 + 1][::-1]
            else:
                # swap
                v[idx1], v[idx2] = v[idx2], v[idx1]
        mut_population[i] = v  # 将 v 赋值给 mut_population 的一行
    return mut_population
@njit(cache=True)
def cross(population, mut_population, CR):
    

    cross_population = population.copy()
    
    for i in range(0, len(population) - len(population) % 2, 2):
        if CR >= np.random.rand():
            cross_population[i], cross_population[i + 1] = _intercross(mut_population[i], mut_population[i + 1])
    return cross_population
@njit(cache=True)
def _intercross(a, b):
    L = len(a)
    r1, r2 = np.random.randint(0, L, 2)
    if r1 != r2:
        a0, b0 = a.copy(), b.copy()
        s, e = min(r1, r2), max(r1, r2)
        for i in range(s, e + 1):
            a1, b1 = a.copy(), b.copy()
            a[i], b[i] = b0[i], a0[i]
            x, y = np.where(a == a[i])[0], np.where(b == b[i])[0]
            i1, i2 = x[x != i], y[y != i]
            if i1.size > 0:
                a[i1[0]] = a1[i]
            if i2.size > 0:
                b[i2[0]] = b1[i]
            if len(a) < 151 or len(b) < 151:
                raise ValueError("Start or end point not in path")
    
    return a, b
@njit(cache=True)
def select(population, fit, pop_size, tournament_size=3,scorelen=24,allstime=20):
    new_population = []
    new_fit = []
    # if np.min(fit) < 2420000:
    #     scorelen = 18
    #     allstime = 20
    # else:
    #     scorelen = 24
    #     allstime = 10
    for _ in range(pop_size):
        tournament = np.random.choice(pop_size, tournament_size, replace=False)
        best_idx = tournament[0]
        for idx in tournament:
            cross_population1 = adaptedLargeLocalSearch(population[idx], allstime, 0.2, scorelen)[0]
            if calculateDistance(cross_population1, dist_matrix1) < calculateDistance(population[idx], dist_matrix1):
                population[idx] = cross_population1
            if calculateDistance(population[idx], dist_matrix1) < calculateDistance(population[best_idx], dist_matrix1):
                best_idx = idx
        new_population.append(population[best_idx])
        new_fit.append(calculateDistance(population[best_idx], dist_matrix1))
    return new_population, np.array(new_fit)
@njit(cache=True)
def get_top_n_individuals(population, fit, n):
    sorted_indices = np.argsort(fit)
    top_n_individuals = np.zeros((n, len(population[0])), dtype=np.int64)
    # print(type(sorted_indices),type(sorted_indices[:n]),population[1])
    for p in range(n):
        top_n_individuals[p] = population[sorted_indices[p]]
    # top_n_individuals = population[sorted_indices[:n]]
    # print(type(top_n_individuals),top_n_individuals)
    return top_n_individuals


