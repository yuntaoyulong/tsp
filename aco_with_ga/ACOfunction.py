from numba import njit
import numpy as np
@njit(cache=True)
def calculate_probability(tau_m, eta_m, taboo_l, current_index, city_number, alpha, beta):
    numerator_lst = np.zeros(city_number)
    for i in range(city_number):
        if taboo_l[i] == 1:
            numerator_lst[i] = (tau_m[current_index, i] ** alpha) * (eta_m[current_index, i] ** beta) + 0.1 ** 50
    total = np.sum(numerator_lst)
    prob_lst = numerator_lst / total if total > 0 else numerator_lst
    return prob_lst

@njit(cache=True)
def random_selection(prob_lst):
    cumulative_prob = np.cumsum(prob_lst)
    r = np.random.rand()
    for i in range(len(cumulative_prob)):
        if r < cumulative_prob[i]:
            return i
    return len(cumulative_prob) - 1

@njit(cache=True)
def ant_colony_optimization(start_index, tau_m, city_number, eta_mtx, alpha, beta):
    current_index = start_index
    path_index = np.empty(city_number, dtype=np.int32)
    path_index[0] = current_index
    taboo_lst = np.ones(city_number, dtype=np.int32)
    taboo_lst[start_index] = 0
    step = 1
    while np.sum(taboo_lst) != 0:
        prob = calculate_probability(tau_m, eta_mtx, taboo_lst, current_index, city_number, alpha, beta)
        next_index = random_selection(prob)
        taboo_lst[next_index] = 0
        path_index[step] = next_index
        current_index = next_index
        step += 1
    return path_index[:step]