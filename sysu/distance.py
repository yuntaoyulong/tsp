
from numba import jit,njit
import numpy as np
dist_matrix1_path="distance_matrix.csv"
dist_matrix1 = np.genfromtxt(dist_matrix1_path, delimiter=',', dtype=int)
@jit(nopython=True)
def save_list_to_txt(my_list, filename):
    with open(filename, 'a+') as file:
        for item in my_list:
            file.write(f"{item}\n")

@njit(cache=True)
def calculateDistance(path,dis_mtx,circle=True):
    distance = 0
    for i in range(0, len(path) - 1):
        distance += dis_mtx[path[i], path[i + 1]]
    if circle:
        distance += dis_mtx[path[-1], path[0]]
    # if distance <= 2400000:
    #     for i in range(10):
    #         print("好极了",path,distance,"feichanghao")
    return distance


