from path import rearrange_path
from numba import njit
import numpy as np
dist_matrix1_path="distance_matrix.csv"
dist_matrix1 = np.genfromtxt(dist_matrix1_path, delimiter=',', dtype=int)
@njit(cache=True)
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

@njit(cache=True)
def distance_calculation(i_index, j_index, dist_matrix1):

    return dist_matrix1[i_index][j_index]
def ceshi():
    path_list = np.array([134, 131, 130, 129, 133, 132, 128, 127, 126, 99, 135, 136, 137, 145, 98, 86, 146, 138,
                85, 84, 119, 105, 120, 147, 83, 82, 117, 118, 101, 100, 116, 150, 115, 113, 141, 92,
                91, 90, 143, 142, 144, 89, 87, 88, 112, 24, 51, 16, 40, 50, 48, 15, 19, 41,
                23, 20, 21, 13, 22, 10, 17, 52, 7, 25, 9, 8, 14, 47, 11, 43, 45, 12,
                44, 54, 124, 94, 114, 96, 97, 95, 93, 55, 57, 56, 58, 59, 60, 53, 46, 6,
                27, 3, 2, 5, 4, 28, 29, 30, 32, 31, 33, 38, 1, 39, 37, 42, 49, 26,
                123, 62, 18, 0, 34, 35, 36, 61, 140, 149, 64, 63, 76, 75, 74, 121, 111, 122,
                110, 109, 108, 73, 65, 66, 67, 68, 69, 148, 77, 107, 106, 72, 70, 71, 78, 79,
                81, 80, 139, 102, 104, 103, 125])

    a=calculateDistance(path_list,dist_matrix1,circle=True)
    b=calculateDistance(rearrange_path(path_list, 115, 116),dist_matrix1,circle=False)
    print(a,b)

if __name__ == '__main__':
    ceshi()
