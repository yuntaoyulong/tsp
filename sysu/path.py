import random
import numpy as np
from numba import njit
@njit(cache =True)
def generate_path(labels, intersections):
    k = len(set(labels))
    current_class = random.randint(0, k - 1)
    # print(labels,intersections  )
    visited_classes = np.empty(0, dtype=np.int64)
    path = []
    special = 0 
    
    while len(visited_classes) < k:
        visited_classes=np.append(visited_classes,current_class)
        class_elements = np.where(labels == current_class)[0]
        # print(current_class)
        np.random.shuffle(class_elements)
        path.extend(class_elements)
        reachable_classes = np.array([x for x in intersections[current_class] if x not in visited_classes])
        # print('abc')
        # print(reachable_classes)
        # print('cba')
        # labels = labels.tolist()
        if reachable_classes.size > 0:
            if current_class == labels[-1]:
                special = 1
            if labels[-1] in intersections[current_class] and current_class != labels[-1] and special == 0:
                special = 1
                current_class = labels[-1]
            else:
                randomNodeSeednum1 = np.int64(np.random.randint(0,len(reachable_classes)))
                current_class = np.int64(reachable_classes[randomNodeSeednum1])
        else:
            remaining_classes = np.array([x for x in range(k) if x not in visited_classes])
            if remaining_classes.size > 0:
                randomNodeSeednum1 = np.int64(np.random.randint(0,len(remaining_classes)))
                current_class = np.int64(remaining_classes[randomNodeSeednum1])
            else:
                break
    return np.array(path)
