import numpy as np

'''
a = np.zeros((4, 2))
print(a)
print(np.float32(a))
print(np.float32([(0, 0), (0, 0), (0, 0), (0, 0)]))
'''

x_list = [3, 4, 0, 1]
y_list = [1, 4, 3, 2]
points = [(x, y) for x, y in zip(x_list, y_list)]
points.sort(key=lambda x: x[0])
print(points)
points_l = points[:2]
points_r = points[-2:]
print(points_r)
print(points_l)