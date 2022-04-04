# imports section
import matplotlib.pyplot as plt
import random


class Point:
    def __init__(self, x_val, y_val):
        self.x_val = x_val
        self.y_val = y_val

    def print_vals(self):
        print('x value: ' + str(self.x_val))
        print('y value: ' + str(self.y_val))

points = [] # array points
x_vals = []  # array x vals
y_vals = []  # array of y vals

for x in range(0, 100):
    x_vals.append(random.randint(1, 250))
    y_vals.append(random.randint(1, 250))

for x in range(0, 100):
    points.append(Point(x_vals[x], y_vals[x]))

# plotting the points (held in array points)
Point.print_vals(points[0])
plt.scatter(x_vals, y_vals)

dist = []
for x in range(0, 99):
    for y in range(1, 100):
        if x != y:
            lists = []
            # variable d that hold a distance between two points
            d = abs(points[x].x_val - points[y].x_val) + abs(points[x].y_val - points[y].y_val)
            if d < 22: # if distance is less than 20
                # print('CURRENT X VAL: ' + str(x)) # prints the location of loop in x
                # print('CURRENT y VAL: ' + str(y)) # prints the location of loop in y
                Point.print_vals(points[x]) # print point 1
                Point.print_vals(points[y]) # print point 2
                # print("Distance between the points:", d) # print the distance

                # connects the points on the graph
                plt.plot([points[x].x_val, points[y].x_val], [points[x].y_val, points[y].y_val])




plt.show()
