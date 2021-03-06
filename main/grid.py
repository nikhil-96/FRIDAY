from typing import List

from matplotlib import pyplot as plt
import numpy as np



class Grid:
    """
    Grid class defining attributes of the environemnt. Implements functions to set and get
    attribues related to the environment.
    """
    def __init__(self, width, height, p_random=0.0):
        self.p_random = p_random
        self.width = width
        self.height = height
        self.obstacles: List[Square] = []
        self.reg_dirt: List[Square] = []
        self.goals: List[Square] = []
        self.robot = None
        self.dirt_places: List[Square] = []
        self.much_dirt_places: List[Square] = []
        self.death: List[Square] = []
        self.walls: List[Square] = []
        self.boundary = None
        # self._robot=deepcopy(robot)

    def is_in_bounds(self, x, y, size_x, size_y):
        return x >= 0 and x + size_x <= self.width and y >= 0 and y + size_y <= self.height

    def put_obstacle(self, x, y, size_x, size_y):

        assert self.is_in_bounds(x, y, size_x, size_y)
        sx = 0.5
        sy = 0.5
        for x_i in np.arange(size_x // sx):
            for y_i in np.arange(size_y // sy):
                ob = Square(x + (x_i * sx), x + (x_i * sx) + sx, y + (y_i * sy), y + (y_i * sy) + sy)
                self.obstacles.append(ob)
        # Then add all remainder goals
        if size_x % sx != 0:
            for y_i in np.arange(size_y // sy):
                ob = Square(x + size_x - (size_x % sx), x + size_x, y + (y_i * sy), y + (y_i * sy) + sy)
                self.obstacles.append(ob)

        if size_y % sy != 0:
            for x_i in np.arange(size_x // sx):
                ob = Square(x + (x_i * sx), x + (x_i * sx) + sx, y + size_y - (size_y % sy), y + size_y)
                self.obstacles.append(ob)

        if size_y % sy != 0 and size_x % sx != 0:
            ob = Square(x + size_x - (size_x % sx), x + size_x, y + size_y - (size_y % sy), y + size_y)
            self.obstacles.append(ob)

        # ob = Square(x, x + size_x, y, y + size_y)
        # self.obstacles.append(ob)

    def put_wall(self, x, y, size_x, size_y):
        assert self.is_in_bounds(x, y, size_x, size_y)

        sx = 0.5
        sy = 0.5

        for x_i in np.arange(size_x // sx):
            for y_i in np.arange(size_y // sy):
                ob = Square(x + (x_i * sx), x + (x_i * sx) + sx, y + (y_i * sy), y + (y_i * sy) + sy)
                self.walls.append(ob)
        # Then add all remainder goals
        if size_x % sx != 0:
            for y_i in np.arange(size_y // sy):
                ob = Square(x + size_x - (size_x % sx), x + size_x, y + (y_i * sy), y + (y_i * sy) + sy)
                self.walls.append(ob)

        if size_y % sy != 0:
            for x_i in np.arange(size_x // sx):
                ob = Square(x + (x_i * sx), x + (x_i * sx) + sx, y + size_y - (size_y % sy), y + size_y)
                self.walls.append(ob)

        if size_y % sy != 0 and size_x % sx != 0:
            ob = Square(x + size_x - (size_x % sx), x + size_x, y + size_y - (size_y % sy), y + size_y)
            self.walls.append(ob)

        # ob = Square(x, x + size_x, y, y + size_y)
        # self.walls.append(ob)

    def put_death(self, x, y, size_x, size_y):
        assert self.is_in_bounds(x, y, size_x, size_y)
        sx = 0.5
        sy = 0.5

        for x_i in np.arange(size_x // sx):
            for y_i in np.arange(size_y // sy):
                ob = Square(x + (x_i * sx), x + (x_i * sx) + sx, y + (y_i * sy), y + (y_i * sy) + sy)
                self.death.append(ob)
        # Then add all remainder goals
        if size_x % sx != 0:
            for y_i in np.arange(size_y // sy):
                ob = Square(x + size_x - (size_x % sx), x + size_x, y + (y_i * sy), y + (y_i * sy) + sy)
                self.death.append(ob)

        if size_y % sy != 0:
            for x_i in np.arange(size_x // sx):
                ob = Square(x + (x_i * sx), x + (x_i * sx) + sx, y + size_y - (size_y % sy), y + size_y)
                self.death.append(ob)

        if size_y % sy != 0 and size_x % sx != 0:
            ob = Square(x + size_x - (size_x % sx), x + size_x, y + size_y - (size_y % sy), y + size_y)
            self.death.append(ob)


        # ob = Square(x, x + size_x, y, y + size_y)
        # self.death.append(ob)

    def put_dirt(self, x, y, size_x, size_y):
        x = np.random.randint(0, high=27, size=1, dtype=int)
        y = np.random.randint(0, high=27, size=1, dtype=int)
        while self.is_in_bounds(x, y, size_x, size_y) != True:
            x = np.random.randint(0, high=27, size=1, dtype=int)
            y = np.random.randint(0, high=27, size=1, dtype=int)
        # assert self.is_in_bounds(x, y, size_x, size_y)
        # We split the dirt tile into sx by sy blocks
        sx = 0.5
        sy = 0.5
        for x_i in np.arange(size_x // sx):
            for y_i in np.arange(size_y // sy):
                dirt = Square(x + (x_i * sx), x + (x_i * sx) + sx, y + (y_i * sy), y + (y_i * sy) + sy)
                self.dirt_places.append(dirt)
        # Then add all remainder goals
        if size_x % sx != 0:
            for y_i in np.arange(size_y // sy):
                dirt = Square(x + size_x - (size_x % sx), x + size_x, y + (y_i * sy), y + (y_i * sy) + sy)
                self.dirt_places.append(dirt)

        if size_y % sy != 0:
            for x_i in np.arange(size_x // sx):
                dirt = Square(x + (x_i * sx), x + (x_i * sx) + sx, y + size_y - (size_y % sy), y + size_y)
                self.dirt_places.append(dirt)

        if size_y % sy != 0 and size_x % sx != 0:
            dirt = Square(x + size_x - (size_x % sx), x + size_x, y + size_y - (size_y % sy), y + size_y)
            self.dirt_places.append(dirt)

        self.finalize_dirt()

    def put_much_dirt(self, x, y, size_x, size_y):
        x = np.random.randint(10, high=22, size=1, dtype=int)
        y = np.random.randint(2, high=27, size=1, dtype=int)
        while self.is_in_bounds(x, y, size_x, size_y) != True:
            x = np.random.randint(10, high=22, size=1, dtype=int)
            y = np.random.randint(2, high=27, size=1, dtype=int)
        # assert self.is_in_bounds(x, y, size_x, size_y)
        # We split the dirt tile into sx by sy blocks
        sx = 0.5
        sy = 0.5
        for x_i in np.arange(size_x // sx):
            for y_i in np.arange(size_y // sy):
                dirt = Square(x + (x_i * sx), x + (x_i * sx) + sx, y + (y_i * sy), y + (y_i * sy) + sy)
                self.much_dirt_places.append(dirt)
        # Then add all remainder goals
        if size_x % sx != 0:
            for y_i in np.arange(size_y // sy):
                dirt = Square(x + size_x - (size_x % sx), x + size_x, y + (y_i * sy), y + (y_i * sy) + sy)
                self.much_dirt_places.append(dirt)

        if size_y % sy != 0:
            for x_i in np.arange(size_x // sx):
                dirt = Square(x + (x_i * sx), x + (x_i * sx) + sx, y + size_y - (size_y % sy), y + size_y)
                self.much_dirt_places.append(dirt)

        if size_y % sy != 0 and size_x % sx != 0:
            dirt = Square(x + size_x - (size_x % sx), x + size_x, y + size_y - (size_y % sy), y + size_y)
            self.much_dirt_places.append(dirt)

        self.finalize_much_dirt()

    def put_goal(self, x, y, size_x, size_y):
        assert self.is_in_bounds(x, y, size_x, size_y)
        # We split the dirt tile into sx by sy blocks
        sx = 1.0
        sy = 1.0
        for x_i in np.arange(size_x//sx):
            for y_i in np.arange(size_y//sy):
                goal = Square(x+(x_i*sx), x+(x_i*sx)+sx, y+(y_i*sy), y+(y_i*sy)+sy)
                self.goals.append(goal)
        # Then add all remainder goals
        if size_x % sx != 0:
            for y_i in np.arange(size_y // sy):
                goal = Square(x + size_x - (size_x % sx), x + size_x, y + (y_i*sy), y + (y_i*sy) + sy)
                self.goals.append(goal)

        if size_y % sy != 0:
            for x_i in np.arange(size_x // sx):
                goal = Square(x + (x_i*sx), x + (x_i*sx) + sx, y + size_y - (size_y % sy), y + size_y)
                self.goals.append(goal)

        if size_y % sy != 0 and size_x % sx != 0:
            goal = Square(x + size_x - (size_x % sx), x + size_x, y + size_y - (size_y % sy), y + size_y)
            self.goals.append(goal)

    def put_reg_dirt(self, x, y, size_x, size_y):
        x = np.random.randint(1, high=8, size=1, dtype=int)
        y = np.random.randint(2, high=17, size=1, dtype=int)
        while self.is_in_bounds(x, y, size_x, size_y) != True:
            x = np.random.randint(1, high=8, size=1, dtype=int)
            y = np.random.randint(2, high=17, size=1, dtype=int)
        # assert self.is_in_bounds(x, y, size_x, size_y)
        # We split the dirt tile into sx by sy blocks
        sx = 0.5
        sy = 0.5
        for x_i in np.arange(size_x // sx):
            for y_i in np.arange(size_y // sy):
                dirt = Square(x + (x_i * sx), x + (x_i * sx) + sx, y + (y_i * sy), y + (y_i * sy) + sy)
                self.reg_dirt.append(dirt)
        # Then add all remainder goals
        if size_x % sx != 0:
            for y_i in np.arange(size_y // sy):
                dirt = Square(x + size_x - (size_x % sx), x + size_x, y + (y_i * sy), y + (y_i * sy) + sy)
                self.reg_dirt.append(dirt)

        if size_y % sy != 0:
            for x_i in np.arange(size_x // sx):
                dirt = Square(x + (x_i * sx), x + (x_i * sx) + sx, y + size_y - (size_y % sy), y + size_y)
                self.reg_dirt.append(dirt)

        if size_y % sy != 0 and size_x % sx != 0:
            dirt = Square(x + size_x - (size_x % sx), x + size_x, y + size_y - (size_y % sy), y + size_y)
            self.reg_dirt.append(dirt)

        self.finalize_reg_dirt()

    def check_delete_goals(self, robot):
        for i, goal in enumerate(self.goals):
            if goal.intersect(robot.bounding_box):
                self.goals.remove(goal)
                # self._robot.battery_lvl = 0
                return True
        return False

    def check_death(self, robot):
        for i, goal in enumerate(self.goals):
            if goal.intersect(robot.bounding_box):
                return True
        return False

    def is_blocked(self, box: "Square"):
        return any([ob.intersect(box) for ob in self.obstacles])

    def is_wall(self, box: "Square"):
        return any([ob.intersect(box) for ob in self.walls])

    def boundaries(self, start_x, start_y, start_x2 , start_y2 ):
        self.boundary = Square(
            start_x,
            start_x2,
            start_y,
            start_y2
        )

    def finalize_dirt(self):
        for i, dirt in enumerate(self.dirt_places):
            for ob in self.obstacles:
                if dirt.intersect_dirt(ob):
                    self.dirt_places.remove(dirt)
            for ob in self.death:
                if dirt.intersect_dirt(ob):
                    self.dirt_places.remove(dirt)
            for ob in self.walls:
                if dirt.intersect_dirt(ob):
                    self.dirt_places.remove(dirt)
            # for ob in self.much_dirt_places:
            #     if dirt.intersect_dirt(ob):
            #         self.dirt_places.remove(dirt)
            for ob in self.reg_dirt:
                if dirt.intersect_dirt(ob):
                    self.dirt_places.remove(dirt)
        return False

    def finalize_much_dirt(self):
        for i, dirt in enumerate(self.much_dirt_places):
            for ob in self.obstacles:
                if dirt.intersect_dirt(ob):
                    self.much_dirt_places.remove(dirt)
            for ob in self.walls:
                if dirt.intersect_dirt(ob):
                    self.much_dirt_places.remove(dirt)
            for ob in self.death:
                if dirt.intersect_dirt(ob):
                    self.much_dirt_places.remove(dirt)
            for ob in self.reg_dirt:
                if dirt.intersect_dirt(ob):
                    self.much_dirt_places.remove(dirt)
        return False

    def finalize_reg_dirt(self):
        for i, dirt in enumerate(self.reg_dirt):
            for ob in self.obstacles:
                if dirt.intersect_dirt(ob):
                    self.reg_dirt.remove(dirt)
            for ob in self.death:
                if dirt.intersect_dirt(ob):
                    self.reg_dirt.remove(dirt)
            for ob in self.walls:
                if dirt.intersect_dirt(ob):
                    self.reg_dirt.remove(dirt)
            for ob in self.much_dirt_places:
                if dirt.intersect_dirt(ob):
                    self.reg_dirt.remove(dirt)
        return False

    def check_delete_dirt(self, robot):
        for i, dirt in enumerate(self.dirt_places):
            if dirt.intersect(robot.bounding_box):
                self.dirt_places.remove(dirt)
                return True
        return False

    def check_delete_much_dirt(self, robot):
        for i, dirt in enumerate(self.much_dirt_places):
            if dirt.intersect(robot.bounding_box):
                self.much_dirt_places.remove(dirt)
                return True
        return False

    def check_delete_reg_dirt(self, robot):
        for i, dirt in enumerate(self.reg_dirt):
            if dirt.intersect(robot.bounding_box):
                self.reg_dirt.remove(dirt)
                return True
        return False

    def get_border_coords(self):
        return [0, self.width, self.width, 0, 0], [0, 0, self.height, self.height, 0]


class Square:
    """
    Utility class for grid, initializes rectangular objects and tracks interactions
    """
    def __init__(self, x1, x2, y1, y2):
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        self.x_size = x2 - x1
        self.y_size = y2 - y1

    def intersect(self, other):
        intersecting = not (self.x2 <= other.x1 -0.2 or self.x1 >= other.x2+0.2 or self.y2 <= other.y1-0.2 or self.y1 >= other.y2+0.2)
        inside = (other.x1 >= self.x1 and other.x2 <= self.x2 and other.y1 >= self.y1 and other.y2 <= self.y2)
        return intersecting or inside

    def intersect_dirt(self, other):
        # intersecting = not (self.x2 <= other.x1 - 0.2 or self.x1 >= other.x2 + 0.2 or self.y2 <= other.y1 - 0.2 or self.y1 >= other.y2 + 0.2)
        # intersecting = not (self.x2 <= other.x1 or self.x1 >= other.x2 or self.y2 <= other.y1 or self.y1 >= other.y2)
        inside = (other.x1 >= self.x1 and other.x2 <= self.x2 and other.y1 >= self.y1 and other.y2 <= self.y2)
        return inside

    def update_pos(self, x, y):
        self.x1, self.x2, self.y1, self.y2 = x, x + self.x_size, y, y + self.y_size