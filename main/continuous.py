from copy import deepcopy
from pathlib import Path

import numpy as np
from gym import Env
from gym import spaces
from matplotlib import pyplot as plt

from main.robot import Robot
from main.parser import parse_config


class FridayCleaning(Env):
    def __init__(self, config):
        self.rendering_init = False
        grid = config["grid"]
        robot = config["robot"]

        self._original_grid = deepcopy(grid)
        self._original_robot = deepcopy(robot)
        self._robot = deepcopy(robot)
        self._grid = deepcopy(grid)
        self.observation_space = spaces.Dict({
            "distances to borders": spaces.Box(
                low=0,
                high=max(grid.width, grid.height),
                shape=(4,),
                dtype=np.float64
            ),
            "distances to patches": spaces.Box(
                low=0,
                high=max(grid.width, grid.height),
                shape=(4,),
                dtype=np.float64
            )
        })
        self.action_space = spaces.Box(low=0, high=2 * np.pi, shape=(1,))
        self.reward_structure = {
            "wall": -5.,
            "obstacle": -3.,
            "regular": 0.,
            "goal": 10.,
            "dirt": 3.,
            "much_dirt": 5.,
            "death": -5.,
            "reg_dirt": 5.
        }

        assert self._grid.is_in_bounds(
            self._robot.bounding_box.x1,
            self._robot.bounding_box.y1,
            self._robot.bounding_box.x_size,
            self._robot.bounding_box.y_size
        )
        assert not self._grid.is_blocked(robot.bounding_box)

    def step(self, action):
        assert self._robot.alive
        done = False
        #print("inside step action", action)

        # Compute the move vector
        if np.random.binomial(n=1, p=self._grid.p_random) == 1:
            action = self.action_space.sample()
            #print("inside random step action", action)

        #print("inside step robot move dis", self._robot.move_distance)
        move_vector = np.array([np.cos(action[0]), np.sin(action[0])]) * self._robot.move_distance
        #print("inside step move vector", move_vector)

        # Set the new position
        new_pos = tuple(np.array(self._robot.pos) + move_vector)
        #print("inside step robot.pos", self._robot.pos)
        #print("inside step new pos", new_pos)
        #breakpoint()
        # Temporarily set the new bounding box to check if it is valid
        new_box = deepcopy(self._robot.bounding_box)
        new_box.update_pos(*new_pos)

        if self._grid.is_blocked(new_box):
            return self._make_observation(), self.reward_structure["obstacle"], False, {}
        elif self._grid.is_wall(new_box):
            return self._make_observation(), self.reward_structure["wall"], False, {}
        elif not self._grid.is_in_bounds(new_pos[0], new_pos[1], self._robot.size, self._robot.size):
            return self._make_observation(), self.reward_structure["wall"], False, {}
        else:
            do_battery_drain = np.random.binomial(1, self._robot.battery_drain_p)

            if do_battery_drain == 1 and self._robot.battery_lvl > 0:
                self._robot.battery_lvl -= (np.random.exponential(self._robot.battery_drain_lam))
                if self._robot.battery_lvl <= 0:
                    self._robot.alive = False
                    self._robot.battery_lvl = 0

                    return self._make_observation(), self.reward_structure["regular"], True, {"reason": "battery drain"}

            del new_box
            self._robot.pos = new_pos
            self._robot.bounding_box.update_pos(*self._robot.pos)

            # What to do if the robot made a valid move with enough battery:
            if self._grid.check_delete_goals(self._robot) and len(self._grid.goals) == 0:
                self._robot.alive = False
                return self._make_observation(), self.reward_structure["goal"], True, {"reason": "goals cleared"}
            elif self._grid.check_death(self._robot):
                self._robot.alive = False
                return self._make_observation(), self.reward_structure["death"], True, {"reason": "died"}
            elif self._grid.check_delete_goals(self._robot) and len(self._grid.goals) > 0:
                return self._make_observation(), self.reward_structure["goal"], False, {}
            elif self._grid.check_delete_dirt(self._robot):
                return self._make_observation(), self.reward_structure["dirt"], False, {}
            elif self._grid.check_delete_reg_dirt(self._robot):
                return self._make_observation(), self.reward_structure["reg_dirt"], False, {}
            elif self._grid.check_delete_much_dirt(self._robot):
                return self._make_observation(), self.reward_structure["much_dirt"], False, {}
            else:
                return self._make_observation(), self.reward_structure["regular"], False, {}

    def _make_observation(self):
        robot_center = (self._robot.bounding_box.x1 + self._robot.bounding_box.x2) / 2, \
                       (self._robot.bounding_box.y1 + self._robot.bounding_box.y2) / 2
        # Compute the distances (n, e, s, w) to either obstacles or walls depending on which one comes first
        distances_to_obstacles = self._get_distances_from_reference(
            reference_point=robot_center,
            squares=self._grid.obstacles,
            fallback_n=robot_center[1],
            fallback_e=self._grid.width - robot_center[0],
            fallback_s=self._grid.height - robot_center[1],
            fallback_w=robot_center[0]
        )
        # Compute the distances to patches. If there are none, the sensor would return the maximum visible distance.
        distances_to_patches = self._get_distances_from_reference(
            reference_point=robot_center,
            squares=self._grid.goals,
            fallback_n=distances_to_obstacles[0],
            fallback_e=distances_to_obstacles[1],
            fallback_s=distances_to_obstacles[2],
            fallback_w=distances_to_obstacles[3]
        )

        return {
            "distances to borders": np.array(distances_to_obstacles),
            "distances to patches": np.array(distances_to_patches)
        }

    @staticmethod
    def _get_distances_from_reference(reference_point, squares, fallback_n, fallback_e, fallback_s, fallback_w):
        distances_e = [ob.x1 - reference_point[0] for ob in squares
                       if ob.y1 <= reference_point[1] <= ob.y2 and ob.x1 >= reference_point[0]]
        distances_w = [reference_point[0] - ob.x2 for ob in squares
                       if ob.y1 <= reference_point[1] <= ob.y2 and ob.x2 <= reference_point[0]]
        distances_n = [reference_point[1] - ob.y2 for ob in squares
                       if ob.x1 <= reference_point[0] <= ob.x2 and ob.y2 <= reference_point[1]]
        distances_s = [ob.y1 - reference_point[1] for ob in squares
                       if ob.x1 <= reference_point[0] <= ob.x2 and ob.y1 >= reference_point[1]]
        nearest_distance_e = fallback_e if not distances_e else min(distances_e)
        nearest_distance_w = fallback_w if not distances_w else min(distances_w)
        nearest_distance_n = fallback_n if not distances_n else min(distances_n)
        nearest_distance_s = fallback_s if not distances_s else min(distances_s)

        return nearest_distance_n, nearest_distance_e, nearest_distance_s, nearest_distance_w

    def render(self, mode="human"):
        if self.rendering_init == False:
            plt.ion()
            plt.gcf()
            plt.show()
            self.rendering_init = True

        plt.gcf()
        plt.plot(*self._grid.get_border_coords(), color='black', alpha = 0.5)
        plt.axis('off')

        for ob in self._grid.dirt_places:
            plt.fill([ob.x1, ob.x2, ob.x2, ob.x1, ob.x1], [ob.y1, ob.y1, ob.y2, ob.y2, ob.y1], color='#ffcdb2', alpha = 1)

        for ob in self._grid.death:
            plt.fill([ob.x1, ob.x2, ob.x2, ob.x1, ob.x1], [ob.y1, ob.y1, ob.y2, ob.y2, ob.y1], color='#ad050e', alpha = 0.5, linewidth = 0.0)

        for ob in self._grid.much_dirt_places:
            plt.fill([ob.x1, ob.x2, ob.x2, ob.x1, ob.x1], [ob.y1, ob.y1, ob.y2, ob.y2, ob.y1], color='#ffb4a2', alpha = 1)

        for goal in self._grid.goals:
            plt.fill(
                [goal.x1, goal.x2, goal.x2, goal.x1, goal.x1],
                [goal.y1, goal.y1, goal.y2, goal.y2, goal.y1],
                color='#036316', alpha = 0.5 , linewidth = 0.0
            )

        for ob in self._grid.obstacles:
            plt.fill_between([ob.x1, ob.x2, ob.x2, ob.x1, ob.x1], [ob.y1, ob.y1, ob.y2, ob.y2, ob.y1], color='#702702', alpha = 0.5, linewidth = 0.0) ##702702  #4d4730


        for ob in self._grid.reg_dirt:
            plt.fill([ob.x1, ob.x2, ob.x2, ob.x1, ob.x1], [ob.y1, ob.y1, ob.y2, ob.y2, ob.y1], color='#ffb4a2', alpha = 1)

        for ob in self._grid.walls:
            plt.fill([ob.x1, ob.x2, ob.x2, ob.x1, ob.x1], [ob.y1, ob.y1, ob.y2, ob.y2, ob.y1], color='#121212', alpha = 0.5, linewidth = 0.0)


        robot_box = self._robot.bounding_box
        plt.plot(
            [robot_box.x1, robot_box.x2, robot_box.x2, robot_box.x1, robot_box.x1],
            [robot_box.y1, robot_box.y1, robot_box.y2, robot_box.y2, robot_box.y1],
            color='blue'
        )
        plt.title(f"Battery level: {str(round(self._robot.battery_lvl, 2))}")
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    def reset(self):
        # TODO: select a random location for the robot
        self.__init__(dict(grid=self._original_grid, robot=self._original_robot))

        return self._make_observation()

    @property
    def grid(self):
        return deepcopy(self._grid)


if __name__ == "__main__":
    from gym.utils.env_checker import check_env

    # Check if the environment conforms to the Gym API
    grid = parse_config(Path("../grids") / "house.grid")
    robot = Robot(init_position=(0, 8))
    check_env(FridayCleaning(grid, robot))