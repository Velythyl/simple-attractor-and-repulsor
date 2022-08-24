import numpy as np
import torch
import matplotlib.pyplot as plt

class Planner:
    def __init__(self):
        self.attractors = []
        self.repulsors = []

    def set_attractor(self, weight_mat, target_state):
        def attractor(robot_state):
            state_diff = robot_state - target_state
            x = torch.matmul(state_diff.T, weight_mat)
            x2 = torch.matmul(x, state_diff)
            return 0.5 * x2
        self.attractors.append(attractor)

    def set_repulsor(self, weight_mat, target_state):
        def repulsor(robot_state):
            state_diff = robot_state - target_state
            denom = 2 * state_diff.T @ state_diff
            return weight_mat / denom
        self.repulsors.append(repulsor)

    def potential(self, robot_state):
        def grad(robot_state, func):
            input = torch.tensor(robot_state, requires_grad=True)
            output = func(input)
            output.backward()
            _grad = input.grad
            return _grad

        velocities = []
        all_funcs = self.attractors + self.repulsors
        for func in all_funcs:
            velocities.append(-grad(robot_state, func))
        final_velocities = sum(velocities)
        return final_velocities

class Episode:
    def __init__(self, robot_init, goals, obstacles, dt, decimation):
        self.robot_init = robot_init
        self.goals_mats, self.goals = list(zip(*goals))
        self.obstacles_weight, self.obstacles = list(zip(*obstacles))
        self.dt = dt
        self.decimation = decimation

        self.planner = Planner()
        for obs in obstacles:
            self.planner.set_repulsor(*obs)
        for goal in goals:
            self.planner.set_attractor(*goal)

    def run(self):
        robot = self.robot_init.clone()
        robot_history = []
        for i in range(1000):
            robot_history.append(robot.clone().numpy())
            velocities = self.planner.potential(robot)
            for _ in range(self.decimation):
                robot += velocities * self.dt
        robot_history = np.array(robot_history)

        plt.plot(robot_history[:, 0], robot_history[:, 1])
        circs = []
        for goal in self.goals:
            c = plt.Circle(goal, 0.1, color='green')
            circs.append(c)
        for obs in self.obstacles:
            c = plt.Circle(obs, 0.1, color='red')
            circs.append(c)
        for c in (circs + [plt.Circle(self.robot_init, 0.1, color="blue")]):
            plt.gca().add_patch(c)
        plt.show()


if __name__ == "__main__":
    episode = Episode(
        robot_init=torch.tensor([9.5, 9.9]),
        goals=[
            (torch.eye(2), torch.tensor([0.1,0.1]))
        ],
        obstacles=[
            (2, torch.tensor([5,5]))
        ],
        dt=0.01,
        decimation=4
    )
    episode.run()

