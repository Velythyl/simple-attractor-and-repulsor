import numpy as np
import torch
import matplotlib.pyplot as plt

class Planner:
    def __init__(self, max_vel):
        self.attractors = []
        self.repulsors = []

        self.max_vel = max_vel

    def set_attractor(self, target_state, weight_mat):
        def attractor(robot_state):
            state_diff = robot_state - target_state
            x = torch.matmul(state_diff.T, weight_mat)
            x2 = torch.matmul(x, state_diff)
            attract = 0.5 * x2

            #attract = torch.clip(attract, -self.max_vel, self.max_vel)

            return attract
        self.attractors.append(attractor)

    def set_repulsor(self, target_state, weight_mat, min_dist=0.1):
        def repulsor(robot_state):
            state_diff = robot_state - target_state
            denom = 2 * state_diff.T @ state_diff
            repulse = weight_mat / denom

            #repulse = torch.clip(repulse, -self.max_vel, self.max_vel)
            return repulse
            abs_diff = torch.abs(state_diff)
            thresh = torch.sum(abs_diff < min_dist).float()

            repulse_thresholded = repulse * thresh
            return repulse_thresholded
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
    def __init__(self, robot_init, dt, decimation):
        self.robot_init = robot_init

        self.goals_mats = []
        self.goals = []

        self.obstacles_weights = []
        self.obstacles = []
        self.obstacles_thresh = []

        self.dt = dt
        self.decimation = decimation

        self.planner = Planner(0.2)

    def add_goal(self, goal, mat):
        self.goals.append(goal)
        self.goals_mats.append(mat)

    def add_obstacle(self, obs, w):
        self.obstacles.append(obs)
        self.obstacles_weights.append(w)

    def compile(self):
        for obs in zip(self.obstacles, self.obstacles_weights):
            self.planner.set_repulsor(*obs)
        for goal in zip(self.goals, self.goals_mats):
            self.planner.set_attractor(*goal)

    def query(self, robot_pose):
        velocities = torch.clip(self.planner.potential(robot_pose), -self.planner.max_vel, self.planner.max_vel)
        return velocities.clone().detach().cpu().numpy()

    def run(self):
        robot = self.robot_init.clone()
        robot_history = []
        for i in range(1000):
            pos = robot.clone().numpy()
            robot_history.append(pos)
            velocities = torch.clip(self.planner.potential(robot), -self.planner.max_vel, self.planner.max_vel)
            for _ in range(self.decimation):
                robot += velocities * self.dt
            print("des_vel:", velocities)
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
    # squares
    def square(center, l, n):
        top_left = np.array([center[0] - l / 2, center[1] - l / 2])
        # https://stackoverflow.com/questions/53548996/how-to-return-points-on-a-square-as-an-array
        top = np.stack(
            [np.linspace(top_left[0], top_left[0] + l, n//4 + 1),
             np.full(n//4 + 1, top_left[1])],
            axis=1
        )[:-1]
        left = np.stack(
            [np.full(n//4 + 1, top_left[0]),
             np.linspace(top_left[1], top_left[1] - l, n//4 + 1)],
            axis=1
        )[:-1]
        right = left.copy()
        right[:, 0] += l
        bottom = top.copy()
        bottom[:, 1] -= l
        return torch.tensor(np.concatenate([top, right, bottom, left, np.array([[top_left[0] + l, top_left[1] - l]])]))
    """
    # test case
    episode = Episode(
        robot_init=torch.tensor([9.5, 9.9]),
        dt=0.01,
        decimation=4
    )
    episode.add_goal(torch.tensor([0.1,0.1]), torch.eye(2))
    episode.add_obstacle(torch.tensor([5,5]), 2)
    episode.compile()
    episode.run()


    episode = Episode(
        robot_init=torch.tensor([9.5, 9.9]),
        dt=0.01,
        decimation=4
    )
    sq = square([5.0, 5.0], 1, 10)
    for sq_point in sq:
        episode.add_obstacle(sq_point, 1)
    episode.add_goal(torch.tensor([0.1,0.1]), torch.eye(2))
    episode.compile()"""

    episode = Episode(
        robot_init=torch.tensor([2.065, 1.475]),
        dt=0.01,
        decimation=4
    )

    def obs_sq(pos):
        return square(pos, 0.59, 10)

    def add_square(sq):
        for sq_point in obs_sq(sq):
            episode.add_obstacle(sq_point, 0.1)

    for temp in [
        [1.575,2.655], # 1
        [2.065,2.655], # 2
        [3.245,2.655]
    ]:
        add_square(temp)

    #sq = square([2.065, 2.655], 2.065, 10)
    #for sq_point in sq:
    #    episode.add_obstacle(sq_point, 0.1)
    episode.add_obstacle(torch.tensor([3.64, 0]), 4)
    #episode.add_obstacle(torch.tensor([2.065, 0]), 4)

    goal_w = torch.eye(2)
    goal_w[1,1] = 0.5
    episode.add_goal(torch.tensor([2.065,3.9825]), torch.eye(2))
    episode.compile()


    episode.run()
