import os
import math
import copy
import numpy as np
from itertools import product
from classes import PRMController, Utils
from matplotlib import pyplot as plt

from ..gaussian_process.gp_st import GaussianProcessWrapper

from numba import jit
import time
from scipy.ndimage import gaussian_filter
from ..fire_commander.Fire_2D import FireCommanderExtreme as Fire


def add_t(X, t: float):
    return np.concatenate((X, np.zeros((X.shape[0], 1)) + t), axis=1)


class Env:
    def __init__(
        self,
        sample_size=500,
        k_size=10,
        start=None,
        destination=None,
        obstacle=[],
        budget_range=None,
        save_image=False,
        seed=None,
        fixed_env=None,
        adaptive_th=None,
        adaptive_area=True,
        n_agents=1,
        fuel=None,
    ):
        self.ADAPTIVE_TH = adaptive_th
        self.ADAPTIVE_AREA = adaptive_area
        self.sample_size = sample_size
        self.k_size = k_size
        self.budget_range = budget_range
        self.budget = self.budget_init = np.random.uniform(*self.budget_range)
        if start is None:
            self.start = np.random.rand(1, 2)
        else:
            self.start = np.array([start])
        if destination is None:
            self.destination = np.random.rand(1, 2)
        else:
            self.destination = np.array([destination])

        if seed is not None:
            np.random.seed(seed)

        self.obstacle = obstacle
        self.seed = seed
        self.curr_t = 0.0
        self.n_agents = 1
        self.env_size = 30
        self.fuel = fuel
        # # generate Fire environment
        self.fire = Fire(
            world_size=self.env_size,
            online_vis=True,
            start=self.start[0],
            seed=seed,
            fuel=self.fuel,
        )
        self.fire.env_init()

        # generate PRM
        # self.prm = None
        # self.node_coords, self.graph = None, None
        # self.start = np.random.rand(1, 2)
        # self.destination = np.random.rand(1, 2)
        self.prm = PRMController(
            self.sample_size, self.obstacle, self.start, self.destination, self.k_size
        )
        self.node_coords, self.graph = self.prm.runPRM(saveImage=False, seed=seed)

        # self.graph_crtl = GraphController(self.sample_size, self.start, self.k_size)
        # self.node_coords, self.graph = self.graph_crtl.generate_graph()

        # underlying distribution
        self.underlying_distribution = None
        self.ground_truth = None
        # self.high_info_area = None

        # GP
        self.gp_wrapper = None
        self.node_feature = None
        # self.gp_ipp = None
        self.node_info, self.node_std = None, None
        self.node_info0, self.node_std0, self.budget0 = copy.deepcopy(
            (self.node_info, self.node_std, self.budget)
        )
        self.JS, self.JS_init, self.JS_list, self.KL, self.KL_init, self.KL_list = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        self.cov_trace, self.cov_trace_init = None, None
        self.unc, self.unc_list, self.unc_init, self.unc_sum, self.unc_sum_list = (
            None,
            None,
            None,
            None,
            None,
        )
        self.RMSE = None
        self.F1score = None
        self.MI = None
        self.MI0 = None

        # start point
        self.current_node_index = 1  # 1  # 0 used in STAMP
        self.sample = self.start
        self.dist_residual = 0
        self.route = []

        self.save_image = save_image
        self.frame_files = []

        self.env_sim_params = self.fire.fire_params

    def reset(self, seed=None):
        # generate PRM
        # self.start = np.random.rand(1, 2)
        # self.destination = np.random.rand(1, 2)
        # self.prm = PRMController(self.sample_size, self.obstacle, self.start, self.destination, self.budget_range, self.k_size)
        # self.budget = np.random.uniform(*self.budget_range)
        # self.node_coords, self.graph = self.prm.runPRM(saveImage=False)
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed(self.seed)

        # self.fire.env_close()

        # # generate Fire environment
        self.fire = Fire(
            world_size=30,
            online_vis=True,
            start=self.start[0],
            seed=seed if seed else self.seed,
            fuel=self.fuel,
        )
        self.fire.env_init()

        # underlying distribution
        self.underlying_distribution = self.fire

        self.ground_truth = self.get_ground_truth()
        self.ground_truth = gaussian_filter(self.ground_truth, sigma=1.5)
        self.high_info_idx = self.get_high_info_idx() if self.ADAPTIVE_AREA else None

        # initialize gp
        self.curr_t = 0.0
        self.gp_wrapper = GaussianProcessWrapper(self.n_agents, self.node_coords)
        #     if arg.prior_measurement:
        # # randomly choose 50 samples from ground truth as initial data
        #         sample_index = np.random.choice(900, 50, replace=False)
        #         x1x2 = np.array(list(product(np.linspace(0, 1, 30), np.linspace(0, 1, 30))))
        #         samples = x1x2[sample_index]
        #         observed_values = np.array([self.ground_truth[sample_index]]).T

        self.node_feature = self.gp_wrapper.update_node_feature(self.curr_t)
        # node_info, node_info_future = self.node_feature[:, :2], self.node_feature[:, 2:]
        # node_pred, node_std = node_info[:, 0], node_info[:, 1]
        # node_info = node_pred

        # self.node_feature = self.gp_wrapper.update_node_feature(self.curr_t)
        # node_feature = self.node_feature[:, self.n_agents].reshape(-1, 1, self.node_feature.shape[1])
        # node_pred, node_std = node_feature[:, :, 0], node_feature[:, :, 1]

        # initialize evaluations
        self.RMSE = self.gp_wrapper.eval_avg_RMSE(self.ground_truth, self.curr_t)
        self.cov_trace = self.gp_wrapper.eval_avg_cov_trace(
            self.curr_t, self.high_info_idx
        )
        self.unc, self.unc_list = self.gp_wrapper.eval_avg_unc(
            self.curr_t, self.high_info_idx, return_all=True
        )
        self.JS, self.JS_list = self.gp_wrapper.eval_avg_JS(
            self.ground_truth, self.curr_t, return_all=True
        )
        self.KL, self.KL_list = self.gp_wrapper.eval_avg_KL(
            self.ground_truth, self.curr_t, return_all=True
        )
        self.unc_sum, self.unc_sum_list = self.gp_wrapper.eval_avg_unc_sum(
            self.unc_list, self.high_info_idx, return_all=True
        )
        self.JS_init = self.JS
        self.KL_init = self.KL
        self.cov_trace_init = self.cov_trace
        self.unc_init = self.unc
        self.budget = self.budget_init

        # start point
        self.current_node_index = 1  # 1
        self.dist_residual = 0
        self.sample = self.start
        # self.random_speed_factor = np.random.rand()
        self.route = []

        # self.gp_ipp = GaussianProcessForIPP(self.node_coords)
        # # for sample, observed_value in zip(samples, observed_values):
        # #     self.gp_ipp.add_observed_point(sample, observed_value[0])
        # # self.gp_ipp.update_gp()

        # self.high_info_area = self.gp_ipp.get_high_info_area(t=self.ADAPTIVE_TH) if self.ADAPTIVE_AREA else None
        # self.node_info, self.node_std = self.gp_ipp.update_node()

        # # initialize evaluations
        # #self.F1score = self.gp_ipp.evaluate_F1score(self.ground_truth)
        # self.RMSE = self.gp_ipp.evaluate_RMSE(self.ground_truth)
        # self.cov_trace = self.gp_ipp.evaluate_cov_trace(self.high_info_area)
        # self.MI = self.gp_ipp.evaluate_mutual_info(self.high_info_area)
        # self.cov_trace0 = self.cov_trace

        # # save initial state
        # self.node_info0, self.node_std0, self.budget = copy.deepcopy((self.node_info, self.node_std,self.budget0))

        # # start point
        # self.current_node_index = 1
        # self.sample = self.start
        # self.dist_residual = 0
        # self.route = []
        # np.random.seed(None)

        # return self.node_coords, self.graph, self.node_info, self.node_std, self.budget
        return self.node_coords, self.graph, self.node_feature, self.budget

    def step(self, next_node_index, sample_length, measurement=True, eval_speed=None):
        dist = np.linalg.norm(
            self.node_coords[self.current_node_index]
            - self.node_coords[next_node_index]
        )
        remain_length = dist
        next_length = sample_length - self.dist_residual
        reward = 0
        done = True if next_node_index == 0 else False
        no_sample = True
        while remain_length > next_length:
            if no_sample:
                self.sample = (
                    self.node_coords[next_node_index]
                    - self.node_coords[self.current_node_index]
                ) * next_length / dist + self.node_coords[self.current_node_index]
            else:
                self.sample = (
                    self.node_coords[next_node_index]
                    - self.node_coords[self.current_node_index]
                ) * next_length / dist + self.sample
            if measurement:
                observed_value = self.underlying_distribution.return_fire_at_location(
                    self.sample.reshape(-1, 2)[0]
                )  # + np.random.normal(0, 1e-10)
            else:
                observed_value = np.array([0])
            # print(">>> Observed value is ", observed_value)

            self.curr_t += next_length
            # self.budget -= next_length

            remain_length -= next_length
            next_length = sample_length
            no_sample = False

            self.underlying_distribution.single_agent_state_update(
                self.sample.reshape(-1, 2)[0]
            )
            # time1 = time.time()
            (
                fire_state,
                fire_reward,
                fire_done,
                fire_perception_complete,
                fire_action_complete,
                interp_fire_intensity,
            ) = self.fire.env_step(r_func="RF4")
            # time2 = time.time()
            # print(f">>> Time to FIRE env_step: {time2 - time1:.4f}")
            # reward += fire_reward
            # self.set_ground_truth(fire_map=interp_fire_intensity)
            self.set_momentum_GT(fire_map=interp_fire_intensity)

            # time1 = time.time()
            for i in range(self.n_agents):
                self.gp_wrapper.GPs[i].add_observed_point(
                    add_t(self.sample.reshape(-1, 2), self.curr_t), observed_value
                )
            # time2 = time.time()
            # print(f">>> Time to add obs points: {time2 - time1:.4f}")

        # time1 = time.time()
        # self.node_info, self.node_std = self.gp_ipp.update_node()
        if self.gp_wrapper.GPs[0].observed_points:
            self.gp_wrapper.update_gps()
        # time2 = time.time()
        # print(f">>> Time to update GPs: {time2 - time1:.4f}")

        self.dist_residual = (
            self.dist_residual + remain_length if no_sample else remain_length
        )
        self.budget -= dist
        # actual_t = self.curr_t + self.dist_residual
        # actual_budget = self.budget - self.dist_residual
        actual_t = self.curr_t
        actual_budget = self.budget

        # time1 = time.time()
        self.node_feature = self.gp_wrapper.update_node_feature(actual_t)
        # node_info, node_info_future = self.node_feature[:, :2], self.node_feature[:, 2:]
        # node_pred, node_std = node_info[:, 0], node_info[:, 1]
        # node_info = node_pred

        # time2 = time.time()
        # print(f">>> Time to update node feature: {time2 - time1:.4f}")

        # time1 = time.time()
        self.high_info_idx = self.get_high_info_idx() if self.ADAPTIVE_TH else None
        # time2 = time.time()
        # print(f">>> Time to get high info idx: {time2 - time1:.4f}")

        # # evaluate metrics

        self.RMSE = self.gp_wrapper.eval_avg_RMSE(self.ground_truth, actual_t)

        cov_trace = self.gp_wrapper.eval_avg_cov_trace(actual_t, self.high_info_idx)

        unc, unc_list = self.gp_wrapper.eval_avg_unc(
            actual_t, self.high_info_idx, return_all=True
        )

        unc_sum, unc_sum_list = self.gp_wrapper.eval_avg_unc_sum(
            self.unc_list, self.high_info_idx, return_all=True
        )

        JS, JS_list = self.gp_wrapper.eval_avg_JS(
            self.ground_truth, actual_t, return_all=True
        )

        KL, KL_list = self.gp_wrapper.eval_avg_KL(
            self.ground_truth, actual_t, return_all=True
        )

        # if measurement:
        #     self.high_info_area = self.gp_ipp.get_high_info_area(t=self.ADAPTIVE_TH) if self.ADAPTIVE_AREA else None
        # #F1score = self.gp_ipp.evaluate_F1score(self.ground_truth)
        #     RMSE = self.gp_ipp.evaluate_RMSE(self.ground_truth)
        #     self.RMSE = RMSE
        # cov_trace = self.gp_ipp.evaluate_cov_trace(self.high_info_area)
        # self.F1score = F1score
        if next_node_index in self.route[-2:]:
            reward += -0.1

        elif self.cov_trace > cov_trace:
            reward += (self.cov_trace - cov_trace) / self.cov_trace
        self.cov_trace = cov_trace

        if done:
            reward -= cov_trace / 900

        self.JS, self.JS_list = JS, JS_list
        self.KL, self.KL_list = KL, KL_list
        # self.cov_trace = cov_trace
        self.unc, self.unc_list = unc, unc_list
        self.unc_sum, self.unc_sum_list = unc_sum, unc_sum_list
        self.route += [next_node_index]
        self.current_node_index = next_node_index
        if not done and actual_budget <= 0.0005:
            done = True
        # done = True if actual_budget <= 0.0005 else False
        # print("actual budget : ", actual_budget, self.current_node_index, done)

        return reward, done, self.node_feature, actual_budget
        # return reward, done, self.node_info, self.node_std, self.budget

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def calc_fire_intensity_in_field(fire_map, world_size=100):
        field_intensity = np.zeros((world_size, world_size))

        for i in range(fire_map.shape[0]):
            x, y, intensity = fire_map[i]
            m = min(int(x), field_intensity.shape[0] - 1)
            n = min(int(y), field_intensity.shape[1] - 1)

            if field_intensity[m, n] > 0:
                field_intensity[m, n] = max(field_intensity[m, n], intensity)
            else:
                field_intensity[m, n] = intensity

        not_on_fire = np.ones((world_size, world_size))

        for i in range(world_size):
            for j in range(world_size):
                if field_intensity[i, j] > 0:
                    not_on_fire[i, j] = 0

        on_fire_indices = np.nonzero(
            field_intensity
        )  # Get indices of points that are on fire

        for i in range(not_on_fire.shape[0]):
            for j in range(not_on_fire.shape[1]):
                if not_on_fire[i, j] == 1:  # If the point is not on fire
                    neighboring_points = []
                    for k in range(len(on_fire_indices[0])):
                        x_, y_ = on_fire_indices[0][k], on_fire_indices[1][k]
                        distance_squared = (x_ - i) ** 2 + (y_ - j) ** 2
                        if distance_squared <= 2:
                            neighboring_points.append(field_intensity[x_, y_])

                    if len(neighboring_points) != 0:
                        field_intensity[i, j] = np.mean(np.array(neighboring_points))

        return field_intensity

    def get_ground_truth(self, scale=1):
        scale = self.underlying_distribution.world_size

        ground_truth = Fire.calc_fire_intensity_in_field(
            self.underlying_distribution.fire_map,
            self.underlying_distribution.world_size,
        )
        # ground_truth = self.underlying_distribution.fire_map[:, 2]
        ground_truth = Utils.compress_and_average(
            array=ground_truth, new_shape=(30, 30)
        )
        ground_truth = ground_truth.reshape(-1)

        return ground_truth / ground_truth.max()

    def set_ground_truth(self, fire_map):
        # print(">>> Max fire intensity: ", np.max(fire_map))
        # time1 = time.time()
        ground_truth = Utils.compress_and_average(array=fire_map, new_shape=(30, 30))
        # time2 = time.time()
        # print(f">>> Time to compress and average: {time2 - time1:.4f}")
        # print(">>> Max fire intensity after compression: ", np.max(ground_truth))
        self.ground_truth = ground_truth.reshape(-1)
        del ground_truth

    def set_momentum_GT(self, fire_map):
        ground_truth = Utils.compress_and_average(array=fire_map, new_shape=(30, 30))
        # new GT = 0.9 * old GT + 0.1 * new GT
        self.ground_truth = 0.9 * self.ground_truth + 0.1 * ground_truth.reshape(-1)
        del ground_truth

    def get_high_info_idx(self):
        # high_info_idx = []
        # idx = np.argwhere(self.ground_truth > self.ADAPTIVE_TH)
        # high_info_idx += [idx.squeeze(1)]
        # return high_info_idx
        # return entire grid
        return [np.arange(900)]

    def plot(
        self, route, n, step, path, testID=0, CMAES_route=False, sampling_path=False
    ):
        # Plotting shorest path
        plt.switch_backend("agg")
        self.gp_wrapper.plot(self.ground_truth, curr_t=self.curr_t)
        # plt.subplot(1,3,1)
        # plt.scatter(self.start[:,0], self.start[:,1], c='r', s=15)
        # plt.scatter(self.destination[:,0], self.destination[:,1], c='r', s=15)
        if CMAES_route:
            pointsToDisplay = route
        else:
            pointsToDisplay = [(self.prm.findPointsFromNode(path)) for path in route]
        x = [item[0] for item in pointsToDisplay]
        y = [item[1] for item in pointsToDisplay]
        for i in range(len(x) - 1):
            plt.plot(
                x[i : i + 2],
                y[i : i + 2],
                c="black",
                linewidth=4,
                zorder=5,
                alpha=0.25 + 0.6 * i / len(x),
            )
        if sampling_path:
            pointsToDisplay2 = [
                (self.prm.findPointsFromNode(path)) for path in sampling_path
            ]
            x0 = [item[0] for item in pointsToDisplay2]
            y0 = [item[1] for item in pointsToDisplay2]
            x1 = [item[0] for item in pointsToDisplay2[:3]]
            y1 = [item[1] for item in pointsToDisplay2[:3]]
            for i in range(len(x0) - 1):
                plt.plot(
                    x0[i : i + 2],
                    y0[i : i + 2],
                    c="white",
                    linewidth=4,
                    zorder=5,
                    alpha=1 - 0.2 * i / len(x0),
                )
            for i in range(len(x1) - 1):
                plt.plot(x1[i : i + 2], y1[i : i + 2], c="red", linewidth=4, zorder=6)

        # plt.subplot(2, 2, 4)
        # plt.title("High interest area")
        # high_area = self.gp_wrapper.get_high_info_area(
        #     curr_t=self.curr_t, adaptive_t=self.ADAPTIVE_TH
        # )
        # try:
        #     xh = high_area[0][:, 0]
        #     yh = high_area[0][:, 1]
        # except:
        #     # print("No high info area")
        #     xh, yh = [], []
        # plt.hist2d(
        #     xh, yh, bins=30, range=[[0, 1], [0, 1]], vmin=0, vmax=1, rasterized=True
        # )
        # # plt.scatter(self.start[:,0], self.start[:,1], c='r', s=15)
        # # plt.scatter(self.destination[:,0], self.destination[:,1], c='r', s=15)

        # # x = [item[0] for item in pointsToDisplay]
        # # y = [item[1] for item in pointsToDisplay]

        # for i in range(len(x) - 1):
        #     plt.plot(
        #         x[i : i + 2],
        #         y[i : i + 2],
        #         c="black",
        #         linewidth=4,
        #         zorder=5,
        #         alpha=0.25 + 0.6 * i / len(x),
        #     )
        plt.suptitle(
            "Budget: {:.4g}/{:.4g},  RMSE: {:.4g}".format(
                self.budget, self.budget0, self.RMSE
            )
        )
        plt.tight_layout()
        plt.savefig(
            "{}/{}_{}_{}_samples.png".format(path, n, testID, step, self.sample_size),
            dpi=150,
        )
        # plt.show()
        frame = "{}/{}_{}_{}_samples.png".format(
            path, n, testID, step, self.sample_size
        )
        self.frame_files.append(frame)

        plt.close()

    def route_step(self, route, sample_length, measurement=True):
        current_node = route[0]
        # print(route, route[1:], route[0])
        for next_node in route[1:]:
            dist = np.linalg.norm(current_node - next_node)
            remain_length = dist
            next_length = sample_length - self.dist_residual
            no_sample = True
            while remain_length > next_length:
                if no_sample:
                    self.sample = (
                        next_node - current_node
                    ) * next_length / dist + current_node
                else:
                    self.sample = (
                        next_node - current_node
                    ) * next_length / dist + self.sample
                observed_value = self.underlying_distribution.return_fire_at_location(
                    self.sample.reshape(-1, 2)[0]
                )  # + np.random.normal(0, 1e-10)
                self.curr_t += next_length
                for i in range(self.n_agents):
                    # print('inside loop adding obs points', self.sample)
                    # print('self.sample: ', self.sample)
                    # print('reshape: ', self.sample.reshape(-1, 2))
                    self.gp_wrapper.GPs[i].add_observed_point(
                        add_t(self.sample.reshape(-1, 2), self.curr_t), observed_value
                    )
                remain_length -= next_length
                next_length = sample_length
                no_sample = False

            self.dist_residual = (
                self.dist_residual + remain_length if no_sample else remain_length
            )
            self.dist_residual_tmp = self.dist_residual
            if measurement:
                self.budget -= dist
            current_node = next_node

        self.gp_wrapper.update_gps()

        if measurement:
            # self.high_info_area = (
            #     self.gp_wrapper.get_high_info_area(self.curr_t, adaptive_t=self.ADAPTIVE_TH) if self.ADAPTIVE_AREA else None
            # )
            cov_trace = self.gp_wrapper.eval_avg_cov_trace(
                self.curr_t, self.get_high_info_idx()
            )
            self.cov_trace = cov_trace
        else:
            cov_trace = self.gp_wrapper.eval_avg_cov_trace(
                self.curr_t, self.get_high_info_idx()
            )

        rmse = self.gp_wrapper.eval_avg_RMSE(self.ground_truth, self.curr_t)

        return cov_trace, rmse


if __name__ == "__main__":
    env = Env(sample_size=200, budget_range=(7.999, 8), save_image=True)
    nodes, graph, info, std, budget = env.reset()
    print(nodes)
    print(graph)
