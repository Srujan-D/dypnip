import copy
import os

import imageio
import numpy as np
import torch

from ...environments.env_fire import Env

# from attention_net import AttentionNet
from ...neural_networks.attention_net import AttentionNet
from ...neural_networks.dynamics_pred_net import PredictNextBelief

from ...parameters.params_robust_attention_SARL import *
import scipy.signal as signal

import time


def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Worker:
    def __init__(
        self,
        metaAgentID,
        localNetwork,
        global_step,
        budget_range,
        sample_size=SAMPLE_SIZE,
        sample_length=None,
        # device=f"cuda:{CUDA_DEVICE[0]}",
        device="cuda",
        greedy=False,
        save_image=False,
        belief_predictor=PredictNextBelief(),
    ):

        self.device = device
        self.greedy = greedy
        self.metaAgentID = metaAgentID
        self.global_step = global_step
        self.save_image = save_image
        self.sample_length = sample_length
        self.sample_size = sample_size
        self.n_agents = 1

        # index = fuel_map_ablation[np.sum(global_step >= index_map)]

        # # if index_map >= len(fuel_map_episode):
        # #     index_map = len(fuel_map_episode) - 1

        # print(">>> fuel map episode is ", global_step, index)
        # # after every 1000 episodes, the fuel map is changed
        # # if global_step % 1000 == 0:
        # self.env = Env(
        #     sample_size=self.sample_size,
        #     k_size=K_SIZE,
        #     budget_range=budget_range,
        #     save_image=self.save_image,
        #     adaptive_th=ADAPTIVE_TH,
        #     adaptive_area=ADAPTIVE_AREA,
        #     n_agents=self.n_agents,
        #     fuel=index,
        # )

        self.env = Env(
            sample_size=self.sample_size,
            k_size=K_SIZE,
            budget_range=budget_range,
            save_image=self.save_image,
            adaptive_th=ADAPTIVE_TH,
            adaptive_area=ADAPTIVE_AREA,
            n_agents=self.n_agents,
        )

        # self.local_net = AttentionNet(2, 128, device=self.device)
        # self.local_net.to(device)
        self.local_net = localNetwork
        self.experience = None

        self.belief_predictor = belief_predictor

        self.episode_buffer_keys = [
            "node_inputs",
            "edge_inputs",
            "current_index",
            "action_index",
            "value",
            "reward",
            "value_prime",
            "target_v",
            "budget_inputs",
            "LSTM_h",
            "LSTM_c",
            "mask",
            "pos_encoding",
            "pred_next_belief",
            "KL_diff_beliefs",
            "belief_lstm_h",
            "belief_lstm_c",
            "next_policy_feature",
            "env_sim_params",
        ]

    def run_episode(self, currEpisode):
        # episode_buffer = []
        episode_buffer = {k: [] for k in self.episode_buffer_keys}

        perf_metrics = dict()
        # for i in range(13):
        #     episode_buffer.append([])

        done = False
        node_coords, graph, node_feature, budget = self.env.reset()
        # get node_pred, node_std, node_pred_future, node_std_futre from node_feature
        # node_feature = node_feature[:, self.n_agents].reshape(-1, self.n_agents, node_feature.shape[1])
        # node_info, node_std = node_feature[:, :, 0], node_feature[:, :, 1]

        node_info, node_info_future = node_feature[:, :2], node_feature[:, 2:]
        node_pred, node_std = node_info[:, 0], node_info[:, 1]
        node_info = node_pred

        # get the current mean of the grid
        env_grid_mean0, env_grid_std0 = self.env.gp_wrapper.return_grid()

        n_nodes = node_coords.shape[0]
        node_info_inputs = node_info.reshape((n_nodes, 1))
        node_std_inputs = node_std.reshape((n_nodes, 1))
        budget_inputs = self.calc_estimate_budget(budget, current_idx=1)
        node_inputs = np.concatenate(
            (node_coords, node_info_inputs, node_std_inputs), axis=1
        )
        node_inputs = (
            torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
        )  # (1, sample_size+2, 4)
        budget_inputs = (
            torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device)
        )  # (1, sample_size+2, 1)

        graph = list(graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            if len(node_edges) >= 4:
                edge_inputs.append(node_edges)

        pos_encoding = self.calculate_position_embedding(edge_inputs)
        pos_encoding = (
            torch.from_numpy(pos_encoding).float().unsqueeze(0).to(self.device)
        )  # (1, sample_size+2, 32)

        edge_inputs = (
            torch.tensor(edge_inputs).unsqueeze(0).to(self.device)
        )  # (1, sample_size+2, k_size)

        current_index = (
            torch.tensor([self.env.current_node_index])
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )  # (1,1,1)
        route = [current_index.item()]

        LSTM_h = torch.zeros((1, 1, EMBEDDING_DIM)).to(self.device)
        LSTM_c = torch.zeros((1, 1, EMBEDDING_DIM)).to(self.device)

        belief_lstm_h = torch.zeros((1, 1, BELIEF_EMBEDDING_DIM)).to(self.device)
        belief_lstm_c = torch.zeros((1, 1, BELIEF_EMBEDDING_DIM)).to(self.device)

        mask = torch.zeros((1, self.sample_size + 2, K_SIZE), dtype=torch.int64).to(
            self.device
        )

        # perf metrics lists
        rmse_list = [self.env.RMSE]
        unc_list = [self.env.unc_list]
        jsd_list = [self.env.JS_list]
        kld_list = [self.env.KL_list]
        unc_stddev_list = [np.std(self.env.unc_list)]
        jsd_stddev_list = [np.std(self.env.JS_list)]
        budget_list = [0]

        for i in range(256):
            episode_buffer["LSTM_h"] += LSTM_h
            episode_buffer["LSTM_c"] += LSTM_c
            episode_buffer["mask"] += mask
            episode_buffer["pos_encoding"] += pos_encoding
            episode_buffer["belief_lstm_h"] += belief_lstm_h
            episode_buffer["belief_lstm_c"] += belief_lstm_c

            with torch.no_grad():
                # print('node input size is ', node_inputs.size())
                # print('edge input size is ', edge_inputs.size())
                # print('budget input size is ', budget_inputs.size())
                # print('pos_encoding size is ', pos_encoding.size())
                # print('mask size is ', mask.size())
                # quit()
                # import pdb
                # pdb.set_trace()
                # print("bud get : ", budget_inputs)

                # env_grid_0 = self.env.gp_wrapper.return_grid()
                # change shape to 1, 30, 30
                # env_grid_mean0 = torch.Tensor(env_grid_mean0).unsqueeze(0).to(self.device)
                # env_grid_std0 = torch.Tensor(env_grid_std0).unsqueeze(0).to(self.device)

                next_belief, belief_lstm_h, belief_lstm_c = self.belief_predictor(
                    torch.Tensor(env_grid_mean0).unsqueeze(0).to(self.device),
                    belief_lstm_h,
                    belief_lstm_c,
                )
                episode_buffer["pred_next_belief"] += [next_belief]
                next_policy_feature = self.belief_predictor.return_policy_feature()
                episode_buffer["next_policy_feature"] += [next_policy_feature]

                episode_buffer["env_sim_params"] += [
                    torch.Tensor(self.env.env_sim_params)
                ]

                logp_list, value, LSTM_h, LSTM_c = self.local_net(
                    node_inputs,
                    edge_inputs,
                    budget_inputs,
                    current_index,
                    LSTM_h,
                    LSTM_c,
                    pos_encoding,
                    mask,
                    i,
                    next_belief=next_policy_feature,
                )
            # next_node (1), logp_list (1, 10), value (1,1,1)
            if self.greedy:
                action_index = torch.argmax(logp_list, dim=1).long()
            else:
                action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

            episode_buffer["node_inputs"] += node_inputs
            episode_buffer["edge_inputs"] += edge_inputs
            episode_buffer["current_index"] += current_index
            episode_buffer["action_index"] += action_index.unsqueeze(0).unsqueeze(0)
            episode_buffer["value"] += value
            episode_buffer["budget_inputs"] += budget_inputs

            next_node_index = edge_inputs[:, current_index.item(), action_index.item()]
            route.append(next_node_index.item())
            # reward, done, node_info, node_std, remain_budget = self.env.step(next_node_index.item(), self.sample_length)
            # time1 = time.time()
            reward, done, node_feature, remain_budget = self.env.step(
                next_node_index.item(), self.sample_length
            )
            # time2 = time.time()
            # print(f"--------------------> time for step is {time2 - time1:.4f}")

            # # # get node_pred, node_std, node_pred_future, node_std_future from node_feature
            # node_feature = node_feature[:, self.n_agents].reshape(-1, self.n_agents, node_feature.shape[1])
            # node_info, node_std = node_feature[:, :, 0], node_feature[:, :, 1]

            node_info, node_info_future = node_feature[:, :2], node_feature[:, 2:]

            env_grid_mean1, env_grid_std1 = self.env.gp_wrapper.return_grid()
            # env_grid_mean1 = torch.Tensor(env_grid_mean1).unsqueeze(0).to(self.device)
            # env_grid_std1 = torch.Tensor(env_grid_std1).unsqueeze(0).to(self.device)

            episode_buffer["KL_diff_beliefs"] += [
                self.find_KL_GP(
                    env_grid_mean0, env_grid_std0, env_grid_mean1, env_grid_std1
                )
            ]
            env_grid_mean0, env_grid_std0 = env_grid_mean1, env_grid_std1
            # episode_buffer["KL_diff_beliefs"] += self.find_KL_GP(
            #     node_pred, node_std, node_info[:, 0], node_info[:, 1]
            # )
            # episode_buffer["KL_diff_beliefs"] += (node_info[:, 0] - node_pred)
            # print(">>> KL diff next belief = ", episode_buffer["KL_diff_beliefs"])

            node_pred, node_std = node_info[:, 0], node_info[:, 1]
            node_info = node_pred

            rmse_list += [self.env.RMSE]
            unc_list += [self.env.unc_list]
            jsd_list += [self.env.JS_list]
            kld_list += [self.env.KL_list]
            unc_stddev_list += [np.std(self.env.unc_list)]
            jsd_stddev_list += [np.std(self.env.JS_list)]
            budget_list += [self.env.budget_init - remain_budget]

            # if (not done and i==127):
            # reward += -np.linalg.norm(self.env.node_coords[self.env.current_node_index,:]-self.env.node_coords[0,:])

            episode_buffer["reward"] += torch.FloatTensor([[[reward]]]).to(self.device)

            current_index = next_node_index.unsqueeze(0).unsqueeze(0)
            node_info_inputs = node_info.reshape(n_nodes, 1)
            node_std_inputs = node_std.reshape(n_nodes, 1)
            budget_inputs = self.calc_estimate_budget(
                remain_budget, current_idx=current_index.item()
            )
            node_inputs = np.concatenate(
                (node_coords, node_info_inputs, node_std_inputs), axis=1
            )
            node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
            budget_inputs = (
                torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device)
            )
            # current_edge = torch.gather(
            #     edge_inputs, 1, current_index.repeat(1, 1, edge_inputs.size()[2])
            # ).permute(0, 2, 1)
            # connected_nodes_budget = torch.gather(budget_inputs, 1, current_edge)
            # if all(connected_nodes_budget.squeeze(0).squeeze(1)[1:] <= 0):
            #     print("================Overbudget!")
            #     print("remain_budget", remain_budget)
            #     print("current_index", current_index)
            # else:
            #     print("------>connected_nodes_budget", connected_nodes_budget.permute(0, 2, 1), remain_budget)

            # mask last five node
            mask = torch.zeros((1, self.sample_size + 2, K_SIZE), dtype=torch.int64).to(
                self.device
            )
            # connected_nodes = edge_inputs[0, current_index.item()]
            # current_edge = torch.gather(edge_inputs, 1, current_index.repeat(1, 1, K_SIZE))
            # current_edge = current_edge.permute(0, 2, 1)
            # connected_nodes_budget = torch.gather(budget_inputs, 1, current_edge) # (1, k_size, 1)
            # n_available_node = sum(int(x>0) for x in connected_nodes_budget.squeeze(0))
            # if n_available_node > 5:
            #    for j, node in enumerate(connected_nodes.squeeze(0)):
            #        if node.item() in route[-2:]:
            #            mask[0, route[-1], j] = 1

            # save a frame
            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot(route, self.global_step, i, gifs_path)

            if done:
                print("done with current node index ", self.env.current_node_index)
                episode_buffer["value_prime"] = episode_buffer["value"][1:]
                episode_buffer["value_prime"].append(
                    torch.FloatTensor([[0]]).to(self.device)
                )
                if self.env.current_node_index == 0:

                    budget_list = [0]
                    perf_metrics["success_rate"] = True
                    print("{} Goodbye world! We did it!".format(i))
                else:
                    budget_list = [np.nan]
                    perf_metrics["success_rate"] = False
                    # print("{} Overbudget!".format(i))
                # self.env.fire.env_close()
                rmse_list = [self.env.RMSE]
                jsd_list = [self.env.JS_list]
                kld_list = [self.env.KL_list]
                unc_list = [self.env.unc_list]
                perf_metrics["avgrmse"] = np.mean(rmse_list)
                perf_metrics["avgunc"] = np.mean(unc_list)
                perf_metrics["avgjsd"] = np.mean(jsd_list)
                perf_metrics["avgkld"] = np.mean(kld_list)
                perf_metrics["stdunc"] = np.mean(unc_stddev_list)
                perf_metrics["stdjsd"] = np.mean(jsd_stddev_list)
                perf_metrics["f1"] = self.env.gp_wrapper.eval_avg_F1(
                    self.env.ground_truth, self.env.curr_t
                )
                perf_metrics["mi"] = self.env.gp_wrapper.eval_avg_MI(self.env.curr_t)
                # perf_metrics['covtr'] = self.env.cov_trace
                perf_metrics["js"] = self.env.JS
                perf_metrics["rmse"] = self.env.RMSE
                perf_metrics["scalex"] = self.env.gp_wrapper.GPs[0].kernel.length_scale[
                    0
                ]  # 0.1  # self.env.GPs.gp.kernel_.length_scale[0]
                perf_metrics["scaley"] = self.env.gp_wrapper.GPs[0].kernel.length_scale[
                    1
                ]  # 0.1  # self.env.GPs.gp.kernel_.length_scale[0]
                perf_metrics["scalet"] = self.env.gp_wrapper.GPs[0].kernel.length_scale[
                    2
                ]  # 3  # scale_t

                perf_metrics["cov_trace"] = self.env.cov_trace
                break
        if not done:
            episode_buffer["value_prime"] = episode_buffer["value"][1:]
            with torch.no_grad():
                _, value, LSTM_h, LSTM_c = self.local_net(
                    node_inputs,
                    edge_inputs,
                    budget_inputs,
                    current_index,
                    LSTM_h,
                    LSTM_c,
                    pos_encoding,
                    mask,
                    next_belief=next_policy_feature,
                )
            episode_buffer["value_prime"].append(value.squeeze(0))
            perf_metrics["remain_budget"] = remain_budget / budget
            perf_metrics["avgrmse"] = np.mean(rmse_list)
            perf_metrics["avgunc"] = np.mean(unc_list)
            perf_metrics["avgjsd"] = np.mean(jsd_list)
            perf_metrics["avgkld"] = np.mean(kld_list)
            perf_metrics["stdunc"] = np.mean(unc_stddev_list)
            perf_metrics["stdjsd"] = np.mean(jsd_stddev_list)
            perf_metrics["f1"] = self.env.gp_wrapper.eval_avg_F1(
                self.env.ground_truth, self.env.curr_t
            )
            perf_metrics["mi"] = self.env.gp_wrapper.eval_avg_MI(self.env.curr_t)
            perf_metrics["cov_trace"] = self.env.cov_trace
            perf_metrics["js"] = self.env.JS
            perf_metrics["rmse"] = self.env.RMSE
            perf_metrics["success_rate"] = False
            perf_metrics["scalex"] = self.env.gp_wrapper.GPs[0].kernel.length_scale[
                0
            ]  # 0.1  # self.env.GPs.gp.kernel_.length_scale[0]
            perf_metrics["scaley"] = self.env.gp_wrapper.GPs[0].kernel.length_scale[
                1
            ]  # 0.1  # self.env.GPs.gp.kernel_.length_scale[0]
            perf_metrics["scalet"] = self.env.gp_wrapper.GPs[0].kernel.length_scale[
                2
            ]  # 3  # scale_t

        print("route is ", route)
        reward = copy.deepcopy(episode_buffer["reward"])
        reward.append(episode_buffer["value_prime"][-1])
        for i in range(len(reward)):
            reward[i] = reward[i].cpu().numpy()
        reward_plus = np.array(reward, dtype=object).reshape(-1)
        discounted_rewards = discount(reward_plus, GAMMA)[:-1]
        discounted_rewards = discounted_rewards.tolist()
        target_v = (
            torch.FloatTensor(discounted_rewards)
            .unsqueeze(1)
            .unsqueeze(1)
            .to(self.device)
        )

        for i in range(target_v.size()[0]):
            episode_buffer["target_v"].append(target_v[i, :, :])

        # save gif
        if self.save_image:
            if self.greedy:
                from test_driver import result_path as path
            else:
                path = gifs_path
            self.make_gif(path, currEpisode)

        self.experience = episode_buffer
        return perf_metrics

    def work(self, currEpisode):
        """
        Interacts with the environment. The agent gets either gradients or experience buffer
        """
        self.currEpisode = currEpisode
        self.perf_metrics = self.run_episode(currEpisode)

    def calc_estimate_budget(self, budget, current_idx):
        all_budget = []
        current_coord = self.env.node_coords[current_idx]
        end_coord = self.env.node_coords[0]
        for i, point_coord in enumerate(self.env.node_coords):
            dist_current2point = self.env.prm.calcDistance(current_coord, point_coord)
            dist_point2end = self.env.prm.calcDistance(point_coord, end_coord)
            estimate_budget = (budget - dist_current2point - dist_point2end) / 10
            # estimate_budget = (budget - dist_current2point - dist_point2end) / budget
            all_budget.append(estimate_budget)
        return np.asarray(all_budget).reshape(i + 1, 1)

    def calculate_position_embedding(self, edge_inputs):
        A_matrix = np.zeros((self.sample_size + 2, self.sample_size + 2))
        D_matrix = np.zeros((self.sample_size + 2, self.sample_size + 2))
        for i in range(self.sample_size + 2):
            for j in range(self.sample_size + 2):
                if j in edge_inputs[i] and i != j:
                    A_matrix[i][j] = 1.0
        for i in range(self.sample_size + 2):
            D_matrix[i][i] = 1 / np.sqrt(len(edge_inputs[i]) - 1)
        L = np.eye(self.sample_size + 2) - np.matmul(D_matrix, A_matrix, D_matrix)
        eigen_values, eigen_vector = np.linalg.eig(L)
        idx = eigen_values.argsort()
        eigen_values, eigen_vector = eigen_values[idx], np.real(eigen_vector[:, idx])
        eigen_vector = eigen_vector[:, 1 : 32 + 1]
        return eigen_vector

    def make_gif(self, path, n):
        with imageio.get_writer(
            "{}/{}_cov_trace_{:.4g}.gif".format(path, n, self.env.cov_trace),
            mode="I",
            duration=2.5,
        ) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print("++++++++++++++++++++++++++++++++++++++++ gif complete\n")

        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)

    def find_KL_GP(self, mu1, K1, mu2, K2):
        """
        find KL divergence between two Gaussian processes
        """
        # try:
        #     if np.linalg.det(K1) == 0 or np.linalg.det(K2) == 0:
        #         return 1e6
        #     else:
        #         print("k2/k1", np.linalg.det(K2) / np.linalg.det(K1))
        #         return 0.5 * (
        #             np.log(np.linalg.det(K2) / np.linalg.det(K1))
        #             - mu1.shape[0]
        #             + np.trace(np.matmul(np.linalg.inv(K2), K1))
        #             + np.matmul(
        #                 np.matmul(np.transpose(mu2 - mu1), np.linalg.inv(K2)), mu2 - mu1
        #             )
        #         )
        # except:
        #     # in case coovariance has invalid entries, just return dist(mu1, mu2)
        # print("mu1 - mu2 shape is ", (mu1 - mu2).shape)

        ## predicting difference in beliefs
        # return torch.Tensor(mu1 - mu2).to(self.device)

        ## predicting the next belief
        return torch.Tensor(mu2).to(self.device)
    
        # ## reconstructing the belief
        # return torch.Tensor(mu1).to(self.device)


if __name__ == "__main__":
    device = torch.device(f"cuda:{CUDA_DEVICE[0]}")
    localNetwork = AttentionNet(INPUT_DIM, EMBEDDING_DIM).cuda()
    worker = Worker(
        1, localNetwork, 0, budget_range=(4, 6), save_image=False, sample_length=0.05
    )
    worker.run_episode(0)
