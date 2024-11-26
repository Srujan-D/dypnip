import copy

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast

# from attention_net import AttentionNet
from ...neural_networks.attention_net import AttentionNet
from ...neural_networks.dynamics_pred_net import PredictNextBelief

# from runner import RLRunner
from ..runner.runner_SARL import RLRunner
from ...parameters.params_robust_attention_SARL import *
import wandb
import pprint

ray.init(num_cpus=NUM_META_AGENT)
print("Welcome to PRM-AN!")

writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

global_step = None


def writeToTensorBoard(writer, tensorboardData, curr_episode, plotMeans=True):
    # each row in tensorboardData represents an episode
    # each column is a specific metric

    if plotMeans == True:
        tensorboardData = np.array(tensorboardData)
        tensorboardData = list(np.nanmean(tensorboardData, axis=0))
        metric_name = [
            "avgrmse",
            "avgunc",
            "avgjsd",
            "avgkld",
            "stdunc",
            "stdjsd",
            "cov_trace",
            "f1",
            "mi",
            "js",
            "rmse",
            # "scalex",
            # "scaley",
            # "scalet",
            "success_rate",
        ]
        try:
            # print("number of lists in tensorboardData:", len(tensorboardData))
            # for i, n in enumerate(metric_name):
            #     print(f"list of metrics: {i} of length {len(n)}")
            (
                reward,
                value,
                policyLoss,
                valueLoss,
                entropy,
                gradNorm,
                returns,
                # remain_budget,
                belief_loss,
                avg_rmse,
                avg_unc,
                avg_jsd,
                avg_kld,
                std_unc,
                std_jsd,
                cov_tr,
                F1,
                MI,
                JS,
                rmse,
                scalex,
                scaley,
                scalet,
                success_rate,
            ) = tensorboardData
        except:
            print("number of lists in tensorboardData:", len(tensorboardData))
            for i, n in enumerate(metric_name):
                print(f"list of metrics: {i} of length {len(n)}")
            (
                reward,
                value,
                policyLoss,
                valueLoss,
                entropy,
                gradNorm,
                returns,
                # remain_budget,
                belief_loss,
                avg_rmse,
                avg_unc,
                avg_jsd,
                avg_kld,
                std_unc,
                std_jsd,
                cov_tr,
                F1,
                MI,
                JS,
                rmse,
                scalex,
                scaley,
                scalet,
                success_rate,
            ) = tensorboardData
    else:
        (
            reward,
            value,
            policyLoss,
            valueLoss,
            entropy,
            gradNorm,
            returns,
            # remain_budget,
            belief_loss,
            avg_rmse,
            avg_unc,
            avg_jsd,
            avg_kld,
            std_unc,
            std_jsd,
            cov_tr,
            F1,
            MI,
            JS,
            rmse,
            scalex,
            scaley,
            scalet,
            success_rate,
        ) = tensorboardData
    if use_wandb:
        wandb.log({"Losses/Value": value}, step=curr_episode)
        wandb.log({"Losses/Policy Loss": policyLoss}, step=curr_episode)
        wandb.log({"Losses/Value Loss": valueLoss}, step=curr_episode)
        wandb.log({"Losses/Entropy": entropy}, step=curr_episode)
        wandb.log({"Losses/Grad Norm": gradNorm}, step=curr_episode)
        wandb.log({"Losses/Belief Loss": belief_loss}, step=curr_episode)
        wandb.log({"Perf/Reward": reward}, step=curr_episode)
        wandb.log({"Perf/Returns": returns}, step=curr_episode)
        wandb.log({"Perf/Success Rate": success_rate}, step=curr_episode)
        wandb.log({"Perf/Avg RMSE": avg_rmse}, step=curr_episode)
        wandb.log({"Perf/Avg Uncertainty": avg_unc}, step=curr_episode)
        wandb.log({"Perf/Avg JSD": avg_jsd}, step=curr_episode)
        wandb.log({"Perf/Avg KLD": avg_kld}, step=curr_episode)
        wandb.log({"Perf/Std Uncertainty": std_unc}, step=curr_episode)
        wandb.log({"Perf/Std JSD": std_jsd}, step=curr_episode)
        wandb.log({"Perf/JS": JS}, step=curr_episode)
        wandb.log({"Perf/RMSE": rmse}, step=curr_episode)
        wandb.log({"Perf/F1 Score": F1}, step=curr_episode)
        wandb.log({"GP/MI": MI}, step=curr_episode)
        wandb.log({"GP/Cov Trace": cov_tr}, step=curr_episode)


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(torch.cuda.device_count())
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE[0])
    # torch.cuda.set_device(CUDA_DEVICE[0])
    pprint.pprint(ray.cluster_resources())
    pprint.pprint(os.environ["CUDA_VISIBLE_DEVICES"])

    device = (
        # torch.device(f"cuda:{CUDA_DEVICE[0]}")
        torch.device("cuda")
        if USE_GPU_GLOBAL
        else torch.device("cpu")
    )
    local_device = (
        # torch.device(f"cuda:{CUDA_DEVICE[0]}") if USE_GPU else torch.device("cpu")
        torch.device("cuda")
        if USE_GPU
        else torch.device("cpu")
    )

    global_network = AttentionNet(INPUT_DIM, EMBEDDING_DIM, device).to(device)
    # global_network.share_memory()
    global_optimizer = optim.Adam(global_network.parameters(), lr=LR)
    lr_decay = optim.lr_scheduler.StepLR(
        global_optimizer, step_size=DECAY_STEP, gamma=0.96
    )

    belief_predictor = PredictNextBelief(device).to(device)
    belief_optimizer = optim.Adam(belief_predictor.parameters(), lr=LR)
    belief_lr_decay = optim.lr_scheduler.StepLR(
        belief_optimizer, step_size=DECAY_STEP, gamma=0.96
    )
    # Automatically logs gradients of pytorch model
    # wandb.watch(global_network, log_freq = SUMMARY_WINDOW)
    if use_wandb:
        wandb.init(name=FOLDER_NAME, project="st_catnipp")

    best_perf = 900
    curr_episode = 0
    if LOAD_MODEL:
        print("Loading Model...")
        checkpoint = torch.load(pretrain_model_path + "/checkpoint.pth")
        global_network.load_state_dict(checkpoint["model"])
        global_optimizer.load_state_dict(checkpoint["optimizer"])
        lr_decay.load_state_dict(checkpoint["lr_decay"])
        # curr_episode = checkpoint["episode"]
        print("curr_episode set to ", curr_episode)

        best_model_checkpoint = torch.load(pretrain_model_path + "/best_model_checkpoint.pth")
        best_perf = best_model_checkpoint["best_perf"]
        print("best performance so far:", best_perf)
        print(global_optimizer.state_dict()["param_groups"][0]["lr"])

        belief_checkpoint = torch.load(pretrain_model_path + "/belief_checkpoint.pth")
        belief_predictor.load_state_dict(belief_checkpoint["model"])
        belief_optimizer.load_state_dict(belief_checkpoint["optimizer"])
        belief_lr_decay.load_state_dict(belief_checkpoint["lr_decay"])

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # get initial weigths
    if device != local_device:
        weights = global_network.to(local_device).state_dict()
        global_network.to(device)

        belief_weights = belief_predictor.to(local_device).state_dict()
        belief_predictor.to(device)
    else:
        weights = global_network.state_dict()
        belief_weights = belief_predictor.state_dict()

    # breakpoint()
    # launch the first job on each runner
    dp_model = nn.DataParallel(global_network)
    # belief_model = nn.DataParallel(belief_predictor)

    jobList = []
    sample_size = np.random.randint(200, 400)
    for i, meta_agent in enumerate(meta_agents):
        jobList.append(
            meta_agent.job.remote(
                weights, curr_episode, BUDGET_RANGE, sample_size, SAMPLE_LENGTH, belief_predictor_weights=belief_weights,
            )
        )
        curr_episode += 1
    # metric_name = ['remain_budget', 'success_rate', 'RMSE', 'delta_cov_trace', 'MI', 'F1Score', 'cov_trace']

    metric_name = [
        "avgrmse",
        "avgunc",
        "avgjsd",
        "avgkld",
        "stdunc",
        "stdjsd",
        "cov_trace",
        "f1",
        "mi",
        "js",
        "rmse",
        "scalex",
        "scaley",
        "scalet",
        "success_rate",
    ]

    rollouts_keys = [
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
    ]
    tensorboardData = []
    trainingData = []
    experience_buffer = []
    # for i in range(13):
    #     experience_buffer.append([])
    experience_buffer = {k: [] for k in rollouts_keys}

    try:
        while True:
            # wait for any job to be completed
            done_id, jobList = ray.wait(jobList, num_returns=NUM_META_AGENT)
            # get the results
            # jobResults, metrics, info = ray.get(done_id)[0]
            done_jobs = ray.get(done_id)
            random.shuffle(done_jobs)
            # done_jobs = list(reversed(done_jobs))
            perf_metrics = {}
            for n in metric_name:
                perf_metrics[n] = []
            for job in done_jobs:
                jobResults, metrics, info = job
                for key in rollouts_keys:
                    experience_buffer[key].extend(jobResults[key])
                for n in metric_name:
                    # print("================metrics consist of:", metrics.keys())
                    perf_metrics[n].append(metrics[n])

            if (
                np.mean(perf_metrics["cov_trace"]) < best_perf
                and curr_episode % 32 == 0
            ):
                best_perf = np.mean(perf_metrics["cov_trace"])
                print("Saving best model", end="\n")
                checkpoint = {
                    "model": global_network.state_dict(),
                    "optimizer": global_optimizer.state_dict(),
                    "episode": curr_episode,
                    "lr_decay": lr_decay.state_dict(),
                    "best_perf": best_perf,
                }
                path_checkpoint = "./" + model_path + "/best_model_checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print("Saved model", end="\n")

                belief_checkpoint = {
                    "model": belief_predictor.state_dict(),
                    "optimizer": global_optimizer.state_dict(),
                    "episode": curr_episode,
                    "lr_decay": lr_decay.state_dict(),
                }
                path_belief_checkpoint = "./" + model_path + "/belief_checkpoint.pth"
                torch.save(belief_checkpoint, path_belief_checkpoint)

            update_done = False
            while len(experience_buffer["node_inputs"]) >= BATCH_SIZE:
                # print("Training on experience buffer")
                rollouts = copy.deepcopy(experience_buffer)
                for key in rollouts_keys:
                    rollouts[key] = rollouts[key][:BATCH_SIZE]
                for i in rollouts_keys:
                    experience_buffer[i] = experience_buffer[i][BATCH_SIZE:]
                if len(experience_buffer["node_inputs"]) < BATCH_SIZE:
                    update_done = True
                if update_done:
                    experience_buffer = {k: [] for k in rollouts_keys}
                    sample_size = np.random.randint(200, 400)

                node_inputs_batch = torch.stack(
                    rollouts["node_inputs"], dim=0
                )  # (batch,sample_size+2,2)
                edge_inputs_batch = torch.stack(
                    rollouts["edge_inputs"], dim=0
                )  # (batch,sample_size+2,k_size)
                current_inputs_batch = torch.stack(
                    rollouts["current_index"], dim=0
                )  # (batch,1,1)
                action_batch = torch.stack(
                    rollouts["action_index"], dim=0
                )  # (batch,1,1)
                value_batch = torch.stack(rollouts["value"], dim=0)  # (batch,1,1)
                reward_batch = torch.stack(rollouts["reward"], dim=0)  # (batch,1,1)
                value_prime_batch = torch.stack(
                    rollouts["value_prime"], dim=0
                )  # (batch,1,1)
                target_v_batch = torch.stack(rollouts["target_v"])
                budget_inputs_batch = torch.stack(rollouts["budget_inputs"], dim=0)
                LSTM_h_batch = torch.stack(rollouts["LSTM_h"])
                LSTM_c_batch = torch.stack(rollouts["LSTM_c"])
                mask_batch = torch.stack(rollouts["mask"])
                pos_encoding_batch = torch.stack(rollouts["pos_encoding"])
                pred_next_belief_batch = torch.stack(rollouts["pred_next_belief"])
                KL_diff_beliefs_batch = torch.stack(rollouts["KL_diff_beliefs"])
                belief_lstm_h_batch = torch.stack(rollouts["belief_lstm_h"])
                belief_lstm_c_batch = torch.stack(rollouts["belief_lstm_c"])
                next_policy_feature_batch = torch.stack(
                    rollouts["next_policy_feature"]
                ).squeeze(1)

                if device != local_device:
                    node_inputs_batch = node_inputs_batch.to(device)
                    edge_inputs_batch = edge_inputs_batch.to(device)
                    current_inputs_batch = current_inputs_batch.to(device)
                    action_batch = action_batch.to(device)
                    value_batch = value_batch.to(device)
                    reward_batch = reward_batch.to(device)
                    value_prime_batch = value_prime_batch.to(device)
                    target_v_batch = target_v_batch.to(device)
                    budget_inputs_batch = budget_inputs_batch.to(device)
                    LSTM_h_batch = LSTM_h_batch.to(device)
                    LSTM_c_batch = LSTM_c_batch.to(device)
                    mask_batch = mask_batch.to(device)
                    pos_encoding_batch = pos_encoding_batch.to(device)
                    pred_next_belief_batch = pred_next_belief_batch.to(device)
                    KL_diff_beliefs_batch = KL_diff_beliefs_batch.to(device)
                    belief_lstm_h_batch = belief_lstm_h_batch.to(device)
                    belief_lstm_c_batch = belief_lstm_c_batch.to(device)
                    next_policy_feature_batch = next_policy_feature_batch.to(device)

                # PPO
                with torch.no_grad():
                    logp_list, value, _, _ = global_network(
                        node_inputs_batch,
                        edge_inputs_batch,
                        budget_inputs_batch,
                        current_inputs_batch,
                        LSTM_h_batch,
                        LSTM_c_batch,
                        pos_encoding_batch,
                        mask_batch,
                        next_belief=next_policy_feature_batch,
                    )
                old_logp = torch.gather(
                    logp_list, 1, action_batch.squeeze(1)
                ).unsqueeze(
                    1
                )  # (batch_size,1,1)
                advantage = (
                    reward_batch + GAMMA * value_prime_batch - value_batch
                )  # (batch_size, 1, 1)
                # advantage = target_v_batch - value_batch

                entropy = (logp_list * logp_list.exp()).sum(dim=-1).mean()

                scaler = GradScaler()
                # belief_scaler = GradScaler()

                for i in range(UPDATE_EPOCHS):
                    with autocast():
                        # print("==calling global network==")
                        logp_list, value, _, _ = dp_model(
                            node_inputs_batch,
                            edge_inputs_batch,
                            budget_inputs_batch,
                            current_inputs_batch,
                            LSTM_h_batch,
                            LSTM_c_batch,
                            pos_encoding_batch,
                            mask_batch,
                            next_belief=next_policy_feature_batch,
                        )
                        # print("==done calling global network==")
                        logp = torch.gather(
                            logp_list, 1, action_batch.squeeze(1)
                        ).unsqueeze(1)
                        ratios = torch.exp(logp - old_logp.detach())
                        surr1 = ratios * advantage.detach()
                        surr2 = (
                            torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * advantage.detach()
                        )
                        policy_loss = -torch.min(surr1, surr2)
                        policy_loss = policy_loss.mean()

                        # value_clipped = value + (target_v_batch - value).clamp(-0.2, 0.2)
                        # value_clipped_loss = (value_clipped-target_v_batch).pow(2)
                        # value_loss =(value-target_v_batch).pow(2).mean()
                        # value_loss = torch.max(value_loss, value_clipped_loss).mean()

                        mse_loss = nn.MSELoss()
                        value_loss = mse_loss(value, target_v_batch).mean()

                        entropy_loss = (logp_list * logp_list.exp()).sum(dim=-1).mean()

                        loss = policy_loss + 0.5 * value_loss + 0.0 * entropy_loss

                        # print("calc belief loss")
                        # print("pred_next_belief_batch:", pred_next_belief_batch.size())
                        # print("KL_diff_beliefs_batch:", KL_diff_beliefs_batch.size())
                        if (KL_diff_beliefs_batch.size() == torch.Size([32, 30, 30])):
                            KL_diff_beliefs_batch = KL_diff_beliefs_batch.unsqueeze(1).unsqueeze(1)
                                                    
                        belief_loss = mse_loss(
                            pred_next_belief_batch, KL_diff_beliefs_batch
                        )
                        belief_loss.requires_grad = True
                        
                        # print("belief_loss:", belief_loss.item())

                    global_optimizer.zero_grad()
                    # belief_optimizer.zero_grad()
                    # loss.backward()
                    scaler.scale(loss).backward()

                    # scaler.scale(belief_loss).backward()
                    
                    scaler.unscale_(global_optimizer)
                    # scaler.unscale_(belief_optimizer)

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        global_network.parameters(), max_norm=10, norm_type=2
                    )
                    # global_optimizer.step()
                    scaler.step(global_optimizer)
                    # scaler.step(belief_optimizer)

                    scaler.update()

                    belief_optimizer.zero_grad()
                    belief_loss.backward()
                    belief_optimizer.step()

                    # belief_scaler.scale(belief_loss).backward()
                    # belief_scaler.unscale_(belief_optimizer)
                    # belief_scaler.step(belief_optimizer)
                    # belief_scaler.update()

                lr_decay.step()
                belief_lr_decay.step()

                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [
                    reward_batch.mean().item(),
                    value_batch.mean().item(),
                    policy_loss.item(),
                    value_loss.item(),
                    entropy.item(),
                    grad_norm.item(),
                    target_v_batch.mean().item(),
                    belief_loss.item(),
                    # np.zeros_like(target_v_batch.mean().item()),
                    *perf_data,
                ]
                trainingData.append(data)

                # experience_buffer = []
                # for i in range(8):
                #    experience_buffer.append([])

            if len(trainingData) >= SUMMARY_WINDOW:
                # print(trainingData)
                writeToTensorBoard(writer, trainingData, curr_episode)
                trainingData = []
            else:
                print(
                    "trainingData length:",
                    len(trainingData),
                    "SUMMARY_WINDOW:",
                    SUMMARY_WINDOW,
                )

            # get the updated global weights
            if update_done == True:
                if device != local_device:
                    weights = global_network.to(local_device).state_dict()
                    global_network.to(device)

                    belief_weights = belief_predictor.to(local_device).state_dict()
                    belief_predictor.to(device)
                else:
                    weights = global_network.state_dict()
                    belief_weights = belief_predictor.state_dict()

            jobList = []
            for i, meta_agent in enumerate(meta_agents):
                jobList.append(
                    meta_agent.job.remote(
                        weights,
                        curr_episode,
                        BUDGET_RANGE,
                        sample_size,
                        SAMPLE_LENGTH,
                        belief_predictor_weights=belief_weights,
                    )
                )
                curr_episode += 1

                print("+++++++Starting episode", curr_episode, "on metaAgent", i)

            if curr_episode % 32 == 0:
                print("Saving model", end="\n")
                checkpoint = {
                    "model": global_network.state_dict(),
                    "optimizer": global_optimizer.state_dict(),
                    "episode": curr_episode,
                    "lr_decay": lr_decay.state_dict(),
                }
                path_checkpoint = "./" + model_path + "/checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                belief_checkpoint = {
                    "model": belief_predictor.state_dict(),
                    "optimizer": global_optimizer.state_dict(),
                    "episode": curr_episode,
                    "lr_decay": lr_decay.state_dict(),
                }
                path_belief_checkpoint = "./" + model_path + "/belief_checkpoint.pth"
                torch.save(belief_checkpoint, path_belief_checkpoint)
                print("Saved model", end="\n")

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        if use_wandb:
            wandb.finish(quiet=True)
        for a in meta_agents:
            ray.kill(a)


if __name__ == "__main__":
    main()
