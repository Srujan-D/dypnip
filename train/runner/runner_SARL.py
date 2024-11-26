import torch
import numpy as np
import ray
import os

# from attention_net import AttentionNet
from ...neural_networks.attention_net import AttentionNet
from ...neural_networks.dynamics_pred_net import PredictNextBelief

# from worker import Worker
from ..worker.worker_SARL import Worker
from ...parameters.params_robust_attention_SARL import *


class Runner(object):
    """Actor object to start running simulation on workers.
    Gradient computation is also executed on this object."""

    def __init__(self, metaAgentID):
        self.metaAgentID = metaAgentID
        # self.device = torch.device(f"cuda:{CUDA_DEVICE[0]}") if USE_GPU else torch.device('cpu')
        self.device = torch.device("cuda") if USE_GPU else torch.device("cpu")
        self.localNetwork = AttentionNet(INPUT_DIM, EMBEDDING_DIM, device=self.device)
        self.localNetwork.to(self.device)

        self.belief_predictor = PredictNextBelief(self.device)
        self.belief_predictor.to(self.device)

    def get_weights(self):
        return self.localNetwork.state_dict()

    def set_weights(self, weights):
        self.localNetwork.load_state_dict(weights)

    def get_belief_predictor_weights(self):
        return self.belief_predictor.state_dict()

    def set_belief_predictor_weights(self, weights):
        self.belief_predictor.load_state_dict(weights)

    def singleThreadedJob(
        self,
        episodeNumber,
        budget_range,
        sample_size,
        sample_length,
        history_size=None,
        target_size=None,
    ):
        save_img = (
            False
            if (SAVE_IMG_GAP != 0 and episodeNumber % SAVE_IMG_GAP == 0)
            else False
        )
        # save_img = False
        worker = Worker(
            self.metaAgentID,
            self.localNetwork,
            episodeNumber,
            budget_range,
            sample_size,
            sample_length,
            self.device,
            save_image=save_img,
            greedy=False,
            belief_predictor=self.belief_predictor,
        )
        worker.work(episodeNumber)

        jobResults = worker.experience
        perf_metrics = worker.perf_metrics
        return jobResults, perf_metrics

    def job(
        self,
        global_weights,
        episodeNumber,
        budget_range,
        sample_size=SAMPLE_SIZE,
        sample_length=None,
        belief_predictor_weights=None,
    ):
        print(
            "starting episode {} on metaAgent {}".format(
                episodeNumber, self.metaAgentID
            )
        )
        # set the local weights to the global weight values from the master network
        self.set_weights(global_weights)

        if belief_predictor_weights is not None:
            self.set_belief_predictor_weights(belief_predictor_weights)

        jobResults, metrics = self.singleThreadedJob(
            episodeNumber, budget_range, sample_size, sample_length
        )

        info = {
            "id": self.metaAgentID,
            "episode_number": episodeNumber,
        }
        print(
            "finished episode {} on metaAgent {}".format(
                episodeNumber, self.metaAgentID
            )
        )
        return jobResults, metrics, info


@ray.remote(num_cpus=1, num_gpus=len(CUDA_DEVICE) / NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, metaAgentID):
        super().__init__(metaAgentID)


if __name__ == "__main__":
    ray.init()
    runner = RLRunner.remote(0)
    job_id = runner.singleThreadedJob.remote(1)
    out = ray.get(job_id)
    print(out[1])
