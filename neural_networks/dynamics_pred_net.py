
import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp.autocast_mode import autocast
from ..parameters.params_robust_attention_SARL import *
import os

class PredictNextBelief(nn.Module):
    def __init__(self, device="cuda"):
        super(PredictNextBelief, self).__init__()
        self.device = device
        # self.conv encoder --> what features of GP to use? predicted mean, uncertainty,
        # --> do we want to encode history (just regress GP over time) explicitly?
        # self.lstm layer
        # self.MLP layer

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(8, 4, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.lstm = nn.LSTM(
            input_size=4 * 6 * 6, hidden_size=BELIEF_EMBEDDING_DIM, num_layers=1, batch_first=True
        )

        self.fc = nn.Linear(BELIEF_EMBEDDING_DIM, 4 * 6 * 6)

        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=1, padding=1, output_padding=0),
            nn.Sigmoid()
        )
        
        self.policy_feature = None
        
        if SAVE_FEATURES:
            self.features_file = 'features.npy'
            self.labels_file = 'labels.npy'
            if not os.path.exists(self.features_file):
                np.save(self.features_file, np.empty((0, 128), dtype=np.float32))
            if not os.path.exists(self.labels_file):
                np.save(self.labels_file, np.empty((0,), dtype=np.float32))
        else:
            self.features_file = None
            self.labels_file = None

        self.freeze_parameters()
    
    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False


    def forward(self, x, lstm_h=torch.zeros(1, 1, BELIEF_EMBEDDING_DIM), lstm_c=torch.zeros(1, 1, BELIEF_EMBEDDING_DIM), fuel=None):
        batch_size = 1  # x.size(0)
        x = self.conv_encoder(x)
        x = x.view(batch_size, -1)
        # x = self.linear(x)
        x, (lstm_h, lstm_c) = self.lstm(x.unsqueeze(1), (lstm_h, lstm_c))
        x = x[:, -1, :]
        self.policy_feature = x
        # self.policy_feature= self.policy_feature.detach()
        
        x = self.fc(x)

        x = x.view(batch_size, 4, 6, 6)
        x = self.conv_decoder(x)
        return x, lstm_h, lstm_c
    

    def return_policy_feature(self):
        # print("policy feature size", self.policy_feature.size(), self.policy_feature.unsqueeze(0).size())
        # detached_policy_feature = self.policy_feature.detach()
        # return detached_policy_feature.unsqueeze(0)

        return self.policy_feature.unsqueeze(0)
    
    def save_policy_feature(self, fuel):
        # Load existing features and labels
        features = np.load(self.features_file)
        labels = np.load(self.labels_file)

        # Append the latest policy feature and fuel label
        new_feature = self.policy_feature.detach().numpy()
        new_label = np.array([fuel], dtype=np.float32)

        features = np.vstack((features, new_feature))
        labels = np.concatenate((labels, new_label))

        # Save updated features and labels
        np.save(self.features_file, features)
        np.save(self.labels_file, labels)