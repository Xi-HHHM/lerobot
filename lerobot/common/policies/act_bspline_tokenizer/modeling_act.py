#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://arxiv.org/abs/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from itertools import chain
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.common.policies.act_bspline_tokenizer.configuration_act import ACTBsplineTokenizerConfig, get_action_tokenizer
from lerobot.common.policies.act.modeling_act import ACT, ACTTemporalEnsembler
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy



class ACTBsplineTokenizerPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://arxiv.org/abs/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    config_class = ACTBsplineTokenizerConfig
    name = "act_bspline_tokenizer"

    def __init__(
        self,
        config: ACTBsplineTokenizerConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Tokenizer
        self.action_tokenizer = get_action_tokenizer()
        config.chunk_size = self.action_tokenizer.num_basis

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.model = ACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [batch[key] for key in self.config.image_features]

        # If we are doing temporal ensembling, do online updates where we keep track of the number of actions
        # we are ensembling over.
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.model(batch)[0]  # (batch_size, chunk_size, action_dim)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            # Recontruct actions from tokenizer
            tokens = self.model(batch)[0]
            tokens = einops.rearrange(tokens, "b t d -> b (d t)", t=self.action_tokenizer.num_basis)
            actions = self.action_tokenizer.reconstruct_traj_continuous(tokens)

            # TODO(rcadene): make _forward return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [batch[key] for key in self.config.image_features]

        batch = self.normalize_targets(batch)
        tokenized_action = self.action_tokenizer.encode_continuous(batch["action"], update_bounds=True)

        # Reconstruction
        # Sanity Check, check if the reconstructed tokens are correct            
        traj = self.action_tokenizer.reconstruct_traj_continuous(tokenized_action)
        # traj = traj.cpu().numpy()
        # a = batch["action"].cpu().numpy()
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(14, 1)
        # for i in range(14):
        #     axs[i].plot(traj[0, :, i], "b", label="Trajectory" if i == 0 else "")
        #     axs[i].plot(a[0, :, i], "r--", label="Action" if i == 0 else "")

        # # Create one legend for the whole figure
        # lines = [
        #     axs[0].lines[0],  # First line plotted (trajectory)
        #     axs[0].lines[1],  # Second line plotted (action)
        # ]
        # labels = ["Trajectory", "Action"]
        # fig.legend(lines, labels, loc="upper right")
        # plt.tight_layout()
        
        # plt.savefig("/home/huang/other_workspace/hongyi/rec.png")

        traj_pad = batch["action_is_pad"]
        batch["action_is_pad"] = torch.zeros((batch["action"].shape[0], self.action_tokenizer.num_basis),
                                    dtype=torch.bool, device=batch["action"].device)

        batch["action"] = einops.rearrange(tokenized_action, "b (d t) -> b t d", t=self.action_tokenizer.num_basis)

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)
        traj_actions_hat = einops.rearrange(actions_hat, "b t d -> b (d t)", t=self.action_tokenizer.num_basis)
        with torch.no_grad():
            traj_hat = self.action_tokenizer.reconstruct_traj_continuous(traj_actions_hat)

        ## MP weight level L1 loss
        ## ToCheck!!!
        batch["action_is_pad"] = torch.zeros((batch["action"].shape[0], self.action_tokenizer.num_basis),
                                            dtype=torch.bool, device=batch["action"].device)
        l1_loss = (
            F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()
                
        # Trajectory level L1 loss
        traj_l1_loss = (
            F.l1_loss(traj, traj_hat, reduction="none") * ~traj_pad.unsqueeze(-1)
        ).mean()

        loss_dict = {"traj_l1_loss": traj_l1_loss.item()}
        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://arxiv.org/abs/1312.6114 for more details).
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict

