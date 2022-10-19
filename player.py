import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import utils


class Player:
    def __init__(self, model, lr=2e-3, weight_decay=1e-4):
        self.model = model
        self.policy_network = self.model.policy_network
        self.policy_head = self.model.policy_head
        self.value_head = self.model.value_head

        params = {'weight': [], 'bias': []}
        for stage in self.policy_network:
            for name, param in stage.named_parameters():
                if 'bias' in name:
                    params['bias'].append(param)
                else:
                    params['weight'].append(param)
        self.policy_optim = torch.optim.NAdam([{'params': params['weight'], 'weight_decay': weight_decay},
                                               {'params': params['bias'], 'weight_decay': 0}],
                                              lr=lr)
        self.value_optim = torch.optim.NAdam(self.value_head.parameters(),
                                             lr=lr, weight_decay=weight_decay)

        self.dataset = {'s': [], 'p': [], 'a': [], 'r': []}
        self.history_done = []

    def save_state_dict(self, model_dir):
        torch.save(self.model.state_dict(), f'{model_dir}/agent')
        torch.save(self.policy_optim.state_dict(), f'{model_dir}/policy_optim')
        torch.save(self.value_optim.state_dict(), f'{model_dir}/value_optim')

    @staticmethod
    def select_action(model, state):
        _state = utils.from_numpy(state)  # (n, 2, 6, 6)
        mask = _state[:, 1].reshape(-1, 36) != 0
        _state = utils.zero_center(_state)
        with torch.no_grad():
            out = model(_state)
            policy = model.policy_head(out, mask).cpu()  # (n, 36)
        return policy.numpy(), Categorical(policy).sample().numpy()

    def remember(self, state, policy, action, done):
        self.dataset['s'].append(state)
        self.dataset['p'].append(policy)
        self.dataset['a'].append(action)
        self.history_done.append(done)

    def assign_reward(self, reward):
        for done in self.history_done:
            self.dataset['r'].append(reward[~done])
        self.history_done.clear()

    def learn(self, minibatch_size, eps):
        dataset = TensorDataset(*(
            utils.from_numpy(np.concatenate(data, axis=0)) for data in self.dataset.values()
        ))
        batch = DataLoader(
            dataset=dataset,
            batch_size=minibatch_size,
            shuffle=True,
            drop_last=True,
        )
        for data in self.dataset.values():
            data.clear()
        total_step = round(100 * 1024 / minibatch_size)
        info = {'pi': [], 'pi_ratio': [], 'total_norm': []}

        with tqdm(total=total_step, ncols=80, leave=False, desc='Updating policy') as pbar:
            for step, (state, old_pi, action, reward) in enumerate(batch):
                pbar.update(1)
                # state = utils.from_numpy(state)  # utils.from_numpy(utils.augment_data(state))
                mask = state[:, 1].reshape(-1, 36) != 0
                state = utils.zero_center(state)
                out = self.model(state)
                policy = self.policy_head(out, mask)
                # old_pi = utils.from_numpy(old_pi)  # utils.augment_data(old_pi.reshape((-1, 1, 6, 6))).reshape((-1, 36))
                action = utils.to_mask(action)  # (n,) --> (n, 36)
                # _action = torch.from_numpy(action)  # utils.augment_data(action.reshape((-1, 1, 6, 6))).reshape((-1, 36))
                # reward = utils.from_numpy(reward)  # .repeat(6)

                pi_ratio = policy[action] / old_pi[action]  # (n,)
                torch.mean(
                    -torch.min(pi_ratio * reward, pi_ratio.clip(1 - eps, 1 + eps) * reward)
                ).backward()

                info['pi'].append(old_pi[action].detach())
                info['pi_ratio'].append(pi_ratio.detach())

                total_norm = clip_grad_norm_(self.model.parameters(), max_norm=10, error_if_nonfinite=True)
                info['total_norm'].append(total_norm.unsqueeze(0))
                # for stage in self.policy_network:
                #     for name, param in stage.named_parameters():
                #         print(name, param.grad)
                self.policy_optim.step()
                self.policy_optim.zero_grad()

        return info
