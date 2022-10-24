import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import utils


class Agent:
    def __init__(self, model, minibatch_size=1024,
                 clip=0.2, entropy_coeff=0.01, max_norm=0.5,
                 lr=2e-3, weight_decay=1e-4, eps=1e-5,
                 t_max=20000, min_lr=1e-5):
        self.minibatch_size = minibatch_size
        self.clip = clip
        self.entropy_coeff = entropy_coeff
        self.max_norm = max_norm

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
                                              lr=lr, eps=eps)
        self.value_optim = torch.optim.NAdam(self.value_head.parameters(),
                                             lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.policy_optim,
                                                                    T_max=t_max, eta_min=min_lr)

        self.dataset = {'s': [], 'p': [], 'a': [], 'r': []}
        self.history_done = []

    def save_checkpoint(self, model_dir, ep):
        torch.save({
            'epoch': ep,
            'model': self.model.state_dict(),
            'policy_optim': self.policy_optim.state_dict(),
            'value_optim': self.value_optim.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, f'{model_dir}/checkpoint')

    def save_as_opponent(self, model_dir, ep):
        model_path = f'{model_dir}/opp/model-{ep}'
        torch.save(self.model.state_dict(), model_path)
        tqdm.write(f'New model state saved to {model_path}')

    @staticmethod
    def choose_action(model, state):
        mask = torch.from_numpy(state[:, 1].reshape(-1, 36) != 0).to(device=utils.device, non_blocking=True)
        state = utils.zero_center(state)
        _state = torch.from_numpy(state.astype(np.single)).to(device=utils.device)  # (n, 2, 6, 6)
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

    def gather_data(self):
        raw_data = [np.concatenate(data, axis=0) for data in self.dataset.values()]  # [state, policy, action, reward]
        raw_data.append(raw_data[0][:, 1].reshape(-1, 36) != 0)  # create masks for softmax
        raw_data[0] = utils.zero_center(raw_data[0].astype(np.single))  # zero-center states
        raw_data[2] = utils.to_mask(raw_data[2])  # convert actions (n,) to masks (n, 36)
        return map(torch.from_numpy, raw_data)

    def learn(self):
        dataset = TensorDataset(*self.gather_data())
        batch = DataLoader(
            dataset=dataset,
            batch_size=self.minibatch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8,
            pin_memory=True,
        )
        for data in self.dataset.values():
            data.clear()
        info = {'pi': [], 'entropy': [], 'pi_ratio': [], 'total_norm': []}

        with tqdm(total=len(batch), ncols=80, leave=False, desc='Updating policy') as tbar:
            for dataset in batch:
                tbar.update(1)
                state, old_pi, action, reward, mask = map(lambda data: data.to(device=utils.device, non_blocking=True),
                                                          dataset)
                out = self.model(state)
                policy = self.policy_head(out, mask)
                pi_ratio = policy[action] / old_pi[action]  # (n,)
                obj = torch.min(pi_ratio * reward, pi_ratio.clip(1 - self.clip, 1 + self.clip) * reward)
                entropy = self.entropy_coeff * Categorical(policy).entropy()
                loss = -torch.mean(obj + entropy)
                loss.backward()
                total_norm = clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm, error_if_nonfinite=True)

                info['entropy'].append(entropy.detach())
                info['total_norm'].append(total_norm.unsqueeze(0))
                info['pi'].append(old_pi[action].detach())
                info['pi_ratio'].append(pi_ratio.detach())
                # for stage in self.policy_network:
                #     for name, param in stage.named_parameters():
                #         print(name, param.grad)
                self.policy_optim.step()
                self.policy_optim.zero_grad()
        return info

    def rollout(self, state, action, num_turns):
        state_t = state.copy()
        done = np.full(len(state_t), False)

        def step(state, action, player_idx):
            nonlocal state_t, done, num_turns
            state = utils.update_game_state(state, action)
            state_t[~done] = state if player_idx == 0 else state[:, [1, 0]]
            num_turns += 1
            pieces = state_t[:, :2].sum(axis=(2, 3))
            if num_turns > 2: done = ~pieces.all(axis=1)
            reward = np.where(pieces[:, 0] > 0, 1, -1) if np.all(done) else None
            return state_t[~done].copy(), reward, done

        state, reward, done = step(state.copy(), action, 0)
        player_idx = 1
        while np.any(~done):
            if player_idx == 1:
                state[:, [0, 1]] = state[:, [1, 0]]
            _, action = Agent.choose_action(self.model, state)
            state, reward, done = step(state.copy(), action, player_idx)
            player_idx = 1 - player_idx
        return reward
