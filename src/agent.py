import numpy as np
import torch
import torch.nn.functional as f
from torch.distributions.categorical import Categorical
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader
from tqdm.autonotebook import tqdm

import utils
from globals import glob


class Agent:
    def __init__(self, model, minibatch_size=1024,
                 clip=0.2, entropy_coeff=0.01, max_norm=0.5,
                 lr=1e-3, weight_decay=1e-4, eps=1e-5):
        self.minibatch_size = minibatch_size
        self.clip = clip
        self.entropy_coeff = entropy_coeff
        self.max_norm = max_norm

        self.model = model
        self.policy_network = self.model.policy_network
        self.policy_head = self.model.policy_head
        self.value_head = self.model.value_head
        params = {'weight': [], 'bias': []}
        for module in self.policy_network:
            for name, param in module.named_parameters():
                if 'bias' in name:
                    params['bias'].append(param)
                else:
                    params['weight'].append(param)
        self.policy_optim = torch.optim.AdamW([{'params': params['weight'], 'weight_decay': weight_decay},
                                               {'params': params['bias'], 'weight_decay': 0}],
                                              lr=lr, eps=eps)
        self.value_optim = torch.optim.AdamW(self.value_head.parameters(),
                                             lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.policy_optim, step_size=6000)

        self.dataset = {'s': [], 'p': [], 'a': [], 'r': []}
        self.history_done = []

    def load_checkpoint(self):
        import os
        os.makedirs(f'{glob.MODEL_DIR}/opp', exist_ok=True)
        if os.path.exists(f'{glob.MODEL_DIR}/checkpoint'):
            checkpoint = torch.load(f'{glob.MODEL_DIR}/checkpoint')
            self.model.load_state_dict(checkpoint['model'])
            self.policy_optim.load_state_dict(checkpoint['policy_optim'])
            self.value_optim.load_state_dict(checkpoint['value_optim'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            ep_start = checkpoint['epoch']
            ema_win_rate = checkpoint['win_rate_dict']
        else:
            ep_start = 0
            ema_win_rate = {}
        return ep_start, ema_win_rate

    def save_checkpoint(self, model_dir, ep, win_rate_dict):
        torch.save({
            'epoch': ep,
            'win_rate_dict': win_rate_dict,
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
    def choose_action(model, state, return_state_value=False):
        with torch.no_grad():
            mask = torch.from_numpy(state[:, 1].reshape(-1, 36) != 0).to(device=glob.device, non_blocking=True)
            state = utils.zero_center(state)
            _state = torch.from_numpy(state.astype(np.float32)).to(device=glob.device)  # (n, 2, 6, 6)
            out = model(_state)
            policy = f.softmax(model.policy_head(out).masked_fill(mask, -torch.inf), dim=1).cpu()  # (n, 36)
            if return_state_value:
                return model.value_head(out).squeeze(1).cpu().numpy(), Categorical(policy).sample().numpy()
            else:
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
        raw_data[0] = utils.zero_center(raw_data[0].astype(np.float32))  # zero-center states
        raw_data[2] = utils.to_mask(raw_data[2])  # convert actions (n,) to masks (n, 36)
        return map(torch.from_numpy, raw_data)

    def learn(self):
        dataset = TensorDataset(*self.gather_data())
        batch = DataLoader(
            dataset=dataset,
            batch_size=self.minibatch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        )
        for data in self.dataset.values():
            data.clear()
        info = {'value_loss': [], 'pi': [], 'entropy': [], 'pi_ratio': [], 'total_norm': []}

        with tqdm(total=len(batch), ncols=80, position=1, leave=False, desc='Updating policy') as tbar:
            for dataset in batch:
                tbar.update(1)
                state, old_pi, action, reward, mask = map(lambda data: data.to(device=glob.device, non_blocking=True),
                                                          dataset)
                out = self.model(state)
                policy = f.softmax(self.policy_head(out).masked_fill(mask, -torch.inf), dim=1)
                state_value = self.value_head(out.detach()).squeeze(1)
                adv = reward - state_value.detach()
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)
                pi_ratio = policy[action] / old_pi[action]  # (n,)
                obj = torch.min(pi_ratio * adv, pi_ratio.clip(1 - self.clip, 1 + self.clip) * adv)
                entropy = self.entropy_coeff * Categorical(policy).entropy()

                value_loss = f.mse_loss(state_value, reward)
                loss = -torch.mean(obj + entropy) + value_loss
                loss.backward()
                total_norm = clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm, error_if_nonfinite=True)

                info['value_loss'].append(value_loss.detach().unsqueeze(0))
                info['entropy'].append(entropy.detach())
                info['total_norm'].append(total_norm.unsqueeze(0))
                info['pi'].append(old_pi[action].detach())
                info['pi_ratio'].append(pi_ratio.detach())
                self.policy_optim.step()
                self.policy_optim.zero_grad()
                self.value_optim.step()
                self.value_optim.zero_grad()
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
            reward = np.where(pieces[:, 0] > 0, 1, -1).astype(np.float32) if np.all(done) else None
            return state_t[~done].copy(), reward, done

        state, reward, done = step(state, action, 0)
        state_value = None
        sel = None
        player_idx = 1
        while np.any(~done):
            if player_idx == 1:
                state[:, [0, 1]] = state[:, [1, 0]]
            if state_value is None and player_idx == 0:
                state_value, action = self.choose_action(self.model, state, return_state_value=True)
                sel = ~done.copy()
            else:
                _, action = self.choose_action(self.model, state)
            state, reward, done = step(state, action, player_idx)
            player_idx = 1 - player_idx
        if state_value is not None:
            reward[sel] = (state_value + reward[sel]) / 2
        return reward
