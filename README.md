# rl-popit
Welcome to my website to play the game. Give it a try!

http://crema.evalieben.cn:18002/game/

<img width="400" alt="屏幕截图 2023-01-03 032403" src="https://user-images.githubusercontent.com/100750226/210271646-0e533114-2949-4059-b341-8876f448d539.png">

The game is currently only available in Chinese.

# Neural Network Architecture
The model structure references DeepMind's work on AlphaGo Zero:
> Silver, D., Schrittwieser, J., Simonyan, K. et al. Mastering the game of Go without human knowledge. Nature 550, 354–359 (2017). https://doi.org/10.1038/nature24270

The best model, resnet3-64, is comprised of:
- 1 input convolutional layer
- 3 residual blocks (2 convolutional layers + 1 skip connection for each block)
- 1 policy head (1 convolutional layer + 1 fully connected layers)
- 1 value head (1 convolutional layer + 2 fully connected layers)

Each of the convolutional layer has 64 features (the input layer has 2 features).
The network structure resembles AlphaGo Zero's but has much less features and residual blocks (My game is too simple after all).

# Method
The neural network is trained by Proximal Policy Optimization, PPO. I also tweaked the original PPO implementation according to this webpage:
> The 37 Implementation Details of Proximal Policy Optimization
> https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

Some of the implementations work very well for my model: entropy maximization, normalization of advantages, global gradient clipping, etc.
Each training cycle begins with 128 vectorized environments sampling game states, and then do backpropagation with a minibatch size of 2048. I simply set the reward of each action to 1 if agent wins in one game, else the reward will be -1.

# Training Curve
The horizontal axis shows the number of epochs and the vertial axis shows the win rate (%). Each curve represents an opponent using the old model. Once the win rate reaches 90%, drop the old model and then save the latest for opponent to use. Training is much tougher after 1000 epochs, and the model finally converges in 15000 epochs.
#### Win Rate Change during Epoch 0 - 1.4k
![win_rate](https://user-images.githubusercontent.com/100750226/210269864-a5c86b04-e1d5-4ee9-8a1d-19e5fad3be6b.svg)
#### Win Rate Change during Epoch 1k - 15k
![win_rate(1)](https://user-images.githubusercontent.com/100750226/210269870-fe495f24-a317-4716-8e90-9267ae997b6a.svg)
