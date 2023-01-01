if __name__ == '__main__':
    from game import Env

    env = Env(graphics=True, fps=None, num_envs=1)
    env.run('../best_models/cnn2-64')
    # env.self_play('../best_models/cnn2-64', '../best_models/resnet3-64')
