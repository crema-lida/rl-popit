FROM nvcr.io/nvidia/pytorch:22.12-py3

CMD mkdir -p /workspace/rl-popit/runs && \
    tensorboard --logdir rl-popit/runs & disown && \
    jupyter lab --NotebookApp.token=''