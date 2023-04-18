import wandb

from tensorboardX import SummaryWriter


class Logger:
    """Abstract class for logger."""

    def add_scalar(self, tag: str, scalar_value, global_step=None):
        """Logs scalar value."""
        raise Exception("Not implemented for this logger.")


class WandbLogger(Logger):
    """Weight and Biases logger."""

    def __init__(self, project: str, entity: str, name: str = None, group: str = None, config=None, notes: str = None):
        """
        notes: A longer description of the run. helps you remember what you were doing when you
            ran this run.
        """
        wandb.init(project=project, entity=entity, name=name, group=group, config=config, notes=notes)

    def add_scalar(self, tag: str, scalar_value, global_step=None):
        wandb.log({tag: scalar_value}, step=global_step)


class TensorboardLogger(Logger):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(self, tag, scalar_value, global_step=None):
        self.writer.add_scalar(tag, scalar_value, global_step)