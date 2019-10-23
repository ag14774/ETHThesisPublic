import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from base.base_trainer import BaseTrainer
from logger.visualization import WriterTensorboardX
from matplotlib.lines import Line2D
from torchvision.utils import make_grid
from utils.util import get_global_rank, get_world_size
from model.util import all_tensors_to
import shutil


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            print(n, ave_grads[-1], max_grads[-1])
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([
        Line2D([0], [0], color="c", lw=4),
        Line2D([0], [0], color="b", lw=4),
        Line2D([0], [0], color="k", lw=4)
    ], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def make_grid2(tensor,
               nrow=8,
               padding=2,
               normalize=False,
               range=None,
               scale_each=False,
               pad_value=0):
    while len(tensor.shape) < 4:
        tensor = torch.unsqueeze(tensor, -1)
    return make_grid(tensor, nrow, padding, normalize, range, scale_each,
                     pad_value)


def memory_stats(logger):
    logger.info(f'Memory allocated: {torch.cuda.memory_allocated()}')
    logger.info(f'Max memory allocated: {torch.cuda.max_memory_allocated()}')
    logger.info(f'Memory cached: {torch.cuda.memory_cached()}')
    logger.info(f'Max memory allocated: {torch.cuda.max_memory_cached()}')


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader, lr_scheduler, main_device):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, config,
                                      data_loader, valid_data_loader,
                                      lr_scheduler, main_device)

    def create_checkpoint_dict(self, epoch):
        return super().create_checkpoint_dict(epoch)

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=strict)

    def train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_metrics = torch.zeros(len(self.metrics_labels))

        for batch_idx, (data, target) in enumerate(self.data_loader):
            target = all_tensors_to(target,
                                    self.main_device,
                                    non_blocking=True)
            data = data.to(self.main_device, non_blocking=True)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) +
                                 batch_idx)
            self.writer.add_scalar('loss', loss.item())

            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        epoch, batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        loss.item()))
                # self.writer.add_image(
                #     'input', make_grid2(data.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        return log

    def valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()

        total_val_loss = 0
        total_val_metrics = torch.zeros(len(self.metrics_labels))

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                target = all_tensors_to(target,
                                        self.main_device,
                                        non_blocking=True)
                data = data.to(self.main_device, non_blocking=True)

                output = self.model(data)
                loss = self.loss(output, target)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx,
                    'valid')
                self.writer.add_scalar('loss', loss.item())

                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)

                # self.writer.add_image(
                #     'input', make_grid2(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        if not self.disable_hist:
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss':
            total_val_loss / len(self.valid_data_loader),
            'val_metrics':
            (total_val_metrics / len(self.valid_data_loader)).tolist()
        }


class DistTrainer(Trainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader, lr_scheduler, main_device):
        super(DistTrainer,
              self).__init__(model, loss, metrics, optimizer, config,
                             data_loader, valid_data_loader, lr_scheduler,
                             main_device)
        # TODO: BROADCAST PARAMETERS HERE TO CHECK IF THEY ARE THE SAME

    def create_checkpoint_dict(self, epoch):
        arch = type(self.model).__name__
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        lr_sched_dict = None
        if self.lr_scheduler:
            lr_sched_dict = self.lr_scheduler.state_dict()
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': lr_sched_dict,
            'monitor_best': self.mnt_best,
            'config': self.config.raw
        }
        return state

    def load_state_dict(self, state_dict, strict=True):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.module.load_state_dict(state_dict, strict=strict)
        else:
            return self.model.load_state_dict(state_dict, strict=strict)

    def train_epoch(self, epoch):
        return super().train_epoch(epoch)

    def valid_epoch(self, epoch):
        return super().valid_epoch(epoch)


class DistTrainerMoE(Trainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader, lr_scheduler, main_device):
        super(DistTrainerMoE,
              self).__init__(model, loss, metrics, optimizer, config,
                             data_loader, valid_data_loader, lr_scheduler,
                             main_device)
        self._set_defaults(model, loss, metrics, optimizer, config,
                           data_loader, valid_data_loader, lr_scheduler,
                           main_device)
        cfg_trainer = config['trainer']['extra_args']

        self.checkpoint_dir = config.save_dir
        self.log_dir = config.log_dir
        # setup visualization writer instance
        if get_global_rank() == 0:
            enable_board = cfg_trainer['tensorboardX']
        else:
            enable_board = False

        self.writer = WriterTensorboardX(self.log_dir, self.logger,
                                         enable_board)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    def create_checkpoint_dict(self, epoch):
        arch = type(self.model).__name__
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        lr_sched_dict = None
        if self.lr_scheduler:
            lr_sched_dict = self.lr_scheduler.state_dict()
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': lr_sched_dict,
            'monitor_best': self.mnt_best,
            'world_size': get_world_size(),
            'config': self.config.raw
        }
        return state

    def load_state_dict(self, state_dict, strict=True):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.module.load_state_dict(state_dict, strict=strict)
        else:
            return self.model.load_state_dict(state_dict, strict=strict)

    def _check_early_stop(self, log, epoch):
        # evaluate model performance according to configured metric,
        # save best checkpoint as model_best
        early_stop_and_save_checkpoint = torch.tensor([0., 0.])
        early_stop_and_save_checkpoint[1] = self.improved_since_last_save
        if get_global_rank() == 0 and self.mnt_mode != 'off':
            try:
                # check whether model performance improved or not,
                # according to specified metric(mnt_metric)
                improved = (self.mnt_mode == 'min'
                            and log[self.mnt_metric] <= self.mnt_best) or (
                                self.mnt_mode == 'max'
                                and log[self.mnt_metric] >= self.mnt_best)
            except KeyError:
                self.logger.warning(
                    "Warning: Metric '{}' is not found. "
                    "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                self.mnt_mode = 'off'
                improved = False
                self.not_improved_count = 0

            if improved:
                self.mnt_best = log[self.mnt_metric]
                self.not_improved_count = 0
                self.improved_since_last_save = True
                early_stop_and_save_checkpoint[
                    1] = self.improved_since_last_save
            else:
                self.not_improved_count += 1

            if self.not_improved_count > self.early_stop:
                self.logger.info(
                    "Validation performance didn\'t improve for {} epochs. "
                    "Training stops.".format(self.early_stop))
                early_stop_and_save_checkpoint[0] = True

        dist.broadcast(early_stop_and_save_checkpoint, src=0)
        early_stop = early_stop_and_save_checkpoint[0]
        improved_since_last_save = early_stop_and_save_checkpoint[1]
        if epoch % self.save_period == 0:
            self._save_checkpoint(epoch, save_best=improved_since_last_save)
            self.improved_since_last_save = False

        return early_stop

    def train(self):
        """
        Full training logic
        """
        self.not_improved_count = 0
        self.improved_since_last_save = False
        for epoch in range(self.start_epoch, self.epochs + 1):
            print()
            self.data_loader.step(epoch)

            result = self.train_epoch(epoch)
            if self.do_validation:
                self.valid_data_loader.step(epoch)
                val_log = self.valid_epoch(epoch)
                result = {**result, **val_log}

            if self.lr_scheduler is not None:
                self.logger.info(
                    f"Learning rate: {self.lr_scheduler.get_lr()}")
                self.lr_scheduler.step(epoch=epoch)

            log = self._log_info(result, epoch)
            early_stop = self._check_early_stop(log, epoch)

            if early_stop:
                break

    def train_epoch(self, epoch):
        if get_global_rank() == 0:
            checkpoint_dir = self.checkpoint_dir / f'checkpoint-epoch{epoch}'
            best_dir = self.checkpoint_dir / 'model_best'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_dir.mkdir(parents=True, exist_ok=True)
        return super().train_epoch(epoch)

    def valid_epoch(self, epoch):
        return super().valid_epoch(epoch)

    def _log_info(self, result, epoch):
        # save logged informations into log dict
        log = {'epoch': epoch}
        for key, value in result.items():
            if key == 'metrics':
                log.update({
                    mtr_label: value[i]
                    for i, mtr_label in enumerate(self.metrics_labels)
                })
            elif key == 'val_metrics':
                log.update({
                    'val_' + mtr_label: value[i]
                    for i, mtr_label in enumerate(self.metrics_labels)
                })
            else:
                log[key] = value

        # TODO: Reduce self.mnt_metric to make early stopping reliable

        # print logged informations to the screen
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

        return log

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint
            to 'model_best.pth'
        """
        state = self.create_checkpoint_dict(epoch)
        checkpoint_dir = self.checkpoint_dir / f'checkpoint-epoch{epoch}'
        rank = get_global_rank()
        filename = str(checkpoint_dir / f'rank{rank}.pth')

        self.logger.info("Saving checkpoint: {} ...".format(filename))
        torch.save(state, filename)

        world_size = get_world_size()

        if rank == 0:
            last_checked = world_size
            file = checkpoint_dir / f'rank{last_checked}.pth'
            while file.is_file():
                file.unlink()
                last_checked += 1
                file = checkpoint_dir / f'rank{last_checked}.pth'
            timestamp = checkpoint_dir / "timestamp.txt"
            timestamp.touch()

        if save_best:
            best_dir = self.checkpoint_dir / 'model_best'
            best_path = str(best_dir / f'rank{rank}.pth')
            self.logger.info("Saving current best: model_best.pth ...")
            torch.save(state, best_path)

            if rank == 0:
                last_checked = world_size
                file = best_dir / f'rank{last_checked}.pth'
                while file.is_file():
                    file.unlink()
                    last_checked += 1
                    file = best_dir / f'rank{last_checked}.pth'
                timestamp = best_dir / "timestamp_best.txt"
                timestamp.touch()

        if rank == 0:
            self._remove_checkpoints()

    def _remove_checkpoints(self):
        timestamps = self.checkpoint_dir.parent.rglob("timestamp.txt")
        timestamps = sorted(timestamps, key=lambda f: f.stat().st_mtime)
        timestamps = timestamps[:-self.keep_last]
        for t in timestamps:
            shutil.rmtree(t.parent)

        timestamps_b = self.checkpoint_dir.parent.rglob("timestamp_best.txt")
        timestamps_b = sorted(timestamps_b, key=lambda f: f.stat().st_mtime)
        timestamps_b = timestamps_b[:-1]
        for t in timestamps_b:
            shutil.rmtree(t.parent)

        configs = self.checkpoint_dir.parent.rglob("config.json")
        configs = sorted(configs, key=lambda f: f.stat().st_mtime)
        configs = configs[:-1]
        for c in configs:
            if len(list(c.parent.glob('checkpoint-epoch*'))) == 0 and not (
                    c.parent / 'model_best').is_dir():
                c.unlink()

        for f in self.checkpoint_dir.parent.glob('*'):
            if f.is_dir() and len(list(f.glob('*'))) == 0:
                f.rmdir()

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """

        rank = get_global_rank()
        resume_path = str(resume_path / f"rank{rank}.pth")
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)

        if int(checkpoint['world_size']) != int(get_world_size()):
            raise RuntimeError(f"Error: Checkpoint created with world_size "
                               f"{checkpoint['world_size']} but current "
                               f"world_size is {get_world_size()}.")

        if self.strict_load:
            self.start_epoch = checkpoint['epoch'] + 1
            self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config.raw['arch']:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is "
                "different from that of checkpoint. This may yield "
                "an exception while state_dict is being loaded.")
        try:
            missing, unexpected = self.load_state_dict(
                checkpoint['state_dict'], strict=self.strict_load)
            if not self.strict_load:
                self.logger.info(f"Missing layers: {missing}")
                self.logger.info(f"Unexpected layers: {unexpected}")
        except RuntimeError:
            checkpoint['state_dict'] = self._attempt_to_fix_state_dict(
                checkpoint['state_dict'], missing, unexpected)
            self.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when
        # optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config.raw[
                'optimizer']['type']:
            self.logger.warning(
                "Warning: Optimizer type given in config file is different "
                "from that of checkpoint. Optimizer parameters not resumed.")
        elif not self.strict_load:
            self.logger.warning("Warning: Strict loading is disabled. "
                                "Optimizer parameters not resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        if checkpoint['config']['lr_scheduler']['type'] != self.config.raw[
                'lr_scheduler']['type']:
            self.logger.warning(
                "Warning: LR Scheduler type given in config file is different "
                "from that of checkpoint. Parameters not resumed.")
        elif not self.strict_load:
            self.logger.warning("Warning: Strict loading is disabled. "
                                "LR scheduler state not resumed.")
        else:
            if self.lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(
                self.start_epoch))
