from detectron2.engine import LRScheduler

class LRScheduler_(LRScheduler):
    '''
    add tag to lr and show on tensorboard
    '''
    def __init__(self, optimizer=None, scheduler=None, tag=''):
        super().__init__(optimizer=optimizer, scheduler=scheduler)
        self.tag=tag
    def after_step(self):
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        if self.tag:
            lr_tag = "lr_" + self.tag
        else:
            lr_tag = "lr"
        self.trainer.storage.put_scalar(lr_tag, lr, smoothing_hint=False)
        self.scheduler.step()