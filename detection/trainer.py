import logging
import weakref
import time
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, create_ddp_model, SimpleTrainer, hooks
from detectron2.checkpoint import DetectionCheckpointer

from fvcore.nn.precise_bn import get_bn_modules
from .evaluation.pascal_voc import PascalVOCDetectionEvaluator_
from .data.build import build_DA_detection_train_loader
import torch

class _DATrainer(SimpleTrainer):
    # one2one domain adpatation trainer
    def __init__(self, model, source_domain_data_loader, target_domain_data_loader, loss_weight, optimizer):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super(SimpleTrainer).__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.source_domain_data_loader = source_domain_data_loader
        self.target_domain_data_loader = target_domain_data_loader
        self._source_domain_data_loader_iter = iter(source_domain_data_loader)
        self._target_domain_data_loader_iter = iter(target_domain_data_loader)
        self.loss_weight = loss_weight
        self.optimizer = optimizer

    def run_step(self):
        assert self.model.training, "[DASimpleTrainer] model was changed to eval mode!"

        start = time.perf_counter()
        s_data = next(self._source_domain_data_loader_iter)
        data_time = time.perf_counter() - start

        start = time.perf_counter()
        t_data = next(self._target_domain_data_loader_iter)
        data_time = time.perf_counter() - start + data_time

        loss_dict = self.model(s_data, t_data)

        loss_dict = {l: self.loss_weight[l] * loss_dict[l] for l in self.loss_weight}
        losses = sum(loss_dict.values())
        self.optimizer.zero_grad()
        losses.backward()
        self._write_metrics(loss_dict, data_time)
        self.optimizer.step()
    

class DATrainer(DefaultTrainer):
    # one2one domain adpatation trainer
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        source_domain_data_loader = self.build_train_loader(cfg, 'source')
        target_domain_data_loader = self.build_train_loader(cfg, 'target')

        model = create_ddp_model(model, broadcast_buffers=False)

        loss_weight = {'loss_cls': 1, 'loss_box_reg': 1, 'loss_rpn_cls': 1, 'loss_rpn_loc': 1,\
        'loss_sap_source_domain': cfg.MODEL.DA_HEAD.LOSS_WEIGHT, 'loss_sap_target_domain': cfg.MODEL.DA_HEAD.LOSS_WEIGHT}

        self._trainer = _DATrainer(
            model, source_domain_data_loader, target_domain_data_loader, loss_weight, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())
    
    @classmethod
    def build_train_loader(cls, cfg, dataset_domain):
        if dataset_domain == 'source':
            return build_DA_detection_train_loader(cfg, dataset_domain=dataset_domain)
        elif dataset_domain == 'target':
            return build_DA_detection_train_loader(cfg, dataset_domain=dataset_domain)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return PascalVOCDetectionEvaluator_(dataset_name)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.LRScheduler(),
            hooks.IterationTimer(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]
        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    @classmethod
    def build_optimizer(cls, cfg, model):
        return torch.optim.Adam(
                [p for name, p in model.named_parameters() if p.requires_grad], 
                lr=cfg.SOLVER.BASE_LR,
                betas=(0.9, 0.999), weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )

class DefaultTrainer_(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return PascalVOCDetectionEvaluator_(dataset_name)
