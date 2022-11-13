# -*- coding: utf-8 -*-

from argparse import  ArgumentParser, REMAINDER
import os
import sys
from datetime import datetime
import random
import logging
from collections import defaultdict
import weakref

from omegaconf import OmegaConf
import numpy as np
import tensorflow as tf
import PIL
from tabulate import tabulate
from .trainers import TrainerBase
from src.config import CfgNode
from src.utils.logger import setup_logger
from src.core.models import build_model

def default_arg_parser(help=None):

    default_help = """
    """
    parser = ArgumentParser(help or default_help)
    parser.add_argument("--config",default="",help="Path to config file")
    parser.add_argument("--resume",action="store_true",help="Resume from checkpoint directory")
    parser.add_argument("--eval_only", action="store_true", help="perform evaluation only")
    parser.add_argument("opts",default=None,nargs=REMAINDER,help="Args to pass into Config")

    return parser

class DefaultTrainer(TrainerBase):
    """
    A trainer with default training logic. It does the following:

    1. Create a :class:`SimpleTrainer` using model, optimizer, dataloader
       defined by the given config. Create a LR scheduler defined by the config.
    2. Load the last checkpoint or `cfg.MODEL.WEIGHTS`, if exists, when
       `resume_or_load` is called.
    3. Register a few common hooks defined by the config.

    """
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()
        logger = logging.getLogger("tfdetectron")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
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
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model


def _try_get_key(cfg, *keys, default=None):
    """
    Try select keys from cfg until the first key that exists. Otherwise return default.
    """
    if isinstance(cfg, CfgNode):
        cfg = OmegaConf.create(cfg.dump())
    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            return p
    return default

def _collect_env_info():
    gpus = tf.config.list_physical_devices('GPU')
    has_gpu = bool(gpus)  # true for both CUDA & ROCM
    tf_version = tf.__version__

    has_rocm = False
    has_cuda = has_gpu and (not has_rocm)

    data = []
    data.append(("sys.platform", sys.platform))  # check-template.yml depends on it
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("numpy", np.__version__))

    try:
        import src  # noqa

        data.append(
            ("Tfdetectron", src.__version__ + " @" + os.path.dirname(src.__file__))
        )
    except ImportError:
        data.append(("Tfdetectron", "failed to import"))
    except AttributeError:
        data.append(("Tfdetectron", "imported a wrong installation"))

    data.append(("Tensorflow", tf_version + " @" + os.path.dirname(tf.__file__)))

    if not has_gpu:
        has_gpu_text = "No: No GPU detected"
    else:
        has_gpu_text = "Yes"
    data.append(("GPU available", has_gpu_text))
    if has_gpu:
        devices = defaultdict(list)
        for k in range(len(gpus)):
            details = tf.config.experimental.get_device_details(gpus[k])
            cap = ".".join((str(x) for x in details["compute_capability"]))
            name = details["device_name"] + f" (arch={cap})"
            devices[name].append(str(k))
        for name, devids in devices.items():
            data.append(("GPU " + ",".join(devids), name))

    data.append(("Pillow", PIL.__version__))

    try:
        import cv2

        data.append(("cv2", cv2.__version__))
    except (ImportError, AttributeError):
        data.append(("cv2", "Not found"))
    env_str = tabulate(data) + "\n"
    return env_str

def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in TF, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = _try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
    if output_dir is not None:
        os.makedirs(output_dir,exist_ok=True)

    setup_logger(output_dir, name="core")
    logger = setup_logger(output_dir)

    logger.info("Rank of current process: {}. World size: {}".format(1,1))
    logger.info("Environment info:\n" + _collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                (open(args.config_file, "r").read(), args.config_file),
            )
        )

    # Note: some of our scripts may expect the existence of
    # config.yaml in output directory
    path = os.path.join(output_dir, "config.yaml")
    if isinstance(cfg, CfgNode):
        logger.info("Running with full config:\n{}".format((cfg.dump(), ".yaml")))
        with open(path, "w") as f:
            f.write(cfg.dump())
    else:
        raise RuntimeError("Unknow error occured, please fix!")
    logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed = _try_get_key(cfg, "SEED", "train.seed", default=-1)
    seed_all_rng(None if seed < 0 else seed )

