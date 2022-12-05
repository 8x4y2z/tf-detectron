"""Meta network of APM and Centernet.
APM provides proposals and Centernel provides detections
"""
import logging
from collections import OrderedDict, abc
from typing import Union, List
import time
import datetime
from contextlib import ExitStack
import os

import numpy as np
import torch
from torch import nn
from PIL import Image
from detectron2.layers.nms import batched_nms

import detectron2.utils.comm as comm
from detectron2.structures import Instances, Boxes
from detectron2.train_apm import Trainer as APMTrainer
from detectron2.train import Trainer as CenterNetTrainer
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    DatasetEvaluator,
    DatasetEvaluators,
    print_csv_format,
    verify_results,
    inference_context
    )
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data.datasets import register_coco_instances


if "lisa-train-val" not in MetadataCatalog.list():

    register_coco_instances(
        "lisa-train-val",
        {},
        "datasets/lisa/train_val.json",
        "datasets/lisa/train_val"
    )

def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

class LisaTrainer:
    """
    This trainer is suitable only for inference/eval
    """

    @classmethod
    def build_evaluator(cls, cfg1, dataset_name, output_folder=None):
        return build_evaluator(cfg1, dataset_name, output_folder)

    @classmethod
    def build_model(cls,cfg1,cfg2):
        apm = APMTrainer.build_model(cfg1)
        centernet = CenterNetTrainer.build_model(cfg2)
        return apm,centernet

    @classmethod
    def test(cls, cfg1,cfg2,model, evaluators=None):
        apm,centernet = model
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg1.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg1.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg1.DATASETS.TEST):
            # data_loader = cls.build_test_loader(cfg1, dataset_name)
            data_loader = APMTrainer.build_test_loader(cfg1,dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg1, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = modified_inference_on_dataset(apm,centernet, data_loader, evaluator,cfg2)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def modified_inference_on_dataset(
        apm, centernet,data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],cfg
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(apm, nn.Module) and isinstance(centernet,nn.Module):
            stack.enter_context(inference_context(apm))
            stack.enter_context(inference_context(centernet))
        stack.enter_context(torch.no_grad())
        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            apm_outputs = apm(inputs)
            outputs = []
            for proposal in apm_outputs[0].proposal_boxes.tensor:
                inp_dict = {}
                a0,b0,a1,b1 = tuple(int(aa) for aa in proposal)
                cropped_img = inputs[0]["image_orig"][b0:b1,a0:a1]
                cropped_img = np.ascontiguousarray(Image.fromarray(cropped_img)
                                     .convert("YCbCr")
                                     .resize((cfg.INPUT.MIN_SIZE_TEST,cfg.INPUT.MIN_SIZE_TEST))
                                     )
                inp_dict["image"] = torch.tensor(cropped_img).permute([2,0,1])
                inp_dict["height"] = cropped_img.shape[0]
                inp_dict["width"] = cropped_img.shape[1]
                inp_dict["file_name"] = inputs[0]["file_name"]
                inp_dict["image_id"] = inputs[0]["image_id"]
                output = centernet([inp_dict])[0]
                # Bring back to input image dimensions
                output["instances"].pred_boxes.tensor[:,0::2] += a0
                output["instances"].pred_boxes.tensor[:,1::2] += b0
                outputs.append(output)

            # Now perform NMS
            output_boxes = type(outputs[0]["instances"].pred_boxes).cat([a["instances"].pred_boxes for a in outputs])
            output_scores = torch.cat([a["instances"].scores for a in outputs])
            cats = torch.cat([a["instances"].pred_classes for a in outputs])
            out_indices = batched_nms(output_boxes.tensor,output_scores,cats,0.3)

            output_boxes = output_boxes.tensor[out_indices]
            output_scores = output_scores[out_indices]
            output_cats = cats[out_indices]

            outputs = [{"instances":Instances((inputs[0]["height"],inputs[0]["width"]),**{

                "pred_boxes": Boxes(output_boxes),
                "scores": output_scores,
                "pred_classes":output_cats
            })}]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()
            # if idx>5:
            #     break

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results









def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg1 = get_cfg()
    cfg1.merge_from_file(args.config_file1)
    cfg1.merge_from_list(args.opts)

    cfg1.freeze()
    default_setup(cfg1, args)

    cfg2 = get_cfg()
    cfg2.merge_from_file(args.config_file2)
    cfg2.merge_from_list(args.opts)

    cfg2.freeze()
    default_setup(cfg2, args)
    return cfg1, cfg2


def main(args):
    cfg = setup(args)
    cfg1,cfg2 = cfg

    if args.eval_only:
        model = LisaTrainer.build_model(cfg1,cfg2)
        DetectionCheckpointer(model[0], save_dir=cfg1.OUTPUT_DIR).resume_or_load(
            cfg1.MODEL.WEIGHTS, resume=args.resume
        )
        DetectionCheckpointer(model[1], save_dir=cfg2.OUTPUT_DIR).resume_or_load(
            cfg2.MODEL.WEIGHTS, resume=args.resume
        )
        res = LisaTrainer.test(cfg1,cfg2, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(LisaTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg1, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    raise NotImplementedError("This is only for inference/eval")
    # trainer = LisaTrainer(cfg)
    # trainer.resume_or_load(resume=args.resume)
    # if cfg.TEST.AUG.ENABLED:
    #     trainer.register_hooks(
    #         [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
    #     )
    # return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
