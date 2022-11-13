# -*- coding: utf-8 -*-
from src.engine import default_arg_parser, DefaultTrainer, default_setup
from src.config import get_cfg


class Trainer(DefaultTrainer):
    """Trainer that simply calls defalut trainer, but can be customized
    """
    def __init__(self, *args,**kwargs):
        super(Trainer, self).__init__(*args,**kwargs)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg




def main(args):
    config = setup(args)

    if args.eval_only:
        model = Trainer.build_model(config)
        DetectionCheckpointer(model, save_dir=config.OUTPUT_DIR).resume_or_load(
            confif.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(config, model)
        verify_results(config, res)
        return res

    trainer = Trainer(config)
    trainer.resume_or_load(resume=args.resume)
    results = trainer.train()
    return results


if __name__ == '__main__':
    arg_parser = default_arg_parser()
    args = arg_parser.parse_args()
    main(args)
