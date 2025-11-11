import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import clean_cfg, get_cfg_default
from dassl.engine import build_trainer
from yacs.config import CfgNode as CN


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    cfg.TRAINER.OTPCLIP = CN()
    cfg.TRAINER.OTPCLIP.PREC = "amp"

    if "NLPROMPT" not in cfg.TRAINER:
        cfg.TRAINER.NLPROMPT = CN()
    cfg.TRAINER.NLPROMPT.N_CTX = 16
    cfg.TRAINER.NLPROMPT.CTX_INIT = ""
    cfg.TRAINER.NLPROMPT.CSC = False
    cfg.TRAINER.NLPROMPT.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.NLPROMPT.PREC = "fp32"

    cfg.MODEL.ATTR_BANK = ""
    cfg.MODEL.TAU = 0.07

    cfg.MODEL.PROMPT = CN()
    cfg.MODEL.PROMPT.N_CTX = 16
    cfg.MODEL.PROMPT.SPARSE_LAMBDA = 0.0
    cfg.MODEL.PROMPT.GROUP_LAMBDA = 0.0
    cfg.MODEL.PROMPT.ORTH_LAMBDA = 0.0

    cfg.MODEL.PROMPT_NOISE = CN()
    cfg.MODEL.PROMPT_NOISE.GRANULARITY = "token"
    cfg.MODEL.PROMPT_NOISE.S_INIT = 0.05
    cfg.MODEL.PROMPT_NOISE.T_INIT = 0.02
    cfg.MODEL.PROMPT_NOISE.KL_LAMBDA = 0.0
    cfg.MODEL.PROMPT_NOISE.L1_LAMBDA = 0.0
    cfg.MODEL.PROMPT_NOISE.ANNEAL = CN()
    cfg.MODEL.PROMPT_NOISE.ANNEAL.START = 0.0
    cfg.MODEL.PROMPT_NOISE.ANNEAL.END = 0.0
    cfg.MODEL.PROMPT_NOISE.CAP_SIGMA = 0.1

    cfg.MODEL.MTA = CN()
    cfg.MODEL.MTA.H = 0.7
    cfg.MODEL.MTA.T = 7
    cfg.MODEL.MTA.LAMBDA = 0.0
    cfg.MODEL.MTA.ENTROPY = True
    cfg.MODEL.MTA.TRAIN_GRAD = False

    cfg.MODEL.OT = CN()
    cfg.MODEL.OT.SINKHORN = 30
    cfg.MODEL.OT.TAU_CLEAN = 0.5
    cfg.MODEL.OT.TEMP = 0.07

    cfg.TRAIN.LR_PROMPT = 2e-5
    cfg.TRAIN.LR_BACKBONE = 0.0

    cfg.LOSS = CN()
    cfg.LOSS.W = CN()
    cfg.LOSS.W.CE = 1.0
    cfg.LOSS.W.ROBUST = 0.0
    cfg.LOSS.W.OT = 0.0
    cfg.LOSS.W.PROMPT_REG = 0.0
    cfg.LOSS.W.NOISE_REG = 0.0
    cfg.LOSS.ROBUST_TYPE = "MAE"
    cfg.LOSS.GCE_Q = 0.7

    if "METRICS" not in cfg.TEST:
        cfg.TEST.METRICS = []


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    clean_cfg(cfg, args.trainer)
    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument(
        "--output-dir", type=str, default="", help="output directory"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains",
        type=str,
        nargs="+",
        help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains",
        type=str,
        nargs="+",
        help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument(
        "--trainer", type=str, default="", help="name of trainer"
    )
    parser.add_argument(
        "--backbone", type=str, default="", help="name of CNN backbone"
    )
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument(
        "--eval-only", action="store_true", help="evaluation only"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch",
        type=int,
        help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
