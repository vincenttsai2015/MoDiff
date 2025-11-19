import torch
import argparse
import time
from parsers.parser import Parser
from parsers.config import get_config
from trainer import Trainer_G_DT_comp as Trainer
from sampler import Sampler_G_DiT as Sampler
import pynvml, json

def main(parsed_args):
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    ts = time.strftime('%b%d_%H%M', time.localtime())
    args = Parser().parse()
    print(f'args: {args}')

    config_file = f"{parsed_args.config_folder}/{parsed_args.config_prefix}_{parsed_args.scale}"
    config = get_config(config_file, args.seed)
    config.scale = parsed_args.scale
    config.type = args.beta_type

    # -------- Train --------
    if parsed_args.type == 'train_comp':
        trainer = Trainer(config)
        Tsf = f'{ts}_comp'
        ckpt = trainer.train(Tsf)

    # -------- Generation --------
    elif parsed_args.type == 'eval_comp':
        config.ckpt = parsed_args.ckpt_name
        sampler = Sampler(config)
        sampler.evaluation_ByCompound()

    else:
        raise ValueError(f'Wrong type : {parsed_args.type}')


if __name__ == '__main__':
    parsed_args = argparse.ArgumentParser(description="Operation mode and basic parameter configuration")
    parsed_args.add_argument('--type', type=str, required=True, choices=['train_comp', 'eval_comp'], default="train_comp", help="train_comp / eval_comp")
    parsed_args.add_argument('--scale', type=str, default="1h")
    parsed_args.add_argument('--config_folder', type=str, default="Stack")
    parsed_args.add_argument('--config_prefix',type=str, default="SO")  #consistent with folder, always be abbreviation
    parsed_args.add_argument('--ckpt_name',type=str,default="Sep30_SO1h_comp",help="Note that ckpt should be consistent with config")
    parsed_args, _ = parsed_args.parse_known_args()
    print(f'parsed_args: {parsed_args}')
    main(parsed_args)