import argparse

class GSDM_Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='GSDM')
        self.parser.add_argument('--gsdm_type', type=str, default="train")
        self.parser.add_argument('--config', type=str, default="community_small",help="Path of config file")
        self.parser.add_argument('--comment', type=str, default="", help="A single line comment for the experiment")
        self.parser.add_argument('--seed', type=int, default=42)
        self.parser.add_argument('--beta_type', type=str, default="linear") # linear, exp, cosine, tanh
        
    def parse(self):
        args, unparsed = self.parser.parse_known_args()        
        print(f'Parsed arguments: {args}')
        print(f'Unparsed arguments: {unparsed}')
        # if len(unparsed) != 0:
        #     raise SystemExit(f'Unknown arguments: {unparsed}')
        return args