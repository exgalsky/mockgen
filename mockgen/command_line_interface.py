import lpt
import mockgen
from mockgen.util import MockgenDefaults

def main():

    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Commandline interface to mockgen')

    parser.add_argument('modelID', type=str)
    parser.add_argument('--N',       type=int, help=f'grid dimention [default = {MockgenDefaults["N"]}]', 
                        default=MockgenDefaults["N"])
    parser.add_argument('--seed',    type=int, help=   f'random seed [default = {MockgenDefaults["seed"]}]', 
                        default=MockgenDefaults['seed'])
    parser.add_argument('--ityp',    type=str, help=f'lpt input type [default = {MockgenDefaults["ityp"]}]',  
                        default=MockgenDefaults['ityp'])

    args = parser.parse_args()

    N     = args.N
    seed  = args.seed
    input = args.input
    ID    = args.modelID

    mocksky = mockgen.Sky(ID=ID,N=N,seed=seed,input=input)

    return mocksky.generate()

if __name__ == "__main__":
    import sys
    sys.exit(main())