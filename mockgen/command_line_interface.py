

def main():
    import xgutil.log_util as xglogutil

    import mockgen
    import mockgen.defaults as mgd
    import sys
    import argparse

    errcount = 0

    parser = argparse.ArgumentParser(description='Commandline interface to mockgen')

    parser.add_argument('ID',                            help=f'model ID [{mgd.ID}]',            type=str)
    parser.add_argument('--N',        default=mgd.N,     help=f'grid dimension [{mgd.N}]',       type=int)
    parser.add_argument('--seed',     default=mgd.seed,  help=f'seed [{mgd.seed}]',              type=int)
    parser.add_argument('--input',    default=mgd.input, help=f'input type [{mgd.input}]',       type=str)
    parser.add_argument('--Lbox',     default=mgd.Lbox,  help=f'box size in Mpc [{mgd.Lbox}]',   type=int)
    parser.add_argument('--Nside',    default=mgd.Nside, help=f'healpix Nside [{mgd.Nside}]',    type=int)
    parser.add_argument('--laststep', default=mgd.laststep, help=f'input type [{mgd.laststep}]', type=str)
    parser.add_argument('--gpu',      default=mgd.gpu,   help=f'use GPU [{mgd.gpu}]', action=argparse.BooleanOptionalAction)
    parser.add_argument('--mpi',      default=mgd.mpi,   help=f'use MPI [{mgd.mpi}]', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    mocksky = mockgen.Sky(ID = args.ID,
                           N = args.N,
                        seed = args.seed,
                        Lbox = args.Lbox,
                       input = args.input,
                    laststep = args.laststep,
                       Nside = args.Nside,
                         gpu = args.gpu,
                         mpi = args.mpi)

    errcount += mocksky.generate()

    return errcount

if __name__ == "__main__":
    import sys
    sys.exit(main())