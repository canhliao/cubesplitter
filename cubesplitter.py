from argparse import ArgumentParser, Namespace
import os
import cubes

parser = ArgumentParser()
parser.add_argument('cube', help='Cube to be split into alpha real, alpha imaginary, beta real, and beta imaginary', type=str)
args: Namespace = parser.parse_args()

mycube = cubes.Cube(args.cube)
basename = args.cube.split(".")[0]
spins = ['A', 'B']
for spin in spins:
    mycube.compute_2c()
    mycube.write_out(f"{basename}_{spin}R.cube", data=f'R{spin}')
    mycube.write_out(f"{basename}_{spin}I.cube", data=f'I{spin}')