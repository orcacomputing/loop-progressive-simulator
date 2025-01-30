import sys
import os
import argparse
import base64
import json

from datetime import datetime
from random import randbytes

import numpy as np

sys.path.append(os.path.abspath("../loop_progressive_simulator"))
from bscircuits import ConcreteBSCircuit, multiple_loops
from complexity_analysis import fc_nonprogressive_compare_lp, fc_progressive
from fock_states import AbstractBosonicFockState, SparseBosonicFockState
from lattice_paths import PSCSpace


def int_tuple(s):
    try:
        return tuple(int(c) for c in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("Comma-separated list of integers expected.")


def angle(s):
    ERR = argparse.ArgumentTypeError("Comma-separated list of floats or fractions expected.")
    try:
        return float(s)

    except ValueError:
        frac = s.split("/")

        if len(frac) != 2:
            raise ERR

        try:
            p, q = tuple(int(x) for x in frac)
        except:
            raise ERR

        return p * np.pi / q


def angle_tuple(s):
    return [angle(c) for c in s.split(",")]


def generate_identifier():
    dt = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    rs = randbytes(6)
    rs = base64.urlsafe_b64encode(rs).decode("ascii")
    return f"{dt}_{rs}"


def parse_args():
    parser = argparse.ArgumentParser("complexity_experiment",
                                     description = "Run a single complexity calculation for a given loop and input configuration.",
                                     epilog="If ran as part of a batch of experiments, provide --experiment-id; in this case, all instances should have the same --number-modes, --loop-lengths, and --input-state. The --output-state will be allowed to vary.\n\nIf ran individually, --experiment-id can be omitted.")

    parser.add_argument("-ell", "--loop-lengths", type=int_tuple, required=True,
                        help="Comma-separated ordered list of lengths of loops composed in sequence from left to right.")
    parser.add_argument("-m", "--number-modes", type=int,
                        help="Number of modes.")

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("-n", "--input-state", type=int_tuple, required=False,
                             help="Comma-separated number basis state to be used as input. Default: 1,0,1,0,...,1,0")

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("-N", "--output_state", type=int_tuple,
                              help="Comma-separated number basis state to be post-selected on output.")
    output_group.add_argument("-Nh", "--output-state-heuristic",
                              choices=["paths", "uniform"],
                              help="Use a heuristic probability to choose the output state. Requires the knowledge of the number of modes (e.g. from -m or -n).")

    wavefunction_group = parser.add_mutually_exclusive_group(required=True)
    wavefunction_group.add_argument("-wA", "--amplitudes", action="store_true",
                                    help="Use when you want to simulate amplitudes. Angles can be supplied using -th/-thf/-tha/-thr/-thR. If no output is selected (-N/-Nh), then the computed amplitudes are used for measurement probabilities.")
    wavefunction_group.add_argument("-wP", "--possibilistic", action="store_true",
                                    help="Run the simulator in abstract mode, without computing amplitudes. Angles not needed. Requires selecting output (-N/-Nh).")
    wavefunction_group.add_argument("-wN", "--no-wavefunction", action="store_true",
                                    help="No wavefunction evolution at all. Angles not needed. Requires selecting output (-N/-Nh).")


    angle_group = parser.add_mutually_exclusive_group()
    angle_group.add_argument("-th", "--beamsplitter-angles", type=angle_tuple, metavar="ANGLES",
                             help="Comma-separated list of beamsplitter angles, given as absolute floats (e.g. 3.14), or as fractions of pi (e.g. -2/3 which means -2pi/3).")
    angle_group.add_argument("-thf", "--beamsplitter-angles-file", metavar="FILE",
                             help="File containing beamsplitter angles, in the same format as -th.")
    angle_group.add_argument("-tha", "--beamsplitter-angles-all", type=angle, metavar="ANGLE",
                             help="A single angle, same for all beamsplitters.")
    angle_group.add_argument("-thr", "--beamsplitter-angles-random", action="store_true",
                             help="Generate uniformly random BS angles from [0, 2pi).")
    angle_group.add_argument("-thR", "--beamsplitter-angles-random-range", type=angle, nargs=2, metavar="ANGLE",
                             help="Generate uniformly random BS angles from [α, β); usage is `-thR α β`.")


    prog_group = parser.add_mutually_exclusive_group(required=True)
    prog_group.add_argument("-p", "--progressive", action="store_true", dest="progressive",
                            help="Run the simulator with progressive measurements.")
    prog_group.add_argument("-np", "--non-progressive", action="store_false", dest="progressive",
                            help="Run the simulator without progressive measurements, i.e. entire circuit as one component.")

    parser.add_argument("-LP", "--lattice-paths", action="store_true",
                        help="Run step simulator and lattice path state enumeration simultaneously, and compare bases.")
    parser.add_argument("-C", "--only-complexities", action="store_true",
                        help="Only store the Fock complexities (and measurement parameters), instead of full information (e.g. diagrams).")

    parser.add_argument("-i", "--experiment-id",
                        help="Identifier of the experiment run. Default: generate a new unique identifier.")
    parser.add_argument("-o", "--output",
                        help="Output file. If an output state is selected, then the default is 'experiments/{experiment_id}/{output_state}.json'. Otherwise, the file is '{experiment_id}/any_{random identifier}.json'.")

    args = parser.parse_args()



    if args.progressive and args.output_state is None and args.output_state_heuristic is None and not args.amplitudes:
        parser.error("In progressive (-p) abstract (no -A) mode, output state (-N) is required.")

    if not args.progressive and not args.lattice_paths:
        print("WIP Currently, non-progressive simulation (-np) implies lattice paths (-LP).")
        args.lattice_paths = True

    if args.lattice_paths and args.loop_lengths[0] != 1:
        parser.error(f"When using lattice paths (-LP), the first loop must have length 1, but {args.loop_lengths[0]} was used.")

    if all(x is None for x in (args.number_modes,
                               args.input_state,
                               args.output_state)):
        parser.error("Unknown number of modes. Specify using -m, or imply from -n/-N.")


    if args.number_modes is None:
        if args.output_state is not None:
            args.number_modes = len(args.output_state)
            if args.input_state is not None and len(args.input_state) != args.number_modes:
                parser.error("Length of input state must match the number of modes.")
        else:
            args.number_modes = len(args.input_state)
            if args.output_state is not None and len(args.output_state) != args.number_modes:
                parser.error("Length of output state must match the number of modes.")

    if args.input_state is None:
        args.input_state = (1,0) * (args.number_modes // 2) + ((1,) if args.number_modes % 2 == 1 else ())

    elif args.number_modes != len(args.input_state):
        parser.error("The provided input state (-n) does not match the number of modes (from -m or -N).")


    if args.output_state is not None and sum(args.input_state) != sum(args.output_state):
        parser.error(f"Input and output states have inequal numbers of photons: Input {args.input_state} has {sum(args.input_state)} and output {args.output_state} has {sum(args.output_state)}.")


    angles_provided = any([args.beamsplitter_angles_random,
                           args.beamsplitter_angles is not None,
                           args.beamsplitter_angles_file is not None,
                           args.beamsplitter_angles_all is not None,
                           args.beamsplitter_angles_random_range is not None])
    if args.amplitudes:
        if not angles_provided:
            parser.error("-A requires one of -th/-thr/-thf/-tha.")
        else:
            _X = multiple_loops(*args.loop_lengths, modes=args.number_modes)
            number_beamsplitters = len(_X)

            if args.beamsplitter_angles_random:
                args.beamsplitter_angles = list(2 * np.pi * np.random.random(number_beamsplitters))

            elif args.beamsplitter_angles_random_range is not None:
                alpha, beta = args.beamsplitter_angles_random_range
                args.beamsplitter_angles = list(alpha + (beta - alpha) * np.random.random(number_beamsplitters))

            elif args.beamsplitter_angles_all is not None:
                args.beamsplitter_angles = [args.beamsplitter_angles_all] * number_beamsplitters

            elif args.beamsplitter_angles_file is not None:
                with open(args.beamsplitter_angles_file, "r") as f:
                    s = f.read().strip()
                    args.beamsplitter_angles = angle_tuple(s)

            if len(args.beamsplitter_angles) != number_beamsplitters:
                parser.error(f"Expected {number_beamsplitters} angles; received {len(args.beamsplitter_angles)}.")

    else:
        if angles_provided:
            print("Warning: Beamsplitter angles were provided, but -A is off. Angles will be ignored.")
            args.beamsplitter_angles = None


    if args.experiment_id is None:
        args.experiment_id = generate_identifier()


    if args.output is None:
        dir = f"experiments/{args.experiment_id}"
        os.makedirs(dir, exist_ok=True)

        if args.output_state is None:
            file_name = "any"
        else:
            file_name = "-".join(str(n) for n in args.output_state)

        if len(file_name) > 255:
            file_name = "select"

        # when we have random angles, we assume that the experiment will be repeated with the same parameters;
        # thus we add another identifier to files to not overwrite them, but to be able to have them in the same directory
        if args.amplitudes or args.output_state_heuristic or file_name == "select":
            file_name += f"_{base64.urlsafe_b64encode(randbytes(6)).decode('ascii')}"

        args.output = f"{dir}/{file_name}.json"

    return args


def fc_multi_loop(args):
    assert len(args.loop_lengths) >= 2

    if args.amplitudes:
        Psi = SparseBosonicFockState({args.input_state: 1})
    else:
        Psi = AbstractBosonicFockState({args.input_state})

    if args.progressive:
        X = multiple_loops(*args.loop_lengths, modes=args.number_modes, thetas=args.beamsplitter_angles)
        return fc_progressive(X, Psi, args.output_state, args.lattice_paths,
                              heuristic = args.output_state_heuristic,
                              evolve_vector = args.amplitudes)

    else:
        X1 = multiple_loops(args.loop_lengths[0], modes=args.number_modes)
        Xr = multiple_loops(*args.loop_lengths[1:], modes=args.number_modes)

        if args.amplitudes:
            X1 = ConcreteBSCircuit.from_abstract(X1, args.beamsplitter_angles[:len(X1)])
            Xr = ConcreteBSCircuit.from_abstract(Xr, args.beamsplitter_angles[len(X1):])

        S = PSCSpace.from_basis_state(args.input_state)
        return fc_nonprogressive_compare_lp(X1, Xr, Psi, S)[0]



def print_configuration(args):
    print(f"""\
\t=======>    COMPLEXITY EXPERIMENT    <=======

Time: {datetime.now()}
(Parent) run ID: {args.experiment_id}
Saving in: {args.output}
Progressive: {'Y' if not args.progressive else 'N'}\t\tLattice paths: {'Y' if not args.progressive else 'N'}

Modes: {args.number_modes}\t\tLoop lengths: {args.loop_lengths}
Amplitudes: {'Y' if args.amplitudes else 'N'}\t\tHeuristic: {args.output_state_heuristic}\
""", end="")

    if args.amplitudes:
        print("\t\tAngles: ", end="")
        if args.beamsplitter_angles_file is not None:
            print(f"from file {args.beamsplitter_angles_file}")
        elif args.beamsplitter_angles_random:
            print("random")
        else:
            print("entered in command-line")
    else:
        print()

    print(f"""\
Input state: {args.input_state}
Output state: {args.output_state}

\t=============================================\n""")


def run_experiment(args):
    print_configuration(args)
    time_start = datetime.now()
    fc_log = fc_multi_loop(args)
    time_end = datetime.now()

    with open(args.output, "w") as f:
        json.dump(fp = f,
                  obj = {"meta": args.__dict__,
                         "time_start": str(time_start),
                         "time_end": str(time_end),
                         "fc_log": fc_log})


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
