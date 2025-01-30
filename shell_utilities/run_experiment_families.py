import sys
import os
import argparse
import re
import subprocess

from threading import Thread
from copy import copy
from itertools import chain, product
from queue import Queue

sys.path.append(os.path.abspath("../loop_progressive_simulator"))
from bscircuits import multiple_loops
from lattice_paths import PermutableCumulativeSpace, apply_bscircuit_and_permute
from utils import cpu_count

from complexity_experiment import generate_identifier, int_tuple


ARG_RANGE_RE = re.compile(r"^([a-zA-Z]+)=(\d+):(\d+)(?::(\d+))?$")
def arg_range(s):
    m = re.match(ARG_RANGE_RE, s)
    if not m:
        raise argparse.ArgumentTypeError("Expecting VAR=START:STOP[:STEP].")

    var = m.group(1)
    rng = (int(x) for x in m.groups()[1:] if x is not None)

    return var, range(*rng)


def arg_int_or_var(s):
    try:
        return int(s)
    except ValueError:
        if not re.match(r"^[a-zA-Z]+$", s):
            raise argparse.ArgumentTypeError("Expecting int or variable.")
        return s


ARG_SYM_TUPLE = re.compile(r"^((?:(?:[a-zA-Z]+|\d+),)*(?:[a-zA-Z]+|\d+))$")
ARG_SYM_RANGE = re.compile(r"^([a-zA-Z]+|\d+):([a-zA-Z]+|\d+)(?::([a-zA-Z]+|\d+))?$")
def arg_sym_tuple_range_or_ellipsis(s):
    m = re.match(ARG_SYM_TUPLE, s)
    if m:
        return tuple(arg_int_or_var(x) for x in s.split(","))

    m = re.match(ARG_SYM_RANGE, s)
    if m:
        return slice(*(arg_int_or_var(g) for g in m.groups() if g is not None))

    raise argparse.ArgumentTypeError("Expecting a range syntax START:STOP[:STEP] or a comma-separated list. Both will allow letter symbols.")


ARG_EVAL = re.compile(r"^([a-zA-Z]+)=([\w@+\-*/%()]+)$")
def arg_eval(s):
    print(s)
    m = re.match(ARG_EVAL, s)
    if m:
        return m.groups()

    raise argparse.ArgumentTypeError("Expecting an evaluation syntax VAR=EXPR.")


def parse_args():
    parser = argparse.ArgumentParser("run_experiment_families",
                                     description="Run a families of experiments.",
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-P", "--processes", type=int,
                        help="Maximum number of processes. Default: number of CPUs from nproc.")

    family_parser = argparse.ArgumentParser("Experiment family specification")

    family_parser.add_argument("-r", "--range", metavar=("VAR=START:STOP[:STEP]"), type=arg_range, action="append",
                               help="Define a variable VAR and its range using Python slice syntax.")
    family_parser.add_argument("-e", "--eval", action="append", dest="evals", type=arg_eval,
                               help="Evaluate expressions; expects VAR=EXPR (without spaces). Only simple arithmetic is allowed. DANGER: will execute code.")
    family_parser.add_argument("-ell", "--loops", type=arg_sym_tuple_range_or_ellipsis,
                               help="Lengths of loops composed in sequence.")
    family_parser.add_argument("-m", "--modes", type=arg_int_or_var, default=10)
    family_parser.add_argument("-n", "--input-state", type=int_tuple,
                               help="Comma-separated number basis state to be used as input. Default: 1,0,1,0,...,1,0")
    family_parser.add_argument("-rep", type=int, dest="random_angles_repeats", # legacy destination, the argument used to be -A-thr-rep, implying random angles; now, angles are provided in direct arguments, for example to support -tha/-thf arguments
                               help="Use when you want to simulate amplitudes. Suppy the number of repetitions. Use further arguments after --.")
    family_parser.add_argument("-no-N", "--no-output-selection", action="store_true", dest="single",
                               help="Do not iterate over all possible output states with the -N option, run the simulation just once.")

    family_parser.epilog = "Any arguments to be passed directly to the experiment code should go after --, e.g. '-- -LP'."

    def parse_family(s):
        parser_args = []
        direct_args = []
        split_args = s.split(" ")

        for i, a in enumerate(split_args):
            if a == "--":
                parser_args = split_args[:i]
                direct_args = split_args[i+1:]
                break
        else:
            parser_args = split_args

        args = family_parser.parse_args(parser_args)

        if args.range:
            args.range = dict(args.range)

        if args.evals is None:
            args.evals = []

        args.direct_args = direct_args

        return args


    parser.add_argument("-E", "--experiment_specs", action="append", type=parse_family, required=True,
                        help="Experiment family specification. Repeat -E multiple times for multiple experiment families. See usage below.")

    parser.epilog = family_parser.format_help()
    return parser.parse_args()


def csv(lst):
    return ",".join(str(x) for x in lst)


def gen_fam(ells, modes=10, state=None,
            single=False, direct_args=(), uid=None):
    assert len(ells) >= 2
    assert ells[0] == 1

    if not uid:
        uid = generate_identifier()

    if state is None:
        state = (1,0) * (modes // 2)
        if modes % 2 == 1:
            state = state + (1,)

    if single:
        yield ["-i", uid,
               "-ell", csv(ells),
               "-m", str(modes),
               "-n", csv(state),
               *direct_args]

    else:
        Y = multiple_loops(*ells[1:], modes=modes)
        S = PermutableCumulativeSpace.from_L1(state, modes)
        S = apply_bscircuit_and_permute(Y, S)

        for ns in S.enumerate_basis():
            yield ["-i", uid,
                "-ell", csv(ells),
                "-m", str(modes),
                "-n", csv(state),
                "-N", csv(ns),
                *direct_args]


def gen_fam_random_angles(number_repetitions, ells, modes=10, state=None,
                          single=False, direct_args=(), uid=None):

    if uid is None:
        # share identifier
        uid = generate_identifier()

    for _ in range(number_repetitions):
        yield from gen_fam(ells, modes, state,
                           single=single,
                           direct_args=direct_args,
                           uid=uid)



def parameter_valuations(ranges, evals):
    vars, gens = zip(*list(ranges.items()))
    prod_gen = product(*gens)

    for vals in prod_gen:
        valuation = {var:val for var, val in zip(vars, vals)}
        for e, expr in evals:
            valuation[e] = eval(expr, copy(valuation))
        yield valuation


def _loops_valuation(args, valuation):
    try:
        if isinstance(args.loops, tuple):
            loops = []
            for ell in args.loops:
                if isinstance(ell, int):
                    loops.append(ell)
                else:
                    loops.append(valuation[ell])

        elif isinstance(args.loops, slice):
            slc = args.loops.start, args.loops.stop, args.loops.step
            slc = [s if isinstance(s, int) else valuation[s]
                   for s in slc if s is not None]
            loops = range(*slc)

        else:
            raise TypeError(f"Loop vector {repr(args.loops)} must be a tuple or a slice.")

    except KeyError as exc:
        raise AttributeError(f"Loop vector {args.loops} contains an undefined variable {exc.args[0]}.") from None

    return tuple(loops)


def _mode_valuation(args, valuation):
    if isinstance(args.modes, int):
        return args.modes
    else:
        try:
            return valuation[args.modes]
        except KeyError:
            raise AttributeError(f"Number of modes {args.modes} contains an undefined variable.")


def experiment_generator_parametrized(args):
    print(args)
    if args.random_angles_repeats is not None:
        generator = lambda *in_args, **in_kwargs:\
            gen_fam_random_angles(args.random_angles_repeats, *in_args, **in_kwargs)
    else:
        generator = lambda *in_args, **in_kwargs:\
            gen_fam(*in_args, **in_kwargs)

    if args.range:
        valuations = parameter_valuations(args.range, args.evals)
        for val in valuations:
            print(val)
            ell = _loops_valuation(args, val)
            m = _mode_valuation(args, val)

            yield from generator(ell,
                                 modes = m,
                                 state = args.input_state,
                                 single = args.single,
                                 direct_args = args.direct_args)
    else:
        # only one family
        yield from generator(args.loops,
                             modes = args.modes,
                             state = args.input_state,
                             single = args.single,
                             direct_args = args.direct_args)



def full_command(args):
    return ["python3", "complexity_experiment.py", *args]


def queue_tasks(task_queue, args, number_processes):
    for cmd in map(full_command,
                   chain(*(experiment_generator_parametrized(e)
                           for e in args.experiment_specs))):
        print(f"Queueing task {cmd}")
        task_queue.put(cmd)

    # sigkill to all threads
    for _ in range(number_processes):
        task_queue.put("DONE")


def execute_tasks(task_queue, print_queue):
    def print(*args, **kwargs):
        print_queue.put((args, kwargs))

    thread_id = generate_identifier()
    log_filename = f"logs/thread_{thread_id}.log"
    print(f"Starting thread {thread_id}. Logging in {log_filename}.")

    while True:
        task = task_queue.get()
        if task == "DONE":
            print("DONE; ending thread")
            return

        print("Starting task:", task)
        # start subprocess and block
        proc = subprocess.Popen(task, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        print(f"Task finished. Appending output and error to {log_filename}.")
        with open(log_filename, "ab") as f:
            f.write(out)
            f.write(err)


def printer(print_queue, number_processes):
    done_counter = 0
    while True:
        if done_counter == number_processes:
            print("All processes signalled DONE. Shutting down printer thread.")
            return

        args, kwargs = print_queue.get()
        match args:
            case ["DONE; ending thread"]:
                done_counter += 1
        print(*args, **kwargs)


def run_experiments(args, number_processes):
    if not os.path.exists("logs"):
        os.mkdir("logs")

    task_queue = Queue(maxsize=100)
    print_queue = Queue()

    threads = [
        Thread(target=printer, args=(print_queue, number_processes)),
        Thread(target=queue_tasks, args=(task_queue, args, number_processes)),
        *[Thread(target=execute_tasks, args=(task_queue, print_queue))
          for _ in range(number_processes)]]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print("All threads have terminated.")



if __name__ == "__main__":
    args = parse_args()
    print(args)

    number_processes = cpu_count()
    print(f"Detected {number_processes} available CPUs using nproc.")

    if args.processes is not None and args.processes != number_processes:
        if args.processes < number_processes:
            print(f"Override number of processes using -P {args.processes}. Using only {args.processes} CPU.")
            number_processes = args.processes
        elif args.processes > number_processes:
            print(f"Override number of processes using -P {args.processes}, but only {number_processes} are available.")
            exit(1)

    run_experiments(args, number_processes)
