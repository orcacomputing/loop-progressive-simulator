# Progressive simulation algorithm for loop-based Boson Sampling systems

This repository contains software to accompany the paper
> Samo Novák and David D. Roberts and Alexander Makarovskiy and Raúl García-Patrón and William R. Clements.
> "Boundaries for quantum advantage with single photons and loop-based time-bin interferometers" (2024).
> [arXiv:2411.16873 [quant-ph]](https://arxiv.org/abs/2411.16873)


The repository contains:

1. a reference implementation of the progressive simulation algorithm for loop-based Boson Sampling systems (see [`loop_progressive_simulator/`](loop_progressive_simulator/)),
2. code and utilities used to compute the memory complexity of such simulation using the lattice path formalism introduced in the paper (see [`shell_utilities/run_experiment_families.py`](shell_utilities/run_experiment_families.py),
3. a Jupyter notebook used to generate the figures in the paper ([`notebooks/paper_plot_generation.ipynb`](notebooks/paper_plot_generation.ipynb)).



## How to use the main experiment-generation script

The utilities to run experiments and compute their sampling complexities are found in [`shell_utilities/`](shell_utilities/). To allow for easy generation of a large amount of different experiments, we use a two-layered architecture consisting of an experiment scheduler [`run_experiment_families.py`](shell_utilities/run_experiment_families.py) acting as the recommended user interface, and the code to run an individual experiment [`complexity_experiment.py`](shell_utilities/complexity_experiment.py).
The scheduler `run_experiment_families.py` will run `complexity_experiment.py` several times to generate many instances of given experiments. The scheduler will use multiple CPUs (if available) to do this, parallelizing the work.

In the following, we call the set of instances of a given experiment (a specified number of modes, input state, loops, etc.) an *experiment family*.

The recommended way to use the utilities is to run [`shell_utilities/run_experiment_families.py`](shell_utilities/run_experiment_families.py)supplied with arguments defining families of experiments, possibly parametrized on the number of modes, photons, loop lengths, etc. The general syntax is:

```sh
python3 run_experiment_families.py -E "<experiment specification>" [-E "<...>"] ... -P <max_proc>
```

Here, the argument `-E` defines experiment families, and its contents are a string (enclosed in quotation marks) consisting of experiment arguments defined below. A call like above may contain one or more occurences of `-E`. The argument `-P` specifies the maximum number of processors to use. It can be omitted, in which case all CPUs will be used.


### Example experiments 

Here, we show a few examples of the usage of `run_experiment_families.py`. The full specification can be found by running:
```sh
python3 run_experiment_families.py --help
```
or if needed also:
```sh
python3 complexity_experiment.py --help
```



#### Simple (non-parametric) experiment

Say we want to simulate an interferometer on $m=5$ modes composed of loops of lengths $\underline\ell = (1,2,3)$ and with the input state $\ket{\underline n} = \ket{1,1,2,0,0}$. We want to validate the heuristic, so we run both the wavefunction simulator (with random beamsplitter angles), as well as the lattice-path based heuristic. In each case, we generate 100 samples, and we use 10 processors for this operation. We invoke the following:

```sh
python3 run_experiment_families.py \
    -E "-m 5 -ell 1,2,3 -n 1,1,2,0,0 -rep 100 -no-N -- -wA -thr -p" \
    -E "-m 5 -ell 1,2,3 -n 1,1,2,0,0 -rep 100 -no-N -- -wN -LP -Nh paths -p" \
    -P 10
```

Above, the arguments within `-E` are divided into two parts, separated by `--`. The first part contains arguments to the experiment scheduler itself, while the second part contains arguments passed directly to the complexity experiment code (see description of the layers above).

In this example, the scheduler arguments (before `--`) are:
- `-m 5` the number of modes $m = 5$,
- `-ell 1,2,3` the loop lengths $\underline\ell = (1,2,3)$, notice the comma-separated values,
- `-n 1,1,2,0,0` the input state $\ket{\underline n} = \ket{1,1,2,0,0}$ in the number (mode occupation) basis, with a photon each in the first two modes, two photons in mode 2, and zero elsewhere,
- `-rep 100` number of samples per experiment,
- `-no-N` saying we do not want to generate a sample for each possible measurement outcome $\ket{\underline N}$. If omitted, the scheduler will run an experiment instance for each possible $\ket{\underline N}$, postselecting on that outcome to compute its complexity. In most cases, we recommend using `-no-N` which selects outcomes using true amplitudes or a heuristic probability (see below).

The arguments passed directly to the complexity simulator (after `--`) are:
- in the first line with `-E` above:
  - `-wA` run full `w`avefunction simulator, i.e. compute true `A`mplitudes,
  - `-thr` beamsplitter angles $`\{\theta_i\}_i`$ (`th`) are chosen `r`andomly in each instance; see `complexity_experiment.py --help` for other options of selecting $`\{ \theta_i \}_i`$
  - `-p` use `p`rogressive decomposition.
- in the second line:
  - `-wN` do not perform `w`avefunction simulation, i.e. statevector is `N`ot computed,
  - `-LP` compute the `L`attice `P`ath diagram evolution instead,
  - `-Nh paths` choose the measurement outcome $\ket{\underline N}$ using the `h`euristic probability called here (lattice) `paths`.


#### Parametrized experiment family

Now we run a more complex example, more in the spirit of the experiments used in the paper (see the [notebook](notebooks/paper_plot_generation.ipynb)); though a little different to show the framework.

We simulate interferometers with loop lengths $\underline\ell = (1, \ell, \ell^2)$ where $`\ell \in \{ 1, \dots, 5 \}`$. We allow the number of modes to vary as well, taking the values $`m \in \{ R = 2+\ell+\ell^2, R + 10, R + 20, \dots, 280 \}`$. The input state used will be always $\ket{\underline n} = \ket{1,0,1,0,\dots,1,0,(1)}$, where the $(1)$ is present if the number of modes is odd. This is a large family of experiments, the majority of which are too resource and time-heavy to do wavefunction simulation, so we only use the lattice path formalism. The command invoked is:

```sh
python3 run_experiment_families.py \
    -E "-m m -ell 1,l,ll -rep 100 -no-N -r m=R:281:10 -r l=1:6 -e ll=l*l -e R=2+l+ll -- -wN -LP -Nh paths -p"
```

Above:
- `-m m` means the number of modes in an experiment will be defined by a variable `m`,
- `-ell 1,l,ll` means the loop lengths are given by 1, variable `l`, and variable `ll`,
- `-r m=R:281:10` defines a variable `m` taking the values in `range(R, 281, 10)` in the Python sense, i.e. $R \le m < 281$ (note never equal to 281), and taking steps of size 10,
- `-r l=1:6` defines a variable `l` taking values in `range(1, 6)`, i.e. $l = \{ 1, 2, \dots, 5 \}$
- `-e ll=l*l` defines a variable `ll` that, in each instance, evaluates to `l*l`,
- `-e R=2+l+ll` define a variable `R` that, in each instance, evaluates to `2+l+ll`,
- the absence of `-n` implies the default input state $\ket{\underline n} = \ket{1,0,1,0,\dots,1,0,(1)}$.

The variable/evaluation framework is intentionally limited. Variable names must match the regular expression `[a-zA-Z]+`, and they can be used only in `-m`, `-ell`, `-e`, `-r`. 
The range definition (`-r`) syntax is `VAR=START:STOP[:STEP]` and the semantics are those of Python's `range`. The evaluation (`-e`) allows only basic arithmetic, however ***it will execute code***.



### Output data

The utilities, as used above, will create two subdirectories in the current working directory:
- `logs/` &mdash; the scheduler threads (if using multiple CPUs) will write logs saying what experiments they are running, and where the output data can be found. If errors occur, these will be found here as well.
- `experiments/` &mdash; the complexity experiment families will each get a subdirectory here, by default identified by the start time and a random string. Each such subdirectory will contain output data from individual runs of that experiment family, as json files. At the moment, these contain a lot of information by default. See in the [notebook](notebooks/paper_plot_generation.ipynb) how to process this data.



### Full argument specification

In order to see the full specification of the scheduler, call:
```sh
python3 run_experiment_families.py --help
```

In order to see the full specification of the complexity experiment code itself, in particular arguments that may go after `--` in the above examples, call:
```sh
python3 complexity_experiment.py --help
```

Note that some arguments to `complexity_experiment.py` are passed by the scheduler `run_experiment_families.py`, and should not be passed in `-E "... -- <here>"` unless you specifically want to overwrite them.
