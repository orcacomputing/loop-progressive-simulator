import json
from argparse import Namespace
from contextlib import AbstractContextManager
from datetime import datetime
from functools import reduce
from itertools import chain, tee
from types import NoneType
from typing import Callable, List, Tuple, Union

import numpy as np

import number_basis
from bscircuits import AbstractBSCircuit, BSCircuit, ConcreteBSCircuit, multiple_loops, progressive_decomposition, split_decomposition_L1_rest
from fock_states import AbstractBosonicFockState, SparseBosonicFockState, map_fock_basis, project_number_mode
from lattice_paths import PSCSpace, PermutableCumulativeSpace, apply_beamsplitter_and_permute, apply_beamsplitter_extensible, apply_bscircuit_and_permute, apply_nn_beamsplitter, path_heuristic_probability, projection_contraction
from step_simulator_nosym import apply_abstract_beamsplitter, apply_beamsplitter, mode_marginals_only, sample_project_collapse



def log_sim_lp_data(state, S):
    state_basis = set(state.basis())
    S_basis = set(number_basis.make_basis_state(b)
                  for b in S.enumerate_basis())

    state_not_S = state_basis.difference(S_basis)
    S_not_state = S_basis.difference(state_basis)

    return {"sim": state.dim,
            "lp": int(S.dim),
            "same": state.dim == int(S.dim),
            "sim\\lp": list(state_not_S),
            "lp\\sim": list(S_not_state),
            "diagram": {"mu": S.mu,
                        "nu": S.nu if isinstance(S, PSCSpace) else None,
                        "permutation": S.permutation.list()}}


def log_lp(diagram):
    return {"lp": int(diagram.dim),
            "diagram": {"mu": diagram.mu,
                        "nu": diagram.nu if isinstance(diagram, PSCSpace) else None,
                        "permutation": diagram.permutation.list()}}


def fc_circuit(circuit: BSCircuit,
               state: Union[SparseBosonicFockState, NoneType] = None,
               diagram: Union[PermutableCumulativeSpace, NoneType] = None,
               input_basis: Tuple[int,...] = None):

    if state is None:
        def log_data(_, S):
            return log_lp(S)

    elif diagram is not None:
        def log_data(Psi, diagram):
            m = diagram.modes
            relevant_state = map_fock_basis(lambda basis: basis[:m], Psi)
            return log_sim_lp_data(relevant_state, diagram)

    else:
        def log_data(Psi, _):
            return int(Psi.dim)


    dims = [log_data(state, diagram)]


    if state is None:
        def apply_step(*_args):
            return None

    elif isinstance(circuit, ConcreteBSCircuit):
        apply_step = apply_beamsplitter

    else: # AbstractBSCircuit
        apply_step = apply_abstract_beamsplitter


    for (a, b, *theta) in circuit:
        args = (a, b, *theta) + (state,)
        state = apply_step(*args)

        if diagram is not None:
            diagram = apply_beamsplitter_extensible(a, b, diagram, input_basis)

        dims.append(log_data(state, diagram))

    return (state, diagram), dims



def fc_measurement(state: SparseBosonicFockState, mode: int, number: Union[int, NoneType] = None):
    if isinstance(state, AbstractBosonicFockState):
        if number is None:
            raise ValueError("You must specify the output for a progressive simulator in abstract mode.")

        new_state = project_number_mode(mode, number, state)
        probability = None

    else: # SparseBosonicFockState
        marginals = mode_marginals_only(state, mode)

        # if no output is selected, sample one based on the marginals (as usual in PS)
        if number is None:
            number, new_state = sample_project_collapse(state, mode, marginals)

        else:
            new_state = project_number_mode(mode, number, state)
            new_state.normalize_in_place()

        probability = marginals[number]

    return new_state, probability, int(number)



def fc_nonprogressive_compare_lp(X1: BSCircuit, Xr: BSCircuit,
                                 state: SparseBosonicFockState, S: PermutableCumulativeSpace,
                                 lattice_paths : bool = True):

    dims = [log_sim_lp_data(state, S)]

    if isinstance(X1, ConcreteBSCircuit):

        for (a, b, theta) in X1:
            assert b == a+1, f"Something went wrong, X1 contains a beamsplitter of length {b-a}."
            state = apply_beamsplitter(a, b, theta, state)
            S = apply_nn_beamsplitter(a, S)

            dims.append(log_sim_lp_data(state, S))

        S = PermutableCumulativeSpace.from_cumulative_space(S)
        for (a, b, theta) in Xr:
            state = apply_beamsplitter(a, b, theta, state)
            S = apply_beamsplitter_and_permute(a, b, S)

            dims.append(log_sim_lp_data(state, S))

    else:

        for (a, b) in X1:
            assert b == a+1, f"Something went wrong, X1 contains a beamsplitter of length {b-a}."
            state = apply_abstract_beamsplitter(a, b, state)
            S = apply_nn_beamsplitter(a, S)

            dims.append(log_sim_lp_data(state, S))

        S = PermutableCumulativeSpace.from_cumulative_space(S)
        for (a, b) in Xr:
            state = apply_abstract_beamsplitter(a, b, state)
            S = apply_beamsplitter_and_permute(a, b, S)

            dims.append(log_sim_lp_data(state, S))

    return dims, state, S



def fc_progressive_nonlp(prog_circ: List[BSCircuit],
                         state: SparseBosonicFockState,
                         ns: Union[List[int], NoneType]):

    log = []

    for i, component in enumerate(prog_circ):
        mode = 0
        component = component.shift_unsafe(-i)

        (state, _), ld = fc_circuit(component, state)

        number = ns[i] if ns is not None else None
        projected_state, measurement_probability, number = fc_measurement(state, mode, number=number)

        projected_state = map_fock_basis(lambda basis: tuple(basis[:mode] + basis[mode+1:]), projected_state)

        log.append({"fc@BS": ld,
                    "n<M": state.photons,
                    "n=M": number,
                    "prob": measurement_probability,
                    "n>M": projected_state.photons})

        if projected_state.dim == 0:
            break

        state = projected_state

    return log



def fc_progressive_lp(prog_circ: BSCircuit,
                      state: SparseBosonicFockState,
                      ns: Union[List[int], NoneType],
                      heuristic = False,
                      evolve_vector = True):
    if state.dim != 1:
        raise ValueError(f"Lattice path evaluation requires an input state of Fock complexity 1. Instead, fc = {state.dim} was used.")

    input_basis = list(state.basis())[0]
    diagram = PermutableCumulativeSpace(mu = (input_basis[0],))

    if not evolve_vector:
        state = None

    log = []

    for i, component in enumerate(prog_circ):
        # measured modes are deleted, so the mode to measure is always the top (0);
        # shift circuits accordingly, i.e. delete earlier modes
        mode = 0
        component = component.shift_unsafe(-i)

        if evolve_vector:
            photons_before = state.photons
        else:
            photons_before = diagram.photons

        (state, diagram), ld = fc_circuit(component, state, diagram,
                                          input_basis[i:]) # only new modes!!


        heuristic_marginals = {"paths": path_heuristic_probability(mode, diagram)} # dict for compatibility with testing other heuristics; those have been removed in this version

        if heuristic is not None:
            marginals = heuristic_marginals[heuristic]
            numbers = sorted(list(marginals.keys()))
            probabilities = [marginals[n] for n in numbers]
            number = int(np.random.choice(numbers, p=probabilities))

        elif ns is not None:
            number = ns[i]

        else:
            number = None

        # TODO some of the following is shared with nonlp code; join them eventually
        if evolve_vector:
            projected_state, measurement_probability, number = fc_measurement(state, mode, number=number)
            projected_state = map_fock_basis(lambda basis: tuple(basis[:mode] + basis[mode+1:]), projected_state)

        else:
            projected_state = None
            measurement_probability = None

        diagram = projection_contraction(0, number, diagram)

        if evolve_vector:
            photons_after = projected_state.photons
        else:
            photons_after = diagram.photons

        log.append({"fc@BS": ld,
                    "n<M": photons_before,
                    "n=M": number,
                    "prob": float(measurement_probability) if measurement_probability is not None else None,
                    "hprob": {heur: float(heuristic_marginals[heur][number]) for heur in heuristic_marginals.keys()},
                    "heuristic": heuristic,
                    "n>M": photons_after,
                    "diagram": {"mu": diagram.mu,
                                "nu": diagram.nu if isinstance(diagram, PSCSpace) else None,
                                "permutation": diagram.permutation.list()}})

        if projected_state is not None and projected_state.dim == 0:
            break

        state = projected_state

    return log



def fc_progressive(circ: BSCircuit,
                   state: SparseBosonicFockState,
                   ns: Union[List[int], NoneType],
                   lattice_paths : bool = False,
                   heuristic = None,
                   evolve_vector = True):

    prog_circ = progressive_decomposition(circ)

    if lattice_paths:
        return fc_progressive_lp(prog_circ, state, ns,
                                 heuristic=heuristic,
                                 evolve_vector=evolve_vector)
    else:
        return fc_progressive_nonlp(prog_circ, state, ns)
