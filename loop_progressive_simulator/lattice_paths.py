from functools import cached_property
from itertools import accumulate, chain, tee
from types import NoneType
from typing import Dict, Generator, Iterable, Sequence, Tuple, TypeVar, Union
from warnings import warn, filterwarnings

import numpy as np
from sympy.combinatorics import Permutation

from bscircuits import AbstractBSCircuit
import number_basis
from utils import compose



filterwarnings("error")


def enumerate_with_initial_segment(mu, lamda):
    ell = len(mu)
    fix = len(lamda)

    if fix == ell - 1:
        yield lamda + (mu[fix],)
    else:
        bottom = 0 if fix == 0 else lamda[fix - 1]
        top = mu[fix]

        for lam in range(bottom, top + 1):
            yield from enumerate_with_initial_segment(mu, lamda + (lam,))


def _tuple_from_any_args(mu, *args):
    if not isinstance(mu, tuple):
        return (mu, *args)
    else:
        return mu + args


def enumerate_downset(mu, *args):
    mu = _tuple_from_any_args(mu, *args)
    if len(mu) >= 2:
        yield from enumerate_with_initial_segment(mu, ())
    else:
        yield mu


def count_downset_DP(mu, *args, dtype=np.ulonglong):
    mu = _tuple_from_any_args(mu, *args)
    m = len(mu)
    n = mu[-1]

    array = np.zeros((m, n+1), dtype=dtype)
    array[m-1,:] = 1

    for x in range(m-2, -1, -1):
        for y in range(mu[x], -1, -1):
            if y < n:
                array[x, y] = array[x + 1, y] + array[x, y + 1]
            else:
                array[x, y] = array[x + 1, y]

    return array[0,0]


def difference(lamda):
    lamda1, lamda2 = tee(lamda, 2)
    yield from map(lambda x: x[1] - x[0],
                   zip(chain((0,), lamda1),
                       lamda2))


def paths_leq(lamda, kappa):
    if len(lamda) != len(kappa) or lamda[-1] != kappa[-1]:
        raise ValueError(f"Lattice dimensions do not match: λ in L({len(lamda)}-1, {lamda[-1]}) and κ in L({len(kappa)}-1, {kappa[-1]}).")

    return all(lamda[a] <= kappa[a] for a in range(len(lamda)))


def paths_lne(lamda, kappa):
    return paths_leq(lamda, kappa) and not all(lamda[a] == kappa[a] for a in range(len(lamda)))


def path_meet(lamda, kappa):
    if len(lamda) != len(kappa) or lamda[-1] != kappa[-1]:
        raise ValueError(f"Lattice dimensions do not match: λ in L({len(lamda)}-1, {lamda[-1]}) and κ in L({len(kappa)}-1, {kappa[-1]}).")

    return tuple(min(lamda[a], kappa[a]) for a in range(len(lamda)))


def path_join(lamda, kappa):
    if len(lamda) != len(kappa) or lamda[-1] != kappa[-1]:
        raise ValueError(f"Lattice dimensions do not match: λ in L({len(lamda)}-1, {lamda[-1]}) and κ in L({len(kappa)}-1, {kappa[-1]}).")

    return tuple(max(lamda[a], kappa[a]) for a in range(len(lamda)))




class CumulativeSpace:

    def __init__(self, mu):
        self.mu = tuple(mu)


    @classmethod
    def from_maximum_photons(cls, nmax):
        return cls(accumulate(nmax))


    @classmethod
    def from_L1(cls, state: Tuple[int,...], modes: int = None):
        if not modes:
            modes = len(state)
        else:
            assert modes >= len(state)

        mu = [0] * modes

        if len(state) > 0:
            mu[0] = state[0]
        if len(state) > 1:
            mu[0] += state[1]

        for a in range(len(state) - 2):
            mu[a+1] = mu[a] + state[a+2]

        n = mu[len(state) - 2]
        for a in range(len(state) - 2, modes - 1):
            mu[a+1] = n

        return cls(mu)


    def __repr__(self):
        return f"{self.__class__.__name__}({self.mu})"


    @property
    def photons(self):
        return self.mu[-1]


    @property
    def modes(self):
        return len(self.mu)


    @property
    def top_path(self):
        return self.mu


    @cached_property
    def top_basis(self):
        return tuple(difference(self.mu))


    def enumerate_basis_paths(self):
        yield from enumerate_downset(self.mu)


    def enumerate_basis(self):
        yield from map(compose(tuple, difference), self.enumerate_basis_paths())


    @cached_property
    def dim(self):
        try:
            return count_downset_DP(self.mu)

        except RuntimeWarning:
            return self.inexact_large_dim


    @cached_property
    def inexact_large_dim(self):
        return count_downset_DP(self.mu, dtype=np.longdouble)


    def __eq__(self, other) -> bool:
        return self.top_path == other.top_path


    def __lt__(self, other) -> bool:
        return paths_lne(self.top_path, other.top_path)


    def __le__(self, other) -> bool:
        return paths_leq(self.top_path, other.top_path)



T = TypeVar("T")

def permute_sequence(p: Permutation, seq: Sequence[T]) -> Generator[T, None, None]:
    """Permute a sequence seq of length k by a permutation p from S_k by sending the element at position i to p(i). Equivalently, it generates a sequence (seq[~p(0)], ..., seq[~p(k-1)]).
    """
    seq = list(seq)

    inv = ~p
    for i in range(inv.size):
        yield seq[inv(i)]



class PermutableCumulativeSpace(CumulativeSpace):

    permutation = None

    def __init__(self, mu, permutation: Union[Permutation, NoneType] = None):
        if permutation is not None:
            self.permutation = permutation
        else:
            # identity
            self.permutation = Permutation(len(list(mu)) - 1)

        super().__init__(mu)


    @classmethod
    def from_cumulative_space(cls, S: CumulativeSpace):
        return cls(S.mu, None)


    def __repr__(self):
        return f"{self.__class__.__name__}({self.mu}, {self.permutation})"


    @property
    def top_path(self):
        return tuple(permute_sequence(self.permutation, self.mu))


    @cached_property
    def top_basis(self):
        return tuple(permute_sequence(self.permutation, difference(self.mu)))


    def enumerate_basis_paths(self):
        yield from map(compose(tuple, accumulate), self.enumerate_basis())


    def enumerate_basis(self):
        yield from map(lambda x: tuple(permute_sequence(self.permutation, difference(x))),
                       super().enumerate_basis_paths())



def sort_and_compute_permutation(seq: Iterable[int]) -> Tuple[Iterable[int], Permutation]:
    """Sort a list of integers and return also the permutation"""
    sp = sorted(enumerate(seq), key=lambda x: x[1])
    per, sor = zip(*sp)
    per = Permutation(per)
    return sor, per


def apply_beamsplitter_without_permutation(a: int, b: int,
                                           lamda: Tuple[int,...]) -> PermutableCumulativeSpace:
    n_ab = max(lamda[a], lamda[b])
    return tuple(n_ab if i in (a,b) else lamda[i] for i in range(len(lamda)))


def apply_beamsplitter_and_permute(a: int, b: int,
                                   S: PermutableCumulativeSpace) \
                                  -> PermutableCumulativeSpace:

    kappa = apply_beamsplitter_without_permutation(a, b, S.top_path)
    kappa, per = sort_and_compute_permutation(kappa)
    return PermutableCumulativeSpace(kappa, per)



# NB: this requires that ℓ_1 = 1
def apply_beamsplitter_new_mode(a:int,
                                n_new: int,
                                S: PermutableCumulativeSpace) \
                               -> PermutableCumulativeSpace:

    assert a == S.modes - 1 == (~S.permutation)(a), "Only the last mode of a PCS can be coupled to a new mode."

    n = S.photons

    n_ab = n + n_new
    mu = S.mu[:-1] + (n_ab, n_ab)
    per = Permutation(S.permutation.list() + [len(mu) - 1])
    return PermutableCumulativeSpace(mu, per)



def apply_beamsplitter_extensible(a: int, b: int,
                                  S: PermutableCumulativeSpace,
                                  input_basis: Tuple[int, ...]) -> PermutableCumulativeSpace:

    if a > b:
        a, b = b, a

    if b < S.modes:
        # existing mode
        return apply_beamsplitter_and_permute(a, b, S)

    elif a < S.modes:
        # one new mode
        return apply_beamsplitter_new_mode(a, number_basis.number_in_mode(input_basis, b), S)

    else:
        raise ValueError(f"At least one mode must be within a PCS. Was given a beamsplitter {a,b} on a space {S}.")



def apply_bscircuit_and_permute(circuit: AbstractBSCircuit,
                                S: Union[CumulativeSpace, PermutableCumulativeSpace]) \
                                -> PermutableCumulativeSpace:

    if not isinstance(S, PermutableCumulativeSpace):
        S = PermutableCumulativeSpace.from_cumulative_space(S)

    mu = S.top_path
    for a, b in circuit:
        mu = apply_beamsplitter_without_permutation(a, b, mu)

    mu, per = sort_and_compute_permutation(mu)
    return PermutableCumulativeSpace(mu, per)



def min_downset(mu):
    n = mu[-1]
    return (0,) * (len(mu) - 1) + (n,)



def enumerate_interval(nu, mu):
    assert len(nu) == len(mu)
    # naive implementation, filter over enumerate_downset(mu)
    yield from filter(lambda lamda: all(lamda[a] >= nu[a]
                                        for a in range(len(mu))),
                      enumerate_downset(mu))



def count_interval_DP(nu, mu, dtype=np.ulonglong):
    if len(mu) != len(nu) or mu[-1] != nu[-1]:
        raise ValueError(f"Lattice dimensions do not match: mu in L({len(mu)}-1, {mu[-1]}) and nu in L({len(nu)}-1, {nu[-1]}).")

    m = len(mu)
    n = mu[-1]

    if not all(nu[a] <= mu[a] for a in range(m)):
        raise ValueError(f"Path nu = {nu} is not less than mu = {mu} in the Young order.")

    if m == 1 or n == 0:
        return 1

    array = np.zeros((m, n+1), dtype=dtype)
    array[m-1, nu[-2]:] = 1

    for x in range(m-2, -1, -1):
        if x == 0:
            min_y = 0
        else:
            min_y = nu[x-1]

        for y in range(mu[x], min_y - 1, -1):
            if y < n:
                array[x, y] = array[x + 1, y] + array[x, y + 1]
            else:
                array[x, y] = array[x + 1, y]

    for y in range(nu[0] - 1, -1, -1):
        array[0, y] = array[0, y+1]

    return array[0,0]



class PSCSpace(PermutableCumulativeSpace):
    """Permutable skew cumulative space."""

    nu = None

    def __init__(self, mu, nu, permutation: Permutation = None):
        if not paths_leq(nu, mu):
            raise ValueError(f"nu = {nu} </= {mu} = mu")
        super().__init__(mu, permutation)
        self.nu = nu


    @classmethod
    def from_cumulative_space(cls, S: CumulativeSpace):
        if isinstance(S, cls):
            return S

        if isinstance(S, PermutableCumulativeSpace):
            per = S.permutation
        else:
            per = None

        return cls(S.mu, min_downset(S.mu), per)


    @classmethod
    def from_L1(cls, state: Tuple[int,...], modes: int = None):
        CS = CumulativeSpace.from_L1(state, modes)
        return cls(CS.mu, min_downset(CS.mu))


    @classmethod
    def from_basis_state(cls, basis_state: Tuple[int,...]):
        mu = tuple(accumulate(basis_state))
        return cls(mu, mu)


    def __repr__(self):
        return f"{self.__class__.__name__}({self.mu}, {self.nu}, {self.permutation})"


    @property
    def bottom_path(self):
        return tuple(permute_sequence(self.permutation, self.nu))


    @cached_property
    def bottom_basis(self):
        return tuple(permute_sequence(self.permutation, difference(self.nu)))


    def enumerate_basis(self):
        yield from map(lambda x: tuple(permute_sequence(self.permutation, difference(x))),
                       enumerate_interval(self.nu, self.mu))


    @cached_property
    def dim(self):
        return count_interval_DP(self.nu, self.mu)


    @cached_property
    def inexact_large_dim(self):
        return count_interval_DP(self.nu, self.mu, dtype=np.longdouble)


    def __eq__(self, other) -> bool:
        return super().__eq__(other) and (self.bottom_path == other.bottom_path)


    def __le__(self, other) -> bool:
        return super(PermutableCumulativeSpace).__le__(other) and paths_leq(other.bottom_path, self.bottom_path)


    def __lt__(self, other) -> bool:
        # a bit inefficient but simple
        return self <= other and (any(self.top_path[a] != other.top_path[a]
                                      or self.bottom_path[a] != other.bottom_path[a]
                                      for a in range(self.modes)))



def apply_nn_beamsplitter(a: int, S: PSCSpace) -> PSCSpace:
    if not S.permutation.is_Identity:
        raise NotImplementedError("Action on PSC space with nontrivial permutation is unknown.")

    assert 0 <= a < S.modes - 1

    mu = tuple(S.mu[a+1] if i in (a, a+1) else S.mu[i] for i in range(S.modes))

    if a > 0:
        nu_ab = S.nu[a-1]
    else:
        nu_ab = 0
    nu = tuple(nu_ab if i in (a,) else S.nu[i] for i in range(S.modes - 1)) + (mu[-1],)

    return PSCSpace(mu, nu, S.permutation)



def number_projection(mode: int, number: int, S: PSCSpace):
    # iterate over the starting height of the controlled path segment
    # this ranges between nu[mode-1] <= q <= mu[mode-1]
    # such that the next step can have the appropriate height

    # NB: by convention nu[-1] = mu[-1] = 0
    # (where this is not the Python's modular access list[-1] = list[length - 1],
    # but the -1st component of the lattice path)

    if mode == 0:
        q_range = (0,)

    else:
        q_range = range(S.nu[mode - 1], S.mu[mode - 1] + 1)


    for q in q_range:

        if not (S.nu[mode] <= q + number <= S.mu[mode]):
            continue

        eps = (q,) * mode + (q + number, *(S.photons,)*(S.modes - mode - 1))
        eta = (0,) * (mode - 1) + ((q,) if mode > 0 else ()) + (q + number,) * (S.modes - mode - 1) + (S.photons,)

        yield PSCSpace(path_meet(S.mu, eps), path_join(S.nu, eta), S.permutation)



def contract_path(mode, number, lamda):
    assert lamda[mode] == (lamda[mode - 1] if mode > 0 else 0) + number,\
        f"Path {lamda} is not in the domain of the contraction operator c_{{{mode}}}^{{{number}}}."

    return lamda[:mode] + tuple(x - number for x in lamda[mode+1:])



def project_permutation(mode, permutation):
    """Used for deleting modes."""
    per_mode = permutation(mode)
    new_per = []

    for b, per_b in enumerate(list(permutation)):
        if b == mode:
            continue # SKIP

        new_per.append(per_b - 1
                       if per_b > per_mode
                       else per_b)

    return Permutation(new_per)



def projection_contraction(mode: int, number: int, S: PermutableCumulativeSpace) -> PermutableCumulativeSpace:
    assert number is not None and isinstance(number, int)

    # 'mode' is the index of a mode seen on the outside, i.e. mode of <mu↓>^sigma, while 'a' is the internal mode within the lattice path representation, i.e. mode of <mu↓>
    # these are related as sigma(a) = mode
    a = (~S.permutation)(mode)
    Qs = list(number_projection(a, number, PSCSpace.from_cumulative_space(S)))

    if len(Qs) == 0:
        raise ValueError(f"Impossible measurement projection of mode {mode} to {number} photons from space {S}.")

    else:
        return PermutableCumulativeSpace(mu = contract_path(a, number, Qs[-1].mu),
                                         permutation = project_permutation(a, Qs[-1].permutation))



def path_heuristic_probability(mode: int, S: PermutableCumulativeSpace) -> Dict[int, float]:
    a = (~S.permutation)(mode)

    possible_n = range(S.nu[a] if isinstance(S, PSCSpace) else 0,
                       S.mu[a] + 1)

    subdiagram_weights = {}
    total = np.float128(0)

    for n in possible_n:
        diagram = projection_contraction(mode, n, S)
        subdiagram_weights[n] = np.float128(diagram.dim)
        total += np.float128(diagram.dim)

    return {n: float(w/total) for n, w in subdiagram_weights.items()}
