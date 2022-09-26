import numpy as np
from ..core import CycleEquation
from scipy.optimize import root
from scipy.special import comb


def n_prime_cycles(n, k):
    if n == 1:
        return k
    elif n == 2:
        return int(comb(k, 2))
    else:
        return int((1. / n) * (k ** n - np.sum([n_prime_cycles(j, k) * j for j in range(1, n // 2 + 1) if n % j == 0])))



__all__ = ['Henon']


class Henon(CycleEquation):


    def __init__(self, n, a, b, states=None, basis='symbolic', **kwargs):
        """
        :param n: int
            Cycle length
        :param a: float
        :param b: float
        :param states: np.ndarray
            NumPy array of states; array of shape (M, N) indicates that there are M different cycles of length N which
            are of interest to the user.
        :param basis: str
            The basis of the values of the states array; used as a means of tracking the types of values in the states
            array.
        Notes
        -----
        'string' basis implies that all values equal one of the following: ['1', '2', '3'] or ['-1', '0', '1']

        'symbolic' basis implies that all values equal one of the following: [-1, 0, 1]

        'proxy' basis implies that all values equal one of the following: [1, 2, 3]

        """
        self.n = n
        self.a = a
        self.b = b
        self.states = states
        self.basis = basis

    @property
    def symbols(self):
        return [0, 1]

    def eqn(self):
        return -np.roll(self.states, -1) + self.a * self.states ** 2 + self.b * np.roll(self.states, 1) - 1

    def jacobian(self):
        return 2 * self.a * np.diag(self.states) - np.roll(np.eye(self.n), -1, axis=1) + self.b * np.roll(np.eye(self.n), 1, axis=1)

    def generate_states(self):
        shadow_states = np.concatenate([coord.ravel().reshape(-1, 1) for coord in np.meshgrid(*(self.symbols for i in
                                                                                                range(self.n)))],
                                       axis=1)
        self.states = shadow_states

    def eqn_wrapper(self):

        def _wrapper(x):
            return self.__class__(self.n, self.a, self.b, states=x).eqn()

        return _wrapper

    def hunt(self, method='hybr', scipy_options=None):
        """ A convenience wrapper for calling scipy.optimize.minimize

        :param method: str
            The name of a numerical algorithm provided by 'minimize'
        :param scipy_options: None or dict
            Dictionary or None (default) the options/hyperparameters of the particular scipy algorithm defined by 'method'

        :return:
        """
        results_list = []
        for state in self.states:
            result = root(self.eqn_wrapper(), state, method=method,
                          options=scipy_options)
            results_list.append(result)
        converged_states = np.concatenate([result.x for result in results_list])
        return self.__class__(**{**vars(self), 'states': converged_states})

    def _check_cyclic(self, cycle_1, cycle_2):
        """ Checks if two cycles are members of the same group cycle

        """
        return ', '.join(map(str, cycle_1)) in ', '.join(map(str, cycle_2))

    def prime_cycles(self, check_neg=False, check_rev=False):
        initial_conditions = self.states.copy()
        double_cycles = np.append(initial_conditions, initial_conditions, axis=1)
        # double_cycles is each shadow state repeated so that it is twice its length. This is used show checking for cyclic
        # permutations as every permunation exists in the cycle as if it goes through it twice. Ex: all cyclic permutation of 01
        # exist somwhere in 0101
        i = 0
        while i < np.shape(initial_conditions)[0]:  # looping through each row of the initial conditions
            j = np.shape(initial_conditions)[0] - 1
            while j > i:  # looping rows of double_cycles, starting at the bottomw and ending before the row of the current
                # cycle we are checking
                if self._check_cyclic(initial_conditions[i], double_cycles[j]) == True:
                    initial_conditions = np.delete(initial_conditions, j, 0)
                    double_cycles = np.delete(double_cycles, j,
                                              0)  # if a cycle string exists in the double_cycle of of another
                j = j - 1  # cycle, delete one of the cycles
            i = i + 1

        copy_of_reversed_initial = initial_conditions.copy()
        i = 0
        del_array = np.zeros(np.shape(initial_conditions)[0])
        while i < np.shape(initial_conditions)[0]:
            j = 1
            while j <= np.shape(initial_conditions)[1] - 1:
                self._rotate(copy_of_reversed_initial[i])
                if self._check_cyclic(copy_of_reversed_initial[i], initial_conditions[i]) == True:
                    del_array[i] = 1
                j = j + 1
            i = i + 1

        initial_conditions = np.delete(initial_conditions, np.where(del_array == 1), 0)
        states = initial_conditions
        return states

    def _rotate(self, a):
        x = a[len(a) - 1]
        for i in range(len(a) - 1, 0, -1):
            a[i] = a[i - 1]
        a[0] = x
        return a

    def shadow_states_mapping(self, symbol_state):
        shadow_states_dict = {0: -0.27429188517743175, 1: 0.6076252185107651}
        mapped_state = np.array([shadow_states_dict[symbol] for symbol in symbol_state])
        return mapped_state