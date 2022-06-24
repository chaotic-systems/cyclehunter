import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eig
from ..core import CycleEquation

__all__ = ['PhiK']


class PhiK(CycleEquation):

    def __init__(self, n, k, musqr, states=None, basis='symbolic'):
        """
        :param n: int
            Cycle length
        :param k: int
            potential exponent
        :param constants: int
            In this implementation of PhiK, this is mu squared. Naming is simply a convention
        """
        self.n = n
        self.k = k
        self.musqr = musqr
        self.s = musqr + 2
        self.states = states
        self.basis = basis

    @property
    def symbols(self):
        return [-1, 0, 1]

    def eqn(self):
        """ Calculate phi-k equations with respect to a tensor of initial conditions
        lattice_state : np.ndarray
            State variable

        """
        s = self.s
        k = self.k
        states_tensor = self.states
        # the equations of motion
        eqn_tensor = (-1 * np.roll(states_tensor, -1, axis=1) + (-1 * (s - 2) * states_tensor ** k +
                                                                 s * states_tensor) - np.roll(states_tensor, 1, axis=1))
        # the l2 norm, giving us a scalar cost functional
        return eqn_tensor

    def cost(self):
        """ L2 norm of equations used as cost function.

        """
        return (0.5 * np.linalg.norm(self.eqn(), axis=1) ** 2).sum()

    def costgrad(self):
        """ Gradient of L2 norm of equations used as cost function.

        """
        s = self.s
        states = self.states
        Ftensor = self.eqn()
        JTF = -np.roll(Ftensor, 1, axis=1) - np.roll(Ftensor, -1, axis=1) + (
                -3 * (s - 2) * states ** 2 + s) * Ftensor
        return JTF.ravel()

    def jac_tensor(self, states=None):
        """ Calculate all Jacobians for cuurent state

        """
        n, k = self.n, self.k
        if states is None:
            states = self.states
        J = np.zeros([3 ** n, n, n])
        upper_rows, upper_cols = self._kth_diag_indices(J[0], -1)
        lower_rows, lower_cols = self._kth_diag_indices(J[0], 1)
        zeroth = np.repeat(np.arange(3 ** n), len(upper_rows))
        upper_rows = np.tile(upper_rows, 3 ** n)
        upper_cols = np.tile(upper_cols, 3 ** n)
        lower_rows = np.tile(lower_rows, 3 ** n)
        lower_cols = np.tile(lower_cols, 3 ** n)

        J[zeroth, upper_rows, upper_cols] = -1
        J[zeroth, lower_rows, lower_cols] = -1
        J[:, 0, -1] = -1
        J[:, -1, 0] = -1
        tensor_diagonal = (-3 * (self.s - 2) * states ** 2 + self.s).ravel()

        rows, cols = np.diag_indices(n)
        zeroth = np.repeat(np.arange(3 ** n), len(rows))

        J[zeroth, np.tile(rows, 3 ** n), np.tile(cols, 3 ** n)] = tensor_diagonal
        return J

    def _kth_diag_indices(self, a, k):
        rows, cols = np.diag_indices_from(a)
        if k < 0:
            return rows[-k:], cols[:k]
        elif k > 0:
            return rows[:-k], cols[k:]
        else:
            return rows, cols

    def n_cycle_everything(self, compute_eig=False):
        converged_cycles = self.hunt()
        n_jacobians = self.jac_tensor(converged_cycles)

        if compute_eig:
            all_eig_val = []
            all_eig_vec = []
            for each_jac in n_jacobians:
                val, vec = eig(each_jac)
                all_eig_val.append(val)
                all_eig_vec.append(vec[np.newaxis])
            all_eig_val = np.concatenate(all_eig_val)
            all_eig_vec = np.concatenate(all_eig_vec)
            return converged_cycles, all_eig_val, all_eig_vec, n_jacobians
        else:
            return converged_cycles, None, None, n_jacobians

    def generate_states(self, prime=True, sort=False ):
        """ Produces all possible combinations of k-ary alphabet, puts them in tensor of shape (k**n, n)

        :return:
        """

        self.states = np.concatenate([coord.ravel().reshape(-1, 1) for coord in np.meshgrid(*(self.symbols for i in
                                                                                              range(self.n)))],
                                     axis=1)
        if sort:
            self.states = np.sort(self.states, axis=0)
        if prime:
            self.states = self.prime_orbits()

        return self

    def hunt(self, method='l-bfgs-b', **kwargs):
        options = kwargs.get('scipy_options')
        cycles = minimize(self.costwrapper(), self.states, jac=self.costgradwrapper(), method=method,
                          options=options)
        cycle_tensor = cycles.x.reshape(-1, self.n)
        return cycle_tensor

    def costwrapper(self):
        """ Functions for scipy routines must take vectors of state variables, not class objects. 


        :return: 
        """

        def minfunc_(x):
            return self.__class__(self.n, self.k, self.musqr, states=x.reshape(-1, self.n)).cost()

        return minfunc_

    def costgradwrapper(self):
        """ Functions for scipy routines must take vectors of state variables, not class objects. 


        :return: 
        """

        def _minjac(x):
            return self.__class__(self.n, self.k, self.musqr, states=x.reshape(-1, self.n)).costgrad()

        return _minjac

    def change_basis(self, to=None):
        if to is None or self.basis == to:
            return self.states
        elif to == 'symbolic':
            return self.states - 2
        elif to == 'proxy':
            return self.states + 2

    def prime_orbits(self, check_neg=False, check_rev=False):
        """ Maps a set of initial conditions for phi-k equations into a set of prime representatives

        check_neg : bool
            If true, quotients -1 -> 1 symmetry

        check_rev : bool
            If true, quotients reversal symmetry, i.e equivariance w.r.t. array[::-1]

        Notes
        -----

        The different flags are essentially equivalent to calling the same function just on modified inputs.
        What the code does is a vectorized comparison using numpy broadcasting. That is, given two vectors v1, v2 of length M,
        all possible pairs are compared, to see if v1 occurs as a substring in v2. This pairwise comparison results in a matrix;
        the ij-th element of the matrix, M_ij, counts the number of occurrences of v1_i in v2_j.

        This can be used along with boolean logic and summation to see if any off-diagonal elements are non-zero; a prime
        set would result in a diagonal matrix under this comparison.

        """
        states = self.states
        #initial conditions should be you entire list of possible shadow state configurations
        #check_neg is a value that takes either 1 or 0 where if it is 1, it will check for phi to negative phi symmetry
        if -1 in np.unique(states):
            states += 2
        # here i am just changing my shadow state values to a different symbolic alphabet that will work better
        double_cycles = np.append(states,states,axis=1)
        # double_cycles is each shadow state repeated so that it is twice its length. This is used show checking for cyclic
        # permutations as every permunation exists in the orbit as if it goes through it twice. Ex: all cyclic permutation of 123
        # exist somwhere in 123123
        i = 0
        while i < np.shape(states)[0]:
            # looping through each row of the initial conditions
            j = np.shape(states)[0] - 1
            while j > i:
                # looping rows of double_cycles, starting at the bottomw and ending before the row of the current
                # orbit we are checking
                if self.check_cyclic(states[i],double_cycles[j]):
                    # if a orbit string exists in the double_cycle of of another orbit, delete one of the orbits
                    states = np.delete(states, j, 0)
                    double_cycles = np.delete(double_cycles, j, 0)
                j = j - 1
            i = i + 1
        if check_neg == 1:
            states = -1 * (states % -4)
            i = 0
            while i < np.shape(states)[0]:
                j = np.shape(states)[0] - 1
                while j > i:
                    if self.check_cyclic(states[i],double_cycles[j]):
                        # does the same process as before but for the comparing the negatives of the orbits
                        #  to the double cycles
                        states = np.delete(states, j, 0)
                        double_cycles = np.delete(double_cycles, j, 0)              #
                    j = j - 1                                                       #
                i = i + 1
            states = -1 * (states % -4)
        if check_rev == 1:
            states = states[..., ::-1]
            i = 0
            while i < np.shape(states)[0]:
                j = np.shape(states)[0] - 1
                while j > i:
                    if self.check_cyclic(states[i],double_cycles[j]):
                        states = np.delete(states, j, 0)
                        double_cycles = np.delete(double_cycles, j, 0)
                    j = j - 1
                i = i + 1
        copy_of_reversed_initial = states.copy()
        i = 0
        del_array = np.zeros(np.shape(states)[0])
        while i < np.shape(states)[0]:
            j = 1
            while j <= np.shape(states)[1] - 1:
                copy_of_reversed_initial[i] = np.roll(copy_of_reversed_initial[i], 1)
                if self.check_cyclic(copy_of_reversed_initial[i], states[i]):
                    del_array[i] = 1
                j = j + 1
            i = i + 1

        states = np.delete(states, np.where(del_array == 1), 0)
        states -= 2
        return states

    def check_cyclic(self, orbit_1, orbit_2):
        """ Checks if two orbits are members of the same group orbit
    
        A: 
    
        """
        return ', '.join(map(str, orbit_1)) in ', '.join(map(str, orbit_2))
