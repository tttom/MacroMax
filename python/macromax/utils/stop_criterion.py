import numpy as np
from pathlib import Path
from examples import log

from macromax.solver import Solution

log = log.getChild(Path(__file__).stem)


class Adaptive:
    """
    A class to represent an adaptive stop criterion.

    It can be used as function, either directly as the callback for the solver, or as part of a custom function.
    """
    def __init__(self, residue: float = 1e-4, iteration: int = 10_000):
        """

        :param residue: The minimum residue required to continue the iteration.
        :param iteration: The maximum interation count.
        """
        self.residue = residue
        self.iteration = iteration

        self.__iterations = np.ones(2, dtype=int)
        self.__residues = np.ones(2)

        self.__first_two_checks_after = np.asarray((25, 50))

        self.__previous_iteration = None  # To reset when applied to a new problem

    def __call__(self, s: Solution):
        """
        This stop criterion tries to minimize the amount of times that the residue is computed by estimating when
        convergence will be reached.
        """
        if self.__previous_iteration is not None and s.iteration < self.__previous_iteration:
            self.__iterations = np.ones(2, dtype=int)  # Reset when applied to a new problem
        self.__previous_iteration = s.iteration
        if s.iteration > max(self.__iterations):  # reset
            self.__iterations = s.iteration + self.__first_two_checks_after
        if s.iteration == self.__iterations[0]:  # fill in the first measurement
            self.__residues[0] = s.residue
        elif s.iteration == self.__iterations[1]:  # do the second measurement and determine how to proceed
            self.__residues[1] = s.residue
            if self.__residues[1] < self.residue:
                return False
            convergence_rate = (self.__residues[1] / self.__residues[0]) ** (1 / (self.__iterations[1] - self.__iterations[0]))
            estimated_iterations_left = np.log(self.residue / self.__residues[1]) / np.log(convergence_rate)
            self.__iterations[0] = self.__iterations[1]
            self.__residues[0] = self.__residues[1]
            self.__iterations[1] = s.iteration + min(max(1, int(estimated_iterations_left + 0.5)), s.iteration)
            # log.info(f'{s.iteration}: Final iteration estimate {s.iteration + estimated_iterations_left:0.1f}, checking at {self.__iterations[1]}')
        elif s.iteration >= self.iteration:
            return False
        return True
