import cvxpy as cp
import typing
import numpy as np
import os
from tqdm import tqdm
import utils


# project vector to the simplex
def project_simplex(y: np.ndarray) -> np.ndarray:
    p = cp.Variable(y.shape)
    objective = cp.Minimize(cp.sum_squares(p - y))
    constraints = [cp.sum(p) == 1, p >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status != cp.OPTIMAL:
        ex = f"Can't project to the simplex: {problem.status}"
        raise cp.error.SolverError(ex)
    return p.value


# project vector to simplex on A
def project_A(A, y):
    p = cp.Variable(y.shape)
    objective = cp.Minimize(cp.quad_form((p - y), A))
    constraints = [cp.sum(p) == 1, p >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status != cp.OPTIMAL:
        ex = f"Can't project to A: {problem.status}"
        raise cp.error.SolverError(ex)
    return p.value


# project to simplex given data
def best_hindsight(Rs):
    p = cp.Variable((Rs.shape[0], 1))
    objective = cp.Maximize(cp.sum(cp.log(Rs.T @ p)))
    constraints = [cp.sum(p) == 1, p >= 0]
    problem = cp.Problem(objective, constraints)
    # https://www.cvxpy.org/tutorial/advanced/index.html#solvers
    # needs better than ECOS
    problem.solve(solver="SCS")
    if problem.status != cp.OPTIMAL:
        ex = f"Can't find best in hindsight: {problem.status}"
        raise cp.error.SolverError(ex)
    return problem.value


# find best fixed distribution in hindsight
def hindsight(X: np.ndarray) -> list:
    T = X.shape[1]
    rewards = []

    for t in tqdm(range(1, T), desc="hindsight"):
        r_t = X[:, t] / X[:, t - 1]
        r_t = r_t[:, None]

        # hindsight
        Rs: np.ndarray
        try:
            Rs = np.hstack((Rs, r_t))
        except:
            Rs = r_t

        rewards += [best_hindsight(Rs)]

    return rewards


def algs() -> list:
    return [oga, ons]


class algorithm:
    def __init__(
        self,
        data: str,  # data folder
        old_dates: str,  # old param dates
        new_dates: str,  # new param dates
    ):
        self.name = "algorithm"
        self.data = data
        self.params_path = os.path.join("data", data, "params")
        self.old_dates = old_dates
        self.new_dates = new_dates

        self.params_strs: list = []

    def params_names(self) -> typing.List[str]:
        return [self.name + "_" + p for p in self.params_strs]

    def load_params(self):
        # load params to dictionary
        self.params = {}
        for p_str, p_name in zip(self.params_strs, self.params_names()):
            try:
                name = p_name + f"_{self.old_dates}.npy"
                full_path = os.path.join(self.params_path, name)
                self.params[p_str] = np.load(full_path, allow_pickle=True)
            except FileNotFoundError as e:
                self.params = {}
                self.date = None
                print(f"{self.name}: no params")
                break

    def save_params(self):
        for p_str, p_name in zip(self.params_strs, self.params_names()):
            full_path = os.path.join(
                self.params_path, p_name + "_" + self.new_dates + ".npy"
            )
            np.save(full_path, self.params[p_str])

    def run(self, X: np.ndarray) -> list:
        self.load_params()
        rewards = self.algorithm(X)
        self.save_params()
        return rewards

    def algorithm(self, X: np.ndarray) -> list:
        return []


class oga(algorithm):
    def __init__(
        self,
        data: str,  # data folder
        old_dates: str,  # old param dates
        new_dates: str,  # new param dates
    ):
        super().__init__(data, old_dates, new_dates)
        self.name = "oga"
        self.params_strs = ["x"]

    def algorithm(self, X: np.ndarray) -> list:
        T = X.shape[1]  # length
        d = X.shape[0]  # dimension
        if not self.params:  # no previous params
            self.params["x"] = np.ones([d, 1]) / d  # how to invest

        rewards = []

        for t in tqdm(range(1, T), desc=self.name):
            r_t = X[:, t] / X[:, t - 1]
            r_t = r_t[:, None]

            multiplier = r_t.T @ self.params["x"]
            rewards += [np.log(multiplier)[0][0]]

            grad = r_t / multiplier
            eta = 1 / (d * np.sqrt(t))
            y = self.params["x"] + eta * grad  # + for ascent
            self.params["x"] = project_simplex(y)

        return rewards


class ons(algorithm):
    def __init__(
        self,
        data: str,  # data folder
        old_dates: str,  # old param dates
        new_dates: str,  # new param dates
    ):
        super().__init__(data, old_dates, new_dates)
        self.name = "ons"
        self.params_strs = ["x", "A", "b", "beta"]

    def algorithm(self, X: np.ndarray, beta: float = 2.0) -> list:
        self
        T = X.shape[1]
        d = X.shape[0]
        if not self.params:  # no previous params
            self.params["x"] = np.ones([d, 1]) / d
            self.params["A"] = np.zeros([d, d])
            self.params["b"] = np.zeros([d, 1])
            self.params["beta"] = beta

        rewards = []

        for t in tqdm(range(1, T), desc=self.name):
            r_t = X[:, t] / X[:, t - 1]
            r_t = r_t[:, None]

            multiplier = r_t.T @ self.params["x"]
            rewards += [np.log(multiplier)[0][0]]

            grad = r_t / multiplier
            hess = grad @ grad.T
            self.params["A"] += hess
            self.params["b"] += (
                hess @ self.params["x"] + (1 / self.params["beta"]) * grad
            )  # + for ascent
            self.params["x"] = project_A(
                self.params["A"], np.linalg.pinv(self.params["A"]) @ self.params["b"]
            )

        return rewards
