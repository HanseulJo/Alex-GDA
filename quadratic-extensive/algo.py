import numpy as np 
from tqdm import tqdm


class QuadraticMinimaxAlgo(object):
    def __init__(self, μ_x, μ_y, μ_xy, L_x, L_y, L_xy, d_x, d_y, seed_init, seed_func):
        np.random.seed(seed_init)
        self.x_0 = np.random.normal(size=(d_x,)) * 2
        self.y_0 = np.random.normal(size=(d_y,)) * 2
        np.random.seed(seed_func)
        U, _ = np.linalg.qr(np.random.normal(size=(d_x, d_x)))
        V, _ = np.linalg.qr(np.random.normal(size=(d_y, d_y)))
        self.A = np.diag([μ_x, L_x] + list(np.random.uniform(μ_x, L_x, d_x-2)))
        self.C = np.diag([μ_y, L_y] + list(np.random.uniform(μ_x, L_x, d_y-2)))
        if d_x >= d_y:
            B1 = np.diag([L_xy, L_xy] + list(np.random.uniform(μ_xy, L_xy, d_y-2)))
            B2 = np.zeros((d_x-d_y, d_y))
            self.B = np.concatenate([B1,B2], axis=0)
        else:
            B1 = np.diag([L_xy, μ_xy] + list(np.random.uniform(μ_xy, L_xy, d_x-2)))
            B2 = np.zeros((d_x, d_y-d_x))
            self.B = np.concatenate([B1,B2], axis=1)
        # key = random.PRNGKey(42)
        # keys = random.split(key, 7)
        # self.x_0 = random.normal(keys[0], (d_x,)) * 10
        # self.y_0 = random.normal(keys[1], (d_y,)) * 10
        # U, _ = np.linalg.qr(random.normal(keys[2], (d_x, d_x)))
        # V, _ = np.linalg.qr(random.normal(keys[3], (d_y, d_y)))
        # self.A = np.diag(np.array([μ_x, L_x] + random.uniform(keys[4], (d_x-2,), minval=μ_x, maxval=L_x).tolist()))
        # self.C = np.diag(np.array([μ_y, L_y] + random.uniform(keys[5], (d_y-2,), minval=μ_y, maxval=L_y).tolist()))
        # if d_x >= d_y:
        #     B1 = np.diag(np.array([L_xy, L_xy] + random.uniform(keys[6], (d_y-2,), minval=μ_xy, maxval=L_xy).tolist()))
        #     B2 = np.zeros((d_x-d_y, d_y))
        #     self.B = np.concatenate([B1,B2], axis=0)
        # else:
        #     B1 = np.diag(np.array([L_xy, μ_xy] + random.uniform(keys[6], (d_x-2,), minval=μ_xy, maxval=L_xy).tolist()))
        #     B2 = np.zeros((d_x, d_y-d_x))
        #     self.B = np.concatenate([B1,B2], axis=1)
        self.A = U @ self.A @ U.T
        self.C = V @ self.C @ V.T
        self.B = U @ self.B @ V.T

    def grad_f(self, x, y):
        return ((self.A @ x + self.B @ y), (self.B.T @ x - self.C @ y))
    
    def run(self, α, β, γ, δ, n_iter, ϵ, verbose, momentum_x=0, momentum_y=0):
        self.α = α
        self.β = β
        self.γ = γ
        self.δ = δ
        self.momentum_x = momentum_x
        self.momentum_y = momentum_y
        self.v_x = np.zeros_like(self.x_0)
        self.v_y = np.zeros_like(self.y_0)

        # Initialization
        x = self.x_0.copy()
        y = self.y_0.copy()
        x_ = self.x_0.copy()
        y_ = self.y_0.copy()

        # Storing values
        distances = [(x @ x) + (y @ y)]

        # Run
        pbar = tqdm(range(1, n_iter+1), disable=not verbose)
        for i in pbar:
            x, y, x_, y_ = self.algo(i, x, y, x_, y_)

            # storing values
            distances.append((x @ x) + (y @ y))

            if distances[-1]>1e6:
                if verbose: print("Diverged")
                # return np.arange(n_iter), None, None
                break
            if (x @ x) + (y @ y) < ϵ:
                break
            pbar.set_description(f"{distances[-1]}")

        return np.array(distances)
    
    def algo(self):
        pass


class GDA(QuadraticMinimaxAlgo):
    grad_complexity = 1
    def algo(self, i, x, y, x_, y_):
        grad_x, _ = self.grad_f(x,y_)
        self.v_x = grad_x + self.momentum_x * self.v_x
        x_ = x - self.γ * self.α * self.v_x
        x += - self.α * self.v_x
        _, grad_y = self.grad_f(x_,y)
        self.v_y = grad_y + self.momentum_y * self.v_y
        y_ = y + self.δ * self.β * self.v_y
        y += self.β * self.v_y
        return x, y, x_, y_


class EG(QuadraticMinimaxAlgo):
    grad_complexity = 2
    def algo(self, i, x, y, x_, y_):
        grad_x, grad_y = self.grad_f(x,y)
        grad_x, grad_y = self.grad_f(x - self.α * grad_x, y + self.β * grad_y)
        self.v_x = grad_x + self.momentum_x * self.v_x
        self.v_y = grad_y - self.momentum_y * self.v_y
        x += -self.γ * self.v_x
        y += self.δ * self.v_y
        return x, y, x_, y_ 


class AltEG(QuadraticMinimaxAlgo):
    grad_complexity = 3
    def algo(self, i, x, y, x_, y_):
        grad_x, grad_y = self.grad_f(x,y)
        grad_x, _ = self.grad_f(x - self.α * grad_x, y + self.β * grad_y)
        self.v_x = grad_x + self.momentum_x * self.v_x
        x += -self.γ * self.v_x
        grad_x, grad_y = self.grad_f(x,y)
        _, grad_y = self.grad_f(x - self.α * grad_x, y + self.β * grad_y)  
        self.v_y = grad_y - self.momentum_y * self.v_y
        y += self.δ * self.v_y
        return x, y, x_, y_


class OGD(QuadraticMinimaxAlgo):
    grad_complexity = 1
    def algo(self, i, x, y, x_, y_):
        grad_x, grad_y = self.grad_f(x,y)
        self.a_x = self.γ * self.v_x
        self.v_x = grad_x + self.momentum_x * self.v_x
        self.a_x += -self.α * self.v_x
        x += self.a_x 
        self.a_y = -self.δ * self.v_y
        self.v_y = grad_y + self.momentum_y * self.v_y
        self.a_y += self.β * self.v_y
        y += self.a_y 
        return x, y, x_, y_


class AltOGD(QuadraticMinimaxAlgo):
    grad_complexity = 1
    def algo(self, i, x, y, x_, y_):
        grad_x, _ = self.grad_f(x,y)
        self.a_x = self.γ * self.v_x
        self.v_x = grad_x + self.momentum_x * self.v_x
        self.a_x += -self.α * self.v_x
        x += self.a_x
        _, grad_y = self.grad_f(x,y)
        self.a_y = -self.δ * self.v_y
        self.v_y = grad_y + self.momentum_y * self.v_y
        self.a_y += self.β * self.v_y
        y += self.a_y 
        return x, y, x_, y_

if __name__ == '__main__':
    runner = AltOGD(0.2, 0.2, 0.2, 1, 1, 1, 3, 3, 999, 999)
    runner.run(0.2, 0.2, 0.1, 0.1, 10000, 0.01, True)
