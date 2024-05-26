from argparse import ArgumentParser
import numpy as np 
from tqdm import tqdm
import pandas as pd
from itertools import product
import logging

from algo import GDA, EG, AltEG, OGD, AltOGD


verbose=False

def custom_linspace(start, stop, interval_length):
    num = round((stop - start) / interval_length) + 1
    linspace = np.linspace(start, stop, num)
    linspace = list(map(lambda x: round(x, 3), linspace))
    return linspace

tuner_momentum = [-0.99, -0.95] + custom_linspace(-0.9, 0.9, 0.1) + [0.95, 0.99]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--mu_x',   type=float, default=0.02)
    parser.add_argument('--mu_y',   type=float, default=0.02)
    parser.add_argument('--mu_xy',  type=float, default=0.01)
    parser.add_argument('--L_x',    type=float, default=1.)
    parser.add_argument('--L_y',    type=float, default=1.)
    parser.add_argument('--L_xy',   type=float, default=1.)
    parser.add_argument('--eps',    type=float, default=1e-4)
    parser.add_argument('--d_x',    type=int, default=100)
    parser.add_argument('--d_y',    type=int, default=100)
    parser.add_argument('--n_iter',    type=int, default=10000)
    parser.add_argument('--num_seeds',    type=int, default=10)
    args = parser.parse_args()

    μ_x, μ_y, μ_xy, L_x, L_y, L_xy = args.mu_x, args.mu_y, args.mu_xy, args.L_x, args.L_y, args.L_xy
    ϵ = args.eps
    d_x, d_y = args.d_x, args.d_y
    n_iter = args.n_iter

    print("μ_x, μ_y, μ_xy, L_x, L_y, L_xy =", μ_x, μ_y, μ_xy, L_x, L_y, L_xy)

    base_lr = {
        'SimGDA':   (min(μ_x/L_x**2, μ_x/L_xy**2), min(μ_y/L_y**2, μ_y/L_xy**2)),
        'AltGDA':   (min(1/L_x, 1/L_xy), min(1/L_y, 1/L_xy)),
    }

    def tune_GDA(key, γ, δ, seed_init, seed_func):
        runner = GDA(μ_x, μ_y, μ_xy, L_x, L_y, L_xy, d_x, d_y, seed_init, seed_func)
        n_iter = 1000 * args.n_iter if key == 'SimGDA' else args.n_iter
        min_len = n_iter
        pbar = tqdm(custom_linspace(0.1, 1.5, 0.1), desc=key)
        result = list(range(n_iter)) 
        for factor in pbar:
            if key.startswith('AlexGDA') or key.startswith('AltGDA'):
                α, β = np.array(base_lr['AltGDA']) * factor
            else:
                α, β = np.array(base_lr['SimGDA']) * factor
            rec = runner.run(α, β, γ, δ, n_iter, ϵ, verbose)
            if len(rec) < min_len and rec[-1] <= ϵ:
                min_len = len(rec)
                result = rec
                pbar.set_description(f'{key} | {min_len} f {factor:.3g}')
        return result

    def tune_GDA_M(key, γ, δ, seed_init, seed_func):
        runner = GDA(μ_x, μ_y, μ_xy, L_x, L_y, L_xy, d_x, d_y, seed_init, seed_func)
        n_iter = 1000 * args.n_iter if key == 'SimGDA+M' else args.n_iter
        min_len = n_iter
        pbar = tqdm(custom_linspace(0.2, 1.5, 0.1), desc=key)
        result = list(range(n_iter)) 
        for factor in pbar:
            for momentum_x in tuner_momentum:
                for momentum_y in tuner_momentum:
                    if momentum_x == momentum_y == 0: continue
                    α, β = np.array(base_lr['AltGDA']) * factor
                    rec = runner.run(α, β, γ, δ, n_iter, ϵ, verbose, momentum_x, momentum_y)
                    if len(rec) < min_len and rec[-1] <= ϵ:
                        min_len = len(rec)
                        result = rec
                        pbar.set_description(f'{key} | {min_len} f {factor:.3g} mx {momentum_x} my {momentum_y}')
        return result


    def tune_baseline(cls, seed_init, seed_func):
        key = cls.__name__
        runner = cls(μ_x, μ_y, μ_xy, L_x, L_y, L_xy, d_x, d_y, seed_init, seed_func)
        min_len = n_iter
        iterator = list(product(custom_linspace(0.1, 1.5, 0.1), custom_linspace(0.1, 1.5, 0.1)))
        pbar = tqdm(iterator, desc=key, total=len(iterator))
        result = list(range(n_iter)) 
        for factor, factor2 in pbar:
            α, β = np.array(base_lr['AltGDA']) * factor
            γ, δ = np.array(base_lr['AltGDA']) * factor2
            if α == γ and 'OGD' in key: continue
            rec = runner.run(α, β, γ, δ, n_iter, ϵ, verbose)
            if len(rec) < min_len and rec[-1] <= ϵ:
                min_len = len(rec)
                result = rec
                pbar.set_description(f'{key} | {min_len*cls.grad_complexity} f {factor:.3g} {factor2:.3g}')
        return result


    def tune_baseline_M(cls, seed_init, seed_func):
        key = cls.__name__
        runner = cls(μ_x, μ_y, μ_xy, L_x, L_y, L_xy, d_x, d_y, seed_init, seed_func)
        min_len = n_iter
        iterator = list(product(custom_linspace(0.1, 1.5, 0.1), custom_linspace(0.1, 1.5, 0.1)))
        pbar = tqdm(iterator, desc=key+'+M', total=len(iterator))
        result = list(range(n_iter)) 
        for factor, factor2 in pbar:
            for momentum_x in tuner_momentum:
                for momentum_y in tuner_momentum:
                    # if momentum_x == momentum_y == 0: continue
                    α, β = np.array(base_lr['AltGDA']) * factor
                    γ, δ = np.array(base_lr['AltGDA']) * factor2
                    rec = runner.run(α, β, γ, δ, n_iter, ϵ, verbose, momentum_x, momentum_y)
                    if len(rec) < min_len and rec[-1] <= ϵ:
                        min_len = len(rec)
                        result = rec
                        pbar.set_description(f'{key}+M | {min_len*cls.grad_complexity} f {factor:.3g} {factor2:3g} mx {momentum_x} my {momentum_y}')
        return result

    gradient_complexities = {
        'SimGDA': [],
        'SimGDA+M': [],
        'AltGDA': [],
        'AltGDA+M': [],
        'EG': [],
        'EG+M': [],
        'AltEG': [],
        'AltEG+M': [],
        'OGD': [],
        'OGD+M': [],
        'AltOGD': [],
        'AltOGD+M': [],
        'AlexGDA': [],
        'AlexGDA+M': [],
    }
    
    save_name = f"μx{args.mu_x}_μy{args.mu_y}_μxy{args.mu_xy}_Lx{args.L_x}_Ly{args.L_y}_Lxy{args.L_xy}_" \
                f"ϵ{args.eps}_dx{args.d_x}_dy{args.d_y}_T{args.n_iter}"
    logging.basicConfig(
        filename='log_'+save_name+'.log', 
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
    )

    for seed_init, seed_func in product(range(3), range(args.num_seeds)):
        print("seed_init", seed_init, "seed_func", seed_func)

        simgda = len(tune_GDA('SimGDA', 1, 0, seed_init, seed_func))
        simgda_m = len(tune_GDA_M('SimGDA+M', 1, 0, seed_init, seed_func))
        logging.info(f'Sim {simgda} Sim+M {simgda_m}')

        altgda = len(tune_GDA('AltGDA', 1, 1, seed_init, seed_func))
        altgda_m = len(tune_GDA_M('AltGDA+M', 1, 1, seed_init, seed_func))
        logging.info(f'Alt {altgda} Alt+M {altgda_m}')

        records = {}
        for γ in custom_linspace(0.5, 3.0, 0.1):
            for δ in custom_linspace(0.5, 3.0, 0.1):
                if (γ == δ == 1): continue
                records[f'AlexGDA[{γ}|{δ}]'] = tune_GDA(f'AlexGDA[{γ}|{δ}]', γ, δ, seed_init, seed_func)
        best_alexgda = min(records, key=lambda k: (len(records[k]), records[k][-1]))
        alexgda = len(records[best_alexgda])
        logging.info(f'Best Alex   {best_alexgda} len={alexgda}')
        
        records_m = {}
        for γ in custom_linspace(1.0, 3.0, 0.1):
            for δ in custom_linspace(1.0, 3.0, 0.1):
                if (γ == δ == 1): continue
                records_m[f'AlexGDA[{γ}|{δ}]+M'] = tune_GDA_M(f'AlexGDA[{γ}|{δ}]+M', γ, δ, seed_init, seed_func)
        best_alexgda_m = min(records_m, key=lambda k: (len(records_m[k]), records_m[k][-1]))
        alexgda_m = len(records_m[best_alexgda_m])
        logging.info(f'Best Alex+M {best_alexgda_m} len={alexgda_m}')

        eg = len(tune_baseline(EG, seed_init, seed_func)) * EG.grad_complexity
        eg_m = len(tune_baseline_M(EG, seed_init, seed_func)) * EG.grad_complexity
        logging.info(f'EG {eg} EG+M {eg_m}')

        ogd = len(tune_baseline(OGD, seed_init, seed_func)) * OGD.grad_complexity
        ogd_m = len(tune_baseline_M(OGD, seed_init, seed_func)) * OGD.grad_complexity
        logging.info(f'OGD {ogd} OGD+M {ogd_m}')

        alteg = len(tune_baseline(AltEG, seed_init, seed_func)) * AltEG.grad_complexity
        alteg_m = len(tune_baseline_M(AltEG, seed_init, seed_func)) * AltEG.grad_complexity
        logging.info(f'AltEG {alteg} AltEG+M {alteg_m}')

        altogd = len(tune_baseline(AltOGD, seed_init, seed_func)) * AltOGD.grad_complexity
        altogd_m = len(tune_baseline_M(AltOGD, seed_init, seed_func)) * AltOGD.grad_complexity
        logging.info(f'AltOGD {altogd} AltOGD+M {altogd_m}')

        gradient_complexities['SimGDA'].append(simgda)
        gradient_complexities['SimGDA+M'].append(simgda_m)
        gradient_complexities['AltGDA'].append(altgda)
        gradient_complexities['AltGDA+M'].append(altgda_m)
        gradient_complexities['EG'].append(eg)
        gradient_complexities['EG+M'].append(eg_m)
        gradient_complexities['AltEG'].append(alteg)
        gradient_complexities['AltEG+M'].append(alteg_m)
        gradient_complexities['OGD'].append(ogd)
        gradient_complexities['OGD+M'].append(ogd_m)
        gradient_complexities['AltOGD'].append(altogd)
        gradient_complexities['AltOGD+M'].append(altogd_m)
        gradient_complexities['AlexGDA'].append(alexgda)
        gradient_complexities['AlexGDA+M'].append(alexgda_m)

    print()
    print()
    print()

    gradient_complexities = pd.DataFrame.from_dict(gradient_complexities)
    gradient_complexities.to_csv('GradCplxt_'+save_name+'.csv')


