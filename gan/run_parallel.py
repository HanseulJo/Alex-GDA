import argparse
import copy
import multiprocessing as mp
from run import run
import time
from datetime import datetime
from itertools import product

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',    type=str,   default='./configs')
    parser.add_argument('--config_name',    type=str,   default='cifar10')
    parser.add_argument('--log_path', type=str, default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument('--seeds',      type=int, default=[0],  nargs='*')
    parser.add_argument('--devices',    type=int, default=[0],  nargs='*')
    parser.add_argument('--lrs_G',    type=float, default=[0.0001],  nargs='*')
    parser.add_argument('--lrs_D',    type=float, default=[0.0003],  nargs='*')
    parser.add_argument('--gammas',    type=float, default=[1],  nargs='*')
    parser.add_argument('--deltas',    type=float, default=[1],  nargs='*')
    parser.add_argument('--num_exp_per_device', type=int,   default=1)
    parser.add_argument('--overrides',  type=str, default=[],   nargs='*')
    args = vars(parser.parse_args())

    log_path = args.pop('log_path')
    seeds = args.pop('seeds')
    available_gpus = args.pop('devices')
    lrs_G = args.pop('lrs_G')
    lrs_D = args.pop('lrs_D')
    gammas = args.pop('gammas')
    deltas = args.pop('deltas')
    num_exp_per_device = args.pop('num_exp_per_device')

    experiments = []
    for lrG, lrD, gamma, delta, seed in product(lrs_G, lrs_D, gammas, deltas, seeds):
        exp = copy.deepcopy(args)
        exp['log_path'] = f"{log_path}_gamma{gamma}_delta{delta}_G{lrG}_D{lrD}/seed{seed}"
        exp['overrides'].append(f'seed={seed}')
        exp['overrides'].append(f'gamma={gamma}')
        exp['overrides'].append(f'delta={delta}')
        exp['overrides'].append(f'optimizer.G.lr={lrG}')
        exp['overrides'].append(f'optimizer.D.lr={lrD}')
        experiments.append(exp)
    
    print(experiments)
    
    # run parallell experiments
    # https://docs.python.org/3.5/library/multiprocessing.html#contexts-and-start-methods
    mp.set_start_method('spawn')
    process_dict = {gpu_id: [] for gpu_id in available_gpus}
    
    for exp in experiments:
        wait = True

        # wait until there exists a finished process
        while wait:
            # Find all finished processes and register available GPU
            for gpu_id, processes in process_dict.items():
                for process in processes:
                    if not process.is_alive():
                        print(f"Process {process.pid} on GPU {gpu_id} finished.")
                        processes.remove(process)
                        if gpu_id not in available_gpus:
                            available_gpus.append(gpu_id)
            
            for gpu_id, processes in process_dict.items():
                if len(processes) < num_exp_per_device:
                    wait = False
                    gpu_id, processes = min(process_dict.items(), key=lambda x: len(x[1]))
                    break
            
            time.sleep(1)

        # get running processes in the gpu
        processes = process_dict[gpu_id]
        exp['overrides'].append(f'gpus=[{gpu_id}]')
        process = mp.Process(target=run, args=(exp, ))
        process.start()
        processes.append(process)
        print(f"Process {process.pid} on GPU {gpu_id} started.")

        # check if the GPU has reached its maximum number of processes
        if len(processes) == num_exp_per_device:
            available_gpus.remove(gpu_id)

