from multiprocessing import Manager
from time import time
import wandb as wb

from grid2op.gym_compat import BoxGymActSpace, DiscreteActSpace # if we import gymnasium, GymEnv will convert to Gymnasium! 

import csv
import numpy as np
from alg.dqn.core import DQN
from alg.lagr_ppo.core import LagrPPO
from alg.ppo.core import PPO
from alg.sac.core import SAC
from alg.td3.core import TD3
from alg.Mask_ppo.core import MASKPPO
from common.checkpoint import DQNCheckpoint, LagrPPOCheckpoint, PPOCheckpoint, SACCheckpoint, TD3Checkpoint, MASKPPOCheckpoint
from common.imports import *
from common.utils import set_random_seed, set_torch, str2bool
from env.config import get_env_args
from env.utils import ENV_DIR, auxiliary_make_env, load_config

# Dictionary mapping algorithm names to their corresponding classes
ALGORITHMS: Dict[str, Type[Any]] = {'DQN': DQN, 'PPO': PPO, 'SAC': SAC, 'TD3': TD3, 'LAGRPPO': LagrPPO, 'MASKPPO': MASKPPO}

def main(args: Namespace) -> None:
    """
    Main function to run the RL algorithms based on the provided arguments.

    Args:
        args (Namespace): Command line arguments parsed by argparse.

    Raises:
        AssertionError: If time limit exceeds 2800 minutes or if number of environments is less than 1.
        AssertionError: If the specified algorithm is not supported.
    """
    assert args.time_limit <= 2800, f"Invalid time limit: {args.time_limit}. Timeout limit is : 2800"
    start_time = time()
    
    # Update args with environment arguments
    args = ap.Namespace(**vars(args), **vars(get_env_args()))
    assert args.n_envs >= 1, f"Invalid n° of environments: {args.n_envs}. Must be >= 1"
    
    alg = args.alg.upper()
    assert alg in ALGORITHMS.keys(), f"Unsupported algorithm: {alg}. Supported algorithms are: {ALGORITHMS}"
    if (alg == "LAGRPPO" and args.constraints_type == 0) or (alg != "LAGRPPO" and args.constraints_type in [1, 2]):
        raise ValueError("Check the constrained version of the alg/env!")

    run_name = args.resume_run_name if args.resume_run_name \
        else f"{args.alg}_{args.env_id}_{'T' if args.action_type == 'topology' else 'R'}_{args.seed}_{args.difficulty}_{'H' if args.use_heuristic else ''}_{'I' if args.heuristic_type == 'idle' else ''}_{'C1' if args.constraints_type == 1 else 'C2' if args.constraints_type == 2 else ''}_{int(time())}_{np.random.randint(0, 50000)}"
        #else f"{args.alg}_{args.env_id}_{"T" if args.action_type == "topology" else "R"}_{args.seed}_{args.difficulty}_{"H" if args.use_heuristic else ""}_{"I" if args.heuristic_type == "idle" else ""}_{"C1" if args.constraints_type == 1 else "C2" if args.constraints_type == 2 else ""}_{int(time())}_{np.random.randint(0, 50000)}"

    # Initialize the appropriate checkpoint based on the algorithm
    if alg == 'LAGRPPO': checkpoint = LagrPPOCheckpoint(run_name, args)
    elif alg == 'DQN': checkpoint = DQNCheckpoint(run_name, args)
    elif alg == 'PPO' : checkpoint = PPOCheckpoint(run_name, args)
    elif alg == 'SAC': checkpoint = SACCheckpoint(run_name, args)
    elif alg == 'TD3': checkpoint = TD3Checkpoint(run_name, args)
    elif alg == 'MASKPPO': checkpoint = MASKPPOCheckpoint(run_name, args)
    else:
        pass  # This case should not occur due to earlier assertion


    # Set random seed and Torch configuration
    #set_random_seed(args.seed)
    #changed to try actual random seed
    set_random_seed(args.seed)  # Random seed for each run
    set_torch(args.n_threads, args.th_deterministic, args.cuda)
    
    # Resume run if checkpoint was resumed
    if checkpoint.resumed: args = checkpoint.loaded_run['args']

    # Create multiple async environments for parallel processing
    main_gym_env, main_g2o_env = auxiliary_make_env(args, checkpoint.resumed)

    with Manager() as manager:
        if args.action_type == "topology":
            print("Initializing the distributed action 'mapper'... (takes a while with big action spaces)")
            shared_action_space = manager.list(main_gym_env.action_space.converter.all_actions)

        def make_vec_subprocess(idx):
            if args.action_type == "topology":
                action_space = DiscreteActSpace(main_g2o_env.action_space,
                                                    action_list=shared_action_space)
                return auxiliary_make_env(args, resume_run=checkpoint.resumed, idx=idx, action_space=action_space)[0]
            
            return auxiliary_make_env(args, resume_run=checkpoint.resumed, idx=idx)[0]
            
        envs = gym.vector.AsyncVectorEnv([lambda i=i: make_vec_subprocess(i) for i in range(args.n_envs)])
        #Modifying so I can apply masking in Mask PPO - change back when using normal code
        #envs = gym.vector.SyncVectorEnv([lambda i=i: make_vec_subprocess(i) for i in range(args.n_envs)])

        if alg == 'MASKPPO':
            difficulty = args.difficulty
            env_id = args.env_id
            loaded_action_space = np.load(f"{ENV_DIR}/action_spaces/{env_id}_action_space.npy", allow_pickle=True)
            config = load_config(args.env_config_path)
            env_config = config['environments']
            max_difficulty = env_config[env_id]['difficulty']
            n_actions = np.geomspace(50, len(loaded_action_space), num=max_difficulty).astype(int)
            action_list=loaded_action_space[:n_actions[difficulty]]
            print(f"Action list for difficulty {difficulty}:", action_list[0:10])  # Print the first 10 actions in the list for verification
            ALGORITHMS[alg](envs, run_name, start_time, args, checkpoint, main_g2o_env, action_list)
        else:
            ALGORITHMS[alg](envs, run_name, start_time, args, checkpoint)

        # survival_data = []

        # with open('survival.csv', 'r', newline='', encoding='utf-8') as file: 
        #     reader = csv.reader(file)
            
        #     survival_data = [float(row[0]) for row in reader]  



        # run_avg_survival = sum(survival_data)/num_runs
        # std_dev = np.std(survival_data)

        #     # Use newline='' to prevent extra blank lines on some platforms
        # with open('output.csv', 'a', newline='', encoding='utf-8') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(["AVG_SURV" + str(run_avg_survival)]) 
        #     writer.writerow(["STD_DEV" + str(std_dev)]) 
                
        
if __name__ == "__main__":
    parser = ap.ArgumentParser()

    # Cluster
    parser.add_argument("--time-limit", type=float, default=1300, help="Time limit for the action ranking")
    parser.add_argument("--checkpoint", type=str2bool, default=True, help="Toggles checkpoint.")
    parser.add_argument("--resume-run-name", type=str, default='', help="Run name to resume")

    # Reproducibility
    parser.add_argument("--alg", type=str, default='PPO', help="Algorithm to run")
    parser.add_argument("--seed", type=int, default=np.random.randint(0, 50000), help="Random seed")

    # Logger
    parser.add_argument("--verbose", type=str2bool, default=True, help="Toggles prints")
    parser.add_argument("--exp-tag", type=str, default='True', help="Tag for logging the experiment")
    parser.add_argument("--track", type=str2bool, default=True, help="Tag for logging the experiment")
    parser.add_argument("--wandb-project", type=str, default="RLGridProject26", help="Wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="DeepRl_2026", help="Entity (team) of wandb's project")
    parser.add_argument("--wandb-mode", type=str, default="online", help="Online or offline wandb mode.")
    #Added for plotting purposes - change between methods
    #parser.add_argument("--group-name", type=str, default='PPO_Group', help="Group name for logging the experiment")

    # Torch
    parser.add_argument("--th-deterministic", type=str2bool, default=True, help="Enable deterministic in Torch.")
    parser.add_argument("--cuda", type=str2bool, default=False, help="Enable CUDA by default.")
    parser.add_argument("--n-threads", type=int, default=4, help="Max number of torch threads.")

    main(parser.parse_known_args()[0])




# from multiprocessing import Manager
# from time import time
# import wandb as wb

# from grid2op.gym_compat import BoxGymActSpace, DiscreteActSpace # if we import gymnasium, GymEnv will convert to Gymnasium! 

# import csv
# import numpy as np
# from alg.dqn.core import DQN
# from alg.lagr_ppo.core import LagrPPO
# from alg.ppo.core import PPO
# from alg.sac.core import SAC
# from alg.td3.core import TD3
# from alg.Mask_ppo.core import MASKPPO
# from common.checkpoint import DQNCheckpoint, LagrPPOCheckpoint, PPOCheckpoint, SACCheckpoint, TD3Checkpoint, MASKPPOCheckpoint
# from common.imports import *
# from common.utils import set_random_seed, set_torch, str2bool
# from env.config import get_env_args
# from env.utils import ENV_DIR, auxiliary_make_env, load_config

# # Dictionary mapping algorithm names to their corresponding classes
# ALGORITHMS: Dict[str, Type[Any]] = {'DQN': DQN, 'PPO': PPO, 'SAC': SAC, 'TD3': TD3, 'LAGRPPO': LagrPPO, 'MASKPPO': MASKPPO}

# def main(args: Namespace) -> None:
#     """
#     Main function to run the RL algorithms based on the provided arguments.

#     Args:
#         args (Namespace): Command line arguments parsed by argparse.

#     Raises:
#         AssertionError: If time limit exceeds 2800 minutes or if number of environments is less than 1.
#         AssertionError: If the specified algorithm is not supported.
#     """
#     assert args.time_limit <= 2800, f"Invalid time limit: {args.time_limit}. Timeout limit is : 2800"
#     start_time = time()
    
#     # Update args with environment arguments
#     args = ap.Namespace(**vars(args), **vars(get_env_args()))
#     assert args.n_envs >= 1, f"Invalid n° of environments: {args.n_envs}. Must be >= 1"
    
#     alg = args.alg.upper()
#     assert alg in ALGORITHMS.keys(), f"Unsupported algorithm: {alg}. Supported algorithms are: {ALGORITHMS}"
#     if (alg == "LAGRPPO" and args.constraints_type == 0) or (alg != "LAGRPPO" and args.constraints_type in [1, 2]):
#         raise ValueError("Check the constrained version of the alg/env!")

#     run_name = args.resume_run_name if args.resume_run_name \
#         else f"{args.alg}_{args.env_id}_{'T' if args.action_type == 'topology' else 'R'}_{args.seed}_{args.difficulty}_{'H' if args.use_heuristic else ''}_{'I' if args.heuristic_type == 'idle' else ''}_{'C1' if args.constraints_type == 1 else 'C2' if args.constraints_type == 2 else ''}_{int(time())}_{np.random.randint(0, 50000)}"
#         #else f"{args.alg}_{args.env_id}_{"T" if args.action_type == "topology" else "R"}_{args.seed}_{args.difficulty}_{"H" if args.use_heuristic else ""}_{"I" if args.heuristic_type == "idle" else ""}_{"C1" if args.constraints_type == 1 else "C2" if args.constraints_type == 2 else ""}_{int(time())}_{np.random.randint(0, 50000)}"

#     # Auto-build wandb group for multi-seed aggregation
#     # if not args.wandb_group:
#     #     parts = [args.alg.upper()]
#     #     if args.constraints_type == 1: parts.append("C1")
#     #     elif args.constraints_type == 2: parts.append("C2")
#     #     parts.append("T" if args.action_type == "topology" else "R")
#     #     args.wandb_group = "_".join(parts)

#     # Initialize the appropriate checkpoint based on the algorithm
#     if alg == 'LAGRPPO': checkpoint = LagrPPOCheckpoint(run_name, args)
#     elif alg == 'DQN': checkpoint = DQNCheckpoint(run_name, args)
#     elif alg == 'PPO' : checkpoint = PPOCheckpoint(run_name, args)
#     elif alg == 'SAC': checkpoint = SACCheckpoint(run_name, args)
#     elif alg == 'TD3': checkpoint = TD3Checkpoint(run_name, args)
#     elif alg == 'MASKPPO': checkpoint = MASKPPOCheckpoint(run_name, args)
#     else:
#         pass  # This case should not occur due to earlier assertion


#     # Set random seed and Torch configuration
#     #set_random_seed(args.seed)
#     #changed to try actual random seed
#     set_random_seed(args.seed)  # Random seed for each run
#     set_torch(args.n_threads, args.th_deterministic, args.cuda)
    
#     # Resume run if checkpoint was resumed
#     if checkpoint.resumed: args = checkpoint.loaded_run['args']

#     # Create multiple async environments for parallel processing
#     main_gym_env, main_g2o_env = auxiliary_make_env(args, checkpoint.resumed)

#     main_gym_env, main_g2o_env = auxiliary_make_env(args, checkpoint.resumed)



#     with Manager() as manager:
#         if args.action_type == "topology":
#             print("Initializing the distributed action 'mapper'... (takes a while with big action spaces)")
#             shared_action_space = manager.list(main_gym_env.action_space.converter.all_actions)

#         def make_vec_subprocess(idx):
#             if args.action_type == "topology":
#                 action_space = DiscreteActSpace(main_g2o_env.action_space,
#                                                     action_list=shared_action_space)
#                 #return auxiliary_make_env(args, resume_run=checkpoint.resumed, idx=idx, action_space=action_space)[0]
#                 return auxiliary_make_env(args, resume_run=checkpoint.resumed, idx=idx, action_space=action_space)[0]
            
#             return auxiliary_make_env(args, resume_run=checkpoint.resumed, idx=idx)[0]
            
#         #envs = gym.vector.AsyncVectorEnv([lambda i=i: make_vec_subprocess(i) for i in range(args.n_envs)])
#         #Modifying so I can apply masking in Mask PPO - change back when using normal code
#         envs = gym.vector.SyncVectorEnv([lambda i=i: make_vec_subprocess(i) for i in range(args.n_envs)])

        
#         if alg == 'MASKPPO':
#             difficulty = args.difficulty
#             env_id = args.env_id
#             loaded_action_space = np.load(f"{ENV_DIR}/action_spaces/{env_id}_action_space.npy", allow_pickle=True)
#             config = load_config(args.env_config_path)
#             env_config = config['environments']
#             max_difficulty = env_config[env_id]['difficulty']
#             n_actions = np.geomspace(50, len(loaded_action_space), num=max_difficulty).astype(int)
#             action_list=loaded_action_space[:n_actions[difficulty]]
#             print(f"Action list for difficulty {difficulty}:", action_list[0:10])  # Print the first 10 actions in the list for verification
#             ALGORITHMS[alg](envs, run_name, start_time, args, checkpoint, main_g2o_env, action_list)
#         else:
#             ALGORITHMS[alg](envs, run_name, start_time, args, checkpoint)


                
        
# if __name__ == "__main__":
#     parser = ap.ArgumentParser()

#     # Cluster
#     parser.add_argument("--time-limit", type=float, default=1300, help="Time limit for the action ranking")
#     parser.add_argument("--checkpoint", type=str2bool, default=True, help="Toggles checkpoint.")
#     parser.add_argument("--resume-run-name", type=str, default='', help="Run name to resume")

#     # Reproducibility
#     parser.add_argument("--alg", type=str, default='PPO', help="Algorithm to run")
#     parser.add_argument("--seed", type=int, nargs='+', default=[0, 1, 2, 3, 4], help="Random seed(s). Pass multiple for multi-seed runs.")
#     # Logger
#     parser.add_argument("--verbose", type=str2bool, default=True, help="Toggles prints")
#     parser.add_argument("--exp-tag", type=str, default='True', help="Tag for logging the experiment")
#     parser.add_argument("--track", type=str2bool, default=True, help="Tag for logging the experiment")
#     parser.add_argument("--wandb-project", type=str, default="RLGridProject26", help="Wandb's project name")
#     parser.add_argument("--wandb-entity", type=str, default="DeepRl_2026", help="Entity (team) of wandb's project")
#     parser.add_argument("--wandb-mode", type=str, default="online", help="Online or offline wandb mode.")
#     #Added for plotting purposes - change between methods
#     #parser.add_argument("--group-name", type=str, default='PPO_Group', help="Group name for logging the experiment")
#     parser.add_argument("--wandb-group", type=str, default="MASK_PPO_Group", help="Wandb group name for multi-seed aggregation.")
    
#     # Torch
#     parser.add_argument("--th-deterministic", type=str2bool, default=True, help="Enable deterministic in Torch.")
#     parser.add_argument("--cuda", type=str2bool, default=False, help="Enable CUDA by default.")
#     parser.add_argument("--n-threads", type=int, default=4, help="Max number of torch threads.")

#     main(parser.parse_known_args()[0])
#     # parsed_args = parser.parse_known_args()[0]
#     # results = []

#     # for seed in parsed_args.seeds:
#     #     seed_args = ap.Namespace(**{k: v for k, v in vars(parsed_args).items() if k != 'seeds'}, seed=seed)
#     #     print(f"\n{'='*50}\n Starting run with seed={seed}\n{'='*50}")
#     #     best_survival = main(seed_args)
#     #     if best_survival is not None:
#     #         results.append(best_survival)
#     #         print(f" Seed {seed} best eval survival: {best_survival:.4f}")

#     # if results:
#     #     arr = np.array(results)
#     #     print(f"\n{'='*50}")
#     #     print(f" FINAL SUMMARY ({len(arr)} seeds)")
#     #     print(f" Best Eval Survival: {arr.mean():.4f} ± {arr.std():.4f}")
#     #     print(f" Individual: {[f'{x:.4f}' for x in arr]}")
#     #     print(f"{'='*50}")

#     #     # Log group summary to wandb
#     #     if parsed_args.track:
#     #         parts = [parsed_args.alg.upper()]
#     #         ct = getattr(parsed_args, 'constraints_type', 0)
#     #         if ct == 1: parts.append("C1")
#     #         elif ct == 2: parts.append("C2")
#     #         if getattr(parsed_args, 'use_cbf', False): parts.append("CBF")
#     #         parts.append("T" if parsed_args.action_type == "topology" else "R")
#     #         group_name = parsed_args.wandb_group or "_".join(parts)

#     #         summary_run = wb.init(
#     #             name=f"{group_name}_summary",
#     #             project=parsed_args.wandb_project,
#     #             entity=parsed_args.wandb_entity,
#     #             mode=parsed_args.wandb_mode,
#     #             group=group_name,
#     #             job_type="summary",
#     #             config=vars(parsed_args),
#     #         )
#     #         wb.log({
#     #             "summary/mean_best_survival": float(arr.mean()),
#     #             "summary/std_best_survival": float(arr.std()),
#     #             **{f"summary/seed_{parsed_args.seeds[i]}_best_survival": float(v) for i, v in enumerate(arr)},
#     #         })
#     #         wb.finish()