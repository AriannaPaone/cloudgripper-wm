

"""Launch Isaac Sim Simulator first."""

import argparse
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # for better error debugging, can be removed for faster training when code is stable

#from isaaclab.app import AppLauncher

# # add argparse arguments
# parser = argparse.ArgumentParser(description="Tutorial on creating a ant base environment.")
# parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to spawn.")
# parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")

# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# # parse the arguments
# args_cli = parser.parse_args()
# # default to launch headless webRTC traces
# args_cli.headless = True
# #args_cli.livestream = 2
# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# import isaaclab_tasks  # noqa: F401
# import isaaclab.envs  # noqa: F401
# from isaaclab_tasks.utils import parse_env_cfg
# import AntMazeManagerv0.tasks  # noqa: F401


# from AntMazeManagerv0.tasks.manager_based.antmazemanagerv0.antmazemanagerv0_env_cfg import Antmazemanagerv0EnvCfg,Antmazemanagerv0SceneCfg_Ground,Antmazemanagerv0SceneCfg_Maze,Antmazemanagerv0SceneCfg_Pyramid,Antmazemanagerv0SceneCfg_Cave # noqa: F401
# from AntMazeManagerv0.tasks.manager_based.franka_stack.franka_stack_env_cfg import FrankaCubeStackEnvCfg
# from AntMazeManagerv0.tasks.manager_based.shadow_hand.shadow_hand_env_cfg import ShadowHandManagerEnvCfg
# #from AntMazeManagerv0.tasks.manager_based.antmazemanagerv0 import Antmazemanagerv0EnvCfg
# from isaaclab.envs import ManagerBasedEnv,ManagerBasedRLEnv

from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter
import time
import torch
import math
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import numpy as np

#import for not shared network architecture
#import AntMazeManagerv0.src.sac_policy as policy
#from AntMazeManagerv0.src.sac_policy import PolicyNetwork,Discretizer
#from AntMazeManagerv0.src  import pl_agent_isaac_sim as pl_agent

# import AntMazeManagerv0.src.policy_multihead as policy
# from AntMazeManagerv0.src.policy_multihead import PolicyMultiheadNetwork, Discretizer
# from AntMazeManagerv0.src import utils as t_utils
# from AntMazeManagerv0.src  import pl_agent_isaac_sim_automatic as pl_agent

from scripts.data.KMyriad.policy_multihead import PolicyMultiheadNetwork, Discretizer
from scripts.data.KMyriad import utils as t_utils
from scripts.data.KMyriad import pl_agent_isaac_sim_automatic as pl_agent
from scripts.data.KMyriad.env_adapter import MaxEntEnvAdapter
from scripts.data.coverage import GridCoverage

from gymnasium import spaces


def main():
    """Main function."""

    #####################################################
    ### Experiment Configurations
    #####################################################

    # task_type = {"empty": Antmazemanagerv0SceneCfg_Ground,"maze": Antmazemanagerv0SceneCfg_Maze,
    #                 "pyramid": Antmazemanagerv0SceneCfg_Pyramid, "cave": Antmazemanagerv0SceneCfg_Cave,
    #                 "franka_stack": FrankaCubeStackEnvCfg, "inhand": ShadowHandManagerEnvCfg}
    

    num_agents = [1]
    multihead = True
    num_epochs = 100  # Number of epochs for training
    name_env =  "cloudgripper_mujoco" #"franka_stack"     #'PointMaze_UMazeDense-v3' #'AntMaze_Large_Diverse_G-v5' #'PointMaze_Large_Diverse_G-v3' #"PointMaze_Open-v3" #"Swimmer-v5" PointMaze_Open-v3 AntMaze_UMaze-v5
    seed = [42] #[0,1,56,123,22,66,55,77,88,99]  # Seed for reproducibility
    ks = [5]
    hidden_sizes = [512,256] # 'AntMaze_Large_Diverse_G-v5'[512,512] # policy network Swimmer [128,128] # Hidden layer sizes for the policy network Swimmer [128,128] # Hidden layer sizes for Ant [512,512]
    traj_len = 300 # Trajectory len for each rollout
    total_trajs = 32 #1000  # Total number of trajectories to collect
    num_envs =  total_trajs ### should be equal to the number of trajectories becuase in one simulation we collect all the trajectories in parallel
    env = None
    chunk_size = 1
    log_entropy = 40
    trunk_lr = 0.0005
    head_lr = 0.0002 if multihead else 0.0002
    #tmp_lr = 0.0004 if multihead else 0.0005
    milestones = [60,150] if not multihead else [80]
    state_filtering = [0,1,5,6] #list(range(3,7)) #[0, 1, 2, 7, 8, 9, 14, 15, 16,57,58,59]
    
    automatic_budget = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    arm_cov   = GridCoverage([(0.0, 1.0), (0.0, 1.0)], [20, 20])
    cube_cov  = GridCoverage([(0.0, 1.0), (0.0, 1.0)], [20, 20])
    cube_disp = GridCoverage([(-1.0, 1.0), (-1.0, 1.0)], [20, 20])

    ######################################################
    ######################################################

    # if "franka_stack" in name_env:
    #     env_cfg = FrankaCubeStackEnvCfg()
    #     state_filtering = [21, 22, 24, 25, 27, 28]#[0,1,7,8,14,15,21,22,23,24,25,26,27,28,29]
    #     discretizer = Discretizer([[-2, 2.0], [-2.0,2.0]], [80, 80])
    #     env_cfg.scene.num_envs = num_envs
    # elif "inhand" in name_env:
    #     env_cfg = ShadowHandManagerEnvCfg()
    #     state_filtering = list(range(3,7))
    #     discretizer = Discretizer([[-2, 2.0], [-2.0,2.0]], [80, 80])
    #     env_cfg.scene.num_envs = num_envs
    # else:
    #     env_cfg = Antmazemanagerv0EnvCfg()
    #     discretizer = Discretizer([[-20, 20.0], [-12.0,12.0]], [50, 50])
    #     state_filtering = [0,1]
    #     env_cfg.scene = task_type[name_env](num_envs=num_envs,env_spacing=0.0)
    #     env_cfg.scene.terrain.env_origins = torch.tensor([(0.0, 0.0, 0.5)], device=device).repeat(num_envs, 1)
    
    env = MaxEntEnvAdapter(num_envs=num_envs, height=64, width=64, device=device)
    discretizer = Discretizer([[0.0, 1.0], [0.0, 1.0]], [50, 50])

    for a in num_agents:
        for s in seed:
            for k in ks:
                t_utils.set_seed(s)  # Set the random seed for reproducibility
                writer,exp_folder = t_utils.init_writer(env_name=name_env, seed=s,MaxEnt=True, 
                                                        num_envs=a, hidden_size=hidden_sizes)  # Initialize TensorBoard writer
                
                # if env is None:
                #     env = ManagerBasedRLEnv(cfg=env_cfg)
                     
                # obs_dim = env.observation_manager.group_obs_dim["policy"][0]
                # act_dim = env.action_manager.action_term_dim[0] * chunk_size
                
                obs_dim = env.num_features
                act_dim = env.num_actions * chunk_size

                env.reset(seed=s)
                
                # setup policy
                if multihead:
                    pl_policy = PolicyMultiheadNetwork(hidden_sizes=hidden_sizes,adapter_hidden=256,
                                                       activation=torch.nn.ReLU,num_envs=num_envs,
                                                       num_agents=a,state_dim=obs_dim, 
                                                       action_dim=act_dim, action_space=env.action_space,
                                                       latent_proj_dim=len(state_filtering)).to(device)
                    behavior_policy = PolicyMultiheadNetwork(hidden_sizes=hidden_sizes,adapter_hidden=256,
                                                             activation=torch.nn.ReLU,num_envs=num_envs,
                                                             num_agents=a,state_dim=obs_dim, 
                                                             action_dim=act_dim, action_space=env.action_space,
                                                             latent_proj_dim=len(state_filtering)).to(device)
                    
                    ################################
                    ### Load pre-trained policy
                    # if a == 1:
                    #     pl_policy.load_state_dict(torch.load('/isaac-lab/logs_new/MaxEnt/20260405-183959_franka_stack_seed_123_pre_False_envs_1_hidden_[512, 256]_goal_None/models/model_agent_0_step_100_seed_123.pt'))  # Load pre-trained policy
                    # elif a == 10:
                    #     pl_policy.load_state_dict(torch.load('/isaac-lab/logs_new/MaxEnt/20260405-120051_franka_stack_seed_22_pre_False_envs_10_hidden_[512, 256]_goal_None/models/model_agent_0_step_100_seed_22.pt'))  # Load pre-trained policy
                    # else:
                    #     pl_policy.load_state_dict(torch.load('/isaac-lab/logs_new/MaxEnt/20260405-135402_franka_stack_seed_1_pre_False_envs_50_hidden_[512, 256]_goal_None/models/model_agent_0_step_100_seed_1.pt'))  # Load pre-trained policy
                    ################################

                    ################################
                    ### Test Loading trunk MEPOL
                    #
                    # load only the trunk of a single agent model
                    #ckpt = torch.load('/isaac-lab/logs_new/MaxEnt/test/20260322-185724_empty_seed_0_pre_False_envs_1_hidden_[512, 256]_goal_None/models/model_agent_0_step_100_seed_0.pt')
                    #trunk_keys = {k: v for k, v in ckpt.items() if k.startswith('net.') or k.startswith('latent_proj.')}
                    #pl_policy.load_state_dict(trunk_keys, strict=False)
                    #
                    #
                    ################################

                    behavior_policy.load_state_dict(pl_policy.state_dict())
                    if a == 1:
                        optimizer = optim.Adam(pl_policy.parameters(), lr=head_lr)
                    elif a == 1000:
                        optimizer = optim.Adam([
                                                {"params": pl_policy.get_trunk_params(), "lr": trunk_lr},    # trunk: slow
                                                {"params": pl_policy.get_head_params(),  "lr": head_lr},    # heads: fast
                                                ])
                    else:
                        optimizer = optim.Adam(pl_policy.parameters(), lr=head_lr)
                    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

                # else:
                #     pl_policy = []
                #     behavior_policy = []
                #     optimizer = []
                #     scheduler = []
                #     tmp_policy = policy.init_single_policy(env,obs_dim, act_dim, hidden_sizes=hidden_sizes)
                #     for i in range(a):
                #         #tmp_policy = policy.init_single_policy(env,obs_dim, act_dim, hidden_sizes=hidden_sizes)
                #         pl_policy.append(PolicyNetwork(hidden_sizes=hidden_sizes,activation=torch.nn.ReLU,state_dim=obs_dim, action_dim=act_dim, action_space=env.cfg.action_space).to(device))
                #         behavior_policy.append(PolicyNetwork(hidden_sizes=hidden_sizes, activation=torch.nn.ReLU, state_dim=obs_dim, action_dim=act_dim, action_space=env.cfg.action_space).to(device))
                #         pl_policy[i].load_state_dict(tmp_policy.state_dict())
                #         behavior_policy[i].load_state_dict(pl_policy[i].state_dict())
                #         optimizer.append(optim.Adam(pl_policy[i].parameters(), lr=head_lr))
                #         scheduler.append(MultiStepLR(optimizer[i], milestones=milestones, gamma=0.5))

                count = 0
                num_trajectories = int(total_trajs / a)  # Number of trajectories per environment 

                if automatic_budget:
                    trajectory_budget = [1 for _ in range(a)]
                    trajectory_budget[0] = total_trajs - sum(trajectory_budget[1:])  # Adjust the first agent's budget to match total_trajs
                else:
                    trajectory_budget = [num_trajectories for _ in range(a)]  # Initial equal budget

                for i in range(num_epochs):
                    start = time.time()
                    invalide_step = False
                    # Store policy parameters before update
                    #old_params = copy.deepcopy([p.clone().detach() for p in pl_policy.parameters()])
                    env2agent =torch.repeat_interleave(torch.arange(a, device=device), torch.tensor(trajectory_budget, device=device))  # [num_envs] maps each env to an agent index based on the allocated budget
                    last_kl_divs,invalide_step = pl_agent.reinforce_collection_and_compute_knn(writer,count,env, pl_policy,behavior_policy,optimizer,
                                                                                            scheduler,discretizer,num_trajectories=num_trajectories, 
                                                                                            trajectory_length=traj_len, state_filter=state_filtering, 
                                                                                            num_agents = a,num_envs=num_envs, k=k, 
                                                                                            env2agent=env2agent,chunk_length=chunk_size,log_entropy_interval=log_entropy)

                    #print("Policy Updated:", params_changed)
                    print("Time Take: ", time.time() - start)
                    count += 1
                    print("Counter",count )
                    writer.flush()
                    if not multihead:
                        if not invalide_step:
                            for j in range(a):
                                behavior_policy[j].load_state_dict(pl_policy[j].state_dict())
                    else:
                        if not invalide_step:
                            behavior_policy.load_state_dict(pl_policy.state_dict())
                            if last_kl_divs is not None :
                                if automatic_budget:
                                    trajectory_budget = t_utils.compute_trajectory_budget_from_kl(
                                                            last_kl_divs,
                                                            total_trajs,
                                                            writer=writer,
                                                            min_budget=5,
                                                            count=count,
                                                            prev_budget=trajectory_budget,  # pass last step's allocation
                                                            alpha=0.3,                      # smoothing strength
                                                            temperature=0.2,                # soften/sharpen KL sensitivity
                                                            cap_step_abs=None,              # optional: limit absolute change per step
                                                            cap_step_frac=None)             # optional: limit to 25% of previous
                                                        
                                else:
                                    pass
                    if i % 10 == 0:
                        with torch.no_grad():
                            s_ev, _, _ = pl_agent.collect_particles(
                                env, pl_policy, 1, traj_len,
                                obs_dim, act_dim, a, num_envs, env2agent)
                        flat = s_ev.reshape(-1, obs_dim).cpu().numpy()
                        a_c = arm_cov.coverage(flat[:, :2])
                        c_c = cube_cov.coverage(flat[:, 5:7])
                        per_env = [arm_cov.coverage(s_ev[0, :, e, :2].cpu().numpy())
                                   for e in range(num_envs)]
                        cube_traj = np.transpose(s_ev[0, :, :, 5:7].cpu().numpy(), (1, 0, 2))
                        c_rel = cube_disp.coverage_relative(cube_traj)

                        writer.add_scalar("Coverage/arm", a_c, count)
                        writer.add_scalar("Coverage/arm_per_env", float(np.mean(per_env)), count)
                        writer.add_scalar("Coverage/cube_absolute", c_c, count)
                        writer.add_scalar("Coverage/cube_displacement", c_rel, count)
                        print(f"  coverage: arm {a_c:.3f}  arm/env {np.mean(per_env):.3f}  "
                              f"cube {c_c:.3f}  cube_disp {c_rel:.4f}")

                # Save the policy after training
                t_utils.save_settings(exp_folder, name_env, s, num_epochs, hidden_sizes, obs_dim, act_dim, a,total_trajs)
                
                if multihead:
                    t_utils.save_policy(pl_policy, exp_folder,0,num_epochs,s)
                else:
                    for i in range(a):
                        t_utils.save_policy(pl_policy[i], exp_folder,i,num_epochs,s)
                writer.close()


    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    #simulation_app.close()