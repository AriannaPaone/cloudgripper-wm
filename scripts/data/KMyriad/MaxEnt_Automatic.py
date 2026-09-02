import argparse
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # for better error debugging, can be removed for faster training when code is stable

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter
import time
import torch
import math
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import numpy as np

from scripts.data.KMyriad.policy_multihead import PolicyMultiheadNetwork, Discretizer
from scripts.data.KMyriad import utils as t_utils
from scripts.data.KMyriad import pl_agent_isaac_sim_automatic as pl_agent
from scripts.data.KMyriad.env_adapter import MaxEntEnvAdapter
from scripts.data.coverage import GridCoverage
from gymnasium import spaces

def train_maxent_policy(object_pos , agent_start_pos, num_agents = 1, multihead = True, num_epochs = 100, name_env=  "cloudgripper_mujoco", seed = 0, k = 5, hidden_sizes= [512,256], traj_len= 300, total_trajs = 32, num_envs=  32, 
                        env= None, chunk_size = 1, log_entropy = 40, 
                        trunk_lr = 0.0005, head_lr = 0.0002, milestones = [80], state_filtering = [0,1,5,6], automatic_budget = False, randomize_object_pos = True):
    """Train a MaxEnt multihead policy and save the checkpoint."""

    a = num_agents  # Assuming single values for simplicity since this is called for comparisons
    t_utils.set_seed(seed)  # Set the random seed for reproducibility

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arm_cov   = GridCoverage([(0.0, 1.0), (0.0, 1.0)], [20, 20])
    cube_cov  = GridCoverage([(0.0, 1.0), (0.0, 1.0)], [20, 20])
    cube_disp = GridCoverage([(-1.0, 1.0), (-1.0, 1.0)], [20, 20])

    env = MaxEntEnvAdapter(num_envs=num_envs, height=64, width=64, max_episode_steps=traj_len, object_pos=object_pos, agent_start_pos=agent_start_pos, device=device)
    print("max_episode_steps:", env.venv.envs[0].spec.max_episode_steps)
    print("traj_len:", traj_len)

    discretizer = Discretizer([[0.0, 1.0], [0.0, 1.0]], [50, 50])

    writer,exp_folder = t_utils.init_writer(env_name=name_env, seed=seed,MaxEnt=True, 
                                            num_envs=a, hidden_size=hidden_sizes
                                            #, goal_position = tag
                                            )  # Initialize TensorBoard writer

    obs_dim = env.num_features
    act_dim = env.num_actions * chunk_size

    env.reset(seed=seed)
    #print the obj pos in the env
    #obs, info = env.reset(seed=seed)
    #print(obs["policy"][:, 5:8])


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


    count = 0
    num_trajectories = int(total_trajs / a)  # Number of trajectories

    if automatic_budget:
        trajectory_budget = [1 for _ in range(a)]
        trajectory_budget[0] = total_trajs - sum(trajectory_budget[1:])  # Adjust the first agent's budget to match total_trajs
    else:
        trajectory_budget = [num_trajectories for _ in range(a)]  # Initial equal budget    

    pos_rng = np.random.default_rng(seed)
    for i in range(num_epochs):
        start = time.time()
        invalide_step = False
        env2agent =torch.repeat_interleave(torch.arange(a, device=device), torch.tensor(trajectory_budget, device=device))  # [num_envs] maps each env to an agent index based on the allocated budget

        #To randomize object pos before each epoch
        xy = pos_rng.uniform([-0.07, -0.05], [0.07, 0.05]) 
        if randomize_object_pos:
            env.epoch_options = {"variation_values": {"object.pos": xy}}

        last_kl_divs,invalide_step = pl_agent.reinforce_collection_and_compute_knn(writer,count,env, pl_policy,behavior_policy,optimizer,
                                                                                scheduler,discretizer,num_trajectories=num_trajectories, 
                                                                                trajectory_length=traj_len, state_filter=state_filtering, 
                                                                                num_agents = a,num_envs=num_envs, k=k, 
                                                                                env2agent=env2agent,chunk_length=chunk_size,log_entropy_interval=log_entropy)

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
                    f"cube {c_c:.3f}  cube_disp {c_rel:.4f} (1/400 = 0.0025 floor)")

    # Save the policy after training
    t_utils.save_settings(exp_folder, name_env, seed, num_epochs, hidden_sizes,
                            obs_dim, act_dim, a, total_trajs)
    if multihead:
        t_utils.save_policy(pl_policy, exp_folder,0,num_epochs,seed)
    else:
        for i in range(a):
            t_utils.save_policy(pl_policy[i], exp_folder,i,num_epochs,seed)
    writer.close()  # Close the TensorBoard writer after training is complete

    env.close()  # Close the environment after training is complete    
    return Path(exp_folder) / "models" / f"model_agent_0_step_{num_epochs}_seed_{seed}.pt"


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
    seed = [0] #[0,1,56,123,22,66,55,77,88,99]  # Seed for reproducibility
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
    randomize_object_pos = True
    automatic_budget = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    arm_cov   = GridCoverage([(0.0, 1.0), (0.0, 1.0)], [20, 20])
    cube_cov  = GridCoverage([(0.0, 1.0), (0.0, 1.0)], [20, 20])
    cube_disp = GridCoverage([(-1.0, 1.0), (-1.0, 1.0)], [20, 20])

    env = MaxEntEnvAdapter(num_envs=num_envs, height=64, width=64, max_episode_steps=traj_len, device=device)
    print("max_episode_steps:", env.venv.envs[0].spec.max_episode_steps)
    print("traj_len:", traj_len)
    discretizer = Discretizer([[0.0, 1.0], [0.0, 1.0]], [50, 50])

    for a in num_agents:
        for s in seed:
            for k in ks:
                t_utils.set_seed(s)  # Set the random seed for reproducibility
                writer,exp_folder = t_utils.init_writer(env_name=name_env, seed=s,MaxEnt=True, 
                                                        num_envs=a, hidden_size=hidden_sizes)  # Initialize TensorBoard writer
                
        
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

                
                count = 0
                num_trajectories = int(total_trajs / a)  # Number of trajectories per environment 

                if automatic_budget:
                    trajectory_budget = [1 for _ in range(a)]
                    trajectory_budget[0] = total_trajs - sum(trajectory_budget[1:])  # Adjust the first agent's budget to match total_trajs
                else:
                    trajectory_budget = [num_trajectories for _ in range(a)]  # Initial equal budget

                pos_rng = np.random.default_rng(s)

                for i in range(num_epochs):
                    start = time.time()
                    invalide_step = False
                    # Store policy parameters before update
                    #old_params = copy.deepcopy([p.clone().detach() for p in pl_policy.parameters()])
                    env2agent =torch.repeat_interleave(torch.arange(a, device=device), torch.tensor(trajectory_budget, device=device))  # [num_envs] maps each env to an agent index based on the allocated budget

                    #To randomize object pos before each epoch
                    xy = pos_rng.uniform([-0.07, -0.05], [0.07, 0.05]) 
                    if randomize_object_pos:
                        env.epoch_options = {"variation_values": {"object.pos": xy}}

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
                    if i % 10 == 0: #every 10th epoch
                        with torch.no_grad():
                            s_ev, _, _ = pl_agent.collect_particles(
                                env, pl_policy, 1, traj_len,
                                obs_dim, act_dim, a, num_envs, env2agent) #get a fresh rollout
                        flat = s_ev.reshape(-1, obs_dim).cpu().numpy() #flatten into a 2D array for coverage computation
                        a_c = arm_cov.coverage(flat[:, :2]) #columns 0 and 1 are the arm x and y
                        c_c = cube_cov.coverage(flat[:, 5:7]) #columns 5 and 6 are the cube x and y
                        per_env = [arm_cov.coverage(s_ev[0, :, e, :2].cpu().numpy())
                                   for e in range(num_envs)] #selects trajectory 0, all timesteps and env e
                        cube_traj = np.transpose(s_ev[0, :, :, 5:7].cpu().numpy(), (1, 0, 2))
                        c_rel = cube_disp.coverage_relative(cube_traj) #metric with no spawn contribution

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