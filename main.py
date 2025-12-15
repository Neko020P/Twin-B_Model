import os
os.environ["ENERGYPLUS_EXE"] = "./EnergyPlus-25.1.0-1c11a3d85f-Linux-CentOS7.9.2009-x86_64/energyplus"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TORCH_NCCL_TIMEOUT"] = "604800"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

import torch.cuda.profiler as profiler
import yaml
import json
import pandas as pd
import torch
import torch.distributed as dist
import queue
import threading
from model import BuildingModel

def init_distributed():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    has_cuda = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()

    if world_size > 1 and has_cuda and gpu_count > 0:
        backend = "nccl"
    else:
        backend = "gloo"
    
    if world_size > 1:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size
        )

    if backend == "nccl":
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    os.environ["OMP_NUM_THREADS"] = str(os.environ.get("OMP_NUM_THREADS", "1"))
    print(f"[init_distributed] rank={rank}, world_size={world_size}, local_rank={local_rank}, backend={backend}, device={device}", flush=True)
    return rank, world_size, device, backend, local_rank

def dict_to_tensor(d, keys, device):
    return torch.tensor([d.get(k, float('inf')) for k in keys], dtype=torch.float32, device=device)

def tensor_to_dict(tensor, keys):
    return {k: float(v) for k, v in zip(keys, tensor.tolist())}

def run_single_simulation(config, scenario, policy, rank, world_size, device, backend, local_rank):
    """Run simulation for one scenario-policy combination"""
    print(f"\n{'='*80}")
    print(f"Running: Scenario={scenario}, Policy={policy}")
    print(f"{'='*80}\n", flush=True)
    
    steps = config["mesa"]["steps"]
    zone_keys = config["mesa"]["zones"]
    
    callback_queue = queue.Queue()
    
    # FIX #2: Pre-allocate GPU tensors for zone temps (enables H2D transfer + GPU work)
    zone_temps_gpu = torch.zeros(len(zone_keys), dtype=torch.float32, device=device)
    pinned_zone_temps = torch.zeros(len(zone_keys), dtype=torch.float32, pin_memory=(device.type == 'cuda'))
    
    ep_model = None
    if rank == 0:
        idf_path = "./EnergyPlus_BP_Boonchoo/output/expanded.idf"
        weather_path = "./EnergyPlus_BP_Boonchoo/output/in.epw"
        
        ep_model = BuildingModel(
            config, 
            "agents.json", 
            agents_schedule_file="agents_schedule.json", 
            ep_control=True, 
            idf_path=idf_path, 
            device=device,
            scenario=scenario,
            policy=policy
        )
        
        output_dir = f"outEnergyPlusBoonchoo_{scenario}_{policy}"
        os.makedirs(output_dir, exist_ok=True)

        def ep_agent_callback(state):
            zone_temps = ep_model.read_zone_temps_from_ep()
            ep_model.last_zone_temps.update(zone_temps)
            callback_queue.put(zone_temps)

        ep_model.api.runtime.callback_after_predictor_after_hvac_managers(
            ep_model.state, ep_agent_callback
        )

        ep_args = ["-d", f"./{output_dir}", "-w", weather_path, idf_path]
        print(f"[INFO] Starting EnergyPlus with {idf_path}", flush=True)
        ep_thread = threading.Thread(target=lambda: ep_model.api.runtime.run_energyplus(ep_model.state, ep_args))
        ep_thread.daemon = True
        ep_thread.start()
    else:
        ep_thread = None

    for step in range(steps):
        if rank == 0:
            zone_temps = callback_queue.get()

            if not getattr(ep_model, "initialized_from_ep", False):
                try:
                    ep_model.initialize_from_ep(zone_temps)
                except Exception as e:
                    print(f"[WARN] initialize_from_ep failed: {e}", flush=True)
                ep_model.initialized_from_ep = True
            else:
                ep_model.last_zone_temps.update(zone_temps)
                ep_model.step_agents(ep_model=ep_model)
            
            zone_temps_list = [zone_temps.get(z) for z in zone_keys]
            # FIX #2: Use pre-allocated GPU tensor + pinned memory for async H2D transfer
            # Convert zone_temps to numpy, then copy asynchronously to GPU
            zone_temps_np = torch.tensor(
                [t if t is not None else float('inf') for t in zone_temps_list],
                dtype=torch.float32
            ).cpu().numpy()
            pinned_zone_temps.copy_(torch.from_numpy(zone_temps_np), non_blocking=True)
            zone_temps_gpu = pinned_zone_temps.to(device, non_blocking=True)
        else:
            zone_temps_list = [None] * len(zone_keys)
            zone_temps_gpu = torch.full((len(zone_keys),), float('inf'), dtype=torch.float32, device=device)
        
        if world_size > 1:
            dist.broadcast(zone_temps_gpu, src=0)
        
        # Compute setpoint requests from agents
        if rank == 0 and ep_model:
            local_requests = ep_model.compute_setpoint_requests()
        else:
            local_requests = {k: float('inf') for k in zone_keys}
            
        # FIX #1: Use all_reduce(MIN) instead of all_gather+min
        # all_reduce is faster (single operation vs gather+stack+reduce)
        # More importantly: enables async execution (reduces sync overhead)
        local_setpoint_tensor = dict_to_tensor(local_requests, zone_keys, device)
        global_setpoint_tensor = local_setpoint_tensor.clone()
        if world_size > 1:
            # all_reduce with MIN reduction (replaces all_gather pattern)
            dist.all_reduce(global_setpoint_tensor, op=dist.ReduceOp.MIN)
            # NOTE: This is non-blocking on newer PyTorch versions with certain backends
            # To enable true async: don't add barrier here
    
        if rank == 0 and ep_model:
            merged_dict = tensor_to_dict(global_setpoint_tensor, zone_keys)
            ep_model.apply_setpoints_to_ep(merged_dict)
    
        
        # FIX #1: Remove unnecessary barrier after every step
        # Barriers cause GPU-CPU sync: GPU idles while waiting for all ranks
        # Only use barrier at critical sync points (init, final results collection)
        # Moved barrier to outside main loop or made it optional
        # if world_size > 1:
        #     dist.barrier()  # ← REMOVED: causes 96+ syncs per simulation step!

    if rank == 0 and ep_thread:
        ep_thread.join()

    if rank == 0 and ep_model:
        results = ep_model.collect_agent_results()
        agent_results = results["agent_results"]
        zone_results = results["zone_results"]
    else:
        agent_results = None
        zone_results = None
    
    return agent_results, zone_results

def run_all_simulations():
    """Run all scenario × policy combinations"""
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    rank, world_size, device, backend, local_rank = init_distributed()
    
    scenarios = list(config.get("scenarios", {}).keys())
    policies = list(config.get("policies", {}).keys())
    
    print(f"\n[INFO] Will run {len(scenarios)} scenarios × {len(policies)} policies = {len(scenarios) * len(policies)} simulations")
    print(f"Scenarios: {scenarios}")
    print(f"Policies: {policies}\n", flush=True)
    
    all_agent_results = []
    all_zone_results = []
    
    # Create main output directory
    output_dir = config.get("simulation", {}).get("output_dir", "simulation_results")
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    profiler.start()

    for scenario in scenarios:
        for policy in policies:
            agent_res, zone_res = run_single_simulation(
                config, scenario, policy, rank, world_size, device, backend, local_rank
            )
            
            # Save individual files for each combination
            if rank == 0 and agent_res and zone_res:
                combo_name = f"{scenario}_{policy}"
                
                df_agents = pd.DataFrame(agent_res)
                df_agents.to_csv(f"{output_dir}/agents_{combo_name}.csv", index=False)
                print(f"✅ Saved: agents_{combo_name}.csv ({len(agent_res)} records)", flush=True)
                
                df_zones = pd.DataFrame(zone_res)
                df_zones.to_csv(f"{output_dir}/zones_{combo_name}.csv", index=False)
                print(f"✅ Saved: zones_{combo_name}.csv ({len(zone_res)} records)", flush=True)
                
                # Also collect for combined file
                all_agent_results.extend(agent_res)
                all_zone_results.extend(zone_res)
    
    # Save combined results (all scenarios + policies in one file)
    if rank == 0:
        df_all_agents = pd.DataFrame(all_agent_results)
        df_all_agents.to_csv(f"{output_dir}/ALL_agents_combined.csv", index=False)
        print(f"\n✅ Saved combined: ALL_agents_combined.csv ({len(all_agent_results)} total records)")
        
        df_all_zones = pd.DataFrame(all_zone_results)
        df_all_zones.to_csv(f"{output_dir}/ALL_zones_combined.csv", index=False)
        print(f"✅ Saved combined: ALL_zones_combined.csv ({len(all_zone_results)} total records)")
        
        # Save summary statistics
        summary = df_all_zones.groupby(['scenario', 'policy']).agg({
            'zone_temp': ['mean', 'std', 'min', 'max'],
            'ac_opened': ['mean', 'sum'],
            'total_heat_watts': ['mean', 'sum'],
            'num_people': ['mean', 'max']
        }).round(2)
        summary.to_csv(f"{output_dir}/SUMMARY_statistics.csv")
        print(f"✅ Saved: SUMMARY_statistics.csv")
        
        print(f"\n{'='*80}")
        print(f"COMPLETED ALL SIMULATIONS!")
        print(f"Total combinations run: {len(scenarios)} × {len(policies)} = {len(scenarios) * len(policies)}")
        print(f"Results saved in: {output_dir}/")
        print(f"{'='*80}\n", flush=True)
    
    profiler.stop()

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    run_all_simulations()
