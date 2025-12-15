#!/bin/bash
#SBATCH --job-name=boonchu-multi-sim
#SBATCH --output=logs/boonchu_multi_%j.out
#SBATCH --error=logs/boonchu_multi_%j.err
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH -p gpu
#SBATCH --mem=128G
#SBATCH -A lt200291

# ---------------------------
# Load environment
# ---------------------------
module load cray-python/3.10.10
source ~/mesa_env/bin/activate

# ---------------------------
# EnergyPlus paths
# ---------------------------
EP_HOME=./EnergyPlus-25.1.0-1c11a3d85f-Linux-CentOS7.9.2009-x86_64
export PATH=$EP_HOME:$PATH
export PYTHONPATH=$PYTHONPATH:$EP_HOME
export ENERGYPLUS_EXE=$EP_HOME/energyplus

# ---------------------------
# Project folder
# ---------------------------
PROJECT_DIR=$(pwd)
cd $PROJECT_DIR

# Create logs directory if not exists
mkdir -p logs

# ---------------------------
# Run Multi-Scenario/Policy Simulation
# ---------------------------
echo "=========================================="
echo "Starting Multi-Scenario/Policy Simulation"
echo "=========================================="
echo "Time: $(date)"
echo ""

# Run the main.py which will handle all scenario x policy combinations
torchrun --standalone --nproc_per_node=2 main.py

echo ""
echo "=========================================="
echo "Simulation Completed"
echo "=========================================="
echo "Time: $(date)"
echo ""
echo "Check results in: simulation_results/"