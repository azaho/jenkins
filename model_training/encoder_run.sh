#!/bin/bash
#SBATCH --job-name=encoder_train          # Name of the job
#SBATCH --ntasks=1             
#SBATCH --cpus-per-task=4    
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G
#SBATCH --constraint=high-capacity
#SBATCH -t 12:00:00         # total run time limit (HH:MM:SS) 
#SBATCH --array=1-24      
#SBATCH --output r/%A_%a.out # STDOUT
#SBATCH --error r/%A_%a.err # STDERR
#SBATCH -p use-everything

source .venv/bin/activate
# Create arrays for each parameter
d_models=(256 512 1024)
latent_dims=(-1) 
model_types=(transformer)
learning_rates=(0.0005)
weight_decays=(0 0.0001)
n_sessions=(1 2)
optimizer_names=('AdamW' 'Muon')

# Get parameters for this array job
idx=$SLURM_ARRAY_TASK_ID-1

# Calculate indices for each parameter
d_model_idx=$((idx % ${#d_models[@]}))
idx=$((idx / ${#d_models[@]}))
latent_dim_idx=$((idx % ${#latent_dims[@]}))
idx=$((idx / ${#latent_dims[@]}))
model_type_idx=$((idx % ${#model_types[@]}))
idx=$((idx / ${#model_types[@]}))
lr_idx=$((idx % ${#learning_rates[@]}))
idx=$((idx / ${#learning_rates[@]}))
wd_idx=$((idx % ${#weight_decays[@]}))
idx=$((idx / ${#weight_decays[@]}))
optimizer_name_idx=$((idx % ${#optimizer_names[@]}))
idx=$((idx / ${#optimizer_names[@]}))
n_sessions_idx=$((idx % ${#n_sessions[@]}))
idx=$((idx / ${#n_sessions[@]}))

# Get parameter values
d_model=${d_models[$d_model_idx]}
latent_dim=${latent_dims[$latent_dim_idx]}
model_type=${model_types[$model_type_idx]}
lr=${learning_rates[$lr_idx]}
weight_decay=${weight_decays[$wd_idx]}
optimizer_name=${optimizer_names[$optimizer_name_idx]}
n_sessions=${n_sessions[$n_sessions_idx]}

echo "d_model: $d_model"
echo "latent_dim: $latent_dim"
echo "model_type: $model_type"
echo "lr: $lr"
echo "weight_decay: $weight_decay"
echo "optimizer_name: $optimizer_name"
echo "n_sessions: $n_sessions"
echo ""

echo "nvidia-smi"
nvidia-smi
echo ""

# Run training with selected parameters
python -u encoder_train.py --d_model $d_model --latent_dim $latent_dim --model_type $model_type --lr $lr --weight_decay $weight_decay --optimizer $optimizer_name --n_sessions $n_sessions