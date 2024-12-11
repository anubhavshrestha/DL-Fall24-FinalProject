# Baseline runs with schedulers
WANDB_NAME="vicreg_baseline_cosine" python train.py --wandb_project JEPA --loss_type vicreg --scheduler cosine --wandb_name "vicreg_baseline_cosine"
WANDB_NAME="vicreg_baseline_onecycle" python train.py --wandb_project JEPA --loss_type vicreg --scheduler onecycle --wandb_name "vicreg_baseline_onecycle"

WANDB_NAME="barlow_baseline_cosine" python train.py --wandb_project JEPA --loss_type barlow --scheduler cosine --wandb_name "barlow_baseline_cosine"
WANDB_NAME="barlow_baseline_onecycle" python train.py --wandb_project JEPA --loss_type barlow --scheduler onecycle --wandb_name "barlow_baseline_onecycle"

# Modified VICReg parameters with different schedulers
WANDB_NAME="vicreg_high_params_cosine" python train.py --wandb_project JEPA --loss_type vicreg --lambda_param 30.0 --mu_param 20.0 --nu_param 2.0 --scheduler cosine --wandb_name "vicreg_high_params_cosine"
WANDB_NAME="vicreg_high_params_onecycle" python train.py --wandb_project JEPA --loss_type vicreg --lambda_param 30.0 --mu_param 20.0 --nu_param 2.0 --scheduler onecycle --wandb_name "vicreg_high_params_onecycle"

# Barlow Twins with modified lambda
WANDB_NAME="barlow_high_lambda_cosine" python train.py --wandb_project JEPA --loss_type barlow --lambda_param 0.01 --scheduler cosine --wandb_name "barlow_high_lambda_cosine"
WANDB_NAME="barlow_high_lambda_onecycle" python train.py --wandb_project JEPA --loss_type barlow --lambda_param 0.01 --scheduler onecycle --wandb_name "barlow_high_lambda_onecycle"

# Learning rate experiments with schedulers
WANDB_NAME="vicreg_high_lr_cosine" python train.py --wandb_project JEPA --loss_type vicreg --learning_rate 1e-4 --prober_lr 5e-3 --scheduler cosine --wandb_name "vicreg_high_lr_cosine"
WANDB_NAME="vicreg_high_lr_onecycle" python train.py --wandb_project JEPA --loss_type vicreg --learning_rate 1e-4 --prober_lr 5e-3 --scheduler onecycle --wandb_name "vicreg_high_lr_onecycle"

# Batch size experiments with schedulers
WANDB_NAME="vicreg_large_batch_cosine" python train.py --wandb_project JEPA --loss_type vicreg --batch_size 256 --learning_rate 6e-5 --scheduler cosine --wandb_name "vicreg_large_batch_cosine"
WANDB_NAME="vicreg_large_batch_onecycle" python train.py --wandb_project JEPA --loss_type vicreg --batch_size 256 --learning_rate 6e-5 --scheduler onecycle --wandb_name "vicreg_large_batch_onecycle"

# Longer training with schedulers
WANDB_NAME="vicreg_long_training_cosine" python train.py --wandb_project JEPA --loss_type vicreg --num_epochs 200 --save_frequency 5 --scheduler cosine --wandb_name "vicreg_long_training_cosine"
WANDB_NAME="vicreg_long_training_onecycle" python train.py --wandb_project JEPA --loss_type vicreg --num_epochs 200 --save_frequency 5 --scheduler onecycle --wandb_name "vicreg_long_training_onecycle"

# Weight decay experiments with schedulers
WANDB_NAME="vicreg_high_wd_cosine" python train.py --wandb_project JEPA --loss_type vicreg --weight_decay 1e-4 --lambda_param 35.0 --scheduler cosine --wandb_name "vicreg_high_wd_cosine"
WANDB_NAME="vicreg_high_wd_onecycle" python train.py --wandb_project JEPA --loss_type vicreg --weight_decay 1e-4 --lambda_param 35.0 --scheduler onecycle --wandb_name "vicreg_high_wd_onecycle"

# Small batch experiments with schedulers
WANDB_NAME="vicreg_small_batch_cosine" python train.py --wandb_project JEPA --loss_type vicreg --batch_size 64 --learning_rate 5e-5 --prober_lr 2e-3 --scheduler cosine --wandb_name "vicreg_small_batch_cosine"
WANDB_NAME="vicreg_small_batch_onecycle" python train.py --wandb_project JEPA --loss_type vicreg --batch_size 64 --learning_rate 5e-5 --prober_lr 2e-3 --scheduler onecycle --wandb_name "vicreg_small_batch_onecycle"

# Full custom configuration with schedulers
WANDB_NAME="barlow_full_custom_cosine" python train.py --wandb_project JEPA --loss_type barlow --learning_rate 2e-4 --weight_decay 5e-5 --batch_size 192 --num_epochs 150 --prober_lr 1e-3 --lambda_param 0.008 --save_frequency 2 --scheduler cosine --wandb_name "barlow_full_custom_cosine"
WANDB_NAME="barlow_full_custom_onecycle" python train.py --wandb_project JEPA --loss_type barlow --learning_rate 2e-4 --weight_decay 5e-5 --batch_size 192 --num_epochs 150 --prober_lr 1e-3 --lambda_param 0.008 --save_frequency 2 --scheduler onecycle --wandb_name "barlow_full_custom_onecycle"
