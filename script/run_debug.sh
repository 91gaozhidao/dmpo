#!/bin/bash
# =============================================================================
# DMPO Comprehensive Debug Script
# =============================================================================
# This script runs all training and fine-tuning pipelines in debug mode with
# minimal resources (tiny batch sizes, 1-2 epochs/iterations, no wandb) to
# verify that the code paths execute correctly.
#
# Usage:
#   bash script/run_debug.sh [OPTIONS]
#
# Options:
#   --gpu           Use GPU (cuda:0) instead of CPU (default: cpu)
#   --timeout SEC   Timeout per command in seconds (default: 300)
#   --logdir DIR    Directory for debug logs (default: ./debug_logs)
#   --suite SUITE   Run only a specific suite: gym, robomimic, d3il, furniture
#   --stage STAGE   Run only a specific stage: pretrain, finetune
#   --dry-run       Print commands without executing
#   --help          Show this help message
#
# Example:
#   bash script/run_debug.sh --gpu --timeout 600
#   bash script/run_debug.sh --suite gym --stage pretrain
#   bash script/run_debug.sh --dry-run
# =============================================================================

set -uo pipefail

# ========================== Configuration ==========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults
DEVICE="cpu"
TIMEOUT=300
DEBUG_LOGDIR="${PROJECT_DIR}/debug_logs"
FILTER_SUITE=""
FILTER_STAGE=""
DRY_RUN=false

# ========================== Argument Parsing ==========================

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            DEVICE="cuda:0"
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --logdir)
            DEBUG_LOGDIR="$2"
            shift 2
            ;;
        --suite)
            FILTER_SUITE="$2"
            shift 2
            ;;
        --stage)
            FILTER_STAGE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            cat <<'HELPEOF'
DMPO Comprehensive Debug Script

Usage:
  bash script/run_debug.sh [OPTIONS]

Options:
  --gpu           Use GPU (cuda:0) instead of CPU (default: cpu)
  --timeout SEC   Timeout per command in seconds (default: 300)
  --logdir DIR    Directory for debug logs (default: ./debug_logs)
  --suite SUITE   Run only a specific suite: gym, robomimic, d3il, furniture
  --stage STAGE   Run only a specific stage: pretrain, finetune
  --dry-run       Print commands without executing
  --help          Show this help message

Example:
  bash script/run_debug.sh --gpu --timeout 600
  bash script/run_debug.sh --suite gym --stage pretrain
  bash script/run_debug.sh --dry-run
HELPEOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ========================== Color Output ==========================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ========================== Environment Setup ==========================

export REINFLOW_DIR="${REINFLOW_DIR:-${PROJECT_DIR}}"
export REINFLOW_DATA_DIR="${REINFLOW_DATA_DIR:-${PROJECT_DIR}/data}"
export REINFLOW_LOG_DIR="${REINFLOW_LOG_DIR:-${DEBUG_LOGDIR}/train_output}"
export REINFLOW_WANDB_ENTITY="${REINFLOW_WANDB_ENTITY:-debug}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export HYDRA_FULL_ERROR=1

echo -e "${CYAN}=============================================${NC}"
echo -e "${CYAN}     DMPO Comprehensive Debug Script${NC}"
echo -e "${CYAN}=============================================${NC}"
echo ""
echo -e "${BLUE}Project directory:${NC}  ${PROJECT_DIR}"
echo -e "${BLUE}Device:${NC}             ${DEVICE}"
echo -e "${BLUE}Timeout per run:${NC}    ${TIMEOUT}s"
echo -e "${BLUE}Debug log dir:${NC}      ${DEBUG_LOGDIR}"
echo -e "${BLUE}Suite filter:${NC}       ${FILTER_SUITE:-all}"
echo -e "${BLUE}Stage filter:${NC}       ${FILTER_STAGE:-all}"
echo -e "${BLUE}Dry run:${NC}            ${DRY_RUN}"
echo ""

mkdir -p "${DEBUG_LOGDIR}"

# ========================== Tracking ==========================

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
TOTAL_COUNT=0
declare -a RESULTS=()

# ========================== Helper Functions ==========================

run_debug_command() {
    # Arguments: <label> <config_dir> <config_name> <extra_overrides...>
    local label="$1"
    shift
    local config_dir="$1"
    shift
    local config_name="$1"
    shift
    local extra_overrides=("$@")

    TOTAL_COUNT=$((TOTAL_COUNT + 1))

    # Check if config exists
    local config_path="${PROJECT_DIR}/${config_dir}/${config_name}.yaml"
    if [[ ! -f "${config_path}" ]]; then
        echo -e "  ${YELLOW}[SKIP]${NC} ${label} — config not found: ${config_path}"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        RESULTS+=("SKIP|${label}|config not found")
        return 0
    fi

    local logfile="${DEBUG_LOGDIR}/${label//\//_}.log"

    # Build command
    local cmd=(
        python "${PROJECT_DIR}/script/run.py"
        "--config-dir=${PROJECT_DIR}/${config_dir}"
        "--config-name=${config_name}"
    )
    cmd+=("${extra_overrides[@]}")

    if [[ "${DRY_RUN}" == true ]]; then
        echo -e "  ${BLUE}[DRY]${NC}  ${label}"
        echo "         ${cmd[*]}"
        RESULTS+=("DRY|${label}|dry run")
        return 0
    fi

    echo -ne "  ${BLUE}[RUN]${NC}  ${label} ..."

    # Run with timeout, capture output
    local start_time
    start_time=$(date +%s)
    if timeout "${TIMEOUT}" "${cmd[@]}" > "${logfile}" 2>&1; then
        local elapsed=$(( $(date +%s) - start_time ))
        echo -e "\r  ${GREEN}[PASS]${NC} ${label} (${elapsed}s)"
        PASS_COUNT=$((PASS_COUNT + 1))
        RESULTS+=("PASS|${label}|${elapsed}s")
    else
        local exit_code=$?
        local elapsed=$(( $(date +%s) - start_time ))
        if [[ ${exit_code} -eq 124 ]]; then
            echo -e "\r  ${YELLOW}[TIME]${NC} ${label} (timeout ${TIMEOUT}s)"
            RESULTS+=("TIMEOUT|${label}|exceeded ${TIMEOUT}s")
            FAIL_COUNT=$((FAIL_COUNT + 1))
        else
            # Extract last few lines of error for quick diagnosis
            local error_hint
            error_hint=$(tail -5 "${logfile}" 2>/dev/null | head -3 | tr '\n' ' ')
            echo -e "\r  ${RED}[FAIL]${NC} ${label} (exit=${exit_code}, ${elapsed}s)"
            echo -e "         ${RED}Hint:${NC} ${error_hint}"
            echo -e "         ${RED}Log:${NC}  ${logfile}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            RESULTS+=("FAIL|${label}|exit=${exit_code}")
        fi
    fi
}

# Common debug overrides for pretrain (supervised learning on dataset)
# Returns overrides as newline-separated values (no spaces in values)
pretrain_overrides() {
    local device="$1"
    local -a overrides=(
        "++wandb=null"
        "++device=${device}"
        "++train.n_epochs=2"
        "++train.save_model_freq=1"
        "++train.lr_scheduler.first_cycle_steps=2"
        "++train.lr_scheduler.warmup_steps=1"
        "++test_in_mujoco=false"
        "++auto_resume=false"
        "++max_n_episodes=2"
    )
    printf '%s\n' "${overrides[@]}"
}

# Common debug overrides for finetune (RL with environment interaction)
# Returns overrides as newline-separated values (no spaces in values)
finetune_overrides() {
    local device="$1"
    local -a overrides=(
        "++wandb=null"
        "++device=${device}"
        "++sim_device=${device}"
        "++train.n_train_itr=2"
        "++train.n_steps=10"
        "++train.n_critic_warmup_itr=0"
        "++train.save_model_freq=1"
        "++train.val_freq=1"
        "++env.n_envs=2"
        "++env.save_video=false"
        "++base_policy_path=null"
    )
    printf '%s\n' "${overrides[@]}"
}

should_run() {
    local suite="$1"
    local stage="$2"
    if [[ -n "${FILTER_SUITE}" && "${FILTER_SUITE}" != "${suite}" ]]; then
        return 1
    fi
    if [[ -n "${FILTER_STAGE}" && "${FILTER_STAGE}" != "${stage}" ]]; then
        return 1
    fi
    return 0
}

# ========================== Test Definitions ==========================

echo -e "${CYAN}Starting debug runs...${NC}"
echo ""

# ------------------------------------------------------------------
# Section 1: Gym Pretrain
# ------------------------------------------------------------------
if should_run "gym" "pretrain"; then
    echo -e "${YELLOW}━━━ [1/8] Gym Pretrain ━━━${NC}"

    # 1.1 Diffusion MLP
    run_debug_command "gym/pretrain/diffusion_mlp" \
        "cfg/gym/pretrain/hopper-medium-v2" "pre_diffusion_mlp" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 1.2 MeanFlow MLP
    run_debug_command "gym/pretrain/meanflow_mlp" \
        "cfg/gym/pretrain/hopper-medium-v2" "pre_meanflow_mlp" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 1.3 MeanFlow Dispersive MLP
    run_debug_command "gym/pretrain/meanflow_dispersive_mlp" \
        "cfg/gym/pretrain/hopper-medium-v2" "pre_meanflow_dispersive_mlp" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 1.4 Improved MeanFlow MLP
    run_debug_command "gym/pretrain/improved_meanflow_mlp" \
        "cfg/gym/pretrain/hopper-medium-v2" "pre_improved_meanflow_mlp" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 1.5 Improved MeanFlow Dispersive MLP (core DMPO)
    run_debug_command "gym/pretrain/improved_meanflow_dispersive_mlp" \
        "cfg/gym/pretrain/hopper-medium-v2" "pre_improved_meanflow_dispersive_mlp" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 1.6 ReFlow MLP
    run_debug_command "gym/pretrain/reflow_mlp" \
        "cfg/gym/pretrain/hopper-medium-v2" "pre_reflow_mlp" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 1.7 Shortcut MLP
    run_debug_command "gym/pretrain/shortcut_mlp" \
        "cfg/gym/pretrain/hopper-medium-v2" "pre_shortcut_mlp" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 1.8 Gaussian MLP (kitchen task)
    run_debug_command "gym/pretrain/gaussian_mlp" \
        "cfg/gym/pretrain/kitchen-mixed-v0" "pre_gaussian_mlp" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 1.9 Test on another env: walker2d
    run_debug_command "gym/pretrain/walker2d_diffusion" \
        "cfg/gym/pretrain/walker2d-medium-v2" "pre_diffusion_mlp" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 1.10 Test on another env: ant
    run_debug_command "gym/pretrain/ant_meanflow" \
        "cfg/gym/pretrain/ant-medium-expert-v2" "pre_meanflow_mlp" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    echo ""
fi

# ------------------------------------------------------------------
# Section 2: Gym Finetune
# ------------------------------------------------------------------
if should_run "gym" "finetune"; then
    echo -e "${YELLOW}━━━ [2/8] Gym Finetune ━━━${NC}"

    # 2.1 PPO Diffusion MLP
    run_debug_command "gym/finetune/ppo_diffusion_mlp" \
        "cfg/gym/finetune/hopper-v2" "ft_ppo_diffusion_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 2.2 PPO DDIM MLP
    run_debug_command "gym/finetune/ppo_ddim_mlp" \
        "cfg/gym/finetune/hopper-v2" "ft_ppo_ddim_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 2.3 PPO MeanFlow MLP (core DMPO)
    run_debug_command "gym/finetune/ppo_meanflow_mlp" \
        "cfg/gym/finetune/hopper-v2" "ft_ppo_meanflow_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 2.4 PPO ReFlow MLP
    run_debug_command "gym/finetune/ppo_reflow_mlp" \
        "cfg/gym/finetune/hopper-v2" "ft_ppo_reflow_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 2.5 PPO Shortcut MLP
    run_debug_command "gym/finetune/ppo_shortcut_mlp" \
        "cfg/gym/finetune/hopper-v2" "ft_ppo_shortcut_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 2.6 FQL MLP
    run_debug_command "gym/finetune/fql_mlp" \
        "cfg/gym/finetune/hopper-v2" "ft_fql_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 2.7 PPO ReFlow Direct Likelihood
    run_debug_command "gym/finetune/ppo_reflow_direct_likelihood" \
        "cfg/gym/finetune/hopper-v2" "ft_ppo_reflow_direct_likelihood_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 2.8 PPO MeanFlow on kitchen
    run_debug_command "gym/finetune/kitchen_ppo_meanflow" \
        "cfg/gym/finetune/kitchen-mixed-v0" "ft_ppo_meanflow_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 2.9 PPO Diffusion on walker2d
    run_debug_command "gym/finetune/walker2d_ppo_diffusion" \
        "cfg/gym/finetune/walker2d-v2" "ft_ppo_diffusion_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    echo ""
fi

# ------------------------------------------------------------------
# Section 3: Robomimic Pretrain (state-only)
# ------------------------------------------------------------------
if should_run "robomimic" "pretrain"; then
    echo -e "${YELLOW}━━━ [3/8] Robomimic Pretrain (state) ━━━${NC}"

    # 3.1 Diffusion MLP
    run_debug_command "robomimic/pretrain/lift_diffusion_mlp" \
        "cfg/robomimic/pretrain/lift" "pre_diffusion_mlp" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 3.2 Gaussian MLP
    run_debug_command "robomimic/pretrain/lift_gaussian_mlp" \
        "cfg/robomimic/pretrain/lift" "pre_gaussian_mlp" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 3.3 GMM MLP
    run_debug_command "robomimic/pretrain/lift_gmm_mlp" \
        "cfg/robomimic/pretrain/lift" "pre_gmm_mlp" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 3.4 Diffusion UNet
    run_debug_command "robomimic/pretrain/lift_diffusion_unet" \
        "cfg/robomimic/pretrain/lift" "pre_diffusion_unet" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 3.5 Gaussian Transformer
    run_debug_command "robomimic/pretrain/lift_gaussian_transformer" \
        "cfg/robomimic/pretrain/lift" "pre_gaussian_transformer" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 3.6 GMM Transformer
    run_debug_command "robomimic/pretrain/lift_gmm_transformer" \
        "cfg/robomimic/pretrain/lift" "pre_gmm_transformer" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    echo ""
fi

# ------------------------------------------------------------------
# Section 4: Robomimic Pretrain (image-based)
# ------------------------------------------------------------------
if should_run "robomimic" "pretrain"; then
    echo -e "${YELLOW}━━━ [4/8] Robomimic Pretrain (image) ━━━${NC}"

    # 4.1 Diffusion MLP Image
    run_debug_command "robomimic/pretrain/lift_diffusion_mlp_img" \
        "cfg/robomimic/pretrain/lift" "pre_diffusion_mlp_img" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 4.2 Diffusion UNet Image
    run_debug_command "robomimic/pretrain/lift_diffusion_unet_img" \
        "cfg/robomimic/pretrain/lift" "pre_diffusion_unet_img" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 4.3 MeanFlow MLP Image
    run_debug_command "robomimic/pretrain/lift_meanflow_mlp_img" \
        "cfg/robomimic/pretrain/lift" "pre_meanflow_mlp_img" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 4.4 MeanFlow Dispersive MLP Image
    run_debug_command "robomimic/pretrain/lift_meanflow_dispersive_mlp_img" \
        "cfg/robomimic/pretrain/lift" "pre_meanflow_dispersive_mlp_img" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 4.5 ReFlow MLP Image
    run_debug_command "robomimic/pretrain/lift_reflow_mlp_img" \
        "cfg/robomimic/pretrain/lift" "pre_reflow_mlp_img" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 4.6 Shortcut MLP Image
    run_debug_command "robomimic/pretrain/lift_shortcut_mlp_img" \
        "cfg/robomimic/pretrain/lift" "pre_shortcut_mlp_img" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 4.7 Shortcut Dispersive MLP Image (InfoNCE-L2)
    run_debug_command "robomimic/pretrain/lift_shortcut_dispersive_mlp_img" \
        "cfg/robomimic/pretrain/lift" "pre_shortcut_dispersive_mlp_img" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 4.8 Shortcut Dispersive Cosine MLP Image
    run_debug_command "robomimic/pretrain/lift_shortcut_dispersive_cosine_mlp_img" \
        "cfg/robomimic/pretrain/lift" "pre_shortcut_dispersive_cosine_mlp_img" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 4.9 Shortcut Dispersive Hinge MLP Image
    run_debug_command "robomimic/pretrain/lift_shortcut_dispersive_hinge_mlp_img" \
        "cfg/robomimic/pretrain/lift" "pre_shortcut_dispersive_hinge_mlp_img" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 4.10 Shortcut Dispersive Covariance MLP Image
    run_debug_command "robomimic/pretrain/lift_shortcut_dispersive_covariance_mlp_img" \
        "cfg/robomimic/pretrain/lift" "pre_shortcut_dispersive_covariance_mlp_img" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 4.11 Consistency MLP Image
    run_debug_command "robomimic/pretrain/lift_consistency_mlp_img" \
        "cfg/robomimic/pretrain/lift" "pre_consistency_mlp_img" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 4.12 Gaussian MLP Image
    run_debug_command "robomimic/pretrain/lift_gaussian_mlp_img" \
        "cfg/robomimic/pretrain/lift" "pre_gaussian_mlp_img" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 4.13 Test on other tasks: can
    run_debug_command "robomimic/pretrain/can_meanflow_mlp_img" \
        "cfg/robomimic/pretrain/can" "pre_meanflow_mlp_img" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    echo ""
fi

# ------------------------------------------------------------------
# Section 5: Robomimic Finetune
# ------------------------------------------------------------------
if should_run "robomimic" "finetune"; then
    echo -e "${YELLOW}━━━ [5/8] Robomimic Finetune ━━━${NC}"

    # 5.1 PPO Diffusion MLP
    run_debug_command "robomimic/finetune/lift_ppo_diffusion_mlp" \
        "cfg/robomimic/finetune/lift" "ft_ppo_diffusion_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 5.2 PPO Diffusion MLP Image
    run_debug_command "robomimic/finetune/lift_ppo_diffusion_mlp_img" \
        "cfg/robomimic/finetune/lift" "ft_ppo_diffusion_mlp_img" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 5.3 PPO Diffusion UNet
    run_debug_command "robomimic/finetune/lift_ppo_diffusion_unet" \
        "cfg/robomimic/finetune/lift" "ft_ppo_diffusion_unet" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 5.4 PPO MeanFlow MLP Image (core DMPO)
    run_debug_command "robomimic/finetune/lift_ppo_meanflow_mlp_img" \
        "cfg/robomimic/finetune/lift" "ft_ppo_meanflow_mlp_img" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 5.5 PPO ReFlow MLP Image
    run_debug_command "robomimic/finetune/lift_ppo_reflow_mlp_img" \
        "cfg/robomimic/finetune/lift" "ft_ppo_reflow_mlp_img" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 5.6 PPO Shortcut MLP Image
    run_debug_command "robomimic/finetune/lift_ppo_shortcut_mlp_img" \
        "cfg/robomimic/finetune/lift" "ft_ppo_shortcut_mlp_img" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 5.7 PPO Gaussian MLP
    run_debug_command "robomimic/finetune/lift_ppo_gaussian_mlp" \
        "cfg/robomimic/finetune/lift" "ft_ppo_gaussian_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 5.8 PPO Gaussian MLP Image
    run_debug_command "robomimic/finetune/lift_ppo_gaussian_mlp_img" \
        "cfg/robomimic/finetune/lift" "ft_ppo_gaussian_mlp_img" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 5.9 PPO GMM MLP
    run_debug_command "robomimic/finetune/lift_ppo_gmm_mlp" \
        "cfg/robomimic/finetune/lift" "ft_ppo_gmm_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 5.10 AWR Diffusion MLP
    run_debug_command "robomimic/finetune/lift_awr_diffusion_mlp" \
        "cfg/robomimic/finetune/lift" "ft_awr_diffusion_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 5.11 DQL Diffusion MLP
    run_debug_command "robomimic/finetune/lift_dql_diffusion_mlp" \
        "cfg/robomimic/finetune/lift" "ft_dql_diffusion_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 5.12 IDQL Diffusion MLP
    run_debug_command "robomimic/finetune/lift_idql_diffusion_mlp" \
        "cfg/robomimic/finetune/lift" "ft_idql_diffusion_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 5.13 DIPO Diffusion MLP
    run_debug_command "robomimic/finetune/lift_dipo_diffusion_mlp" \
        "cfg/robomimic/finetune/lift" "ft_dipo_diffusion_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 5.14 QSM Diffusion MLP
    run_debug_command "robomimic/finetune/lift_qsm_diffusion_mlp" \
        "cfg/robomimic/finetune/lift" "ft_qsm_diffusion_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 5.15 RWR Diffusion MLP
    run_debug_command "robomimic/finetune/lift_rwr_diffusion_mlp" \
        "cfg/robomimic/finetune/lift" "ft_rwr_diffusion_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    echo ""
fi

# ------------------------------------------------------------------
# Section 6: D3IL Pretrain
# ------------------------------------------------------------------
if should_run "d3il" "pretrain"; then
    echo -e "${YELLOW}━━━ [6/8] D3IL Pretrain ━━━${NC}"

    # 6.1 Diffusion MLP
    run_debug_command "d3il/pretrain/avoid_m1_diffusion_mlp" \
        "cfg/d3il/pretrain/avoid_m1" "pre_diffusion_mlp" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 6.2 Gaussian MLP
    run_debug_command "d3il/pretrain/avoid_m1_gaussian_mlp" \
        "cfg/d3il/pretrain/avoid_m1" "pre_gaussian_mlp" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 6.3 GMM MLP
    run_debug_command "d3il/pretrain/avoid_m1_gmm_mlp" \
        "cfg/d3il/pretrain/avoid_m1" "pre_gmm_mlp" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    echo ""
fi

# ------------------------------------------------------------------
# Section 7: D3IL Finetune
# ------------------------------------------------------------------
if should_run "d3il" "finetune"; then
    echo -e "${YELLOW}━━━ [7/8] D3IL Finetune ━━━${NC}"

    # 7.1 PPO Diffusion MLP
    run_debug_command "d3il/finetune/avoid_m1_ppo_diffusion_mlp" \
        "cfg/d3il/finetune/avoid_m1" "ft_ppo_diffusion_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 7.2 PPO Gaussian MLP
    run_debug_command "d3il/finetune/avoid_m1_ppo_gaussian_mlp" \
        "cfg/d3il/finetune/avoid_m1" "ft_ppo_gaussian_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 7.3 PPO GMM MLP
    run_debug_command "d3il/finetune/avoid_m1_ppo_gmm_mlp" \
        "cfg/d3il/finetune/avoid_m1" "ft_ppo_gmm_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    echo ""
fi

# ------------------------------------------------------------------
# Section 8: Furniture Pretrain & Finetune
# ------------------------------------------------------------------
if should_run "furniture" "pretrain"; then
    echo -e "${YELLOW}━━━ [8a/8] Furniture Pretrain ━━━${NC}"

    # 8.1 Diffusion MLP
    run_debug_command "furniture/pretrain/lamp_low_diffusion_mlp" \
        "cfg/furniture/pretrain/lamp_low" "pre_diffusion_mlp" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 8.2 Diffusion UNet
    run_debug_command "furniture/pretrain/lamp_low_diffusion_unet" \
        "cfg/furniture/pretrain/lamp_low" "pre_diffusion_unet" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    # 8.3 Gaussian MLP
    run_debug_command "furniture/pretrain/lamp_low_gaussian_mlp" \
        "cfg/furniture/pretrain/lamp_low" "pre_gaussian_mlp" \
        $(pretrain_overrides "${DEVICE}") \
        "++batch_size=4"

    echo ""
fi

if should_run "furniture" "finetune"; then
    echo -e "${YELLOW}━━━ [8b/8] Furniture Finetune ━━━${NC}"

    # 8.4 PPO Diffusion MLP
    run_debug_command "furniture/finetune/lamp_low_ppo_diffusion_mlp" \
        "cfg/furniture/finetune/lamp_low" "ft_ppo_diffusion_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    # 8.5 PPO Gaussian MLP
    run_debug_command "furniture/finetune/lamp_low_ppo_gaussian_mlp" \
        "cfg/furniture/finetune/lamp_low" "ft_ppo_gaussian_mlp" \
        $(finetune_overrides "${DEVICE}") \
        "++train.batch_size=20"

    echo ""
fi

# ========================== Summary ==========================

echo ""
echo -e "${CYAN}=============================================${NC}"
echo -e "${CYAN}              DEBUG SUMMARY${NC}"
echo -e "${CYAN}=============================================${NC}"
echo ""
echo -e "  Total:    ${TOTAL_COUNT}"
echo -e "  ${GREEN}Passed:${NC}   ${PASS_COUNT}"
echo -e "  ${RED}Failed:${NC}   ${FAIL_COUNT}"
echo -e "  ${YELLOW}Skipped:${NC}  ${SKIP_COUNT}"
echo ""

# Detailed results table
echo -e "${CYAN}─── Detailed Results ───${NC}"
printf "  %-8s  %s\n" "STATUS" "TEST"
echo "  ------   ----"
for result in "${RESULTS[@]}"; do
    IFS='|' read -r status label detail <<< "${result}"
    case "${status}" in
        PASS)    color="${GREEN}" ;;
        FAIL)    color="${RED}" ;;
        SKIP)    color="${YELLOW}" ;;
        TIMEOUT) color="${YELLOW}" ;;
        DRY)     color="${BLUE}" ;;
        *)       color="${NC}" ;;
    esac
    printf "  ${color}%-8s${NC}  %-55s  %s\n" "[${status}]" "${label}" "${detail}"
done

echo ""
echo -e "${CYAN}─── Log Files ───${NC}"
echo -e "  All logs saved to: ${DEBUG_LOGDIR}/"
echo ""

# Save summary to file
SUMMARY_FILE="${DEBUG_LOGDIR}/debug_summary.txt"
{
    echo "DMPO Debug Summary - $(date)"
    echo "================================"
    echo "Total: ${TOTAL_COUNT}, Passed: ${PASS_COUNT}, Failed: ${FAIL_COUNT}, Skipped: ${SKIP_COUNT}"
    echo ""
    for result in "${RESULTS[@]}"; do
        IFS='|' read -r status label detail <<< "${result}"
        printf "%-8s  %-55s  %s\n" "[${status}]" "${label}" "${detail}"
    done
} > "${SUMMARY_FILE}"
echo -e "  Summary saved to: ${SUMMARY_FILE}"
echo ""

# Exit code
if [[ ${FAIL_COUNT} -gt 0 ]]; then
    echo -e "${RED}Some tests failed. Check logs for details.${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
