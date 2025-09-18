#!/bin/bash

# Script to run the S-N-VAE dSprites notebook with different seeds using papermill
# This bash script will execute the notebook for seeds 0-10 and save the output notebooks
# with the seed_{num} suffix.

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_NOTEBOOK="$SCRIPT_DIR/s_n_beta_vae (dsprites).ipynb"
OUTPUT_DIR="$SCRIPT_DIR/outputs"
KERNEL_NAME="main"
SEEDS=(0 1 2 3 4 5 6 7 8 9 10)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if kernel exists
check_kernel_exists() {
    local kernel_name=$1
    if command_exists jupyter; then
        jupyter kernelspec list | grep -q "$kernel_name"
    else
        print_status $YELLOW "Warning: jupyter command not found, cannot verify kernel"
        return 1
    fi
}

echo "S-N-VAE dSprites Notebook Runner with Papermill (Bash Version)"
echo "=============================================================="
echo "Input notebook: $INPUT_NOTEBOOK"
echo "Output directory: $OUTPUT_DIR"
echo "Kernel: $KERNEL_NAME"
echo "Seeds: ${SEEDS[*]}"
echo

# Check if input notebook exists
if [[ ! -f "$INPUT_NOTEBOOK" ]]; then
    print_status $RED "Error: Input notebook not found: $INPUT_NOTEBOOK"
    print_status $RED "Please make sure the notebook file exists in the same directory as this script."
    exit 1
fi

# Check if papermill is installed
if ! command_exists papermill; then
    print_status $RED "Error: papermill is not installed."
    print_status $YELLOW "Please install it with: pip install papermill"
    exit 1
fi

# Check if jupyter is installed
if ! command_exists jupyter; then
    print_status $YELLOW "Warning: jupyter command not found"
else
    print_status $BLUE "Available Jupyter kernels:"
    jupyter kernelspec list
    echo
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
print_status $GREEN "Created output directory: $OUTPUT_DIR"

# Initialize counters
successful_runs=()
failed_runs=()

print_status $BLUE "Starting notebook execution..."
echo "------------------------------------------------"

# Run notebook for each seed
for seed in "${SEEDS[@]}"; do
    output_filename="s_n_beta_vae (dsprites)_seed_${seed}.ipynb"
    output_path="$OUTPUT_DIR/$output_filename"
    
    print_status $BLUE "Running with seed $seed..."
    print_status $BLUE "Output: $output_filename"
    
    if papermill \
        "$INPUT_NOTEBOOK" \
        "$output_path" \
        -p seed "$seed" \
        -k "$KERNEL_NAME" \
        --progress-bar; then
        
        successful_runs+=($seed)
        print_status $GREEN "✓ Successfully completed seed $seed"
    else
        failed_runs+=($seed)
        print_status $RED "✗ Failed to run seed $seed"
    fi
    
    echo "------------------------------"
done

# Summary
echo
echo "=================================================="
echo "EXECUTION SUMMARY"
echo "=================================================="
echo "Total runs attempted: ${#SEEDS[@]}"
echo "Successful runs: ${#successful_runs[@]} - [${successful_runs[*]}]"
echo "Failed runs: ${#failed_runs[@]} - [${failed_runs[*]}]"

if [[ ${#failed_runs[@]} -eq 0 ]]; then
    print_status $GREEN "🎉 All runs completed successfully!"
    exit 0
else
    print_status $YELLOW "⚠️  Some runs failed. Check the error messages above."
    exit 1
fi