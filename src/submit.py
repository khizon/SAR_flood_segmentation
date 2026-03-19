import os
import itertools
from tqdm import tqdm
import textwrap
import subprocess

if __name__ == '__main__':
    sweep = {
        'model': ['linknet'],
        'backbone': ['timm-mobilenetv3_small_100',
                     'timm-mobilenetv3_large_100',
                    #  'timm-mobilenetv3_small_minimal_100',
                     'timm-mobilenetv3_large_minimal_100'],
        'lr': [0.0001,
            #    0.0005,
            #    0.00008
               ],
        'label_type': ['HandLabeled', 'WeaklyLabeled'],
        'transforms': ['flip rotate distort', 'flip rotate elastic griddistort'],
        'expand': [1, 16],
        'debug': ['',
                #   ' --no-debug'
                  ]
    }

    # Ensure output directory exists
    os.makedirs("slurm_scripts", exist_ok=True)

    # Generate all combinations
    keys = list(sweep.keys())
    combinations = itertools.product(*sweep.values())
    for combo in tqdm(combinations, desc='Generating sweep:', unit='runs'):
        combo_dict = dict(zip(keys, combo))

        # Skip combinations
        if (combo_dict['label_type']=='WeaklyLabeled') & (combo_dict['expand']==16):
            continue

        var_script = (
            f"srun python src/train.py{combo_dict['debug']} "
            f"--model {combo_dict['model']} "
            f"--pre_trained imagenet "
            f"--backbone {combo_dict['backbone']} "
            f"--target Flood "
            f"--transforms \"{combo_dict['transforms']}\" "
            f"--label_type {combo_dict['label_type']} "
            f"--lr {combo_dict['lr']} "
            f"--expand {combo_dict['expand']}"
        )

        header = f'''
            #!/bin/bash
            #SBATCH --job-name=train_{combo_dict['backbone']}
            #SBATCH --output=slurm_scripts/%j_{combo_dict['backbone']}.out
            #SBATCH --error=slurm_scripts/%j_{combo_dict['backbone']}.err
            #SBATCH --partition=uperdfi-gpu
            #SBATCH --gres=gpu:1
            #SBATCH --nodes=1
            #SBATCH --requeue

            module load conda
            eval "$(conda shell.bash hook)"
            conda activate sar-env-1

            echo $CONDA_DEFAULT_ENV
            echo $CONDA_PREFIX

            echo "SLURM_JOBID="$SLURM_JOBID
            echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
            echo "SLURM_NNODES"=$SLURM_NNODES
            echo "Working directory = "$SLURM_SUBMIT_DIR

            nvidia-smi
        '''
        
        filename = f"slurm_scripts/train.slurm"
        with open(filename, "w") as f:
            slurm_content = f"{textwrap.dedent(header).strip()}\n{var_script}\n"
            f.write(slurm_content)
        
        # Submit the job to SLURM
        result = subprocess.run(["sbatch", filename], capture_output=True, text=True)

        # Print the scheduler’s response (e.g., "Submitted batch job 12345")
        tqdm.write(result.stdout)
        tqdm.write(result.stderr)