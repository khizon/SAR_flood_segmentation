import wandb
from tqdm import tqdm
import os
import shutil

def cleanup_artifacts_per_run(project_name, entity, dry_run=True):
    api = wandb.Api()
    
    # Get all runs in the project
    runs = api.runs(f"{entity}/{project_name}")
    
    print("Processing runs...")
    for run in tqdm(runs):
        if run.state != "finished":
            print(f"Skipping unfinished run: {run.name}")
            continue
        
        print(f"\nProcessing run: {run.name}")
        
        # Get all model artifacts for this run
        run_artifacts = run.logged_artifacts()
        model_artifacts = [art for art in run_artifacts if art.type == 'model']
        
        if not model_artifacts:
            print(f"No model artifacts found for run: {run.name}")
            continue
        
        # Find the model artifact tagged as "best"
        best_artifact = None
        
        for artifact in model_artifacts:
            if "best" in artifact.aliases:
                best_artifact = artifact
                break
        
        if best_artifact is None:
            print(f"No artifacts found with the 'best' tag for run: {run.name}")
            continue
        
        print(f"Best artifact for run {run.name}: {best_artifact.name}")
        
        # Delete all other model artifacts for this run
        for artifact in model_artifacts:
            if artifact.name != best_artifact.name:
                print(f'DELETING {artifact.name}')
                if not dry_run:
                    artifact.delete()
            else:
                print(f'KEEPING {artifact.name}')
    
    print("\nArtifact cleanup completed.")

def cleanup_cache(ROOT):
    # Go one directory above ROOT
    parent_dir = os.path.dirname(ROOT)
    cache_dir = os.path.join(parent_dir, ".cache")

    if os.path.exists(cache_dir) and os.path.isdir(cache_dir):
        # Define the directory to preserve
        preserve_dir = os.path.join(cache_dir, "torch", "hub", "checkpoints")

        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)

            # Skip the torch/hub/checkpoints directory
            if os.path.commonpath([item_path, preserve_dir]) == preserve_dir:
                continue

            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

        print(f"Cleared contents of {cache_dir}, except {preserve_dir}")
    else:
        print(f"No .cache directory found above {ROOT}")
