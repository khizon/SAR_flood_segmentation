import wandb
from tqdm import tqdm

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