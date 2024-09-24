import wandb
from tqdm import tqdm

def cleanup_artifacts_per_run(project_name, entity, metric_name, dry_run=True):
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
        
        # Find the best model artifact based on the specified metric
        best_artifact = None
        best_metric_value = float('-inf')
        
        for artifact in model_artifacts:
            if metric_name in artifact.metadata:
                metric_value = artifact.metadata[metric_name]
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_artifact = artifact
        
        if best_artifact is None:
            print(f"No artifacts found with the metric '{metric_name}' for run: {run.name}")
            continue
        
        print(f"Best artifact for run {run.name}: {best_artifact.name} with {metric_name} = {best_metric_value}")
        
        # Delete all other model artifacts for this run
        for artifact in model_artifacts:
            if artifact.name != best_artifact.name:
                print(f'DELETING {artifact.name}')
                if not dry_run:
                    artifact.delete()
            else:
                print(f'KEEPING {artifact.name}')
    
    print("\nArtifact cleanup completed.")