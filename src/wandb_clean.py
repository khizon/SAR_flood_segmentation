import wandb
from tqdm import tqdm
import os
from dotenv import load_dotenv
import shutil

def cleanup_artifacts_per_run(project_name, entity, dry_run=True):
    api = wandb.Api()
    
    # Get all runs in the project
    runs = api.runs(f'{entity}/{project_name}')
    

    for run in tqdm(runs, desc="Cleaning Artifacts", unit="runs"):
        if run.state != 'finished':
            tqdm.write(f'Skipping unfinished run: {run.name}')
            continue
        
        tqdm.write(f'Processing run: {run.name}')
        
        # Get all model artifacts for this run
        run_artifacts = run.logged_artifacts()
        model_artifacts = [art for art in run_artifacts if art.type == 'model']
        
        if not model_artifacts:
            tqdm.write(f'No model artifacts found for run: {run.name}')
            continue
        
        if len(model_artifacts) == 1:
            tqdm.write(f'KEEPING {model_artifacts[0].name}')
            continue

        # Find the model artifact tagged as 'best'
        best_artifact = None
        
        for artifact in model_artifacts:
            if 'best' in artifact.aliases:
                best_artifact = artifact
                break

        if best_artifact is None:
            tqdm.write(f'No artifacts found with the BEST tag for run: {run.name}')
            continue
        
        tqdm.write(f'Best artifact for run {run.name}: {best_artifact.name}')
        
        # Delete all other model artifacts for this run
        for artifact in model_artifacts:
            if artifact.name != best_artifact.name:
                tqdm.write(f'DELETING {artifact.name}')
                if not dry_run:
                    artifact.delete()
            else:
                tqdm.write(f'KEEPING {artifact.name}')
    
    print('\nArtifact cleanup completed.')

if __name__ == '__main__':
    # Load environment variables from .env
    load_dotenv()

    # Get API key from .env
    api_key = os.getenv("WANDB_API_KEY")

    # Force re-login with the API key
    wandb.login(key=api_key, relogin=True)
    cleanup_artifacts_per_run('sar_seg_sen1floods11_A100', 'khizon')


