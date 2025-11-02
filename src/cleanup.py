from wandb_clean import cleanup_artifacts_per_run, cleanup_cache
import os

if __name__ == '__main__':
    ROOT = os.getcwd()
    cleanup_cache(ROOT)