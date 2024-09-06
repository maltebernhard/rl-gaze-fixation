import wandb

api = wandb.Api()
artifact = api.artifact("rbo-malte/Sandbox/2024-09-04_20-23_FREE_model:v0")

artifact.download("./training_data/2024-09-04_20-23_FREE")

