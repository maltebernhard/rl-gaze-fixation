import wandb

api = wandb.Api()
artifact = api.artifact("rbo-malte/GazeFixation2d/2024-08-29_19-10_FREE_model:v0")

artifact.download("./training_data/2024-08-29_19-10_FREE")

