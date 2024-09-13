import wandb

project_name = "Sandbox"
artifact_name = "2024-09-04_20-23_FREE"

api = wandb.Api()
artifact = api.artifact("rbo-malte/" + project_name + "/" + artifact_name + "_model" + ":latest")
artifact.download("./training_data/" + artifact_name + "/")

