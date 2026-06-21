import os
import shutil
import bentoml

model_name = "firewatch-detector"
local_weights_path = "models/weights/train/weights/best.pt"

# Creating an isolated, tracked model entity inside BentoML
with bentoml.models.create(model_name) as bento_model:
    dest = os.path.join(bento_model.path, "best.pt")
    shutil.copy(local_weights_path, dest)
    print(f"Model successfully saved to Bento Store as: {bento_model.tag}")
