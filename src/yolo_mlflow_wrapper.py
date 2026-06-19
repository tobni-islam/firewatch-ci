from ultralytics import YOLO
import mlflow.pyfunc


class YOLOWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = YOLO(context.artifacts["weights"])

    def predict(self, context, model_input):
        return self.model(model_input)
