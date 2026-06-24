"""Model manager for loading model-specific inference modules."""

from importlib import util
from pathlib import Path

from config import DEVICE, MODELS_CONFIG


class ModelManager:
    def __init__(self):
        self.models = {}
        self.modules = {}
        self.base_dir = Path(__file__).resolve().parent

    def _resolve_path(self, relative_path):
        path = Path(relative_path)
        if path.is_absolute():
            return path
        return self.base_dir / path

    def _load_module(self, model_type):
        if model_type not in MODELS_CONFIG:
            raise ValueError(f"Unknown model type: {model_type}")

        if model_type in self.modules:
            return self.modules[model_type]

        config = MODELS_CONFIG[model_type]
        module_path = self._resolve_path(config["inference_module"])
        if not module_path.exists():
            print(f"Warning: inference module {module_path} not found.")
            return None

        spec = util.spec_from_file_location(f"mediscan_{model_type}_inference", module_path)
        if spec is None or spec.loader is None:
            print(f"Warning: could not load module spec for {module_path}.")
            return None

        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.modules[model_type] = module
        return module

    def get_model(self, model_type):
        if model_type not in MODELS_CONFIG:
            raise ValueError(f"Unknown model type: {model_type}")

        if model_type in self.models:
            return self.models[model_type]

        config = MODELS_CONFIG[model_type]
        module = self._load_module(model_type)
        if module is None:
            return None

        model_path = self._resolve_path(config["model_path"])
        if not model_path.exists():
            print(f"Warning: model file {model_path} not found. Please train/download it.")
            return None

        model = module.load_model(model_path=str(model_path), device=DEVICE)
        if model is None:
            print(f"Warning: model module for {model_type} did not return a model.")
            return None

        self.models[model_type] = model
        return model

    def predict(self, model_type, image_bytes):
        module = self._load_module(model_type)
        if module is None:
            return "Model not found/loaded", 0.0

        model = self.get_model(model_type)
        if model is None:
            return "Model not found/loaded", 0.0

        predicted_class, confidence = module.predict_image(model, image_bytes, device=DEVICE)
        return predicted_class, confidence


# Singleton instance
manager = ModelManager()
