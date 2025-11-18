
# Infer model family based on model path name
def infer_model_family(model_path: str) -> str:
        """Infer if model is Llama or Qwen based on path name."""
        model_path_lower = model_path.lower()
        if "llama" in model_path_lower:
            return "llama"
        elif "qwen" in model_path_lower:
            return "qwen"
        else:
            return "unknown"
