"""
Inference engine for Sign Language Translation.
Loads the trained SPOTER model and runs predictions on landmark sequences.
"""

import os
import sys
import torch
import numpy as np

# Add project root to path so we can import the model classes
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from web_app.backend.gloss_labels import get_gloss, get_all_glosses


class SignLanguagePredictor:
    """Loads the trained SPOTER model and predicts sign language glosses."""

    def __init__(self, checkpoint_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        if checkpoint_path is None:
            # Use the best validation checkpoint
            checkpoint_path = os.path.join(
                PROJECT_ROOT, "out-checkpoints", "wlasl_train", "checkpoint_v_9.pth"
            )

        self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path: str):
        """Load a trained model checkpoint."""
        if not os.path.exists(checkpoint_path):
            print(f"WARNING: Checkpoint not found at {checkpoint_path}")
            print("The model will not be available for predictions.")
            return

        try:
            self.model = torch.load(checkpoint_path, map_location=self.device)
            self.model.eval()
            self.model.to(self.device)
            print(f"Model loaded successfully from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def preprocess_landmarks(self, landmarks_sequence: list) -> torch.Tensor:
        """
        Preprocess a sequence of landmark frames into a model-ready tensor.

        Args:
            landmarks_sequence: List of frames. Each frame is a list of landmarks.
                              Each landmark is [x, y] coordinates.
                              Expected shape: [num_frames, 75, 2] for MediaPipe (42 hand + 33 pose)
                              or [num_frames, 54, 2] for VisionAPI format.

        Returns:
            Preprocessed tensor ready for model input.
        """
        data = np.array(landmarks_sequence, dtype=np.float32)

        # The model expects [num_frames, n_landmarks, 2]
        # n_landmarks = hidden_dim / 2 = 75 for our training config (hidden_dim=150)
        n_landmarks = data.shape[1] if len(data.shape) == 3 else 75

        if len(data.shape) == 2:
            # Reshape from flat to [num_frames, n_landmarks, 2]
            num_frames = data.shape[0]
            data = data.reshape(num_frames, n_landmarks, 2)

        # Apply the same normalization as training: subtract 0.5
        tensor = torch.from_numpy(data).float() - 0.5
        return tensor

    def predict(self, landmarks_sequence: list) -> dict:
        """
        Predict the sign language gloss from a sequence of landmarks.

        Args:
            landmarks_sequence: List of frames with landmark coordinates.

        Returns:
            Dictionary with prediction results.
        """
        if self.model is None:
            return {
                "success": False,
                "error": "Model not loaded",
                "predicted_label": -1,
                "predicted_gloss": "N/A",
                "confidence": 0.0,
                "top5": [],
            }

        try:
            tensor = self.preprocess_landmarks(landmarks_sequence)
            # The SPOTER model treats each frame as a batch item
            # (training code does squeeze(0), so input is [num_frames, 75, 2])
            tensor = tensor.to(self.device)

            with torch.no_grad():
                output = self.model(tensor)  # [num_frames, 1, num_classes]

            # Average predictions across all frames
            avg_output = output.mean(dim=0)  # [1, num_classes]
            probabilities = torch.softmax(avg_output, dim=-1)  # [1, num_classes]
            confidence, predicted_idx = torch.max(probabilities, dim=-1)

            predicted_label = predicted_idx.item()
            predicted_gloss = get_gloss(predicted_label)
            conf = confidence.item()

            # Get top-5 predictions
            top5_vals, top5_idxs = torch.topk(probabilities, k=min(5, probabilities.shape[-1]), dim=-1)
            top5 = [
                {"label": idx.item(), "gloss": get_gloss(idx.item()), "confidence": round(val.item(), 4)}
                for val, idx in zip(top5_vals[0], top5_idxs[0])
            ]

            return {
                "success": True,
                "predicted_label": predicted_label,
                "predicted_gloss": predicted_gloss,
                "confidence": round(conf, 4),
                "top5": top5,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "predicted_label": -1,
                "predicted_gloss": "N/A",
                "confidence": 0.0,
                "top5": [],
            }
