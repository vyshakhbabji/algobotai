"""Signal normalization utilities."""
from typing import Dict, Optional


def normalize_prediction(pred: Dict) -> Optional[Dict]:
    if not pred:
        return None
    return {
        'expected_return': pred['predicted_return'],
        'confidence': pred['confidence'],
        'signal': pred['signal'],
        'quality': pred.get('quality'),
        'models_used': pred.get('models_used', 0),
        'version': pred.get('version')
    }
