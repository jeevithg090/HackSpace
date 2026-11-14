"""Runway detection in aircraft approach and landing imagery.

The classical detector (:mod:`runway_detection.classical`) needs only NumPy, OpenCV and
Pillow. The U-Net modules (``model``, ``data``, ``train``, ``predict``) need PyTorch and
Albumentations from the ``ml`` extra.
"""

from .classical import ClassicalRunwayDetector, DetectionResult, DetectorConfig

__all__ = ["ClassicalRunwayDetector", "DetectionResult", "DetectorConfig"]
__version__ = "0.1.0"
