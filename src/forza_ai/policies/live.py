"""Bridge causal camera/telemetry observations to a portable steering predictor."""

import math

import numpy as np

from forza_ai.contracts import ModelObservation, ObservationUnavailable, DrivingPrediction


class LiveModelPolicy:
    requires_camera = True
    name = "trained CNN steering policy"

    def __init__(self, export_path=None, *, predictor=None, max_telemetry_age_ns=100_000_000):
        if type(max_telemetry_age_ns) is not int or max_telemetry_age_ns <= 0:
            raise ValueError("max_telemetry_age_ns must be a positive integer")
        if (export_path is None) == (predictor is None):
            raise ValueError("provide exactly one of export_path or predictor")
        if predictor is None:
            # Torch/model loading is unnecessary for placeholder and hardware tests.
            from forza_ai.policies.predictor import load_predictor

            predictor = load_predictor(export_path)
        self.predictor = predictor
        self.driving = getattr(predictor, "driving", False) is True
        if self.driving:
            self.name = "trained CNN steering and pedal policy"
        self.max_telemetry_age_ns = max_telemetry_age_ns

    def predict(self, observation: ModelObservation) -> float:
        frame, vehicle = observation.frame, observation.vehicle
        if frame is None or vehicle is None:
            raise ObservationUnavailable("missing camera frame or telemetry")
        for timestamp in (frame.timestamp_ns, vehicle.timestamp_ns):
            if type(timestamp) is not int or timestamp < 0:
                raise ValueError("observation timestamps must be nonnegative integer nanoseconds")
        if vehicle.is_race_on not in (True, False):
            raise ValueError("is_race_on must be a boolean or 0/1")
        if not vehicle.is_race_on:
            raise ObservationUnavailable("game is paused or race is not active")
        age = frame.timestamp_ns - vehicle.timestamp_ns
        if not 0 <= age <= self.max_telemetry_age_ns:
            raise ObservationUnavailable("telemetry is future-dated or stale relative to the frame")
        if not math.isfinite(vehicle.speed_mps) or vehicle.speed_mps < 0:
            raise ValueError("speed_mps must be finite and nonnegative")
        pixels = frame.rgb
        if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8 or pixels.ndim != 3
                or pixels.shape[2] != 3 or not pixels.shape[0] or not pixels.shape[1]):
            raise ValueError("expected a nonempty uint8 HWC RGB road crop")
        prediction = self.predictor.predict(pixels, vehicle.speed_mps)
        if self.driving:
            if not isinstance(prediction, DrivingPrediction):
                raise ValueError('driving predictor returned a steering-only result')
            return prediction
        angle = float(prediction)
        if not math.isfinite(angle):
            raise ValueError("predictor returned a non-finite steering angle")
        return angle
