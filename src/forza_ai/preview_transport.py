"""Bounded authenticated preview metadata validation, independent of Torch."""
import math

MAX_PREVIEW_BYTES = 100_000


def validate_preview(value, payload, session, request_id):
    if value is None:
        if payload:
            raise ValueError('preview payload without metadata')
        return
    if not isinstance(value, dict) or set(value) != {'session', 'request_id', 'frame_id', 'age_ms', 'active', 'prediction'}:
        raise ValueError('invalid preview fields')
    if value['session'] != session or type(value['request_id']) is not int or not 1 <= value['request_id'] <= request_id:
        raise ValueError('preview from another session or future request')
    if type(value['frame_id']) is not int or not 0 <= value['frame_id'] < 2**63 or type(value['active']) is not bool:
        raise ValueError('invalid preview identity')
    age = value['age_ms']
    if isinstance(age, bool) or not isinstance(age, (int, float)) or not math.isfinite(age) or not 0 <= age <= 2000:
        raise ValueError('invalid preview age')
    prediction = value['prediction']
    if not isinstance(prediction, dict) or set(prediction) not in ({'angle_deg'}, {'angle_deg', 'throttle', 'brake'}):
        raise ValueError('invalid preview prediction')
    for key, number in prediction.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError('nonfinite preview prediction')
        if not (-450 <= number <= 450 if key == 'angle_deg' else 0 <= number <= 1):
            raise ValueError('out of range preview prediction')
    # Do not decode on the control/policy thread. Check fixed PNG IHDR dimensions
    # before exposing authenticated bounded bytes to the browser image decoder.
    if (not 33 <= len(payload) <= MAX_PREVIEW_BYTES or payload[:8] != b'\x89PNG\r\n\x1a\n'
            or payload[12:16] != b'IHDR' or int.from_bytes(payload[16:20], 'big') != 400
            or int.from_bytes(payload[20:24], 'big') != 66 or payload[24:26] != b'\x08\x02'):
        raise ValueError('invalid fixed-size preview PNG')
