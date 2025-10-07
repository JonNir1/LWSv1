import io
from datetime import datetime

from constants import DATE_TIME_FORMAT


_SUBJECT_INFO_FIELD_MAP = {
    "Name": "name", "Subject": "subject_id", "Age": "age", "Sex": "sex", "Handedness": "hand", "DominantEye": "eye",
    "Session": "session", "SessionDate": "session_date", "SessionTime": "session_time", "Distance": "screen_distance_cm",
}


def parse_subject_info(file_path) -> dict:
    """ Reads subject's personal information from the E-Prime log file, and returns a dictionary with the information. """
    f = io.open(file_path, mode="r", encoding="utf-16")
    lines = f.readlines()
    subject_info = {field: None for field in _SUBJECT_INFO_FIELD_MAP.values()}
    for line in lines:
        if ":" not in line:
            continue
        eprime_field, value = line.split(":", 1)
        eprime_field, value = eprime_field.strip(), value.strip()
        if eprime_field in _SUBJECT_INFO_FIELD_MAP.keys():
            field = _SUBJECT_INFO_FIELD_MAP[eprime_field]
            subject_info[field] = value
    f.close()

    # Convert to numeric types
    subject_info["subject_id"] = int(subject_info["subject_id"])
    subject_info["age"] = float(subject_info["age"])
    subject_info["screen_distance_cm"] = float(subject_info["screen_distance_cm"])
    subject_info["session"] = int(subject_info["session"])

    # Convert the session date and time to a datetime object
    date = subject_info.pop("session_date", None)
    time = subject_info.pop("session_time", None)
    if date and time:
        subject_info["date_time"] = datetime.strptime(f"{date} {time}", DATE_TIME_FORMAT)
    else:
        subject_info["date_time"] = None
    return subject_info

