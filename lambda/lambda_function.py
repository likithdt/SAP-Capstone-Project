import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import boto3

dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")

history_table = dynamodb.Table(os.environ["TABLE_NAME"])
latest_table = dynamodb.Table(os.environ["LATEST_TABLE_NAME"])
bucket = os.environ["BUCKET_NAME"]

ALLOWED_STATUS = {"RUNNING", "STOPPED", "FAULT"}

def decimal_number(payload, field, minimum, maximum):
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < minimum or numeric > maximum:
        raise ValueError(f"{field} outside permitted range [{minimum}, {maximum}]")
    return Decimal(str(value))

def parse_timestamp(value):
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO 8601 string with timezone")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp is not valid ISO 8601") from exc
    if dt.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    dt = dt.astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    if (dt - now).total_seconds() > 300:
        raise ValueError("timestamp is more than 5 minutes in the future")
    # Reject excessively old data in this demo; adjust for your replay policy.
    if (now - dt).total_seconds() > 7 * 24 * 3600:
        raise ValueError("timestamp is older than 7 days")
    return dt

def lambda_handler(event, context):
    if not isinstance(event, dict):
        raise ValueError("Telemetry must be a JSON object")

    device_id = event.get("deviceId")
    if not isinstance(device_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", device_id):
        raise ValueError("Invalid deviceId")

    # Example ranges only. Configure them for your simulated machine.
    min_temp = float(os.getenv("MIN_TEMPERATURE", "-20"))
    max_temp = float(os.getenv("MAX_TEMPERATURE", "150"))
    max_vibration = float(os.getenv("MAX_VIBRATION", "100"))

    temperature = decimal_number(event, "temperature", min_temp, max_temp)
    vibration = decimal_number(event, "vibration", 0, max_vibration)
    humidity = decimal_number(event, "humidity", 0, 100)

    motor_status = event.get("motorStatus")
    if motor_status not in ALLOWED_STATUS:
        raise ValueError("motorStatus must be RUNNING, STOPPED or FAULT")

    timestamp_dt = parse_timestamp(event.get("timestamp"))
    timestamp = timestamp_dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    ingested_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    # Illustrative rule-based alert. Tune for your presentation scenario.
    temp_float = float(temperature)
    vibration_float = float(vibration)
    if motor_status == "FAULT" or temp_float >= 100 or vibration_float >= 15:
        maintenance_status = "CRITICAL"
    elif temp_float >= 85 or vibration_float >= 10:
        maintenance_status = "WARNING"
    else:
        maintenance_status = "NORMAL"

    record = {
        "deviceId": device_id,
        "timestamp": timestamp,
        "temperature": temperature,
        "vibration": vibration,
        "humidity": humidity,
        "motorStatus": motor_status,
        "maintenanceStatus": maintenance_status,
        "ingestedAt": ingested_at,
    }

    # Write history and latest snapshot. These writes are not an atomic transaction.
    history_table.put_item(Item=record)
    latest_table.put_item(Item=record)

    # Unique object key prevents accidental overwrite for simultaneous readings.
    date_partition = timestamp_dt.strftime("%Y-%m-%d")
    s3_key = (
        f"telemetry/deviceId={device_id}/date={date_partition}/"
        f"{timestamp_dt.strftime('%H%M%S%f')}-{uuid.uuid4().hex}.json"
    )
    s3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=json.dumps(record, default=str).encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )

    print(json.dumps({
        "accepted": True,
        "deviceId": device_id,
        "timestamp": timestamp,
        "maintenanceStatus": maintenance_status,
        "s3Key": s3_key,
    }))
    return {"accepted": True, "deviceId": device_id, "timestamp": timestamp,
            "maintenanceStatus": maintenance_status, "s3Key": s3_key}
