"""
Cloud Storage Adapter for AWS S3 Data Lake, Amazon Timestream, and DynamoDB.
Autonomous Industrial IoT Digital Twin & Predictive Maintenance Engine with Closed-Loop ERP Action.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json


class CloudStorageAdapter:
    """
    Adapter for AWS telemetry data storage:
      - Amazon S3: Cold raw telemetry data lake & model artifacts
      - Amazon Timestream: Fast time-series metric querying
      - Amazon DynamoDB: Asset registry, alert events & state tracking
    """

    def __init__(
        self,
        s3_bucket: str = "industrial-iot-telemetry-lake",
        timestream_db: str = "industrial_twin_db",
        timestream_table: str = "telemetry_table",
        dynamo_table: str = "industrial_asset_registry",
        simulate: bool = True,
    ):
        self.s3_bucket = s3_bucket
        self.timestream_db = timestream_db
        self.timestream_table = timestream_table
        self.dynamo_table = dynamo_table
        self.simulate = simulate

        # Simulated storage backends
        self.s3_objects: Dict[str, str] = {}
        self.timestream_records: List[Dict[str, Any]] = []
        self.dynamodb_items: Dict[str, Dict[str, Any]] = {}

    def store_raw_s3(self, asset_id: str, payload: Dict[str, Any]) -> str:
        """Store raw JSON telemetry batch in S3 bucket with partition key."""
        now = datetime.now(timezone.utc)
        s3_key = (
            f"raw/year={now.year}/month={now.month:02d}/day={now.day:02d}/"
            f"{asset_id}_{int(now.timestamp())}.json"
        )
        self.s3_objects[s3_key] = json.dumps(payload)
        return f"s3://{self.s3_bucket}/{s3_key}"

    def write_timestream_record(
        self,
        asset_id: str,
        dimensions: Dict[str, str],
        measures: Dict[str, float],
        timestamp_ms: Optional[int] = None,
    ) -> int:
        """Write multi-measure time-series telemetry record to Amazon Timestream."""
        ts = timestamp_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
        record = {
            "Database": self.timestream_db,
            "Table": self.timestream_table,
            "Dimensions": {"asset_id": asset_id, **dimensions},
            "Measures": measures,
            "Time": str(ts),
            "TimeUnit": "MILLISECONDS",
        }
        self.timestream_records.append(record)
        return len(measures)

    def put_dynamodb_item(self, key: str, item: Dict[str, Any]) -> None:
        """Store or update asset metadata / alert event in DynamoDB."""
        self.dynamodb_items[key] = item.copy()

    def get_dynamodb_item(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve item from DynamoDB table."""
        return self.dynamodb_items.get(key)
