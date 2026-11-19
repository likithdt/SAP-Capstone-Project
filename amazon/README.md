# Industrial IoT Cloud Ingestion (Software-only AWS demo)

A software-only demo pipeline:
Python MQTT simulator -> AWS IoT Core -> IoT Rule -> Lambda -> DynamoDB + S3 -> Streamlit dashboard.

## Important
- This is a demo, not a safety-certified industrial control system.
- Configure sensor ranges and maintenance thresholds for your actual scenario.
- Do not commit private keys, AWS credentials, `.env`, or downloaded IoT certificates.
- AWS usage may incur charges. Set a budget alert and delete resources after the demo.
- The Lambda uses an at-least-once event environment: duplicate deliveries are possible. The example creates unique S3 keys; production systems should add an explicit idempotency key and retry/DLQ strategy.

## Project structure
- `simulator/sensor_simulator.py`: generates and publishes MQTT telemetry.
- `lambda/lambda_function.py`: validates and writes readings to DynamoDB and S3.
- `dashboard/app.py`: Streamlit dashboard that reads DynamoDB.
- `aws/iot_rule.sql`: IoT SQL rule query.
- `aws/iot_policy_example.json`: restrictive device policy template.
- `.env.example`: local simulator/dashboard configuration template.
- `requirements.txt`: Python dependencies.

## 1. Create Python environment (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked for this terminal only:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Configure AWS CLI for your own account
Install AWS CLI v2, then:
```powershell
aws configure
aws sts get-caller-identity
```
Use an IAM identity intended for development, not root. Choose one AWS region and use it consistently. The Streamlit dashboard uses the AWS CLI credential chain; the simulator uses only the IoT device certificate.

## 3. Create cloud resources in the AWS Console
Use the same region for all resources.

### A. S3
Create a bucket with a globally unique name, e.g. `industrial-iot-telemetry-<random-suffix>`.
- Keep Block Public Access ON.
- Enable default server-side encryption.
- Optional: enable versioning for a demo if desired.
- Do not create a public bucket policy.

### B. DynamoDB
Create table `MachineTelemetry`:
- Partition key: `deviceId` (String)
- Sort key: `timestamp` (String)

Create table `MachineLatestState`:
- Partition key: `deviceId` (String)

Use on-demand capacity for a small demo. This example stores telemetry history and the latest snapshot separately.

### C. Lambda
Create function `telemetry-ingestion` using a Python runtime supported by Lambda (for example Python 3.12).
- Paste `lambda/lambda_function.py` into the function editor.
- Set environment variables:
  - `TABLE_NAME=MachineTelemetry`
  - `LATEST_TABLE_NAME=MachineLatestState`
  - `BUCKET_NAME=<your bucket name>`
  - `MIN_TEMPERATURE=-20`
  - `MAX_TEMPERATURE=150`
  - `MAX_VIBRATION=100`
- Give its execution role CloudWatch Logs permissions and least-privilege access to:
  - `dynamodb:PutItem` on both tables
  - `s3:PutObject` on `arn:aws:s3:::<your bucket>/telemetry/*`
- The Lambda function needs no public URL for ingestion.

### D. AWS IoT Core
1. Open IoT Core -> Manage -> Things (menu names may vary) and create Thing `industrial-machine-01`.
2. Create/download a device certificate and private key; download Amazon Root CA 1.
3. Attach an IoT policy to the certificate. Adapt `aws/iot_policy_example.json`: replace `<REGION>`, `<ACCOUNT_ID>`, and `<THING_NAME>` with your values. The policy allows connect and publish to only this device's topic.
4. Attach the certificate to the Thing and attach the policy to the certificate.
5. In IoT Core Settings, copy the device data endpoint, usually shaped like `xxxxxxxxxxxxxx-ats.iot.<region>.amazonaws.com`.
6. In Message routing -> Rules, create rule `MachineTelemetryToLambda` using the contents of `aws/iot_rule.sql`.
7. Add Lambda as the rule action, select `telemetry-ingestion`, and grant IoT permission to invoke it when prompted.
8. Confirm the rule is enabled.

### E. Simulator configuration
Copy `.env.example` to `.env` and fill in:
- `AWS_IOT_ENDPOINT`
- `AWS_IOT_THING_NAME`
- `AWS_IOT_TOPIC`
- `AWS_IOT_CERT`
- `AWS_IOT_PRIVATE_KEY`
- `AWS_IOT_ROOT_CA`

Use absolute Windows paths for the certificate files, e.g. `C:/Users/you/Downloads/device.pem.crt`.
Never share the private key.

## 4. Test cloud ingestion
In one PowerShell terminal:
```powershell
python simulator/sensor_simulator.py
```
In the AWS IoT MQTT test client, subscribe to:
`factory/machines/industrial-machine-01/telemetry`
You should see messages.

Then check:
- Lambda -> Monitor -> CloudWatch logs for invocations/errors.
- DynamoDB -> Explore items in `MachineTelemetry`.
- DynamoDB -> Explore items in `MachineLatestState`.
- S3 -> bucket -> `telemetry/deviceId=industrial-machine-01/date=YYYY-MM-DD/`.

To test invalid payloads, publish a message missing `deviceId` or with `"temperature":"hot"`. Lambda should raise a validation error; verify logs. For a demo, use the simulator's built-in `--once` flag to send one message.

## 5. Run Streamlit dashboard
The dashboard uses AWS CLI credentials, not the IoT device certificate.
Set `AWS_REGION` and `DYNAMODB_TABLE` in `.env`, then run:
```powershell
streamlit run dashboard/app.py
```
The dashboard reads the history table. Your AWS identity needs `dynamodb:Scan` and `dynamodb:Query` on `MachineTelemetry`. For a class demo this is simple; for production, do not expose direct AWS credentials to a public dashboard. Put API Gateway + a read-only Lambda between the dashboard and DynamoDB.

## 6. Suggested presentation flow
1. Show the architecture slide.
2. Start the simulator and show JSON telemetry changing every 5 seconds.
3. Open IoT MQTT test client and show messages arriving.
4. Show a Lambda invocation in CloudWatch.
5. Show records in DynamoDB and JSON objects in S3.
6. Run Streamlit and show latest values and historical charts.
7. Explain that high temperature or vibration creates a rule-based `WARNING` / `CRITICAL` indication.
8. State clearly that sensor data is simulated and the current alert logic is rule-based, not an ML prediction.

## Troubleshooting
- MQTT connection fails: check endpoint, certificate paths, Root CA, certificate attachment, policy resources, topic spelling, local clock and outbound network access.
- MQTT messages arrive but Lambda doesn't run: check IoT rule is enabled, SQL topic filter, Lambda action and invoke permission.
- Lambda errors: inspect CloudWatch logs; verify environment variables and execution-role permissions.
- DynamoDB writes but S3 fails (or vice versa): the two writes are not one atomic transaction. Retry/replay failed messages and add a dead-letter/error path for a robust deployment.
- Dashboard access denied: verify `aws sts get-caller-identity`, region, table name and read permissions.
- Empty chart: send a few readings and click Refresh.
