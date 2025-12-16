import boto3
import sys
from botocore.exceptions import ClientError
import os
import logging

from dotenv import load_dotenv, find_dotenv # Import from python-dotenv

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

home_dir = os.path.expanduser("~") 
home_dotenv_path = os.path.join(home_dir, ".env")

# --- !! IMPORTANT: SET THESE VALUES !! ---

# --- Load Environment Variables from .env file ---
# This will search for a .env file in the current directory or parent directories
# and load its key-value pairs into environment variables.
load_dotenv(home_dotenv_path)

# --- Configuration ---
# Attempt to get credentials from environment variables (set by .env or system)
AWS_ACCESS_KEY_ID_FROM_ENV = os.getenv("AWS_imdpdev_ACCESSKEY_ID")
AWS_SECRET_ACCESS_KEY_FROM_ENV = os.getenv("AWS_imdpdev_SECRET_ACCESSKEY")
AWS_SESSION_TOKEN_FROM_ENV = os.getenv("AWS_SESSION_TOKEN") # If using temporary credentials

# AWS Region - can also be set in .env or directly here
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", 'ap-south-1') # Default to ap-south-1 if not set


boto3_session_params = {
    'region_name': AWS_REGION
}
if AWS_ACCESS_KEY_ID_FROM_ENV and AWS_SECRET_ACCESS_KEY_FROM_ENV:
    logger.info("Using AWS credentials from .env file (AWS_imdpdev_...).")
    boto3_session_params['aws_access_key_id'] = AWS_ACCESS_KEY_ID_FROM_ENV
    boto3_session_params['aws_secret_access_key'] = AWS_SECRET_ACCESS_KEY_FROM_ENV
    if AWS_SESSION_TOKEN_FROM_ENV: # If using temporary creds with a session token
         boto3_session_params['aws_session_token'] = AWS_SESSION_TOKEN_FROM_ENV
else:
    logger.info("Using default Boto3 credential resolution (e.g., ~/.aws/credentials, IAM role).")


try:
    # Create a session with the specified credentials if provided
    session = boto3.Session(**boto3_session_params)

    s3_client = boto3.client('s3', region_name=AWS_REGION)
except Exception as e:
    logger.error(f"Error initializing Boto3 clients: {e}")
    exit(1)

def create_dynamodb_table(table_name="TIH_FormBotSessions", region="ap-south-1"):
    """
    Creates a DynamoDB table with a partition key 'session_id'.
    """
    # Initialize the DynamoDB resource
    # Boto3 will automatically look for credentials in your environment or ~/.aws/credentials
    dynamodb = boto3.resource('dynamodb', region_name=region)

    try:
        print(f"Attempting to create table: {table_name} in {region}...")

        table = dynamodb.create_table(
            TableName=table_name,
            # Schema Definition
            KeySchema=[
                {
                    'AttributeName': 'session_id',
                    'KeyType': 'HASH'  # Partition key
                }
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'session_id',
                    'AttributeType': 'S'  # S = String
                }
            ],
            # Provisioned Throughput (Free Tier eligible: 25 RCU/WCU)
            # We set it to 5 just to be safe and low cost.
            ProvisionedThroughput={
                'ReadCapacityUnits': 2,
                'WriteCapacityUnits': 1
            }
            
            # ALTERNATIVE: Use On-Demand Billing (No capacity planning, pay per request)
            # If you want this, uncomment the line below and remove ProvisionedThroughput above
            # BillingMode='PAY_PER_REQUEST'
        )

        print("Table creation initiated. Waiting for status to be 'ACTIVE'...")
        
        # Wait until the table exists. This prevents using the table before it's ready.
        table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
        
        # Reload table attributes to get the latest status
        table.reload()
        
        print("-" * 40)
        print(f"SUCCESS: Table '{table.table_name}' created successfully.")
        print(f"Status: {table.table_status}")
        print(f"ARN: {table.table_arn}")
        print("-" * 40)

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ResourceInUseException':
            print(f"WARNING: The table '{table_name}' already exists.")
        else:
            print(f"ERROR: Could not create table. {e}")
            sys.exit(1)

if __name__ == "__main__":
    # You can change the region here if needed (e.g., 'us-west-2', 'eu-central-1')
    create_dynamodb_table(region="ap-south-1")