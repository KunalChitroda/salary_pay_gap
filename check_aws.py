import os
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError, NoCredentialsError

# Load all environment variables from .env
load_dotenv()

# --- 1. Load and Check Variables ---
print("--- 1. Checking Environment Variables ---")
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")
ecr_repo = os.getenv("ECR_REPOSITORY")
ecr_registry = os.getenv("ECR_REGISTRY_URL")

if not all([aws_access_key, aws_secret_key, aws_region, ecr_repo, ecr_registry]):
    print("❌ ERROR: One or more environment variables are missing from your .env file.")
    exit(1)
else:
    print("✅ All environment variables are loaded.")
    print(f"   Using Region: {aws_region}")
    print(f"   ECR Repo: {ecr_repo}")


# --- 2. Test AWS Connection and Credentials ---
print("\n--- 2. Testing AWS STS Connection ---")
try:
    # Use STS GetCallerIdentity as a simple, low-permission "who am I" check
    sts_client = boto3.client(
        'sts',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )
    response = sts_client.get_caller_identity()
    print(f"✅ AWS Connection Successful. You are: {response['Arn']}")
except ClientError as e:
    if "InvalidClientTokenId" in str(e):
        print("❌ ERROR: Invalid AWS_ACCESS_KEY_ID. Please check your .env file.")
    elif "SignatureDoesNotMatch" in str(e):
        print("❌ ERROR: Invalid AWS_SECRET_ACCESS_KEY. Please check your .env file.")
    else:
        print(f"❌ AWS Connection Error: {e}")
    exit(1)
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
    exit(1)


# --- 3. Test ECR Access ---
print("\n--- 3. Testing ECR Repository Access ---")
try:
    ecr_client = boto3.client(
        'ecr',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )
    
    # Check if the repository exists
    ecr_client.describe_repositories(repositoryNames=[ecr_repo])
    print(f"✅ ECR repository '{ecr_repo}' found.")
except ClientError as e:
    if "RepositoryNotFoundException" in str(e):
        print(f"❌ ERROR: ECR repository '{ecr_repo}' was not found.")
    else:
        print(f"❌ ECR Error: {e}")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")


# --- 4. Test S3 Access (for DVC) ---
print("\n--- 4. Testing S3/DVC Access ---")
print("(This test just checks S3 permissions, not your DVC remote name.)")
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )
    
    # List S3 buckets to check if S3 access is working
    s3_client.list_buckets()
    print("✅ S3 Connection Successful (ListBuckets permission is active).")
except ClientError as e:
    print(f"❌ S3 Error: {e}. You may be missing 's3:ListAllMyBuckets' permissions.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")

print("\n--- Check Complete ---")