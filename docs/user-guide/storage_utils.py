"""Object-storage cleanup utilities for aifs-modal notebooks."""

import os

import boto3


def delete_prefixes(bucket: str, *prefixes: str) -> None:
    """Delete all S3 objects under each prefix and print a per-prefix summary.

    Parameters
    ----------
    bucket:
        S3/Tigris bucket name.
    *prefixes:
        One or more key prefixes to wipe.  Each prefix is listed separately
        so the caller can tell at a glance what was removed.
    """
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    paginator = s3.get_paginator("list_objects_v2")
    total = 0
    for prefix in prefixes:
        deleted = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            objects = page.get("Contents", [])
            if objects:
                s3.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": [{"Key": o["Key"]} for o in objects]},
                )
                deleted += len(objects)
        total += deleted
        print(f"deleted {deleted} object(s) under s3://{bucket}/{prefix}")
    print(f"done — removed {total} objects from s3://{bucket}/")
