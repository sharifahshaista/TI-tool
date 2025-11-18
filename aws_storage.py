"""
AWS S3 Storage Module for TI-Tool
Handles all file operations with S3 bucket instead of local filesystem
"""

import boto3
import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from io import BytesIO, StringIO
import pandas as pd

# Try to import streamlit for secrets management
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

logger = logging.getLogger(__name__)


class S3Storage:
    """
    Unified S3 storage handler for TI-Tool
    Provides filesystem-like interface for S3 operations
    """
    
    def __init__(self, bucket_name: Optional[str] = None, region: Optional[str] = None):
        """
        Initialize S3 storage client
        
        Args:
            bucket_name: S3 bucket name (defaults to env variable or streamlit secrets)
            region: AWS region (defaults to env variable or streamlit secrets)
        """
        # Try environment variables first, then Streamlit secrets
        self.bucket_name = (
            bucket_name or 
            os.getenv("S3_BUCKET_NAME") or 
            os.getenv("AWS_S3_BUCKET")
        )
        
        self.region = (
            region or 
            os.getenv("AWS_DEFAULT_REGION") or
            os.getenv("AWS_REGION") or
            "us-east-1"
        )
        
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        # If environment variables not found, try Streamlit secrets
        if HAS_STREAMLIT and hasattr(st, 'secrets') and (not self.bucket_name or not aws_access_key or not aws_secret_key):
            try:
                self.bucket_name = self.bucket_name or st.secrets.get("AWS", {}).get("S3_BUCKET_NAME") or st.secrets.get("AWS_S3_BUCKET")
                self.region = self.region or st.secrets.get("AWS", {}).get("AWS_DEFAULT_REGION") or st.secrets.get("AWS_REGION", "us-east-1")
                aws_access_key = aws_access_key or st.secrets.get("AWS", {}).get("AWS_ACCESS_KEY_ID") or st.secrets.get("AWS_ACCESS_KEY_ID")
                aws_secret_key = aws_secret_key or st.secrets.get("AWS", {}).get("AWS_SECRET_ACCESS_KEY") or st.secrets.get("AWS_SECRET_ACCESS_KEY")
            except:
                pass  # Streamlit secrets not available, use what we have
        
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME or AWS_S3_BUCKET not set in environment variables or Streamlit secrets")
        
        if not aws_access_key or not aws_secret_key:
            raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set")
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            region_name=self.region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        
        logger.info(f"S3Storage initialized for bucket: {self.bucket_name}")
    
    def upload_file(self, local_path: str, s3_key: str, metadata: Optional[Dict[str, str]] = None) -> bool:
        """
        Upload a file to S3
        
        Args:
            local_path: Local file path
            s3_key: S3 object key (path in bucket)
            metadata: Optional metadata dict
            
        Returns:
            bool: True if successful
        """
        try:
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key, ExtraArgs=extra_args)
            logger.info(f"Uploaded: {local_path} -> s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False
    
    def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Download a file from S3
        
        Args:
            s3_key: S3 object key
            local_path: Local destination path
            
        Returns:
            bool: True if successful
        """
        try:
            # Create parent directory if needed
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Downloaded: s3://{self.bucket_name}/{s3_key} -> {local_path}")
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def upload_string(self, content: str, s3_key: str, content_type: str = 'text/plain') -> bool:
        """
        Upload string content to S3
        
        Args:
            content: String content
            s3_key: S3 object key
            content_type: MIME type
            
        Returns:
            bool: True if successful
        """
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=content.encode('utf-8'),
                ContentType=content_type
            )
            logger.info(f"Uploaded string to: s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"String upload failed: {e}")
            return False
    
    def download_string(self, s3_key: str) -> Optional[str]:
        """
        Download and return string content from S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            str: File content or None if failed
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            content = response['Body'].read().decode('utf-8')
            return content
        except Exception as e:
            logger.error(f"String download failed: {e}")
            return None
    
    def upload_json(self, data: Any, s3_key: str) -> bool:
        """
        Upload JSON data to S3
        
        Args:
            data: Python object (dict, list, etc.)
            s3_key: S3 object key
            
        Returns:
            bool: True if successful
        """
        try:
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            return self.upload_string(json_str, s3_key, content_type='application/json')
        except Exception as e:
            logger.error(f"JSON upload failed: {e}")
            return False
    
    def download_json(self, s3_key: str) -> Optional[Any]:
        """
        Download and parse JSON from S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Parsed JSON data or None if failed
        """
        try:
            content = self.download_string(s3_key)
            if content:
                return json.loads(content)
            return None
        except Exception as e:
            logger.error(f"JSON download failed: {e}")
            return None
    
    def upload_dataframe(self, df: pd.DataFrame, s3_key: str, format: str = 'csv') -> bool:
        """
        Upload pandas DataFrame to S3
        
        Args:
            df: DataFrame to upload
            s3_key: S3 object key
            format: 'csv' or 'json'
            
        Returns:
            bool: True if successful
        """
        try:
            buffer = BytesIO() if format == 'csv' else StringIO()
            
            if format == 'csv':
                df.to_csv(buffer, index=False, encoding='utf-8')
                content_type = 'text/csv'
            else:
                df.to_json(buffer, orient='records', indent=2, force_ascii=False)
                content_type = 'application/json'
            
            buffer.seek(0)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=buffer.getvalue(),
                ContentType=content_type
            )
            logger.info(f"Uploaded DataFrame to: s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"DataFrame upload failed: {e}")
            return False
    
    def download_dataframe(self, s3_key: str) -> Optional[pd.DataFrame]:
        """
        Download CSV/JSON as pandas DataFrame
        
        Args:
            s3_key: S3 object key
            
        Returns:
            DataFrame or None if failed
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            content = response['Body'].read()
            
            if s3_key.endswith('.csv'):
                df = pd.read_csv(BytesIO(content))
            elif s3_key.endswith('.json'):
                df = pd.read_json(BytesIO(content))
            else:
                # Try CSV first, then JSON
                try:
                    df = pd.read_csv(BytesIO(content))
                except:
                    df = pd.read_json(BytesIO(content))
            
            return df
        except Exception as e:
            logger.error(f"DataFrame download failed: {e}")
            return None
    
    def list_files(self, prefix: str = '', suffix: str = '') -> List[str]:
        """
        List files in S3 bucket with optional prefix/suffix filter
        
        Args:
            prefix: Filter by prefix (folder path)
            suffix: Filter by suffix (file extension)
            
        Returns:
            List of S3 keys
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                return []
            
            files = [obj['Key'] for obj in response['Contents']]
            
            if suffix:
                files = [f for f in files if f.endswith(suffix)]
            
            return files
        except Exception as e:
            logger.error(f"List files failed: {e}")
            return []
    
    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            bool: True if successful
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Deleted: s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False
    
    def file_exists(self, s3_key: str) -> bool:
        """
        Check if file exists in S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            bool: True if exists
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except:
            return False
    
    def get_file_metadata(self, s3_key: str) -> Optional[Dict]:
        """
        Get file metadata from S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Dict with metadata or None if failed
        """
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'content_type': response.get('ContentType', 'unknown'),
                'metadata': response.get('Metadata', {})
            }
        except Exception as e:
            logger.error(f"Get metadata failed: {e}")
            return None
    
    def copy_file(self, source_key: str, dest_key: str) -> bool:
        """
        Copy file within S3 bucket
        
        Args:
            source_key: Source S3 key
            dest_key: Destination S3 key
            
        Returns:
            bool: True if successful
        """
        try:
            copy_source = {'Bucket': self.bucket_name, 'Key': source_key}
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=self.bucket_name,
                Key=dest_key
            )
            logger.info(f"Copied: {source_key} -> {dest_key}")
            return True
        except Exception as e:
            logger.error(f"Copy failed: {e}")
            return False
    
    def get_presigned_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate presigned URL for file download
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration in seconds (default 1 hour)
            
        Returns:
            Presigned URL or None if failed
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.error(f"Presigned URL generation failed: {e}")
            return None


# Singleton instance
_storage_instance: Optional[S3Storage] = None


def get_storage() -> S3Storage:
    """
    Get or create singleton S3Storage instance
    
    Returns:
        S3Storage instance
    """
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = S3Storage()
    return _storage_instance


# Helper functions for common operations
def upload_to_s3(local_path: str, s3_folder: str, filename: Optional[str] = None) -> bool:
    """
    Helper: Upload local file to S3
    
    Args:
        local_path: Local file path
        s3_folder: S3 folder (e.g., 'crawled_data', 'processed_data')
        filename: Optional custom filename (defaults to original)
        
    Returns:
        bool: True if successful
    """
    storage = get_storage()
    filename = filename or Path(local_path).name
    s3_key = f"{s3_folder}/{filename}"
    return storage.upload_file(local_path, s3_key)


def download_from_s3(s3_folder: str, filename: str, local_folder: str) -> bool:
    """
    Helper: Download S3 file to local folder
    
    Args:
        s3_folder: S3 folder
        filename: Filename
        local_folder: Local destination folder
        
    Returns:
        bool: True if successful
    """
    storage = get_storage()
    s3_key = f"{s3_folder}/{filename}"
    local_path = str(Path(local_folder) / filename)
    return storage.download_file(s3_key, local_path)


def list_s3_files(folder: str, extension: str = '') -> List[str]:
    """
    Helper: List files in S3 folder
    
    Args:
        folder: S3 folder path
        extension: File extension filter (e.g., '.csv')
        
    Returns:
        List of filenames (without folder prefix)
    """
    storage = get_storage()
    files = storage.list_files(prefix=folder, suffix=extension)
    return [Path(f).name for f in files if f != folder + '/']
