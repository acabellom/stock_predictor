import pytest
import boto3
from stock_predictor.utils import (
    create_s3_client,
    upload_to_s3,
    get_lowest_date_in_s3,
    store_old_data,
    check_highest_date_in_s3,
    create_bucket_if_not_exists,
)
import pandas as pd
from datetime import datetime, timedelta


def test_create_s3_client_calls_boto3_client(monkeypatch):
    """
    Test that create_s3_client calls boto3.client with correct parameters from environment variables.
    :param monkeypatch: pytest fixture for patching.
    """
    calls = {}

    def mock_boto3_client(
        service_name,
        endpoint_url=None,
        aws_access_key_id=None,
        aws_secret_access_key=None,
    ):
        calls["service_name"] = service_name
        calls["endpoint_url"] = endpoint_url
        calls["aws_access_key_id"] = aws_access_key_id
        calls["aws_secret_access_key"] = aws_secret_access_key
        return "mock_s3_client"

    monkeypatch.setattr(boto3, "client", mock_boto3_client)

    monkeypatch.setenv("ENDPOINT_URL", "http://fake-endpoint")
    monkeypatch.setenv("ROOT_USER", "fake_user")
    monkeypatch.setenv("ROOT_PASSWORD", "fake_password")

    client = create_s3_client()

    assert client == "mock_s3_client"
    assert calls["service_name"] == "s3"
    assert calls["endpoint_url"] == "http://fake-endpoint"
    assert calls["aws_access_key_id"] == "fake_user"
    assert calls["aws_secret_access_key"] == "fake_password"


def test_create_s3_client_without_env_vars(monkeypatch):
    """
    Test that create_s3_client can be called without environment variables and handles it gracefully.
    :param monkeypatch: pytest fixture for patching.
    """
    calls = {}

    def mock_boto3_client(
        service_name,
        endpoint_url=None,
        aws_access_key_id=None,
        aws_secret_access_key=None,
    ):
        calls["endpoint_url"] = endpoint_url
        calls["aws_access_key_id"] = aws_access_key_id
        calls["aws_secret_access_key"] = aws_secret_access_key
        return "mock_s3_client"

    monkeypatch.setattr(boto3, "client", mock_boto3_client)

    monkeypatch.delenv("ENDPOINT_URL", raising=False)
    monkeypatch.delenv("ROOT_USER", raising=False)
    monkeypatch.delenv("ROOT_PASSWORD", raising=False)

    client = create_s3_client()

    assert client == "mock_s3_client"
    assert calls["endpoint_url"] is None
    assert calls["aws_access_key_id"] is None
    assert calls["aws_secret_access_key"] is None


def test_upload_to_s3_creates_bucket_if_not_exists(monkeypatch):
    """
    Test that upload_to_s3 creates the bucket if it does not exist and uploads the file.
        :param monkeypatch: pytest fixture for patching.
    """
    calls = {}

    class MockS3:
        def list_buckets(self):
            return {"Buckets": []}

        def create_bucket(self, Bucket):
            calls["create_bucket"] = Bucket

        def put_object(self, Bucket, Key, Body):
            calls["put_object"] = (Bucket, Key, Body)

    s3_client = MockS3()

    data = pd.DataFrame({"a": [1, 2, 3]})

    upload_to_s3(s3_client, "test-bucket", "file.csv", data)

    assert calls["create_bucket"] == "test-bucket"
    assert calls["put_object"][0] == "test-bucket"
    assert calls["put_object"][1] == "file.csv"
    assert isinstance(calls["put_object"][2], str)


def test_upload_to_s3_does_not_create_bucket_if_exists():
    """
    Test that upload_to_s3 does not create the bucket if it already exists and uploads the file.
     :param monkeypatch: pytest fixture for patching.
    """
    calls = {}

    class MockS3:
        def list_buckets(self):
            return {"Buckets": [{"Name": "test-bucket"}]}

        def create_bucket(self, Bucket):
            calls["create_bucket"] = Bucket

        def put_object(self, Bucket, Key, Body):
            calls["put_object"] = (Bucket, Key, Body)

    s3_client = MockS3()

    data = pd.DataFrame({"a": [1, 2, 3]})

    upload_to_s3(s3_client, "test-bucket", "file.csv", data)

    assert "create_bucket" not in calls
    assert calls["put_object"][0] == "test-bucket"
    assert calls["put_object"][1] == "file.csv"


def test_upload_to_s3_sends_csv_data():
    """
    Test that upload_to_s3 sends the correct CSV data in the Body of the put_object call.
     :param monkeypatch: pytest fixture for patching.
    """

    class MockS3:
        def list_buckets(self):
            return {"Buckets": [{"Name": "test-bucket"}]}

        def put_object(self, Bucket, Key, Body):
            self.body = Body

    s3_client = MockS3()

    data = pd.DataFrame({"a": [1, 2, 3]})

    upload_to_s3(s3_client, "test-bucket", "file.csv", data)

    assert "a" in s3_client.body
    assert "1" in s3_client.body


def test_get_lowest_date_in_s3_single_raw_file():
    """
    Test getting lowest date from a single raw_ file.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {
                "Contents": [
                    {"Key": "raw_20240101_data.csv"},
                ]
            }

    result = get_lowest_date_in_s3("AAPL", MockS3())

    assert result == "20240101"


def test_get_lowest_date_in_s3_multiple_raw_files_returns_last_found():
    """
    Test that when multiple raw_ files exist,
    the function returns the date from the last iterated raw_ file.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {
                "Contents": [
                    {"Key": "raw_20220101_data.csv"},
                    {"Key": "raw_20230101_data.csv"},
                ]
            }

    result = get_lowest_date_in_s3("AAPL", MockS3())

    assert result == "20230101"


def test_get_lowest_date_in_s3_ignores_non_raw_files():
    """
    Test that non raw_ files are ignored.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {
                "Contents": [
                    {"Key": "processed.csv"},
                    {"Key": "raw_20231231_data.csv"},
                ]
            }

    result = get_lowest_date_in_s3("AAPL", MockS3())

    assert result == "20231231"


def test_get_lowest_date_in_s3_no_raw_files():
    """
    Test behavior when there are no raw_ files.
    This exposes a bug in the current implementation.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {
                "Contents": [
                    {"Key": "processed.csv"},
                ]
            }

    with pytest.raises(UnboundLocalError):
        get_lowest_date_in_s3("AAPL", MockS3())


def test_get_lowest_date_in_s3_empty_bucket():
    """
    Test behavior when bucket is empty.
    This exposes a bug in the current implementation.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {}

    result = get_lowest_date_in_s3("AAPL", MockS3())
    assert result == (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")


def test_store_old_data_skips_when_key_empty():
    """
    Test that when old_file_key is empty,
    the function does not call S3 operations.
    """

    class MockS3:
        def copy_object(self, **kwargs):
            raise AssertionError("copy_object should not be called")

        def delete_object(self, **kwargs):
            raise AssertionError("delete_object should not be called")

    # Should not raise
    store_old_data(MockS3(), "AAPL", "")


def test_store_old_data_archives_file_successfully():
    """
    Test that the file is copied to archive/ and then deleted.
    """

    calls = {}

    class MockS3:
        def copy_object(self, Bucket, CopySource, Key):
            calls["copy"] = {
                "Bucket": Bucket,
                "CopySource": CopySource,
                "Key": Key,
            }

        def delete_object(self, Bucket, Key):
            calls["delete"] = {
                "Bucket": Bucket,
                "Key": Key,
            }

    store_old_data(MockS3(), "AAPL", "raw/data.csv")

    assert calls["copy"]["Bucket"] == "aapl"
    assert calls["copy"]["CopySource"] == {
        "Bucket": "aapl",
        "Key": "raw/data.csv",
    }
    assert calls["copy"]["Key"] == "archive/raw/data.csv"

    assert calls["delete"]["Bucket"] == "aapl"
    assert calls["delete"]["Key"] == "raw/data.csv"


def test_store_old_data_uses_lowercase_bucket():
    """
    Test that ticker is always converted to lowercase for bucket name.
    """

    calls = {}

    class MockS3:
        def copy_object(self, Bucket, CopySource, Key):
            calls["bucket"] = Bucket

        def delete_object(self, Bucket, Key):
            pass

    store_old_data(MockS3(), "MsFt", "file.csv")

    assert calls["bucket"] == "msft"


def test_store_old_data_propagates_copy_error():
    """
    Test that if copy_object fails, the exception is propagated
    and delete_object is not called.
    """

    class MockS3:
        def copy_object(self, Bucket, CopySource, Key):
            raise Exception("S3 copy failed")

        def delete_object(self, Bucket, Key):
            raise AssertionError("delete_object should not be called")

    with pytest.raises(Exception, match="S3 copy failed"):
        store_old_data(MockS3(), "AAPL", "file.csv")


def test_check_highest_date_in_s3_multiple_raw_files_returns_last_found():
    """
    Test that when multiple raw_ files exist,
    the function returns the date from the last iterated raw_ file.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {
                "Contents": [
                    {"Key": "raw_data_20220101.csv"},
                    {"Key": "raw_data_20231231.csv"},
                ]
            }

    result = check_highest_date_in_s3("AAPL", MockS3())

    # Returns last iterated raw_ file date (not actually highest)
    assert result == "20231231"


def test_check_highest_date_in_s3_ignores_non_raw_files():
    """
    Test that non raw_ files are ignored.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {
                "Contents": [
                    {"Key": "processed.csv"},
                    {"Key": "raw_data_20240101.csv"},
                ]
            }

    result = check_highest_date_in_s3("AAPL", MockS3())

    assert result == "20240101"


def test_check_highest_date_in_s3_empty_bucket():
    """
    Test behavior when bucket is empty.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {}

    result = check_highest_date_in_s3("AAPL", MockS3())

    assert result == "00000000"


def test_check_highest_date_in_s3_no_raw_files():
    """
    Test behavior when there are no raw_ files.
    This exposes a bug in the current implementation.
    """

    class MockS3:
        def list_objects_v2(self, Bucket):
            return {
                "Contents": [
                    {"Key": "processed.csv"},
                    {"Key": "another_file.txt"},
                ]
            }

    with pytest.raises(UnboundLocalError):
        check_highest_date_in_s3("AAPL", MockS3())


def test_check_highest_date_in_s3_uses_lowercase_bucket():
    """
    Test that ticker is converted to lowercase when calling S3.
    """

    captured_bucket = {}

    class MockS3:
        def list_objects_v2(self, Bucket):
            captured_bucket["value"] = Bucket
            return {"Contents": []}

    check_highest_date_in_s3("MsFt", MockS3())

    assert captured_bucket["value"] == "msft"


def test_create_bucket_if_not_exists_creates_bucket_when_missing():
    """
    Test that the bucket is created if it does not exist.
    """

    calls = {}

    class MockS3:
        def list_buckets(self):
            return {
                "Buckets": [
                    {"Name": "existing-bucket"},
                ]
            }

        def create_bucket(self, Bucket):
            calls["created"] = Bucket

    create_bucket_if_not_exists(MockS3(), "new-bucket")

    assert calls["created"] == "new-bucket"


def test_create_bucket_if_not_exists_does_not_create_if_exists():
    """
    Test that the bucket is not created if it already exists.
    """

    class MockS3:
        def list_buckets(self):
            return {
                "Buckets": [
                    {"Name": "existing-bucket"},
                ]
            }

        def create_bucket(self, Bucket):
            raise AssertionError("create_bucket should not be called")

    create_bucket_if_not_exists(MockS3(), "existing-bucket")


def test_create_bucket_if_not_exists_creates_when_no_buckets():
    """
    Test that the bucket is created when the account has no buckets.
    """

    calls = {}

    class MockS3:
        def list_buckets(self):
            return {"Buckets": []}

        def create_bucket(self, Bucket):
            calls["created"] = Bucket

    create_bucket_if_not_exists(MockS3(), "my-bucket")

    assert calls["created"] == "my-bucket"


def test_create_bucket_if_not_exists_handles_missing_buckets_key():
    """
    Test that the function works when 'Buckets' key is missing in response.
    """

    calls = {}

    class MockS3:
        def list_buckets(self):
            return {}

        def create_bucket(self, Bucket):
            calls["created"] = Bucket

    create_bucket_if_not_exists(MockS3(), "test-bucket")

    assert calls["created"] == "test-bucket"
