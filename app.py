import boto3
import botocore


def lambda_handler(event, context):
    print(f'boto3 version: {boto3.__version__}')
    print(f'botocore version: {botocore.__version__}')
    return {
        'statusCode': 200,
        'body': 'Hello from Lambda!'
    }


if __name__ == '__main__':
    event = {
        'key1': 'value1',
        'key2': 'value2',
        'key3': 'value3'
    }
    lambda_handler(event, None)
