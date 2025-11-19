import boto3.session

if __name__ == '__main__':
    with boto3.session.Session() as session:
        s3 = session.resource('s3')
        bucket = s3.Bucket('dev1-inspect-scans')
        for obj in bucket.objects.all():
            print(obj.key)