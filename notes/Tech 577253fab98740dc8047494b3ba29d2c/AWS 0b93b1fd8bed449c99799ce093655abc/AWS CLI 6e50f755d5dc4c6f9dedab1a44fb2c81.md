# AWS CLI

# Get access id, access key after logged in via SSO auth

So it can be used for boto3

```python
aws sso get-role-credentials --profile test-xxx --role-name AdministratorAccess --account-id <long number> --access-token <token> --region ap-northeast-1
```

# Get region

```python
aws configure get region
```