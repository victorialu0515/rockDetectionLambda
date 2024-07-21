# authentication with AWS ECR
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 339713152729.dkr.ecr.us-east-2.amazonaws.com

# Pull base image
docker pull 339713152729.dkr.ecr.us-east-2.amazonaws.com/rockdetection