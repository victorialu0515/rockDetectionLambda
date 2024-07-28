# authentication with AWS ECR
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 339713152729.dkr.ecr.us-east-2.amazonaws.com

# create a new repository in ECR
#aws ecr create-repository \
#    --repository-name rockdetection \
#    --region us-east-2

# push the image to ECR
docker push 339713152729.dkr.ecr.us-east-2.amazonaws.com/rockdetection:latest
