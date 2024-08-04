FROM public.ecr.aws/lambda/python:3.11

WORKDIR ${LAMBDA_TASK_ROOT}

RUN yum install -y mesa-libGL

# Install requirements
COPY docker/requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt --index-url https://download.pytorch.org/whl/cpu && \
    yum clean all && \
    rm -rf /var/cache/yum

# Copy stuff
COPY models ${LAMBDA_TASK_ROOT}/models
COPY *.py ${LAMBDA_TASK_ROOT}

# Set
CMD [ "app.lambda_handler" ]
