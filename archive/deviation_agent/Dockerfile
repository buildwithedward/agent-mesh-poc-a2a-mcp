FROM public.ecr.aws/lambda/python:3.11

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy function code
COPY deviation_agent.py ${LAMBDA_TASK_ROOT}

# Set Lambda handler
CMD ["deviation_agent.lambda_handler"]
