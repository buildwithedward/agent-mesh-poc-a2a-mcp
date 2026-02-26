# # FROM public.ecr.aws/lambda/python:3.10

# # COPY requirements.txt .
# # RUN pip install --no-cache-dir -r requirements.txt

# # COPY app.py .
# # COPY service.py .

# # ENV _HANDLER=app.lambda_handler


# # CMD ["app.lambda_handler"]

# FROM public.ecr.aws/lambda/python:3.10

# # Copy dependencies
# COPY requirements.txt ${LAMBDA_TASK_ROOT}
# RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# # Copy function code
# COPY app.py ${LAMBDA_TASK_ROOT}
# COPY service.py ${LAMBDA_TASK_ROOT}

# # Set the CMD to your handler
# CMD ["app.lambda_handler"]

FROM public.ecr.aws/lambda/python:3.10

# Set the working directory to the standard Lambda root
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy dependencies first (better for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy function code to the current WORKDIR
COPY app.py service.py ./

# Ensure the files have the correct permissions for the Lambda user
RUN chmod +x app.py service.py

# Just the handler string: filename.function_name
CMD ["app.lambda_handler"]