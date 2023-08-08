FROM public.ecr.aws/lambda/python:3.8

# copy function code and models into /var/task
COPY app.py ${LAMBDA_TASK_ROOT}/

COPY requirements.txt  .
# install our dependencies
RUN python3 -m pip install -r requirements.txt --target ${LAMBDA_TASK_ROOT}

#Copy files 
COPY BEATs.py .
COPY backbone.py .
COPY modules.py .
COPY labels.csv .

# Set the CMD to your handler 
CMD [ "app.lambda_handler"]