# Python Built-Ins:
import logging
import sys
import os
import json


# External Dependencies:
import torch
import torchaudio
import boto3

# Local Dependencies:
from BEATs import BEATs, BEATsConfig
#s3_resource = boto3.resource('s3')
s3 = boto3.client('s3')
# Logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
labels_json = None


def model_load(model_path):
    global model
    if model is not None:
        logger.info("Model already loaded")
    else:
        logger.info("Loading Model")
        checkpoint = torch.load(model_path)
        cfg = BEATsConfig(checkpoint['cfg'])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval()
    return model

def download_audio(event):
    logger.info("Download Audio")
    input_bucket_name  = event['Records'][0]['s3']['bucket']['name']
    file_key = event['Records'][0]['s3']['object']['key']
    #local_input_temp_file = "/tmp/" + file_key
    local_input_temp_file = "/tmp/" + file_key.replace('/','-')
    logger.info(f"File_name: {local_input_temp_file}")
    s3.download_file(input_bucket_name, file_key, local_input_temp_file)
    audio_path = local_input_temp_file
    return audio_path

def pre_process(audio_path):
    logger.info("Pre-process")
    torchaudio.set_audio_backend("soundfile")
    waveform, original_sr = torchaudio.load(audio_path)
    resampled_waveform = torchaudio.transforms.Resample(original_sr, 16000)(waveform)
    return resampled_waveform

def get_label(label_pred):
    logger.info("Get Label")
    global labels_json
    if labels_json is not None:
        logger.info("Labels Already Loaded")
    else:
        with open("labels.json", "r") as f:
            logger.info("Reading JSON")
            json_dict = json.load(f)

    values_list = label_pred[0].tolist()
    indices_list = label_pred[1][0].tolist() 
    logger.info(f"Index List: {indices_list} ")

    filtered_labels = {str(index): json_dict[str(index)] for index in indices_list}

    logger.info(f"Json Loaded, continue with result dict")
    result_dict = {label_name: value for label_name, value in zip(filtered_labels.values(), values_list[0])}
    json_data = json.dumps(result_dict)
    
    return json_data
    

def lambda_handler(event, context):
    # Load model
    model = model_load(os.path.join(os.environ['LAMBDA_TASK_ROOT'], os.environ['MODEL_NAME']))
    # Deal with Audio
    audio_path = download_audio(event)
    data = pre_process(audio_path)
    logger.info("Data Ready")
    # Classify Audio
    try:
        with torch.no_grad():
            prediction = model.extract_features(data, padding_mask=None)[0]
        label_pred = prediction.topk(k=5)
        label = get_label(label_pred)
        logger.info(f"Label: {label}")


        return {
            'statusCode': 200,
            'class': label
        }
    except:
        return {
            'statusCode': 404,
            'class': None
        }