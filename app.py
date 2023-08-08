# Python Built-Ins:
import pandas as pd
import logging
import sys
import os


# External Dependencies:
import torch
import torchaudio
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Local Dependencies:
from BEATs import BEATs, BEATsConfig
s3_resource = boto3.resource('s3')
s3 = boto3.client('s3')
# Logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_model(bucket='', key=''):
    logger.info("Download model")
    location = f'/tmp/{os.path.basename(key)}'
    if not os.path.exists(location):
        s3_resource.Object(bucket, key).download_file(location)
    logger.info(f"Model Location: {location} ")
    
    return location

def model_load(model_path):
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
    if type(label_pred) == list:
        label_pred = label_pred[0]
    df = pd.read_csv("labels.csv")
    result = df.loc[df["index"] == label_pred, "display_name"].item()
    return result

def lambda_handler(event, context):
    # Download model
    model_path = download_model(
        bucket='beats-data',
        key='data/BEATs/BEATs_iter3_plus_AS2M.pt')
    # Load model
    model = model_load(model_path=model_path)
    # Download .wav
    audio_path = download_audio(event)
    #Pre-process audio
    data = pre_process(audio_path)
    logger.info("Data Ready")

    # classify image
    try:
        with torch.no_grad():
            prediction = model.extract_features(data, padding_mask=None)[0]
            label_pred = prediction.topk(k=1)[1].tolist()[0][0]
            logger.info(f"Prediction Successfull: {label_pred}")
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