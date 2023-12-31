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
s3 = boto3.client('s3')
# Logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
#labels_json = None
BUCKET = os.environ['WEIGHTS_BUCKET']
KEY = os.environ['MODEL_NAME']



def download_model(bucket='', key=''):
    location = f'/tmp/{os.path.basename(key)}'
    try:
        if not os.path.exists(location):
            s3_resource = boto3.resource('s3')
            s3_resource.Object(bucket, key).download_file(location)
            logger.info(f"Model downloaded and saved at: {location}")
        else:
            logger.info("Model already exists locally.")
    except Exception as e:
        logger.info(f"Error downloading the model: {e}")


    return location


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
    logger.info(f"Sample rate: {original_sr}")

    resampled_waveform = torchaudio.transforms.Resample(original_sr, 16000)(waveform)
    return resampled_waveform

def get_label(label_pred):
    logger.info("Get Label")
    """
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
    logger.info(f"Labels: {filtered_labels}")

    logger.info(f"Json Loaded, continue with result dict")
    result_dict = {label_name: value for label_name, value in zip(filtered_labels.values(), values_list[0])}
    json_data = json.dumps(result_dict)
    
    return json_data
    """

    # Get final label
    index_list = label_pred[1][0].tolist() 
    logger.info(f"Top indexes: {index_list}")

    for value in range(len(index_list)):
          if index_list[value] in [20, 404, 520, 151, 515, 522, 429, 199, 50, 433, 344, 34, 413, 244, 155, 245, 242]:
            return "Conversacion", 202
          elif index_list[value] in [284, 19, 473, 498, 395, 81, 431, 62, 410]:
            return "Bebe Llorando", 200
          elif index_list[value] in [323, 149, 339, 480, 488, 400, 150, 157]:
            return "Ladrido", 201
          elif index_list[value] in [335, 221, 336, 277]:
            return "Maullido", 202
          elif value == 4:
            return "No value", 100


def lambda_handler(event, context):
    request_id = None

    # Attempt to retrieve the AWS request ID from the context
    try:
        request_id = context.aws_request_id
        file_key = event['Records'][0]['s3']['object']['key']
    except AttributeError:
        pass

    # Download model
    model_path = download_model(bucket=BUCKET, key=KEY)
    # Load model
    model = model_load(model_path)
    # Deal with Audio
    audio_path = download_audio(event)
    data = pre_process(audio_path)
    logger.info("Data Ready")
    # Classify Audio
    try:
        with torch.no_grad():
            prediction = model.extract_features(data, padding_mask=None)[0]
        label_pred = prediction.topk(k=5)
        label, code = get_label(label_pred)
        logger.info(f"Label: {label}")

        return {
        
            'request_id' : request_id,
            'records_s3_object_key': file_key,
            'classification_sound_id': f'{code}',
            'classification_sound_description': label,
        }
    except:
        return {
            'statusCode': 404,
            'class': None
        }
    



