import os
from dotenv import load_dotenv

env = load_dotenv()
# env = load_dotenv('env.example')

class ApplicationConfig:
    MODEL_BASE_CONF = float(os.environ.get('MODEL_BASE_CONF'))
    MODEL_BASE_IOU = float(os.environ.get('MODEL_BASE_IOU'))
    MODEL_BASE_INPUTSIZE = int(os.environ.get('MODEL_BASE_INPUTSIZE'))

app_config = ApplicationConfig()