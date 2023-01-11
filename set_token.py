from huggingface_hub.hf_api import HfFolder

token_file = open('token.txt', 'r')
token = token_file.readlines()[0]

HfFolder.save_token(token)