import jwt
from constants import ENCODING_DECODING_ALGO


def encode_token(email, key):
    token = jwt.encode(payload={"email": email}, key=key, algorithm=[ENCODING_DECODING_ALGO])
    return token


def decode_token(token, key):
    payload = jwt.decode(jwt=token, key=key, algorithms=[ENCODING_DECODING_ALGO])
    return payload
