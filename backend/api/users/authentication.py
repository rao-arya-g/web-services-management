import jwt
from django.conf import settings
from rest_framework import authentication, exceptions
from rest_framework.request import Request
from .models import AuthUser
from .helper import decode_token


class CustomAuthentication(authentication.BaseAuthentication):
    def authenticate(self, request: Request):

        token = request.headers.get('Authorization')
        if not token:
            raise exceptions.AuthenticationFailed('Token is missing for authentication')

        try:
            payload = decode_token(token, settings.SECRET_KEY)
            user = AuthUser.objects.filter(email=payload["email"]).first()

        except jwt.InvalidTokenError:
            raise exceptions.AuthenticationFailed('Invalid token')

        return user, None
