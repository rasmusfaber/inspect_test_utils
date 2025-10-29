import datetime
import logging
import os
import time

import anthropic
import httpx
import inspect_ai.hooks
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput, ModelCall, modelapi, ModelAPI
from inspect_ai.model._providers.anthropic import AnthropicAPI
from inspect_ai.model._providers.openai import OpenAIAPI
from inspect_ai.tool import ToolInfo, ToolChoice
from joserfc import jwk, jwt
from openai import APIStatusError

from inspect_test_utils import hooks

log = logging.getLogger(__name__)

inspect_ai.hooks.hooks("refresh_token", "refresh jwt")(
    hooks.refresh_token_hook(
        refresh_url="https://metr.okta.com/oauth2/aus1ww3m0x41jKp3L1d8/v1/token",
        client_id="0oa1wxy3qxaHOoGxG1d8",
        refresh_token=os.getenv("REFRESH_TOKEN"),
        refresh_delta_seconds=99999999,
    )
)


class FailingOpenAI(OpenAIAPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _check_api_key(self):
        api_key = self.client.api_key
        try:
            key_set = jwk.KeySet.import_key_set({"keys": [
                {"kty": "RSA", "alg": "RS256", "kid": "Inp_CcloTJyZYGzUH1lVSM7rS6MMAp2f4VSsv1Bcyvs", "use": "sig",
                 "e": "AQAB",
                 "n": "yINya8skpnL3eEelENTNVu9NsI79-YvkjbBFESp_10I57BcmCUYqn79WmM4R-566Le1pet5kJEXs6sj48MBB66JkhmrQ0ybTLCU_5kKITLjVqXtDbyzdxTu_FcEz0bWB66xEBeCuuT6wPGh57s5dgQXhxHPrkn-TGWl1bHMGmBplGkTAa3IqgqVl08lBMMKMdN77qLCySEeE1RLihRfq4DfhF-Nczt14ZzV8m1kcIjTm9dBqd8SHyXNO0x43HxUJE23sXMECWlLc8y8oGCXMZZ65lj5ccU7R0gD65geg4RfxtRHb36Py43_j1QUl8jZVAvfYHxFR0j6xPLHPslymVw"}]})
            decoded_token = jwt.decode(api_key, key=key_set)
            exp = decoded_token.claims.get("exp")
            log.info("Decoded JWT expiration: %s", datetime.datetime.fromtimestamp(
                exp, tz=datetime.timezone.utc
            ).isoformat(timespec="seconds"))
            modified_exp = exp - 24 * 60 * 60 + 30
            log.info("Modified JWT expiration: %s", datetime.datetime.fromtimestamp(
                modified_exp, tz=datetime.timezone.utc
            ))
        except Exception as e:
            log.error("Failed to decode API key as JWT: %s", e)
            return
        if modified_exp < time.time():
            raise APIStatusError("expired", response=httpx.Response(status_code=401, request=httpx.Request("POST", "http://example.org")), body={"error": "Invalid API key"})

    async def generate(self, input: list[ChatMessage], tools: list[ToolInfo], tool_choice: ToolChoice,
                       config: GenerateConfig) -> ModelOutput | tuple[ModelOutput | Exception, ModelCall]:
        self._check_api_key()
        return await super().generate(input, tools, tool_choice, config)


class FailingAnthropic(AnthropicAPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _check_api_key(self):
        api_key = self.client.api_key
        try:
            key_set = jwk.KeySet.import_key_set({"keys": [
                {"kty": "RSA", "alg": "RS256", "kid": "Inp_CcloTJyZYGzUH1lVSM7rS6MMAp2f4VSsv1Bcyvs", "use": "sig",
                 "e": "AQAB",
                 "n": "yINya8skpnL3eEelENTNVu9NsI79-YvkjbBFESp_10I57BcmCUYqn79WmM4R-566Le1pet5kJEXs6sj48MBB66JkhmrQ0ybTLCU_5kKITLjVqXtDbyzdxTu_FcEz0bWB66xEBeCuuT6wPGh57s5dgQXhxHPrkn-TGWl1bHMGmBplGkTAa3IqgqVl08lBMMKMdN77qLCySEeE1RLihRfq4DfhF-Nczt14ZzV8m1kcIjTm9dBqd8SHyXNO0x43HxUJE23sXMECWlLc8y8oGCXMZZ65lj5ccU7R0gD65geg4RfxtRHb36Py43_j1QUl8jZVAvfYHxFR0j6xPLHPslymVw"}]})
            decoded_token = jwt.decode(api_key, key=key_set)
            exp = decoded_token.claims.get("exp")
            log.info("Decoded JWT expiration: %s", datetime.datetime.fromtimestamp(
                exp, tz=datetime.timezone.utc
            ).isoformat(timespec="seconds"))
            modified_exp = exp - 24 * 60 * 60 + 30
            log.info("Modified JWT expiration: %s", datetime.datetime.fromtimestamp(
                modified_exp, tz=datetime.timezone.utc
            ))
        except Exception as e:
            log.error("Failed to decode API key as JWT: %s", e)
            return
        if modified_exp < time.time():
            raise anthropic.APIStatusError("expired", response=httpx.Response(status_code=401, request=httpx.Request("POST", "http://example.org")), body={"error": "Invalid API key"})

    async def generate(self, input: list[ChatMessage], tools: list[ToolInfo], tool_choice: ToolChoice,
                       config: GenerateConfig) -> ModelOutput | tuple[ModelOutput | Exception, ModelCall]:
        self._check_api_key()
        return await super().generate(input, tools, tool_choice, config)


@modelapi(name="failing_openai")
def failing_openai() -> type[ModelAPI]:
    return FailingOpenAI

@modelapi(name="failing_anthropic")
def failing_anthropic() -> type[ModelAPI]:
    return FailingAnthropic
