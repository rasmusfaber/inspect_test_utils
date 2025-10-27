import datetime
import logging
import time
from typing import override

import httpx
import inspect_ai.hooks


def refresh_token_hook(
        refresh_url: str,
        client_id: str,
        refresh_token: str,
        refresh_delta_seconds: int = 600,
) -> type[inspect_ai.hooks.Hooks]:
    logger = logging.getLogger("hawk.refresh_token_hook")

    class RefreshTokenHook(inspect_ai.hooks.Hooks):
        _current_expiration_time: float | None = None
        _current_access_token: str | None = None

        def _perform_token_refresh(
                self,
        ) -> None:
            logger.debug("Refreshing access token")
            with httpx.Client() as http_client:
                response = http_client.post(
                    url=refresh_url,
                    headers={
                        "accept": "application/json",
                        "content-type": "application/x-www-form-urlencoded",
                    },
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                        "client_id": client_id,
                    },
                )
                response.raise_for_status()
                data = response.json()
            self._current_access_token = data["access_token"]
            self._current_expiration_time = (
                    time.time() + data["expires_in"] - refresh_delta_seconds
            )
            logger.info(
                "Refreshed access token. New expiration time: %s",
                datetime.datetime.fromtimestamp(
                    self._current_expiration_time, tz=datetime.timezone.utc
                ).isoformat(timespec="seconds")
                if self._current_expiration_time
                else "None",
            )

        @override
        def override_api_key(self, data: inspect_ai.hooks.ApiKeyOverride) -> str | None:
            if not self._is_current_access_token_valid():
                self._perform_token_refresh()

            return self._current_access_token

        def _is_current_access_token_valid(self) -> bool:
            now = time.time()
            return (
                    self._current_access_token is not None
                    and self._current_expiration_time is not None
                    and self._current_expiration_time > now
            )

    return RefreshTokenHook
