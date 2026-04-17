import logging
import os
import time
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


DEFAULT_TIMEOUT_SEC = 20
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF_BASE_SEC = 1.5
DEFAULT_RETRY_BACKOFF_MAX_SEC = 5.0


class KISError(Exception):
    """Base exception for KIS client failures."""


class KISAuthError(KISError):
    """Authentication/token issuance failure."""


class KISHTTPError(KISError):
    """Transport-level or HTTP status failure."""


class KISBusinessError(KISError):
    """KIS business-level error returned in a 200 response."""


@dataclass
class KISResponse:
    data: Dict[str, Any]
    headers: Dict[str, str]
    status_code: int


@dataclass
class KISClient:
    base_url: str
    app_key: str
    app_secret: str
    timeout_sec: int = DEFAULT_TIMEOUT_SEC
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_backoff_base_sec: float = DEFAULT_RETRY_BACKOFF_BASE_SEC
    retry_backoff_max_sec: float = DEFAULT_RETRY_BACKOFF_MAX_SEC

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._session = requests.Session()
        self._access_token: Optional[str] = None

    @classmethod
    def from_env(cls) -> "KISClient":
        if load_dotenv:
            load_dotenv()
        base_url = os.getenv("KIS_BASE_URL")
        app_key = os.getenv("KIS_APP_KEY")
        app_secret = os.getenv("KIS_APP_SECRET")
        if not all([base_url, app_key, app_secret]):
            raise ValueError("KIS env missing or incomplete")
        timeout_sec = max(5, int(os.getenv("KIS_TIMEOUT_SEC", str(DEFAULT_TIMEOUT_SEC))))
        max_retries = max(0, int(os.getenv("KIS_MAX_RETRIES", str(DEFAULT_MAX_RETRIES))))
        retry_backoff_base_sec = max(0.5, float(os.getenv("KIS_RETRY_BACKOFF_BASE_SEC", str(DEFAULT_RETRY_BACKOFF_BASE_SEC))))
        retry_backoff_max_sec = max(
            retry_backoff_base_sec,
            float(os.getenv("KIS_RETRY_BACKOFF_MAX_SEC", str(DEFAULT_RETRY_BACKOFF_MAX_SEC))),
        )
        return cls(
            base_url=base_url,
            app_key=app_key,
            app_secret=app_secret,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            retry_backoff_base_sec=retry_backoff_base_sec,
            retry_backoff_max_sec=retry_backoff_max_sec,
        )

    def issue_access_token(self, *, force_refresh: bool = False) -> str:
        if self._access_token and not force_refresh:
            return self._access_token

        url = f"{self.base_url}/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        headers = {"Content-Type": "application/json"}

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_sec,
                )
                if response.status_code != 200:
                    raise KISAuthError(
                        f"tokenP failed: status={response.status_code} body={response.text[:500]}"
                    )
                data = response.json()
                access_token = data.get("access_token")
                if not access_token:
                    raise KISAuthError(f"tokenP missing access_token: {data}")
                self._access_token = access_token
                return access_token
            except (requests.RequestException, ValueError, KISAuthError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self._sleep_before_retry(attempt)

        raise KISAuthError(f"Failed to issue KIS access token: {last_error}") from last_error

    def build_headers(
        self,
        *,
        tr_id: str,
        access_token: Optional[str] = None,
        tr_cont: str = "",
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        token = access_token or self.issue_access_token()
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
        }
        if tr_cont:
            headers["tr_cont"] = tr_cont
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def get(
        self,
        path: str,
        *,
        tr_id: str,
        params: Optional[Dict[str, Any]] = None,
        tr_cont: str = "",
        timeout_sec: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.get_with_meta(
            path,
            tr_id=tr_id,
            params=params,
            tr_cont=tr_cont,
            timeout_sec=timeout_sec,
        ).data

    def get_with_meta(
        self,
        path: str,
        *,
        tr_id: str,
        params: Optional[Dict[str, Any]] = None,
        tr_cont: str = "",
        timeout_sec: Optional[int] = None,
    ) -> KISResponse:
        url = f"{self.base_url}{path}"
        timeout = timeout_sec or self.timeout_sec
        last_error: Optional[Exception] = None
        refreshed_token = False

        for attempt in range(self.max_retries + 1):
            try:
                headers = self.build_headers(tr_id=tr_id, tr_cont=tr_cont)
                response = self._session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=timeout,
                )

                if response.status_code in (401, 403) and not refreshed_token:
                    logging.info("KIS auth appears expired; refreshing token and retrying")
                    self.issue_access_token(force_refresh=True)
                    refreshed_token = True
                    continue

                if response.status_code >= 400:
                    raise KISHTTPError(
                        f"KIS GET failed: path={path} status={response.status_code} body={response.text[:500]}"
                    )

                data = response.json()
                self._raise_for_business_error(path=path, data=data)
                return KISResponse(
                    data=data,
                    headers=dict(response.headers),
                    status_code=response.status_code,
                )
            except (requests.RequestException, ValueError, KISHTTPError, KISBusinessError) as exc:
                last_error = exc
                if not self._should_retry(exc=exc, attempt=attempt):
                    break
                self._sleep_before_retry(attempt)

        if isinstance(last_error, KISError):
            raise last_error
        raise KISHTTPError(f"KIS GET failed unexpectedly for {path}: {last_error}") from last_error

    def request_hashkey(self, payload: Dict[str, Any]) -> str:
        url = f"{self.base_url}/uapi/hashkey"
        headers = {
            "Content-Type": "application/json",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        response = self._session.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=self.timeout_sec,
        )
        if response.status_code != 200:
            raise KISHTTPError(
                f"KIS hashkey failed: status={response.status_code} body={response.text[:500]}"
            )
        data = response.json()
        hashkey = data.get("HASH")
        if not hashkey:
            raise KISBusinessError(f"KIS hashkey missing HASH field: {data}")
        return str(hashkey)

    def post(
        self,
        path: str,
        *,
        tr_id: str,
        payload: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        require_hashkey: bool = False,
        timeout_sec: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.post_with_meta(
            path,
            tr_id=tr_id,
            payload=payload,
            extra_headers=extra_headers,
            require_hashkey=require_hashkey,
            timeout_sec=timeout_sec,
        ).data

    def post_with_meta(
        self,
        path: str,
        *,
        tr_id: str,
        payload: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        require_hashkey: bool = False,
        timeout_sec: Optional[int] = None,
    ) -> KISResponse:
        url = f"{self.base_url}{path}"
        timeout = timeout_sec or self.timeout_sec
        body = payload or {}
        last_error: Optional[Exception] = None
        refreshed_token = False

        for attempt in range(self.max_retries + 1):
            try:
                headers = self.build_headers(tr_id=tr_id, extra_headers=extra_headers)
                headers["custtype"] = headers.get("custtype", "P")
                if require_hashkey:
                    headers["hashkey"] = self.request_hashkey(body)
                response = self._session.post(
                    url,
                    headers=headers,
                    data=json.dumps(body),
                    timeout=timeout,
                )

                if response.status_code in (401, 403) and not refreshed_token:
                    logging.info("KIS auth appears expired for POST; refreshing token and retrying")
                    self.issue_access_token(force_refresh=True)
                    refreshed_token = True
                    continue

                if response.status_code >= 400:
                    raise KISHTTPError(
                        f"KIS POST failed: path={path} status={response.status_code} body={response.text[:500]}"
                    )

                data = response.json()
                self._raise_for_business_error(path=path, data=data)
                return KISResponse(
                    data=data,
                    headers=dict(response.headers),
                    status_code=response.status_code,
                )
            except (requests.RequestException, ValueError, KISHTTPError, KISBusinessError) as exc:
                last_error = exc
                if not self._should_retry(exc=exc, attempt=attempt):
                    break
                self._sleep_before_retry(attempt)

        if isinstance(last_error, KISError):
            raise last_error
        raise KISHTTPError(f"KIS POST failed unexpectedly for {path}: {last_error}") from last_error

    def _raise_for_business_error(self, *, path: str, data: Dict[str, Any]) -> None:
        rt_cd = str(data.get("rt_cd", "")).strip()
        msg1 = data.get("msg1")
        msg_cd = data.get("msg_cd")
        if rt_cd and rt_cd not in {"0", "0000"}:
            raise KISBusinessError(
                f"KIS business error: path={path} rt_cd={rt_cd} msg_cd={msg_cd} msg1={msg1}"
            )

    def _should_retry(self, *, exc: Exception, attempt: int) -> bool:
        if attempt >= self.max_retries:
            return False
        if isinstance(exc, KISBusinessError):
            return False
        return True

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = min(self.retry_backoff_base_sec * (attempt + 1), self.retry_backoff_max_sec)
        logging.info("KIS retry backoff %.1fs (attempt=%d/%d)", delay, attempt + 1, self.max_retries + 1)
        time.sleep(delay)
