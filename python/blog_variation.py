"""
블로그 콘텐츠 변형(variation) 엔진.

목적: 매일 수동 업로드하는 글의 제목/도입/문구/구조가 너무 비슷하면
네이버 등에서 유사문서(저품질)로 분류될 수 있다. 이를 완화하기 위해
- 날짜+유형+플랫폼(+종목코드)을 시드로 문구 풀에서 결정적으로 선택한다.
- 같은 날 같은 입력이면 항상 같은 결과(재현성, 멱등성) → 재생성해도 동일.
- 날짜가 바뀌면 시드가 바뀌어 제목/도입/연결어/구조가 자연스럽게 달라진다.
- 네이버/티스토리는 살트가 달라 같은 날이라도 서로 다른 표현이 나온다(교차 중복 방지).

면책 고지처럼 법적 일관성이 필요한 문구는 변형하지 않는다(여기서 다루지 않음).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_POOL_PATH = _ROOT / "templates" / "blog" / "variations.json"


def _stable_index(seed: str, n: int) -> int:
    """문자열 시드를 0..n-1 범위의 결정적 정수로 변환."""
    if n <= 0:
        raise ValueError("pool is empty")
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest, 16) % n


class VariationPicker:
    """
    시드(날짜·유형·플랫폼)를 고정한 변형 선택기.

    pick("title")        → title 풀에서 시드 기반으로 하나 선택
    pick("title", "x")   → 추가 살트 "x"를 더해 선택(슬롯별로 다른 항목)
    pick_for(code, key)  → 종목코드까지 시드에 더해 종목마다 다른 표현
    """

    def __init__(self, source_date: str, content_type: str, platform: str,
                 pool_path: Path | None = None) -> None:
        self.source_date = source_date
        self.content_type = content_type
        self.platform = platform
        path = pool_path or _DEFAULT_POOL_PATH
        with open(path, "r", encoding="utf-8") as fh:
            self._pools: dict[str, Any] = json.load(fh)
        self._base_seed = f"{source_date}|{content_type}|{platform}"

    # ---- 내부 ----
    def _resolve_pool(self, key: str) -> list[str]:
        """'title.watch' 같은 점 표기 또는 단일 키를 리스트로 해석."""
        node: Any = self._pools
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"variation pool not found: {key}")
            node = node[part]
        if isinstance(node, dict):
            raise KeyError(f"variation pool '{key}' is a group, not a list")
        if not isinstance(node, list):
            return [str(node)]
        return node

    def _pick_from(self, pool: list[str], *salts: str) -> str:
        seed = "|".join([self._base_seed, *salts])
        return pool[_stable_index(seed, len(pool))]

    # ---- 공개 ----
    def pick(self, key: str, *salts: str) -> str:
        """풀 key에서 시드 기반으로 한 항목 선택."""
        return self._pick_from(self._resolve_pool(key), key, *salts)

    def pick_for(self, code: str, key: str, *salts: str) -> str:
        """종목코드를 시드에 더해 종목별로 다른 항목 선택."""
        return self._pick_from(self._resolve_pool(key), key, code, *salts)

    def pick_group(self, group_key: str, subkey: str, *salts: str) -> str:
        """그룹 맵에서 subkey 풀(없으면 _default)을 골라 시드 기반 선택."""
        node = self._pools.get(group_key, {})
        pool = node.get(subkey) or node.get("_default")
        if not pool:
            raise KeyError(f"group pool not found: {group_key}.{subkey}")
        return self._pick_from(pool, group_key, subkey, *salts)

    def choice_index(self, n: int, *salts: str) -> int:
        """0..n-1 범위의 결정적 인덱스(문장 패턴 분기 등에 사용)."""
        seed = "|".join([self._base_seed, *salts])
        return _stable_index(seed, n)

    def pick_n(self, key: str, n: int, *salts: str) -> list:
        """풀에서 중복 없이 n개를 시드 기반으로 선택(날짜별로 회전)."""
        pool = self._resolve_pool(key)
        if not pool:
            return []
        seed = "|".join([self._base_seed, key, *salts])
        start = _stable_index(seed, len(pool))
        count = min(n, len(pool))
        return [pool[(start + i) % len(pool)] for i in range(count)]

    def map_value(self, group_key: str, value: str, default_key: str = "_default") -> str:
        """regime_word 등 매핑 그룹에서 값 변환(없으면 default 또는 원값)."""
        node = self._pools.get(group_key, {})
        if isinstance(node, dict):
            if value in node:
                return node[value]
            if default_key in node:
                return node[default_key]
        return value

    def reasons_kr(self, codes: Iterable[str]) -> list[str]:
        """영문 사유 코드 리스트를 한글 문구로 변환.

        1) 정확 매핑(reason_kr) → 2) 부분 매핑(reason_kr_contains) →
        3) 그래도 영문이 남으면(아스키 문자 위주) 제외해 블로그에 영문 노출 방지.
        """
        exact = self._pools.get("reason_kr", {})
        contains = self._pools.get("reason_kr_contains", [])
        out: list[str] = []
        for c in codes:
            if c in exact:
                out.append(exact[c])
                continue
            matched = None
            for needle, kr in contains:
                if needle in c:
                    matched = kr
                    break
            if matched:
                out.append(matched)
            elif self._is_mostly_ascii(c):
                # 한글로 변환되지 않은 영문 코드 → 노출하지 않음
                continue
            else:
                out.append(c)
        # 중복 제거(순서 유지)
        seen: set[str] = set()
        deduped: list[str] = []
        for r in out:
            if r not in seen:
                seen.add(r)
                deduped.append(r)
        return deduped

    @staticmethod
    def _is_mostly_ascii(text: str) -> bool:
        letters = [ch for ch in text if ch.isalpha()]
        if not letters:
            return False
        ascii_letters = [ch for ch in letters if ord(ch) < 128]
        return len(ascii_letters) / len(letters) > 0.5

    def as_jinja_globals(self) -> dict[str, Any]:
        """Jinja2 환경에 주입할 함수 묶음."""
        return {
            "pick": self.pick,
            "pick_for": self.pick_for,
            "map_value": self.map_value,
            "reasons_kr": self.reasons_kr,
        }
