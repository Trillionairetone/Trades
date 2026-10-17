"""
kalshi_client.py - Kalshi Trade API v2 client (event markets).

What it handles for you:
  * Auth: signs every request (RSA-PSS/SHA-256 or Ed25519) over
    timestamp + METHOD + path. The path is signed WITHOUT the query string,
    which is the #1 cause of 401s in home-made clients.
  * Rate limits: local token buckets that mirror Kalshi's Read / Write /
    Balance buckets (Basic tier by default), so the bot slows itself down
    instead of collecting 429s.
  * Retries: exponential backoff + jitter on 429, 5xx and network errors;
    one retry on 401 (clock blips). Order placement is only retried when it
    is safe: every order carries a client_order_id, and before re-sending
    the client checks whether the first attempt actually reached Kalshi.
  * Money: prices and counts use Decimal and Kalshi's fixed-point strings
    ("0.5600", "10.00"), never floats.
  * Orders: V2 endpoints (/portfolio/events/orders). V2 quotes everything
    from the YES side: bid = buy YES, ask = sell YES (= buy NO at 1-price).
    buy_yes() / buy_no() hide that conversion.

Environment variables:
  KALSHI_ENV               demo (default) | prod
  KALSHI_API_KEY_ID        the key ID shown in Kalshi's API settings
  KALSHI_PRIVATE_KEY       full PEM text (Railway accepts multi-line values;
                           literal "\\n" sequences are also accepted)
  KALSHI_PRIVATE_KEY_B64   alternative: base64 of the PEM file
  KALSHI_PRIVATE_KEY_PATH  alternative: path to the PEM file
  KALSHI_READ_RATE / KALSHI_WRITE_RATE   tokens per second (default 200/100 = Basic)

Requires: requests, cryptography
"""
from __future__ import annotations

import base64
import logging
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlencode, urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

log = logging.getLogger("kalshi")

BASE_URLS = {
    "prod": "https://api.elections.kalshi.com/trade-api/v2",
    "demo": "https://demo-api.kalshi.co/trade-api/v2",
}

CENT = Decimal("0.01")
ONE = Decimal("1")
ZERO = Decimal("0")


# --------------------------------------------------------------------------
# Fixed-point helpers
# --------------------------------------------------------------------------
def D(x: Any) -> Decimal:
    """Parse Kalshi numbers (fixed-point strings, ints, floats) into Decimal."""
    if x is None or x == "":
        return ZERO
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


def price_str(p: Decimal) -> str:
    """Dollar price as Kalshi wants it, e.g. Decimal('0.56') -> '0.5600'."""
    return f"{D(p).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)}"


def count_str(c: Any) -> str:
    """Contract count as fixed-point, e.g. 10 -> '10.00'."""
    return f"{D(c).quantize(CENT, rounding=ROUND_DOWN)}"


def fee_estimate(contracts: Any, price: Any, coefficient: Any = "0.07") -> Decimal:
    """
    Kalshi's published trading-fee formula: ceil_to_cent(coef * C * P * (1-P)).
    Taker coefficient is 0.07 on most markets; maker is lower (often 0.0175)
    or zero. The real fee is reported back on fills - this is for planning.
    It matches the fees in your Clean Odds log (1 @ 0.50 -> $0.02,
    38 @ 0.50 -> $0.67).
    """
    c, p, k = D(contracts), D(price), D(coefficient)
    raw = k * c * p * (ONE - p)
    return raw.quantize(CENT, rounding=ROUND_CEILING)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class KalshiError(Exception):
    def __init__(self, status: int, message: str, code: str = "", body: Any = None):
        super().__init__(f"HTTP {status} {code}: {message}".strip())
        self.status = status
        self.code = code
        self.message = message
        self.body = body


# --------------------------------------------------------------------------
# Signing
# --------------------------------------------------------------------------
def normalize_pem(text: str) -> bytes:
    """
    Rebuild a PEM private key that was mangled by copy/paste, e.g. on iPhone:
    line breaks collapsed into spaces, literal "\\n", wrapping quotes,
    "smart" dashes (— –) instead of -----, or BEGIN/END lines missing.
    Never logs key material.
    """
    import re
    t = text.replace("\\n", "\n").strip().strip('"').strip("'")
    for ch in ("\u2014", "\u2013", "\u2012", "\u2212"):   # em/en/figure dash, minus
        t = t.replace(ch, "-")
    m = re.search(r"BEGIN ([A-Z ]*?PRIVATE KEY)-*(.*?)-*END \1", t, re.S)
    if m:
        label, body = m.group(1).strip(), m.group(2)
    else:  # no header/footer: assume the bare base64 body of Kalshi's key
        label, body = "", re.sub(r"-+[A-Z ]*-+", "", t)
    body = re.sub(r"[^A-Za-z0-9+/=]", "", body)
    # Ed25519 keys are short (64 base64 chars); RSA-2048 keys are ~1600.
    if len(body) < 60:
        raise ValueError("KALSHI_PRIVATE_KEY looks empty or truncated "
                         f"({len(body)} base64 characters found)")
    lines = "\n".join(body[i:i + 64] for i in range(0, len(body), 64))
    labels = [label] if label else ["RSA PRIVATE KEY", "PRIVATE KEY"]
    for lb in labels:
        pem = f"-----BEGIN {lb}-----\n{lines}\n-----END {lb}-----\n".encode()
        try:
            serialization.load_pem_private_key(pem, password=None)
            return pem
        except ValueError:
            continue
    raise ValueError("KALSHI_PRIVATE_KEY could not be parsed even after cleanup "
                     f"({len(body)} base64 characters). Re-copy it from the .key file.")


class Signer:
    """Loads the private key once and signs request pre-sign strings."""

    def __init__(self, key_id: str, pem: bytes):
        if not key_id:
            raise ValueError("KALSHI_API_KEY_ID is empty")
        self.key_id = key_id.strip()
        self._key = serialization.load_pem_private_key(pem, password=None)
        if not isinstance(self._key, (rsa.RSAPrivateKey, ed25519.Ed25519PrivateKey)):
            raise ValueError("Private key must be RSA or Ed25519")

    @classmethod
    def from_env(cls) -> "Signer":
        key_id = os.getenv("KALSHI_API_KEY_ID", "")
        pem_text = os.getenv("KALSHI_PRIVATE_KEY", "")
        b64 = os.getenv("KALSHI_PRIVATE_KEY_B64", "")
        path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
        if pem_text:
            pem = normalize_pem(pem_text)
        elif b64:
            pem = base64.b64decode(b64)
        elif path:
            with open(path, "rb") as f:
                pem = f.read()
        else:
            raise ValueError(
                "No private key: set KALSHI_PRIVATE_KEY, KALSHI_PRIVATE_KEY_B64 "
                "or KALSHI_PRIVATE_KEY_PATH"
            )
        return cls(key_id, pem)

    def sign(self, timestamp_ms: str, method: str, path: str) -> str:
        msg = f"{timestamp_ms}{method.upper()}{path}".encode()
        if isinstance(self._key, rsa.RSAPrivateKey):
            sig = self._key.sign(
                msg,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.DIGEST_LENGTH),
                hashes.SHA256(),
            )
        else:
            sig = self._key.sign(msg)
        return base64.b64encode(sig).decode()

    def headers(self, method: str, path: str) -> Dict[str, str]:
        ts = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self.sign(ts, method, path),
        }


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------
class TokenBucket:
    """Thread-safe token bucket; acquire() blocks until the cost fits."""

    def __init__(self, rate: float, capacity: float):
        self.rate = float(rate)
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, cost: float = 10.0) -> None:
        cost = min(cost, self.capacity)
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= cost:
                    self.tokens -= cost
                    return
                wait = (cost - self.tokens) / self.rate
            time.sleep(wait)


# --------------------------------------------------------------------------
# Order book
# --------------------------------------------------------------------------
@dataclass
class Orderbook:
    """
    Kalshi returns BIDS ONLY for both sides. Asks are implied:
      YES ask = 1 - best NO bid,   NO ask = 1 - best YES bid.
    Levels are stored as (price, size) sorted best-first.
    """
    ticker: str
    yes_bids: List[Tuple[Decimal, Decimal]] = field(default_factory=list)
    no_bids: List[Tuple[Decimal, Decimal]] = field(default_factory=list)

    @classmethod
    def from_response(cls, ticker: str, data: Dict[str, Any]) -> "Orderbook":
        ob = data.get("orderbook_fp") or data.get("orderbook") or {}
        yes = ob.get("yes_dollars") or []
        no = ob.get("no_dollars") or []
        parse = lambda lv: sorted(((D(p), D(q)) for p, q in lv if D(q) > 0),
                                  key=lambda t: t[0], reverse=True)
        return cls(ticker, parse(yes), parse(no))

    @property
    def best_yes_bid(self) -> Optional[Decimal]:
        return self.yes_bids[0][0] if self.yes_bids else None

    @property
    def best_no_bid(self) -> Optional[Decimal]:
        return self.no_bids[0][0] if self.no_bids else None

    @property
    def yes_ask(self) -> Optional[Decimal]:
        return ONE - self.best_no_bid if self.best_no_bid is not None else None

    @property
    def no_ask(self) -> Optional[Decimal]:
        return ONE - self.best_yes_bid if self.best_yes_bid is not None else None

    def ask(self, side: str) -> Optional[Decimal]:
        return self.yes_ask if side == "yes" else self.no_ask

    def bid(self, side: str) -> Optional[Decimal]:
        return self.best_yes_bid if side == "yes" else self.best_no_bid

    def spread(self, side: str) -> Optional[Decimal]:
        a, b = self.ask(side), self.bid(side)
        return (a - b) if a is not None and b is not None else None

    def ask_levels(self, side: str) -> List[Tuple[Decimal, Decimal]]:
        """Asks for `side`, cheapest first, derived from the opposite bids."""
        opposite = self.no_bids if side == "yes" else self.yes_bids
        return [(ONE - p, q) for p, q in opposite]

    def fill_plan(self, side: str, max_contracts: Decimal, max_price: Decimal
                  ) -> Tuple[Decimal, Decimal]:
        """
        Walk the asks for `side` up to max_price.
        Returns (contracts fillable, volume-weighted avg price).
        """
        got, cost = ZERO, ZERO
        for price, size in self.ask_levels(side):
            if price > max_price or got >= max_contracts:
                break
            take = min(size, max_contracts - got)
            got += take
            cost += take * price
        return got, (cost / got if got > 0 else ZERO)


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------
class KalshiClient:
    MAX_RETRIES = 5

    def __init__(self, env: Optional[str] = None, signer: Optional[Signer] = None,
                 session: Optional[requests.Session] = None, timeout: float = 10.0,
                 public: bool = False):
        """public=True: no signing, for public market-data endpoints only."""
        self.env = (env or os.getenv("KALSHI_ENV", "demo")).lower().strip()
        if self.env not in BASE_URLS:
            raise ValueError(f"KALSHI_ENV must be demo or prod, got {self.env!r}")
        self.base = BASE_URLS[self.env]
        self.base_path = urlparse(self.base).path  # "/trade-api/v2"
        self.public = public
        self.signer = None if public else (signer or Signer.from_env())
        self.http = session or requests.Session()
        self.timeout = timeout
        read_rate = float(os.getenv("KALSHI_READ_RATE", "200"))
        write_rate = float(os.getenv("KALSHI_WRITE_RATE", "100"))
        # Basic tier: read bucket holds 3s of budget, write bucket holds 1s.
        self.read_bucket = TokenBucket(read_rate, read_rate * 3)
        self.write_bucket = TokenBucket(write_rate, write_rate)
        self.balance_bucket = TokenBucket(1.0, 1.0)  # GET /portfolio/balance: 1/s

    # ---------------- low level ----------------
    def _bucket_for(self, method: str, path: str) -> Tuple[TokenBucket, float]:
        if path.endswith("/portfolio/balance"):
            return self.balance_bucket, 1.0
        if method == "GET":
            return self.read_bucket, 10.0
        if method == "DELETE":
            return self.write_bucket, 2.0
        return self.write_bucket, 10.0

    def request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None,
                body: Optional[Dict[str, Any]] = None, retry_post: bool = False) -> Dict[str, Any]:
        """
        path is relative to the API base, e.g. "/portfolio/balance".
        POSTs are not retried unless retry_post=True (only set by create_order,
        which verifies before re-sending).
        """
        method = method.upper()
        clean = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
        url = self.base + path + (("?" + urlencode(clean, doseq=True)) if clean else "")
        sign_path = self.base_path + path  # NO query string in the signature
        bucket, cost = self._bucket_for(method, path)

        auth_retried = False
        for attempt in range(self.MAX_RETRIES + 1):
            bucket.acquire(cost)
            headers = self.signer.headers(method, sign_path) if self.signer else {}
            headers["Content-Type"] = "application/json"
            try:
                resp = self.http.request(method, url, headers=headers, json=body,
                                         timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                if method == "POST" and not retry_post:
                    raise
                if attempt >= self.MAX_RETRIES:
                    raise
                self._sleep_backoff(attempt, f"network error: {e}")
                continue

            if resp.status_code < 300:
                if not resp.content:
                    return {}
                try:
                    return resp.json()
                except ValueError:
                    return {"raw": resp.text}

            err = self._parse_error(resp)
            retryable = resp.status_code == 429 or resp.status_code >= 500
            if resp.status_code == 401 and not auth_retried:
                auth_retried, retryable = True, True
            if method == "POST" and not retry_post:
                retryable = retryable and resp.status_code == 429  # 429 = never reached engine
            if not retryable or attempt >= self.MAX_RETRIES:
                raise err
            self._sleep_backoff(attempt, str(err))
        raise RuntimeError("unreachable")

    @staticmethod
    def _parse_error(resp: requests.Response) -> KalshiError:
        try:
            data = resp.json()
        except ValueError:
            return KalshiError(resp.status_code, resp.text[:300])
        e = data.get("error", data)
        if isinstance(e, dict):
            return KalshiError(resp.status_code, e.get("message") or e.get("details") or str(e),
                               e.get("code", ""), data)
        return KalshiError(resp.status_code, str(e), "", data)

    @staticmethod
    def _sleep_backoff(attempt: int, why: str) -> None:
        delay = min(30.0, 0.5 * (2 ** attempt)) * (0.5 + random.random())
        log.warning("retry %d in %.1fs (%s)", attempt + 1, delay, why)
        time.sleep(delay)

    def paginate(self, path: str, key: str, params: Optional[Dict[str, Any]] = None,
                 max_items: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Cursor pagination; yields items under `key` across all pages."""
        params = dict(params or {})
        n = 0
        while True:
            data = self.request("GET", path, params)
            for item in data.get(key) or []:
                yield item
                n += 1
                if max_items and n >= max_items:
                    return
            cursor = data.get("cursor")
            if not cursor:
                return
            params["cursor"] = cursor

    # ---------------- exchange / account ----------------
    def exchange_status(self) -> Dict[str, Any]:
        return self.request("GET", "/exchange/status")

    def account_limits(self) -> Dict[str, Any]:
        return self.request("GET", "/account/limits")

    # ---------------- markets ----------------
    def get_market(self, ticker: str) -> Dict[str, Any]:
        return self.request("GET", f"/markets/{ticker}").get("market", {})

    def iter_markets(self, series_ticker: Optional[str] = None, status: Optional[str] = None,
                     event_ticker: Optional[str] = None, max_items: Optional[int] = None,
                     **extra: Any) -> Iterator[Dict[str, Any]]:
        params = {"series_ticker": series_ticker, "status": status,
                  "event_ticker": event_ticker, "limit": 1000, **extra}
        return self.paginate("/markets", "markets", params, max_items)

    def iter_historical_markets(self, series_ticker: str, max_items: Optional[int] = None
                                ) -> Iterator[Dict[str, Any]]:
        return self.paginate("/historical/markets", "markets",
                             {"series_ticker": series_ticker, "limit": 1000}, max_items)

    def get_orderbook(self, ticker: str, depth: int = 0) -> Orderbook:
        data = self.request("GET", f"/markets/{ticker}/orderbook", {"depth": depth or None})
        return Orderbook.from_response(ticker, data)

    # ---------------- portfolio ----------------
    def get_balance(self) -> Decimal:
        """Available cash in dollars."""
        data = self.request("GET", "/portfolio/balance")
        if data.get("balance_dollars") is not None:
            return D(data["balance_dollars"])
        return D(data.get("balance", 0)) / 100

    def get_positions(self, ticker: Optional[str] = None,
                      settlement_status: str = "unsettled") -> List[Dict[str, Any]]:
        params = {"ticker": ticker, "settlement_status": settlement_status,
                  "count_filter": "position", "limit": 1000}
        return list(self.paginate("/portfolio/positions", "market_positions", params))

    def get_orders(self, ticker: Optional[str] = None, status: Optional[str] = None,
                   max_items: Optional[int] = 1000) -> List[Dict[str, Any]]:
        params = {"ticker": ticker, "status": status, "limit": 1000}
        return list(self.paginate("/portfolio/orders", "orders", params, max_items))

    def get_fills(self, ticker: Optional[str] = None, min_ts: Optional[int] = None,
                  max_items: Optional[int] = 1000) -> List[Dict[str, Any]]:
        params = {"ticker": ticker, "min_ts": min_ts, "limit": 1000}
        return list(self.paginate("/portfolio/fills", "fills", params, max_items))

    def get_settlements(self, max_items: Optional[int] = 500) -> List[Dict[str, Any]]:
        return list(self.paginate("/portfolio/settlements", "settlements",
                                  {"limit": 1000}, max_items))

    def find_order_by_client_id(self, ticker: str, client_order_id: str
                                ) -> Optional[Dict[str, Any]]:
        for o in self.get_orders(ticker=ticker, max_items=200):
            if o.get("client_order_id") == client_order_id:
                return o
        return None

    # ---------------- orders (V2) ----------------
    def create_order(self, ticker: str, book_side: str, count: Any, price: Any,
                     time_in_force: str = "immediate_or_cancel",
                     post_only: bool = False, reduce_only: bool = False,
                     expiration_time: Optional[int] = None,
                     client_order_id: Optional[str] = None,
                     self_trade_prevention_type: str = "taker_at_cross",
                     cancel_order_on_pause: bool = True) -> Dict[str, Any]:
        """
        book_side: "bid" (buy YES) or "ask" (sell YES / buy NO), price in YES dollars.
        Idempotent: if the network drops after sending, the client looks the
        order up by client_order_id before deciding whether to re-send.
        """
        if book_side not in ("bid", "ask"):
            raise ValueError("book_side must be 'bid' or 'ask'")
        p = D(price)
        if not (ZERO < p < ONE):
            raise ValueError(f"price must be between 0 and 1, got {p}")
        cid = client_order_id or str(uuid.uuid4())
        body: Dict[str, Any] = {
            "ticker": ticker,
            "client_order_id": cid,
            "side": book_side,
            "count": count_str(count),
            "price": price_str(p),
            "time_in_force": time_in_force,
            "self_trade_prevention_type": self_trade_prevention_type,
            "post_only": post_only,
            "reduce_only": reduce_only,
            "cancel_order_on_pause": cancel_order_on_pause,
        }
        if expiration_time:
            body["expiration_time"] = int(expiration_time)

        for attempt in range(3):
            try:
                return self.request("POST", "/portfolio/events/orders", body=body)
            except (requests.ConnectionError, requests.Timeout, KalshiError) as e:
                ambiguous = not isinstance(e, KalshiError) or e.status >= 500
                if isinstance(e, KalshiError) and e.status == 409:
                    ambiguous = True  # duplicate client_order_id -> first attempt landed
                if not ambiguous or attempt == 2:
                    raise
                log.warning("order %s outcome unknown (%s); checking before re-send", cid, e)
                time.sleep(1.0 + attempt)
                existing = self.find_order_by_client_id(ticker, cid)
                if existing:
                    log.info("order %s did reach Kalshi; not re-sending", cid)
                    return {"order_id": existing.get("order_id"), "client_order_id": cid,
                            "recovered": True, "order": existing}
        raise RuntimeError("unreachable")

    def buy_yes(self, ticker: str, count: Any, max_price: Any, **kw: Any) -> Dict[str, Any]:
        return self.create_order(ticker, "bid", count, max_price, **kw)

    def buy_no(self, ticker: str, count: Any, max_no_price: Any, **kw: Any) -> Dict[str, Any]:
        # Buying NO at q == selling YES at (1 - q)
        return self.create_order(ticker, "ask", count, ONE - D(max_no_price), **kw)

    def cancel_order(self, order_id: str, market_ticker: str) -> Dict[str, Any]:
        return self.request("DELETE", f"/portfolio/events/orders/{order_id}",
                            {"market_ticker": market_ticker})


def market_close_ts(market: Dict[str, Any]) -> Optional[float]:
    """Close time as a unix timestamp (handles ISO strings and ints)."""
    from datetime import datetime
    v = market.get("close_time") or market.get("close_ts")
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) / (1000 if v > 1e12 else 1)
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
