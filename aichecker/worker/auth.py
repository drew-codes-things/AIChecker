import hmac
import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass

SECRET_HEADER = "X-Checker-Secret"
STATUS_FORBIDDEN = 403
STATUS_UNAUTHORIZED = 401


@dataclass(frozen=True)
class Refusal:
    status: int
    message: str


def _peer_is_allowed(peer: str, allowed_peers: tuple[str, ...]) -> bool:
    try:
        address = ipaddress.ip_address(peer)
    except ValueError:
        return False
    for entry in allowed_peers:
        if address in ipaddress.ip_network(entry, strict=False):
            return True
    return False


def check_request(peer: str, headers: Mapping[str, str], secret: bytes, allowed_peers: tuple[str, ...]) -> Refusal | None:
    if not _peer_is_allowed(peer or "", allowed_peers):
        return Refusal(STATUS_FORBIDDEN, "not allowed from this address")
    presented = (headers.get(SECRET_HEADER) or "").encode()
    if not presented or not hmac.compare_digest(presented, secret):
        return Refusal(STATUS_UNAUTHORIZED, "missing or wrong shared secret")
    return None
