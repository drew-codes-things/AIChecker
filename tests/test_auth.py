import pytest

from aichecker.worker.auth import SECRET_HEADER, check_request

SECRET = b"s" * 40
APP = ("192.0.2.16",)


def ok_headers():
    return {SECRET_HEADER: SECRET.decode()}


def test_the_app_with_the_right_secret_is_accepted():
    assert check_request("192.0.2.16", ok_headers(), SECRET, APP) is None


def test_a_missing_header_is_401():
    refusal = check_request("192.0.2.16", {}, SECRET, APP)
    assert refusal.status == 401


def test_a_wrong_secret_is_401():
    refusal = check_request("192.0.2.16", {SECRET_HEADER: "x" * 40}, SECRET, APP)
    assert refusal.status == 401


def test_an_empty_secret_header_is_401():
    assert check_request("192.0.2.16", {SECRET_HEADER: ""}, SECRET, APP).status == 401


@pytest.mark.parametrize("peer", ["192.0.2.17", "127.0.0.1", "192.0.2.13", "10.0.0.1", "::1", "not-an-address", ""])
def test_any_other_address_is_403_even_with_the_right_secret(peer):
    assert check_request(peer, ok_headers(), SECRET, APP).status == 403


def test_the_address_check_comes_before_the_secret_check_so_strangers_learn_nothing():
    assert check_request("192.0.2.17", {}, SECRET, APP).status == 403


def test_a_network_entry_allows_every_address_in_it():
    assert check_request("192.0.2.99", ok_headers(), SECRET, ("192.0.2.0/24",)) is None
    assert check_request("198.51.100.99", ok_headers(), SECRET, ("192.0.2.0/24",)).status == 403
