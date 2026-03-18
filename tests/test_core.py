"""Tests for Voicetwin."""
from src.core import Voicetwin
def test_init(): assert Voicetwin().get_stats()["ops"] == 0
def test_op(): c = Voicetwin(); c.process(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Voicetwin(); [c.process() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Voicetwin(); c.process(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Voicetwin(); r = c.process(); assert r["service"] == "voicetwin"
