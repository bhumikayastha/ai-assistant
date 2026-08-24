# app/cache.py
import hashlib

_cache = {}  # simple in-memory cache: {hash: answer}

def get_cache_key(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()

def cache_get(query: str):
    return _cache.get(get_cache_key(query))

def cache_set(query: str, answer):
    _cache[get_cache_key(query)] = answer