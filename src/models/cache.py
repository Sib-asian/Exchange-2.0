"""
cache.py — Sistema di caching per le matrici bivariate.

Implementa un cache LRU (Least Recently Used) per memorizzare le matrici
di probabilità calcolate, evitando ricalcoli per analisi ripetute della
stessa partita con parametri simili.

Il cache è particolarmente utile quando:
- L'utente modifica solo le quote exchange (matrici invariate)
- Si analizza la stessa partita più volte
- Si testano diverse linee Over/Under
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass

# Configurazione cache
CACHE_MAX_SIZE = 100  # Numero massimo di entry
CACHE_TTL_SECONDS = 300  # Time-to-live: 5 minuti


@dataclass
class CacheEntry:
    """Entry del cache con valore e timestamp."""

    value: object
    timestamp: float
    hits: int = 0


class MatrixCache:
    """
    Cache LRU per matrici bivariate con TTL.

    Thread-safe per uso in applicazioni Streamlit (single-threaded per request).
    """

    def __init__(self, max_size: int = CACHE_MAX_SIZE, ttl: float = CACHE_TTL_SECONDS):
        self._cache: OrderedDict[Hashable, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl
        self._hits = 0
        self._misses = 0

    def _hash_params(self, *args) -> Hashable:
        """Genera una chiave hash dai parametri."""
        # Arrotonda i float per evitare cache miss per differenze minime
        rounded = []
        for arg in args:
            if isinstance(arg, float):
                rounded.append(round(arg, 6))
            elif isinstance(arg, dict):
                # Per dict, usa la somma dei valori (semplificato)
                rounded.append(frozenset((k, round(v, 6) if isinstance(v, float) else v)
                                         for k, v in arg.items()))
            else:
                rounded.append(arg)
        return tuple(rounded)

    def get_or_compute(
        self,
        compute_fn: Callable[[], object],
        *params,
    ) -> object:
        """
        Recupera dal cache o calcola e memorizza.

        Args:
            compute_fn: Funzione che calcola il valore se non in cache.
            *params: Parametri che identificano la entry.

        Returns:
            Il valore dal cache o appena calcolato.
        """
        key = self._hash_params(*params)
        current_time = time.time()

        # Check cache hit
        if key in self._cache:
            entry = self._cache[key]

            # Verifica TTL
            if current_time - entry.timestamp < self._ttl:
                # Cache hit valido
                entry.hits += 1
                self._hits += 1

                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return entry.value
            else:
                # Entry scaduta
                del self._cache[key]

        # Cache miss
        self._misses += 1
        value = compute_fn()

        # Memorizza nel cache
        self._cache[key] = CacheEntry(value=value, timestamp=current_time, hits=1)

        # Evict oldest se necessario
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

        return value

    def invalidate(self, *params) -> bool:
        """
        Invalida una specifica entry del cache.

        Returns:
            True se l'entry esisteva, False altrimenti.
        """
        key = self._hash_params(*params)
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Cancella tutto il cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        """Restituisce statistiche del cache."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self._ttl,
        }


# Istanza globale del cache
_matrix_cache = MatrixCache()


def get_matrix_cache() -> MatrixCache:
    """Restituisce l'istanza globale del cache."""
    return _matrix_cache


def cached_bivariate_matrix(
    xg_h: float,
    xg_a: float,
    minuto: int,
    tot_cur: float,
    compute_fn: Callable[[], tuple[dict, dict, float]],
) -> tuple[dict, dict, float]:
    """
    Wrapper cached per build_bivariate_matrix.

    Args:
        xg_h, xg_a: xG casa e trasferta.
        minuto: Minuto attuale.
        tot_cur: Total corrente.
        compute_fn: Funzione che calcola (joint_ind, full, rho).

    Returns:
        (joint_ind, full, rho) dal cache o appena calcolato.
    """
    return _matrix_cache.get_or_compute(
        compute_fn,
        "bivariate",
        xg_h,
        xg_a,
        minuto,
        tot_cur,
    )


def cached_copula_matrix(
    xg_h: float,
    xg_a: float,
    theta: float,
    nu: float,
    compute_fn: Callable[[], dict],
) -> dict:
    """
    Wrapper cached per build_copula_matrix.

    Args:
        xg_h, xg_a: xG casa e trasferta.
        theta: Parametro copula.
        nu: Parametro dispersione CMP.
        compute_fn: Funzione che calcola la matrice.

    Returns:
        Matrice copula dal cache o appena calcolata.
    """
    return _matrix_cache.get_or_compute(
        compute_fn,
        "copula",
        xg_h,
        xg_a,
        theta,
        nu,
    )


def cached_markov_distribution(
    xg_h: float,
    xg_a: float,
    minuto: int,
    gol_h: int,
    gol_a: int,
    compute_fn: Callable[[], dict],
) -> dict:
    """
    Wrapper cached per markov_score_distribution.

    Args:
        xg_h, xg_a: xG casa e trasferta.
        minuto: Minuto attuale.
        gol_h, gol_a: Gol attuali.
        compute_fn: Funzione che calcola la distribuzione.

    Returns:
        Distribuzione Markov dal cache o appena calcolata.
    """
    return _matrix_cache.get_or_compute(
        compute_fn,
        "markov",
        xg_h,
        xg_a,
        minuto,
        gol_h,
        gol_a,
    )
