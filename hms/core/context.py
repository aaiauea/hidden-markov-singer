"""Optional sparse phoneme-context modelling.

HMS models phonemes by default.  When context modelling is enabled
(``context.enabled: true`` in `parameters.yaml`, or ``hms train --context``),
the trainer additionally learns HMMs for the phone contexts that *actually
occur* in the corpus -- a sparse set, never a full triphone inventory:

* the exact triphone ``(pre_phone, curr_phone, future_phone)``,
* one-sided diphone contexts where supported: ``(pre, curr, _)`` modelling the
  current phone given its left neighbour, and ``(_, curr, post)`` given its
  right neighbour.

Context keys always have three ``^``-separated fields; the reserved wildcard
(``_`` unless the inventory uses that symbol, then ``?``, ``*`` or ``~``)
marks the unmodelled side, which keeps left and right diphone keys of the
same phone bigram distinct (they pool different frames: the right phone's in
the left context, the left phone's in the right context).

Utterance boundaries carry no special BOS/EOS tokens: the inventory's existing
silence symbol (``sil``) is the neighbour there, exactly as the labels already
use it.

Context models are trained directly from the pooled raw feature sequences of
their occurrences (the same recipe as the class backoff models), and only when
they clear configurable support thresholds (`context_min_frames`,
`context_min_occurrences`) and fit under the model cap (`context_max_models`).
Selection by support is deterministic, so a given corpus and seed always yield
the same sparse set.

At synthesis time a segment resolves through a fixed fallback hierarchy
(``HMSModel.resolve_unit``):

    exact triphone
        -> best-supported one-sided diphone (ties favour the left context)
        -> the dedicated current-phone HMM
        -> the phone-class backoff model
        -> the optional global backoff model

When context modelling is disabled none of this is active and every code path
behaves exactly as before.
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, Tuple

#: Joins the phones of a context key.
CONTEXT_SEPARATOR = "^"

#: Kinds of context units.
KIND_TRIPHONE = "triphone"      # (pre, curr, post)
KIND_LEFT = "left"              # left diphone: (pre, curr, _)
KIND_RIGHT = "right"            # right diphone: (_, curr, post)

#: Reserved storage key of the optional pooled global backoff model.  It can
#: never collide with a context key (which always contains CONTEXT_SEPARATOR).
GLOBAL_KEY = "__global__"

#: Wildcards tried, in order, for the unmodelled side of a diphone key.
WILDCARD_CANDIDATES = ("_", "?", "*", "~")


def context_wildcard(inventory_symbols: Iterable[str]) -> str:
    """The reserved wildcard symbol for diphone keys.

    Normally ``_``; if an inventory actually defines one of the candidates as
    a phoneme, the next one is used so keys can never collide with real
    triphones.
    """
    symbols = set(inventory_symbols)
    for candidate in WILDCARD_CANDIDATES:
        if candidate not in symbols:
            return candidate
    raise ValueError("no context wildcard symbol left: the phoneme inventory "
                     f"defines all of {WILDCARD_CANDIDATES}")


def triphone_key(pre: str, curr: str, post: str) -> str:
    """Key of the exact ``(pre, curr, post)`` context."""
    return CONTEXT_SEPARATOR.join((pre, curr, post))


def left_diphone_key(pre: str, curr: str, wildcard: str) -> str:
    """Key of the left one-sided context: ``curr`` given left neighbour ``pre``."""
    return CONTEXT_SEPARATOR.join((pre, curr, wildcard))


def right_diphone_key(curr: str, post: str, wildcard: str) -> str:
    """Key of the right one-sided context: ``curr`` given right neighbour ``post``."""
    return CONTEXT_SEPARATOR.join((wildcard, curr, post))


def neighbor_contexts(phones: Iterable[str], silence: str
                      ) -> Iterator[Tuple[str, str, str]]:
    """Yield ``(pre, curr, post)`` for every segment of a phone sequence.

    Utterance boundaries use the inventory's silence symbol -- there are no
    BOS/EOS tokens.
    """
    phones = list(phones)
    n = len(phones)
    for i in range(n):
        pre = phones[i - 1] if i > 0 else silence
        post = phones[i + 1] if i < n - 1 else silence
        yield pre, phones[i], post


def context_keys(pre: str, curr: str, post: str, wildcard: str,
                 partial: bool = True) -> List[Tuple[str, str]]:
    """Candidate context ``(key, kind)`` pairs for one occurrence.

    Always includes the exact triphone; when ``partial`` is true also the two
    one-sided diphone contexts (which are what utterance-initial and
    utterance-final phones mostly train).
    """
    keys = [(triphone_key(pre, curr, post), KIND_TRIPHONE)]
    if partial:
        keys.append((left_diphone_key(pre, curr, wildcard), KIND_LEFT))
        keys.append((right_diphone_key(curr, post, wildcard), KIND_RIGHT))
    return keys


def split_context_key(key: str) -> Tuple[str, ...]:
    """A context key back into its phone parts (mainly for inspection)."""
    return tuple(key.split(CONTEXT_SEPARATOR))
