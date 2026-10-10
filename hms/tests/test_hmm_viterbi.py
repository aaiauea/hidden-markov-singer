"""Regression tests: the vectorised ``LeftToRightHMM.viterbi`` must reproduce the
original per-cell implementation exactly (paths, tie-breaking and scores).

``reference_viterbi`` below is the pre-optimisation loop, kept verbatim as the
oracle. It is deliberately slow and is only used in tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from hms.core.gmm import EPS, DiagGMM
from hms.core.hmm import HMMState, LeftToRightHMM


def reference_viterbi(hmm: LeftToRightHMM, X):
    """Original implementation (ab7d623), kept as the reference oracle."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    t_frames = len(X)
    if t_frames == 0:
        return np.zeros(0, dtype=np.int64), 0.0
    ll = hmm.emission_log_likelihood(X)
    n = hmm.n_states
    a = hmm.transition_matrix()

    score = np.full((t_frames, n), -np.inf)
    back = np.zeros((t_frames, n), dtype=np.int64)
    score[0] = ll[0] + np.log(np.maximum(a[0, :n], EPS))
    for t in range(1, t_frames):
        for j in range(n):
            previous = score[t - 1] + np.log(np.maximum(a[:n, j], EPS))
            if not np.isfinite(previous).any():
                back[t, j] = j
                score[t, j] = -np.inf
                continue
            best_i = int(np.argmax(previous))
            score[t, j] = previous[best_i] + ll[t, j]
            back[t, j] = best_i

    exit_scores = score[-1] + np.log(np.maximum(a[:n, n], EPS))
    last = int(np.argmax(exit_scores))
    path = np.zeros(t_frames, dtype=np.int64)
    path[-1] = last
    for t in range(t_frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path, float(exit_scores[last])


def random_hmm(n_states, *, n_components=2, dim=3, allow_skip=False,
               seed=0, tied_self_loops=False):
    rng = np.random.default_rng(seed)
    hmm = LeftToRightHMM(n_states, allow_skip=allow_skip)
    for i in range(n_states):
        weights = rng.dirichlet(np.ones(n_components))
        means = rng.normal(0.0, 2.0, size=(n_components, dim))
        variances = rng.uniform(0.1, 2.0, size=(n_components, dim))
        hmm.states[i] = HMMState(
            gmm=DiagGMM(weights, means, variances, "diag"))
    hmm.self_loops = (np.full(n_states, 0.6) if tied_self_loops
                      else rng.uniform(0.05, 0.99, size=n_states))
    hmm.dim = dim
    return hmm


def assert_same(hmm, X):
    path, score = hmm.viterbi(X)
    ref_path, ref_score = reference_viterbi(hmm, X)
    assert path.dtype == np.int64
    assert path.shape == ref_path.shape
    np.testing.assert_array_equal(path, ref_path)
    assert score == ref_score or (np.isnan(score) and np.isnan(ref_score))
    return path, score


@pytest.mark.parametrize("n_states", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("allow_skip", [False, True])
@pytest.mark.parametrize("length", [1, 2, 3, 17, 150])
def test_matches_reference_on_random_models(n_states, allow_skip, length):
    hmm = random_hmm(n_states, allow_skip=allow_skip,
                     seed=100 * n_states + length)
    X = np.random.default_rng(length).normal(0.0, 2.0, size=(length, 3))
    assert_same(hmm, X)


def test_matches_reference_on_trained_model():
    rng = np.random.default_rng(3)
    sequences = [np.concatenate([rng.normal([0, 0], 0.3, size=(n, 2))
                                 for n in (12, 30, 7)]) for _ in range(6)]
    hmm = LeftToRightHMM(3)
    hmm.train(sequences, n_components=2, n_iterations=3, seed=1)
    for sequence in sequences + [sequences[0][:5]]:
        assert_same(hmm, sequence)


def test_empty_and_single_frame_inputs():
    hmm = random_hmm(3)
    path, score = hmm.viterbi(np.zeros((0, 3)))
    assert path.shape == (0,) and path.dtype == np.int64 and score == 0.0
    assert_same(hmm, np.array([[0.5, -1.0, 2.0]]))


def test_ties_are_broken_towards_the_lowest_state_index():
    # Identical emissions for every state and a symmetric transition structure
    # make many predecessor candidates exactly equal; the path must be the
    # same one the reference finds (first maximum wins).
    hmm = LeftToRightHMM(4)
    shared = DiagGMM(np.array([1.0]), np.zeros((1, 2)), np.ones((1, 2)), "diag")
    for i in range(4):
        hmm.states[i] = HMMState(gmm=shared)
    hmm.self_loops = np.full(4, 0.5)
    X = np.zeros((9, 2))                     # every frame has the same likelihood
    path, _ = assert_same(hmm, X)
    assert path[0] == 0 and path[-1] == 3
    assert np.all(np.diff(path) >= 0)


def test_exact_ties_on_identical_frames_match_reference():
    hmm = random_hmm(4, seed=11)
    frame = np.array([0.3, -0.2, 1.1])
    assert_same(hmm, np.tile(frame, (25, 1)))


def test_impossible_transitions_are_never_used():
    # Left-to-right without skips: a path may never move backwards or jump
    # two states ahead, even though those log-probabilities are log(EPS).
    hmm = random_hmm(5, seed=5)
    X = np.random.default_rng(5).normal(size=(40, 3))
    path, _ = assert_same(hmm, X)
    steps = np.diff(path)
    assert np.all((steps == 0) | (steps == 1))


def test_zero_probability_states_are_respected():
    # A state with no emission model has log-likelihood -inf for every frame,
    # so no path may visit it. Structurally impossible transitions are clamped
    # to log(EPS) rather than -inf, so the path routes around the dead state
    # with a finite penalty, with or without skip transitions.
    X = np.random.default_rng(7).normal(size=(20, 3))
    for allow_skip in (False, True):
        hmm = random_hmm(4, seed=7, allow_skip=allow_skip)
        hmm.states[1] = None
        path, score = assert_same(hmm, X)
        assert np.isfinite(score)
        assert 1 not in set(path.tolist())


def test_fully_dead_model_matches_reference():
    # Every path has zero probability from the first frame: the reference
    # collapses to the identity back-pointers and -inf scores.
    hmm = LeftToRightHMM(3)
    X = np.random.default_rng(9).normal(size=(12, 2))
    path, score = assert_same(hmm, X)
    assert score == -np.inf


def test_death_in_the_middle_of_the_utterance_matches_reference():
    hmm = random_hmm(3, seed=13)
    X = np.random.default_rng(13).normal(size=(10, 3))
    X[4] = np.nan                            # NaN emissions kill every path
    assert_same(hmm, X)


def test_very_small_likelihoods_are_stable():
    # Frames far from every mean: log-likelihoods of order -1e6 with no
    # underflow to -inf. Scores and paths must still match the reference.
    hmm = random_hmm(3, seed=21)
    X = np.random.default_rng(21).normal(0.0, 1.0, size=(30, 3))
    X[10:20] += 1e3
    path, score = assert_same(hmm, X)
    assert np.isfinite(score) and score < -1e5
    assert np.all(np.isfinite(hmm.emission_log_likelihood(X)))


def test_long_sequence_matches_reference():
    hmm = random_hmm(5, n_components=3, seed=31)
    X = np.random.default_rng(31).normal(0.0, 2.0, size=(400, 3))
    assert_same(hmm, X)


def test_segment_is_unchanged_by_the_faster_viterbi():
    rng = np.random.default_rng(41)
    sequences = [np.concatenate([rng.normal([0, 0], 0.3, size=(n, 2))
                                 for n in (10, 25, 6)]) for _ in range(4)]
    hmm = LeftToRightHMM(3)
    hmm.train(sequences, n_components=1, n_iterations=3, seed=2)
    for sequence in sequences:
        path, _ = hmm.viterbi(sequence)
        ref_path, _ = reference_viterbi(hmm, sequence)
        np.testing.assert_array_equal(path, ref_path)
        segmented = hmm.segment(sequence)
        assert len(segmented) == len(sequence)
        assert np.all(np.diff(segmented) >= 0)
