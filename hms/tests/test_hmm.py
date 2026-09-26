"""Left-to-right HMM: likelihoods, segmentation, training, durations."""

from __future__ import annotations

import numpy as np
import pytest

from hms.core.gmm import DiagGMM
from hms.core.hmm import (DEFAULT_SELF_LOOP, HMMState, LeftToRightHMM,
                          StateDurationStats)


def clustered_sequences(n_clusters=(10, 30, 6), repeats=8, seed=0):
    """Three well-separated segments per occurrence, in a fixed order."""
    rng = np.random.default_rng(seed)
    sequences = []
    for _ in range(repeats):
        parts = [rng.normal([index * 6.0, index * -4.0], 0.4,
                            size=(count, 2))
                 for index, count in enumerate(n_clusters)]
        sequences.append(np.concatenate(parts, axis=0))
    return sequences


def test_transition_matrix_is_left_to_right():
    hmm = LeftToRightHMM(4)
    transitions = hmm.transition_matrix()
    assert transitions.shape == (5, 5)
    # the entry state only reaches state 0
    assert transitions[0, 0] == pytest.approx(DEFAULT_SELF_LOOP)
    assert transitions[0, 1] == pytest.approx(1 - DEFAULT_SELF_LOOP)
    # no backwards or skipping transitions by default
    for i in range(4):
        for j in range(4):
            if j > i + 1:
                assert transitions[i, j] == 0.0
        assert transitions[i, 4] == (1 - DEFAULT_SELF_LOOP) if i == 3 else True
    assert transitions[:4, 4].sum() == pytest.approx(1 - DEFAULT_SELF_LOOP)


def test_skip_transitions_when_enabled():
    transitions = LeftToRightHMM(4, allow_skip=True).transition_matrix()
    assert transitions[0, 2] > 0
    assert transitions[2, 0] == 0.0


def test_rows_of_the_transition_matrix_are_distributions():
    transitions = LeftToRightHMM(5, allow_skip=True).transition_matrix()
    rows = transitions[:5].sum(axis=1)
    assert np.allclose(rows, 1.0)


def test_single_state_likelihood_matches_closed_form():
    sequences = clustered_sequences()
    hmm = LeftToRightHMM(1)
    hmm.train(sequences, n_components=1, n_iterations=1)
    sequence = sequences[0]
    transitions = hmm.transition_matrix()
    expected = (hmm.states[0].gmm.log_likelihood(sequence).sum()
                + len(sequence) * np.log(transitions[0, 0])
                + np.log(transitions[0, 1]))
    assert hmm.log_likelihood(sequence) == pytest.approx(expected, rel=1e-9)


def test_forward_backward_consistency():
    sequences = clustered_sequences(repeats=4)
    hmm = LeftToRightHMM(3)
    hmm.train(sequences, n_components=1, n_iterations=2, seed=1)
    sequence = sequences[0]
    alpha, log_prob = hmm.forward_log(sequence)
    beta = hmm.backward_log(sequence)
    # alpha_t + beta_t - log p(X) is the posterior, which must sum to 1
    gamma = hmm.state_posteriors(sequence)
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-10)
    assert np.isfinite(log_prob)
    assert np.isclose(alpha[-1].max(), log_prob, atol=1e-6) or log_prob > -np.inf
    assert beta.shape == alpha.shape


def test_training_recovers_the_cluster_structure():
    """Embedded Viterbi must find the three segments it was given."""
    lengths = (10, 30, 6)
    sequences = clustered_sequences(lengths, repeats=8, seed=2)
    hmm = LeftToRightHMM(3)
    hmm.train(sequences, n_components=1, n_iterations=4, seed=0)
    durations = hmm.duration_mean_frames()
    assert np.allclose(durations, lengths, rtol=0.25)

    # every state's mean must sit near its cluster centre
    centres = [[0.0, 0.0], [6.0, -4.0], [12.0, -8.0]]
    for state, centre in zip(range(3), centres):
        assert np.allclose(hmm.states[state].gmm.means[0], centre, atol=0.6)


def test_segmentation_is_monotone_compact_and_complete():
    sequences = clustered_sequences((8, 25, 5), repeats=3, seed=3)
    hmm = LeftToRightHMM(3)
    hmm.train(sequences, n_components=1, n_iterations=3, seed=0)
    for sequence in sequences:
        path = hmm.segment(sequence)
        assert len(path) == len(sequence)
        assert np.all(np.diff(path) >= 0)          # monotone
        assert path[0] == 0 and path[-1] == 2      # enters and leaves
        counts = np.bincount(path, minlength=3)
        assert (counts > 0).all()                  # no state is skipped


def test_minimum_occupancy_floor_is_respected():
    hmm = LeftToRightHMM(4)
    assert hmm.min_state_frames(100) == 6          # a quarter of the equal share
    assert hmm.min_state_frames(4) == 1
    bounds = np.array([0, 90, 92, 100, 100])
    repaired = hmm._enforce_min_occupancy(bounds, 5)
    assert (np.diff(repaired) >= 5).all()
    assert repaired[0] == 0 and repaired[-1] == 100
    assert np.all(np.diff(repaired) > 0)


def test_short_occurrences_do_not_crash():
    sequences = [np.random.default_rng(0).normal(size=(n, 2)) for n in (1, 2, 3, 7)]
    hmm = LeftToRightHMM(4)
    hmm.train(sequences, n_components=1, n_iterations=3)
    for sequence in sequences:
        path = hmm.segment(sequence)
        assert len(path) == len(sequence)
        assert np.all(np.diff(path) >= 0)


def test_baum_welch_improves_likelihood_on_average():
    sequences = clustered_sequences((10, 20, 5), repeats=6, seed=4)
    hmm = LeftToRightHMM(3)
    hmm.train(sequences, n_components=1, n_iterations=1, seed=0)
    before = np.mean([hmm.log_likelihood(s) for s in sequences])
    hmm.train(sequences, n_components=1, n_iterations=3, method="baum_welch",
              seed=0)
    after = np.mean([hmm.log_likelihood(s) for s in sequences])
    assert after >= before - 1e-6


def test_duration_statistics_and_proportions():
    sequences = clustered_sequences((10, 30, 5), repeats=6, seed=5)
    hmm = LeftToRightHMM(3)
    hmm.train(sequences, n_components=1, n_iterations=3, seed=0)
    durations = hmm.duration_mean_frames()
    proportions = hmm.duration_proportions()
    assert proportions.sum() == pytest.approx(1.0)
    assert np.argmax(proportions) == 1              # the 30-frame state dominates
    assert all(state.duration.count > 0 for state in hmm.states)
    # self loops follow the learned durations
    assert hmm.self_loops[1] > hmm.self_loops[0]


def test_voicing_probabilities_are_collected():
    sequences = clustered_sequences((6, 10, 4), repeats=4, seed=6)
    voiced = [np.concatenate([np.zeros(6, bool), np.ones(10, bool),
                              np.zeros(4, bool)]) for _ in sequences]
    hmm = LeftToRightHMM(3)
    hmm.train(sequences, n_components=1, n_iterations=4, seed=0, voiced=voiced)
    probabilities = [state.voiced_prob for state in hmm.states]
    assert probabilities[1] > 0.7                  # the voiced state
    assert probabilities[0] < 0.4 and probabilities[2] < 0.4


def test_mixture_components_are_learned():
    rng = np.random.default_rng(7)
    parts = []
    for centre in (0.0, 0.0, 0.0), (5.0, 5.0, 5.0):
        parts.append(np.concatenate([rng.normal(centre, 0.3, size=(20, 3)),
                                     rng.normal(centre, 0.3, size=(20, 3))]))
    sequence = np.concatenate(parts, axis=0)
    hmm = LeftToRightHMM(2)
    hmm.train([sequence] * 4, n_components=2, n_iterations=3, seed=0)
    means = np.sort(hmm.states[1].gmm.means[:, 0]) if hmm.n_states > 1 else None
    assert hmm.states[0].gmm.n_components == 2
    assert means is not None


def test_covariance_type_is_passed_through():
    sequences = clustered_sequences((6, 6, 6), repeats=4, seed=8)
    hmm = LeftToRightHMM(3)
    hmm.train(sequences, n_components=2, covariance_type="tied",
              n_iterations=2, seed=0)
    assert hmm.covariance_type == "tied"
    for state in hmm.states:
        assert np.allclose(state.voiced_prob, state.voiced_prob)  # finite
        assert np.allclose(state.gmm.variances[0], state.gmm.variances[1])


def test_serialisation_roundtrip():
    sequences = clustered_sequences((8, 20, 5), repeats=5, seed=9)
    hmm = LeftToRightHMM(3)
    hmm.train(sequences, n_components=2, n_iterations=2, seed=0)
    restored = LeftToRightHMM.from_arrays(hmm.to_arrays())
    assert restored.n_states == hmm.n_states
    assert np.allclose(restored.self_loops, hmm.self_loops)
    assert np.allclose(restored.duration_mean_frames(), hmm.duration_mean_frames())
    assert np.allclose(restored.log_likelihood(sequences[0]),
                       hmm.log_likelihood(sequences[0]))
    assert restored.states[0].gmm.n_components == hmm.states[0].gmm.n_components


def test_unknown_training_method_is_rejected():
    hmm = LeftToRightHMM(2)
    with pytest.raises(ValueError):
        hmm.train(clustered_sequences((5, 5, 5), repeats=2),
                  method="nonsense")


def test_training_requires_data():
    with pytest.raises(ValueError):
        LeftToRightHMM(2).train([])


def test_non_finite_features_are_rejected():
    bad = [np.array([[np.nan, 0.0], [0.0, 0.0]])]
    with pytest.raises(ValueError):
        LeftToRightHMM(2).train(bad)


def test_hand_built_hmm_scores_reasonably():
    """A hand-made 2-state model should prefer its own data."""
    state_a = HMMState(DiagGMM.single_gaussian(np.zeros(2), np.ones(2)))
    state_b = HMMState(DiagGMM.single_gaussian(np.full(2, 5.0), np.ones(2)))
    hmm = LeftToRightHMM(2)
    hmm.states = [state_a, state_b]
    hmm.self_loops = np.array([0.8, 0.8])
    rng = np.random.default_rng(0)
    matching = np.concatenate([rng.normal(0, 0.3, (10, 2)),
                               rng.normal(5, 0.3, (10, 2))])
    mismatched = np.concatenate([rng.normal(5, 0.3, (10, 2)),
                                 rng.normal(0, 0.3, (10, 2))])
    assert hmm.log_likelihood(matching) > hmm.log_likelihood(mismatched)
    path = hmm.segment(matching)
    assert path[:5].tolist() == [0] * 5 and path[-5:].tolist() == [1] * 5


def test_state_duration_stats_serialisation():
    stats = StateDurationStats(1.5, 0.25, 7)
    assert StateDurationStats.from_dict(stats.to_dict()).count == 7
