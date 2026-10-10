"""Regression tests for Viterbi-embedded HMM training (`LeftToRightHMM.train`).

The Viterbi path no longer fits a throw-away bootstrap GMM on the whole pooled
class for every state.  That must not change the learned model: the pinned
values below were produced by the implementation *before* that change, and the
tests check the current code against them.  The inputs are deterministic
arithmetic sequences (no random number generator), so the pins do not depend on
NumPy's RNG stream.

The structural tests check the cost side: the number of GMM fits and the number
of frames they see, so an accidental return to the eager bootstrap is caught
even when the numbers happen to agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from hms.core.gmm import DiagGMM
from hms.core.hmm import LeftToRightHMM


def _sequences(lengths, dim=2, phase=0.0):
    """Deterministic (T, dim) occurrences with three acoustic regions each."""
    out = []
    for k, length in enumerate(lengths):
        t = np.arange(length, dtype=np.float64)[:, None]
        d = np.arange(dim, dtype=np.float64)[None, :]
        region = np.floor(3.0 * t / max(length, 1))
        out.append(np.sin(0.7 * t + 1.3 * d + k + phase) * 0.5
                   + region * (2.0 + d))
    return out


#: name -> (sequence lengths, training kwargs, n_states)
CASES = {
    "regular": ((40, 35, 50), dict(n_components=1, n_iterations=3, seed=0), 3),
    "mixture": ((60, 80), dict(n_components=2, n_iterations=4, seed=1), 4),
    # Occurrences shorter than the state count leave states 2 and 3 without any
    # frame in the duration-proportional first iteration.
    "short_fill": ((1, 2, 2), dict(n_components=1, n_iterations=2, seed=5), 4),
    "short_fill_one_iteration": ((1, 2), dict(n_components=1, n_iterations=1,
                                              seed=7), 4),
    "baum_welch": ((40, 35), dict(n_components=1, n_iterations=2, seed=2,
                                  method="baum_welch"), 3),
}


def summarise_case(name):
    """Observable model state after training one named case."""
    lengths, kwargs, n_states = CASES[name]
    hmm = LeftToRightHMM(n_states=n_states)
    hmm.train(_sequences(lengths), **kwargs)
    return {
        "self_loops": hmm.self_loops.tolist(),
        "duration_mean": [s.duration.mean for s in hmm.states],
        "duration_count": [s.duration.count for s in hmm.states],
        "voiced_prob": [s.voiced_prob for s in hmm.states],
        "weights": [s.gmm.weights.tolist() for s in hmm.states],
        "means": [s.gmm.means.tolist() for s in hmm.states],
        "variances": [s.gmm.variances.tolist() for s in hmm.states],
    }


# Pinned from the pre-change implementation (origin/main, commit 1075776).
PINNED = {'baum_welch': {'duration_count': [12, 12, 11],
                'duration_mean': [0.8110078063402654,
                                  0.6818755000662828,
                                  0.7912180108207898],
                'means': [[[0.18215129809965902, 0.15834814012547502]],
                          [[1.9241018627046507, 2.9422016429596547]],
                          [[3.978609424280284, 5.993476124583778]]],
                'self_loops': [0.5555900387172892,
                               0.4943322773811809,
                               0.5467076560750925],
                'variances': [[[0.24485854876165342, 0.42167674619068257]],
                              [[0.19416465732155327, 0.38156526148429987]],
                              [[0.24514828342753958, 0.31538135485505675]]],
                'voiced_prob': [1.0, 1.0, 1.0],
                'weights': [[1.0], [1.0], [1.0]]},
 'mixture': {'duration_count': [2, 2, 2, 2],
             'duration_mean': [3.0342127941220554,
                               1.4978661367769954,
                               3.0816574020173206,
                               3.101267758593961],
             'means': [[[0.1095408145048807, 0.22830569745930906],
                        [-0.129910529487852, -0.3808528736629067]],
                       [[2.460988122208539, 3.1138776112457998],
                        [-0.016317505968819104, 0.3925136412597816]],
                       [[4.135452875957942, 5.572470085092595],
                        [1.9793789121206014, 2.9910914665389647]],
                       [[3.709617996083575, 5.882320062054934],
                        [4.341433511868913, 6.098101615480896]]],
             'self_loops': [0.9518874775675312,
                            0.7763932022500211,
                            0.9541168532258877,
                            0.9550078729334153],
             'variances': [[[0.1344818181649537, 0.04915737273767577],
                            [0.08455357229346919, 0.011069038047880455]],
                           [[0.0015555591700710612, 0.0341773124478616],
                            [0.07156633449772631, 0.019063351817387716]],
                           [[0.00022082902378102535, 0.00027294372408981636],
                            [0.12031333211748504, 0.12784771396824196]],
                           [[0.03027289200022869, 0.12143764726032186],
                            [0.021152014833612936, 0.10307415380489134]]],
             'voiced_prob': [1.0, 1.0, 1.0, 1.0],
             'weights': [[0.587095504929281, 0.412904495070719],
                         [0.44444444445555553, 0.5555555555444445],
                         [0.022727272822727275, 0.9772727271772728],
                         [0.5550408007850025, 0.4449591992149975]]},
 'regular': {'duration_count': [3, 3, 3],
             'duration_mean': [2.6523924411531583,
                               2.6276897837685844,
                               2.5784777841665627],
             'means': [[[0.05141883509213419, 0.012352828891775624]],
                       [[1.940072512231175, 2.948143623413351]],
                       [[4.02531962227662, 6.019125586644747]]],
             'self_loops': [0.9295176137745557,
                            0.9277548284346837,
                            0.9241105637543788],
             'variances': [[[0.12206291622989879, 0.12358698355987616]],
                           [[0.12279115645353848, 0.1219066033570709]],
                           [[0.12792722900447623, 0.12142226646649223]]],
             'voiced_prob': [1.0, 1.0, 1.0],
             'weights': [[1.0], [1.0], [1.0]]},
 'short_fill': {'duration_count': [2, 2, 3, 1],
                'duration_mean': [0.0, 0.0, 0.0, 0.0],
                'means': [[[0.43769210290839455, 0.146989879508368]],
                          [[2.3547611726715747, 2.8460793781879845]],
                          [[1.1169813102319879, 1.2935835216202605]],
                          [[0.0, 0.4817790927085965]]],
                'self_loops': [0.05, 0.05, 0.05, 0.05],
                'variances': [[[0.0002875266397994986, 0.05101397125814828]],
                              [[0.019901092654490877, 0.050391551378393074]],
                              [[1.0550180232181412, 1.6623356411312817]],
                              [[1e-12, 1e-12]]],
                'voiced_prob': [1.0, 1.0, 1.0, 1.0],
                'weights': [[1.0], [1.0], [1.0], [1.0]]},
 'short_fill_one_iteration': {'duration_count': [2, 1, 2, 2],
                              'duration_mean': [0.0, 0.0, 0.0, 0.0],
                              'means': [[[0.21036774620197413,
                                          0.42731584939847833]],
                                        [[2.4958324052262344,
                                          3.0705600040299337]],
                                        [[0.9721892992100608,
                                          1.30839723427563]],
                                        [[0.9721892992100608,
                                          1.30839723427563]]],
                              'self_loops': [0.05, 0.05, 0.05, 0.05],
                              'variances': [[[0.0442545886420982,
                                              0.0029662448718571323]],
                                            [[1e-12, 1e-12]],
                                            [[1.1902472163500384,
                                              1.5545863101353172]],
                                            [[1.1902472163500384,
                                              1.5545863101353172]]],
                              'voiced_prob': [1.0, 1.0, 1.0, 1.0],
                              'weights': [[1.0], [1.0], [1.0], [1.0]]}}


def _assert_matches(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(np.asarray(actual[key], dtype=float),
                                   np.asarray(expected[key], dtype=float),
                                   rtol=1e-9, atol=1e-12, err_msg=key)


@pytest.mark.parametrize("name", sorted(CASES))
def test_training_matches_pinned_pre_change_models(name):
    if name not in PINNED:
        pytest.skip("pins not generated yet")
    _assert_matches(summarise_case(name), PINNED[name])


@pytest.fixture
def fit_log(monkeypatch):
    """Record the frame count of every ``DiagGMM.fit`` call."""
    calls = []
    original = DiagGMM.fit.__func__

    def counting(cls, X, *args, **kwargs):
        calls.append(len(X))
        return original(cls, X, *args, **kwargs)

    monkeypatch.setattr(DiagGMM, "fit", classmethod(counting))
    return calls


@pytest.mark.parametrize("name", ["regular", "mixture"])
def test_viterbi_training_fits_only_the_iteration_gmms(name, fit_log):
    """No whole-class bootstrap: one fit per state per iteration, and every
    frame is fitted exactly once per iteration (a partition of the frames)."""
    lengths, kwargs, n_states = CASES[name]
    sequences = _sequences(lengths)
    LeftToRightHMM(n_states=n_states).train(sequences, **kwargs)
    n_iter = kwargs["n_iterations"]
    assert len(fit_log) == n_states * n_iter
    assert sum(fit_log) == n_iter * sum(len(s) for s in sequences)


def test_viterbi_training_bootstraps_only_states_left_without_frames(fit_log):
    lengths, kwargs, n_states = CASES["short_fill"]
    LeftToRightHMM(n_states=n_states).train(_sequences(lengths), **kwargs)
    pooled = sum(lengths)
    # Only the two states that received no frame in the first (proportional)
    # iteration get a whole-pool bootstrap fit; the eager version fitted all four.
    assert fit_log.count(pooled) == 2
    assert len(fit_log) == 7


def test_baum_welch_keeps_its_eager_bootstrap(fit_log):
    """Baum-Welch reads the bootstrap emissions in its first E-step."""
    lengths, kwargs, n_states = CASES["baum_welch"]
    LeftToRightHMM(n_states=n_states).train(_sequences(lengths), **kwargs)
    assert len(fit_log) == n_states + n_states * kwargs["n_iterations"]


def test_viterbi_training_still_validates_its_input():
    hmm = LeftToRightHMM(2)
    with pytest.raises(ValueError, match="non-finite"):
        hmm.train([np.array([[np.nan, 0.0], [0.0, 0.0]])])
    with pytest.raises(ValueError, match="feature dimension"):
        LeftToRightHMM(2).train([np.zeros((4, 2)), np.zeros((4, 3))])
    with pytest.raises(ValueError, match="unknown training method"):
        LeftToRightHMM(2).train(_sequences((5, 5)), method="nonsense")
