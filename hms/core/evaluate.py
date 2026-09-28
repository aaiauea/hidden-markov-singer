"""Objective model evaluation: compare models on an evaluation corpus.

`hms evaluate` renders *no audio* and invents *no quality score*: it reports
separate, objective, comparable numbers per model so a human can judge:

* acoustic log-likelihood of the evaluation corpus's frames under each model's
  own units (with context and pitch-bin resolution, exactly as synthesis would
  select them),
* voicing agreement between each model's state voicing probabilities and the
  analysed voicing,
* duration prediction error (per-phoneme log-normal means vs. real segment
  lengths),
* how many frames were routed to backoff models, and -- for a model trained
  with the optional pitch-conditioned tier -- how many frames were served by a
  model of their own pitch bin versus falling back from a bin with no model.

The command does not check that the evaluation corpus is disjoint from any
model's training data -- it measures whatever corpus it is given, and
interpreting the numbers (e.g. as generalisation) is the caller's job.

Because those numbers are only comparable when the models were trained the
same way, evaluation also *checks* the models against each other: the feature
spec must match exactly (a hard error -- otherwise the analysis means
something different per model), and differences in phoneme inventory,
training method, seed, training corpus paths, and any other non-context
training setting are reported as warnings.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hms.core import labels as labels_module
from hms.core.model import HMSModel
from hms.core.pitch_condition import is_pitch_tier
from hms.core.trainer import Corpus, Trainer, TrainingConfig, UtteranceData
from hms.core.transfer import TIER_TRANSFERRED

#: Training-config keys that describe the optional context feature; they are
#: legitimately different between a baseline and a context model.
_CONTEXT_FIELDS = ("context_enabled", "context_min_frames",
                   "context_min_occurrences", "context_max_models",
                   "context_partial", "context_global_backoff")

#: Training-config keys of the optional pitch-conditioned tier, exempt for the
#: same reason: comparing a pitch-conditioned model against its unconditioned
#: baseline is the usual reason to run the comparison.
_PITCH_CONDITIONING_FIELDS = ("pitch_conditioning_enabled",
                              "pitch_conditioning_bin_size")

#: Training-config keys of the experimental cross-language transfer tier; a
#: transferred voice is *meant* to be compared against its single-language
#: baseline, so its settings are exempt from the "models differ" warnings (the
#: differing unit inventories they come with are still reported).
_TRANSFER_FIELDS = ("transfer_enabled", "transfer_target_language",
                    "transfer_target_speaker", "transfer_auxiliary_language",
                    "transfer_auxiliary_speaker", "transfer_auxiliary_model",
                    "transfer_auxiliary_labels", "transfer_auxiliary_wav_dir",
                    "transfer_adapt_pitch_bins", "transfer_map_prior_strength",
                    "transfer_map_adapt_frames")


def _training_config_of(model: HMSModel) -> Dict[str, object]:
    return dict((model.metadata or {}).get("training_config") or {})


def check_model_compatibility(models: Sequence[HMSModel]
                              ) -> Tuple[List[str], List[str]]:
    """Cross-check models before comparing them.

    Returns ``(errors, warnings)``.  Errors make comparison meaningless
    (different feature spaces); warnings flag differences that may explain
    differing metrics (different training paths, seeds, inventories, or
    non-context settings).
    """
    errors: List[str] = []
    warnings: List[str] = []
    if len(models) < 1:
        raise ValueError("at least one model is required")
    base = models[0]

    base_spec = base.spec.to_dict()
    for model in models[1:]:
        if model.spec.to_dict() != base_spec:
            differences = sorted(
                key for key in set(base_spec) | set(model.spec.to_dict())
                if base_spec.get(key) != model.spec.to_dict().get(key))
            errors.append(
                f"feature spec mismatch between {base.name!r} and "
                f"{model.name!r} ({', '.join(differences)}); models must "
                f"share one feature definition to be compared")

    base_inventory = base.phoneme_set.to_dict()
    for model in models[1:]:
        if model.phoneme_set.to_dict() != base_inventory:
            base_symbols = set(base.phoneme_set.phonemes)
            other_symbols = set(model.phoneme_set.phonemes)
            only_base = sorted(base_symbols - other_symbols)
            only_other = sorted(other_symbols - base_symbols)
            detail = []
            if only_base:
                detail.append(f"only in {base.name!r}: {', '.join(only_base)}")
            if only_other:
                detail.append(f"only in {model.name!r}: "
                              f"{', '.join(only_other)}")
            if not detail:
                detail.append("definitions or aliases differ")
            warnings.append("phoneme inventory differs between "
                            f"{base.name!r} and {model.name!r} ("
                            + "; ".join(detail) + ")")

    configs = [_training_config_of(model) for model in models]
    if all(configs):
        names = [model.name for model in models]
        all_keys = set().union(*(set(config) for config in configs))
        for key in sorted(all_keys):
            if key in _CONTEXT_FIELDS or key in _PITCH_CONDITIONING_FIELDS \
                    or key in _TRANSFER_FIELDS:
                continue                      # the optional tiers are the point
            values = [config.get(key) for config in configs]
            comparable = []
            for value in values:
                comparable.append(tuple(value) if isinstance(value, list)
                                  else value)
            if len({repr(value) for value in comparable}) > 1:
                listing = ", ".join(f"{name}={value!r}"
                                    for name, value in zip(names, values))
                if key in ("label_file", "wav_dir"):
                    warnings.append(f"models were trained on different paths "
                                    f"({key}: {listing})")
                elif key == "training_method":
                    warnings.append(f"models used different training methods "
                                    f"({listing})")
                elif key == "seed":
                    warnings.append(f"models were trained with different "
                                    f"seeds ({listing})")
                else:
                    warnings.append(f"non-context training setting differs: "
                                    f"{key} ({listing})")
    return errors, warnings


def _analysis_trainer(model: HMSModel, label_file: str, wav_dir: str,
                      time_unit: Optional[str], vocoder: Optional[str],
                      audio_extensions: Sequence[str], log) -> Trainer:
    """A Trainer configured to analyse the corpus exactly like ``model`` was.

    Training settings are restored from the model's metadata so the analysis
    matches -- except the ones describing the optional model tiers, which say
    how a model was *built*, not how its corpus is analysed; the feature-spec
    fields are then pinned to the model's own spec.  Label time units follow
    the same precedence: an explicit ``time_unit`` argument wins, then the
    model's recorded training setting, then the configuration default.
    """
    metadata_config = _training_config_of(model)
    known = set(TrainingConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    kwargs = {key: value for key, value in metadata_config.items()
              if key in known and not key.startswith("context_")
              and not key.startswith("pitch_conditioning_")
              and not key.startswith("transfer_")
              and key not in ("label_file", "wav_dir")}
    try:
        config = TrainingConfig.from_dict(kwargs)
    except (TypeError, ValueError):
        config = TrainingConfig()
    spec = model.spec
    config.label_file = label_file
    config.wav_dir = wav_dir
    config.audio_extensions = tuple(audio_extensions)
    if time_unit is None:
        recorded = metadata_config.get("time_unit")
        if recorded is not None:
            time_unit = str(recorded)
    if time_unit is not None:
        config.time_unit = time_unit
    config.fs = spec.fs
    config.frame_period = spec.frame_period
    config.fft_size = spec.fft_size
    config.n_mcep = spec.n_mcep
    config.n_band = spec.n_band
    config.use_delta = spec.use_delta
    config.use_delta2 = spec.use_delta2
    config.f0_floor = spec.f0_floor
    config.f0_ceiling = spec.f0_ceil
    config.f0_estimation = spec.f0_estimation
    config.refine_f0 = spec.refine_f0
    if vocoder:
        config.vocoder = vocoder
    trainer = Trainer(config, model.phoneme_set, log=log)
    trainer.spec = spec
    return trainer


def _evaluate_single(model: HMSModel, utterances: Sequence[UtteranceData],
                     trainer: Trainer) -> Dict[str, object]:
    """Objective metrics for one model on the analysed corpus."""
    total_frames = 0
    total_ll = 0.0
    voicing_agree = 0
    voicing_frames = 0
    duration_abs_error = 0.0
    duration_segments = 0
    backoff_frames = 0
    transferred_frames = 0
    pitch_conditioned_frames = 0
    pitch_fallback_frames = 0
    per_phone_ll: Dict[str, List[float]] = {}
    per_phone_frames: Dict[str, int] = {}
    silence = model.phoneme_set.silence

    for data in utterances:
        normalized = trainer.normalize_features(
            data.features, model.feature_offset, model.feature_scale)
        # The scored note travels with its span: it is the pitch condition an
        # evaluation frame would be rendered under, exactly as in synthesis.
        span_notes = list(data.span_notes
                          or [None] * len(data.phoneme_spans))
        spans = [(phone, lo, hi, span_notes[position])
                 for position, (phone, lo, hi) in enumerate(data.phoneme_spans)
                 if hi > lo]
        canon = [model.phoneme_set.canonical(phone)
                 for phone, _lo, _hi, _note in spans]
        for index, ((phone, lo, hi, note), curr) in enumerate(zip(spans, canon)):
            pre = canon[index - 1] if index > 0 else silence
            post = canon[index + 1] if index < len(spans) - 1 else silence
            pitch_bin = model.segment_pitch_bin(curr, note)
            if model.contexts or model.pitch_models:
                _key, hmm, tier = model.resolve_unit(pre, curr, post,
                                                     pitch_bin=pitch_bin)
            else:
                _key = None
                hmm = model.get_or_backoff(curr)
                if model.get_hmm(curr) is not None:
                    tier = "phone"
                elif model.get_transfer_hmm(curr) is not None:
                    tier = TIER_TRANSFERRED
                else:
                    tier = "class"
            segment = normalized[lo:hi]
            n_frames = len(segment)
            ll = float(hmm.log_likelihood(segment))
            total_ll += ll
            total_frames += n_frames
            if tier in ("class", "global"):
                backoff_frames += n_frames
            if model.unit_is_transferred(_key, tier):
                transferred_frames += n_frames
            if pitch_bin is not None:
                if is_pitch_tier(tier):
                    pitch_conditioned_frames += n_frames
                else:
                    pitch_fallback_frames += n_frames
            per_phone_ll.setdefault(curr, []).append(ll)
            per_phone_frames[curr] = per_phone_frames.get(curr, 0) + n_frames

            # voicing agreement, mirroring Synthesizer.voicing's rules
            definition = model.phoneme_set.resolve(phone)
            state_path = hmm.segment(segment)
            for t, state in enumerate(state_path):
                state = min(int(state), hmm.n_states - 1)
                if definition is not None and not definition.voiced:
                    predicted = False
                elif definition is not None and definition.voiced \
                        and model.pitch_model.voiced_prior.get(curr, 1.0) > 0.9:
                    predicted = True
                else:
                    predicted = hmm.states[state].voiced_prob > 0.5
                voicing_agree += int(predicted == bool(data.voiced[lo + t]))
                voicing_frames += 1

            predicted_frames = float(model.duration_model.predict(
                [curr], model.spec.frame_period, tempo=1.0,
                speak=False)[0])
            duration_abs_error += abs(predicted_frames - n_frames)
            duration_segments += 1

    return {
        "frames": total_frames,
        "total_log_likelihood": float(total_ll),
        "log_likelihood_per_frame": float(total_ll / total_frames)
        if total_frames else 0.0,
        "voicing_agreement": float(voicing_agree / voicing_frames)
        if voicing_frames else 0.0,
        "duration_mae_frames": float(duration_abs_error / duration_segments)
        if duration_segments else 0.0,
        "backoff_frames": backoff_frames,
        # Cross-language transfer tier usage: frames scored under a unit
        # adapted from another voice.  Zero for every model trained without the
        # experimental feature, so it reads the same as it always did.
        "transferred_frames": transferred_frames,
        # Pitch-conditioned tier usage: how many frames were scored under a
        # model trained for their pitch bin, and how many asked for a bin the
        # voice has no model for and fell back.  Zero for an unconditioned
        # model, so comparing a conditioned model against its baseline shows
        # how much of the corpus the conditioning actually reached.
        "pitch_conditioned_frames": pitch_conditioned_frames,
        "pitch_fallback_frames": pitch_fallback_frames,
        "per_phone_log_likelihood_per_frame": {
            phone: float(sum(per_phone_ll[phone]) / per_phone_frames[phone])
            for phone in sorted(per_phone_ll)},
    }


def evaluate_models(models: Sequence[HMSModel], label_file: str,
                    wav_dir: str, log=None,
                    time_unit: Optional[str] = None,
                    audio_extensions: Sequence[str] = (".wav",),
                    vocoder: Optional[str] = None) -> Dict[str, object]:
    """Evaluate every model on the same corpus and return a report dict.

    Raises ``ValueError`` when the models cannot be compared at all
    (mismatched feature specs) or when nothing could be analysed.
    """
    log = log or (lambda message: None)
    models = list(models)
    errors, warnings = check_model_compatibility(models)
    for warning in warnings:
        log(f"  ! {warning}")
    if errors:
        raise ValueError("; ".join(errors))

    base = models[0]
    trainer = _analysis_trainer(base, label_file, wav_dir, time_unit,
                                vocoder, audio_extensions, log)
    score = labels_module.load(label_file, time_unit=trainer.config.time_unit,
                               frame_period=trainer.config.frame_period)
    for diagnostic in score.diagnostics:
        log(f"  ! {diagnostic}")
    corpus = Corpus(score, wav_dir, tuple(audio_extensions))
    utterances = trainer.analyse_corpus(corpus)
    if not utterances:
        raise ValueError("no evaluation utterances could be analysed")

    report_models: Dict[str, object] = {}
    seen_names: Dict[str, int] = {}
    for model in models:
        name = model.name
        if name in seen_names:
            seen_names[name] += 1
            name = f"{name} ({seen_names[name]})"
        else:
            seen_names[name] = 0
        log(f"  evaluating {name}")
        report_models[name] = _evaluate_single(model, utterances, trainer)

    return {
        "corpus": {
            "label_file": str(label_file),
            "wav_dir": str(wav_dir),
            "utterances": len(utterances),
            "frames": int(sum(len(u.features) for u in utterances)),
            "seconds": float(sum(len(u.features) for u in utterances)
                             * base.spec.frame_period / 1000.0),
        },
        "models": report_models,
        "warnings": warnings,
    }
