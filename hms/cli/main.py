"""Command line interface: `hms <command>`.

    hms demo           generate the example corpus, train on it, synthesise it
    hms extract        analyse a corpus into WORLD parameters
    hms train          train a voice model
    hms synth          render a score to a WAV file
    hms evaluate       objectively compare models on an evaluation corpus
    hms inspect-model  print what a model contains
    hms doctor         report available backends and where they came from

The CLI is a thin shell over the library: every command maps to one function in
`hms.core` (or `hms.data`), so anything you can do here you can do in Python.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from hms import __version__
from hms.config import (build_feature_spec, load_parameters, load_phoneme_set,
                        synthesis_config_from_parameters,
                        training_config_from_parameters)
from hms.core import labels as labels_module
from hms.core.pitch import note_relative_pitch
from hms.core.model import HMSModel
from hms.core.synthesizer import Synthesizer
from hms.core.trainer import Corpus, Trainer, TrainingConfig
from hms.data import wavio
from hms.vocoder import available_backends, get_vocoder


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _log(message: str) -> None:
    print(message, flush=True)


def _load_f0_file(path):
    """Read an external F0 trajectory: ``.npy``, or text with one Hz per line.

    Unvoiced frames are ``0.0`` (WORLD's convention); ``#`` starts a comment.
    The frame count itself is checked later, when the render knows how many
    frames the score produces.
    """
    import numpy as np

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"F0 file not found: {path}")
    if path.suffix.lower() == ".npy":
        return np.load(path)
    try:
        return np.loadtxt(path, dtype=np.float64, comments="#", ndmin=1)
    except ValueError as exc:
        raise ValueError(f"could not read {path} as one F0 value per line "
                         f"(in Hz, 0.0 for unvoiced): {exc}") from exc


def _load_lib_config(args) -> Dict:
    parameters = load_parameters(getattr(args, "config", None))
    phonemes = load_phoneme_set(getattr(args, "phonemes", None))
    return {"parameters": parameters, "phonemes": phonemes}


def _training_config(args) -> TrainingConfig:
    parameters = load_parameters(getattr(args, "config", None))
    config = training_config_from_parameters(parameters)
    overrides = {
        "label_file": getattr(args, "labels", None),
        "wav_dir": getattr(args, "wav_dir", None),
        "n_iterations": getattr(args, "iterations", None),
        "training_method": getattr(args, "method", None),
        "covariance_type": getattr(args, "covariance", None),
        "n_mcep": getattr(args, "n_mcep", None),
        "seed": getattr(args, "seed", None),
        "vocoder": getattr(args, "vocoder", None),
        "f0_estimation": getattr(args, "f0_estimation", None),
        "pitch_variation": getattr(args, "pitch_variation", None),
        "fs": getattr(args, "fs", None),
    }
    if getattr(args, "delta2", None) is not None:
        overrides["use_delta2"] = bool(args.delta2)
    if getattr(args, "delta", None) is not None:
        overrides["use_delta"] = bool(args.delta)
    if getattr(args, "vibrato", None) is not None:
        overrides["vibrato_enabled"] = bool(args.vibrato)
    if getattr(args, "context", None) is not None:
        overrides["context_enabled"] = bool(args.context)
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)
    return config


DEFAULT_CONFIG_FIELD = "(from parameters.yaml)"


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_doctor(args) -> int:
    """Report what is available on this machine."""
    from hms.vocoder import BACKENDS

    backends = available_backends()
    _log(f"hms version      : {__version__}")
    _log(f"python           : {sys.version.split()[0]} {sys.executable}")
    try:
        import numpy
        _log(f"numpy            : {numpy.__version__}")
    except Exception as exc:  # pragma: no cover
        _log(f"numpy            : MISSING ({exc})")
    try:
        import yaml
        _log(f"pyyaml           : {yaml.__version__}")
    except Exception:  # pragma: no cover
        _log("pyyaml           : MISSING")
    _log("")
    _log("vocoder backends :")
    for name in ("pyworld", "native", "builtin"):
        status = "available" if backends[name] else "unavailable"
        _log(f"  {name:9s} {status}")
    if backends["native"]:
        from hms.vocoder.world_native import NativeWorldVocoder
        vocoder = NativeWorldVocoder()
        _log(f"  -> native uses {vocoder._lib_path}")
    if not backends["native"] and not backends["pyworld"]:
        _log("")
        _log("No real WORLD backend is available, so HMS would fall back to the")
        _log("builtin approximation. To build WORLD from source (needs a C++")
        _log("compiler, no Python headers), run:")
        _log("    tools/build_world.sh")
    _log("")
    _log(f"backend names accepted by --vocoder: {', '.join(BACKENDS)}")
    vocoder = get_vocoder(args.vocoder or "auto")
    _log(f"default backend  : {vocoder.name} (fft_size={vocoder.fft_size})")
    return 0


def cmd_extract(args) -> int:
    """Analyse a corpus into WORLD parameters (and optional features)."""
    import numpy as np

    parameters = load_parameters(getattr(args, "config", None))
    spec = build_feature_spec(parameters)
    config = training_config_from_parameters(parameters)
    config.label_file = args.labels
    config.wav_dir = args.wav_dir
    if args.vocoder:
        config.vocoder = args.vocoder
    if getattr(args, "fs", None):
        config.fs = int(args.fs)

    vocoder = get_vocoder(config.vocoder, fft_size=config.fft_size)
    # the FFT size follows the sample rate (WORLD), so ask the backend
    spec = build_feature_spec(parameters)
    spec.fs = config.fs
    spec.fft_size = config.fft_size or vocoder.fft_size_for(config.fs)

    score = labels_module.load(args.labels, time_unit=config.time_unit,
                               frame_period=config.frame_period)
    corpus = Corpus(score, Path(args.wav_dir), config.audio_extensions)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"backend : {vocoder.name} (fs={spec.fs}, fft_size={spec.fft_size})")
    _log(f"corpus  : {len(score)} utterances, {corpus.total_seconds:.1f} s")
    summary: Dict[str, dict] = {}
    for utterance in score:
        path = corpus.audio_path(utterance.name)
        if path is None:
            _log(f"  ! {utterance.name}: no audio, skipped")
            continue
        signal, fs = wavio.read_wav(path)
        if fs != spec.fs:
            _log(f"  ! {utterance.name}: {fs} Hz != configured {spec.fs} Hz")
            continue
        t0 = time.time()
        sequence = vocoder.analyze_to_sequence(
            signal, fs, frame_period=spec.frame_period,
            f0_floor=spec.f0_floor, f0_ceil=spec.f0_ceil,
            f0_estimation=spec.f0_estimation, refine_f0=spec.refine_f0)
        wavio.save_params(out_dir / f"{utterance.name}.npz", sequence)
        if args.features:
            phones, notes, _ = utterance.frame_labels(
                spec.frame_period, config.default_note, n_frames=len(sequence))
            voiced = sequence.f0 > spec.voiced_threshold
            features = spec.encode(sequence.f0, sequence.sp, sequence.ap)
            features[:, 0] = note_relative_pitch(
                sequence.f0, labels_module.midi_to_hz(notes), voiced,
                spec.f0_ref_hz)
            np.save(out_dir / f"{utterance.name}.features.npy", features)
            (out_dir / f"{utterance.name}.phones.txt").write_text(
                "\n".join(phones) + "\n", encoding="utf-8")
        summary[utterance.name] = {
            "frames": int(len(sequence)),
            "voiced_fraction": float(sequence.voiced.mean()),
            "f0_median_hz": float(np.median(sequence.f0[sequence.f0 > 0]))
            if sequence.voiced.any() else 0.0,
            "seconds": round(time.time() - t0, 3),
        }
        _log(f"  {utterance.name:22s} {len(sequence):5d} frames "
             f"voiced={summary[utterance.name]['voiced_fraction']:.2f} "
             f"f0med={summary[utterance.name]['f0_median_hz']:.1f} Hz")

    (out_dir / "extract_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"wrote {len(summary)} parameter files to {out_dir}")
    return 0


def cmd_train(args) -> int:
    config = _training_config(args)
    phoneme_set = load_phoneme_set(getattr(args, "phonemes", None))

    if args.resume:
        _log("resuming from an existing model is not supported yet; "
             "starting a fresh model")
    t0 = time.time()
    trainer = Trainer(config, phoneme_set, log=_log)
    model = trainer.train()
    if args.name:
        model.name = args.name
    out = Path(args.out)
    model.save(out)
    _log("")
    _log(f"trained in {time.time() - t0:.1f} s")
    _log(f"model written to {out}")
    _log("")
    _log("parameter budget")
    _log("\n".join(model.parameter_report()))
    if args.evaluate:
        _log("")
        _log("training-set likelihood (higher is better):")
        score = labels_module.load(config.label_file,
                                   time_unit=config.time_unit,
                                   frame_period=config.frame_period)
        corpus = Corpus(score, Path(config.wav_dir or "."),
                        config.audio_extensions)
        utter = trainer.analyse_corpus(corpus)
        offset, scale = trainer.compute_normalization(utter)
        features, _, _, _ = trainer.collect_phoneme_data(utter, offset, scale)
        total = 0.0
        for phone, sequences in features.items():
            hmm = model.get_hmm(phone)
            if hmm is None:
                continue
            total += sum(hmm.log_likelihood(s) for s in sequences)
        _log(f"  total log likelihood: {total:,.1f}")
    return 0


def cmd_synth(args) -> int:
    parameters = load_parameters(getattr(args, "config", None))
    model = HMSModel.load(args.model)
    config = synthesis_config_from_parameters(parameters)
    vibrato = (parameters.get("pitch") or {}).get("vibrato") or {}
    if "enabled" in vibrato:
        model.pitch_model.vibrato.enabled = bool(vibrato["enabled"])

    overrides = {
        "variance_scale": args.variance_scale,
        "pitch_variation": args.pitch_variation,
        "f0_source": args.f0_source,
        "duration_mode": args.duration_mode,
        "transpose": args.transpose,
        "pitch_smoothing": args.pitch_smoothing,
        "tempo": args.tempo,
        "mixture": args.mixture,
        "seed": args.seed,
        "vocoder": args.vocoder,
    }
    if args.vibrato is not None:
        overrides["vibrato"] = bool(args.vibrato)
    if args.vibrato_depth is not None:
        overrides["vibrato_depth"] = float(args.vibrato_depth)
    if args.vibrato_rate is not None:
        overrides["vibrato_rate"] = float(args.vibrato_rate)
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)

    time_unit = (parameters.get("training") or {}).get("time_unit", "seconds")
    score = labels_module.load(args.score, time_unit=time_unit,
                               frame_period=model.spec.frame_period)
    for diagnostic in score.diagnostics:
        _log(f"  ! {diagnostic}")
    if args.utterance:
        selected = [u for u in score if u.name == args.utterance]
        if not selected:
            _log(f"no utterance named {args.utterance!r} in {args.score}")
            return 2
        score = labels_module.Score(selected)

    external_f0 = None
    if getattr(args, "f0_file", None):
        external_f0 = _load_f0_file(args.f0_file)
        _log(f"f0 file : {args.f0_file}")

    synthesizer = Synthesizer(model, config, log=_log)
    _log(f"model   : {model.name} ({len(model.hmms)} phonemes)")
    _log(f"backend : {synthesizer.vocoder.name}")
    t0 = time.time()
    result = synthesizer.synthesize(score, f0=external_f0)
    _log(f"rendered {len(score)} utterance(s), {result.duration:.2f} s "
         f"of audio in {time.time() - t0:.1f} s")
    for diagnostic in result.diagnostics:
        _log(f"  ! {diagnostic}")

    wavio.write_wav(args.out, result.audio, model.spec.fs,
                    bit_depth=32 if args.float_wav else 16)
    _log(f"wrote {args.out}")

    import numpy as np
    voiced = result.f0_semitones[np.isfinite(result.f0_semitones)]
    if len(voiced):
        reference = model.spec.f0_ref_hz
        _log(f"F0 range: {voiced.min():.2f} .. {voiced.max():.2f} semitones "
             f"re. {reference:.1f} Hz "
             f"({reference * 2 ** (voiced.min() / 12):.1f} .. "
             f"{reference * 2 ** (voiced.max() / 12):.1f} Hz)")
    if args.trace:
        _write_trace(args.trace, result, model.spec.f0_ref_hz)
        _log(f"wrote state trace {args.trace}")
    if args.save_params:
        wavio.save_params(args.save_params, result.params)
        _log(f"wrote WORLD parameters {args.save_params}")
    return 0


def _write_trace(path, result, f0_ref_hz: float = 261.6255653) -> None:
    lines = ["frame\ttime_s\tphone\tstate\tnote\tf0_hz"]
    import numpy as np
    for t in range(len(result.state_sequence)):
        time_s = t * result.params.frame_period / 1000.0
        f0 = result.f0_semitones[t]
        note = result.notes[t] if t < len(result.notes) else float("nan")
        lines.append(f"{t}\t{time_s:.3f}\t{result.state_sequence[t]}\t"
                     f"{result.state_ids[t]}\t{note:g}\t"
                     + ("-" if not np.isfinite(f0)
                        else f"{f0_ref_hz * 2 ** (f0 / 12):.2f}"))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_evaluate(args) -> int:
    """Objectively compare models on an evaluation corpus (no quality score)."""
    from hms.core.evaluate import evaluate_models

    model_dirs = list(dict.fromkeys(args.model))     # de-dup, keep order
    models = [HMSModel.load(directory) for directory in model_dirs]

    parameters = load_parameters(getattr(args, "config", None))
    # Prefer the models' recorded training time_unit (the first model that
    # carries it wins); the parameters.yaml value is only the fallback for
    # models with no training metadata of their own.
    time_unit = None
    for model in models:
        recorded = ((model.metadata or {}).get("training_config") or {}) \
            .get("time_unit")
        if recorded is not None:
            time_unit = str(recorded)
            break
    if time_unit is None:
        time_unit = (parameters.get("training") or {}).get("time_unit")

    _log(f"corpus : {args.labels} + {args.wav_dir}")
    for model in models:
        _log(f"model  : {model.name} ({model.metadata.get('label_file', '?')})")
    try:
        report = evaluate_models(models, args.labels, args.wav_dir, log=_log,
                                 time_unit=time_unit, vocoder=args.vocoder)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    corpus = report["corpus"]
    _log("")
    _log(f"evaluated {corpus['utterances']} utterances, "
         f"{corpus['frames']} frames ({corpus['seconds']:.1f} s)")
    _log("")
    _log(f"{'model':24s} {'LL total':>14s} {'LL/frame':>10s} "
         f"{'voicing':>9s} {'dur MAE':>10s} {'backoff':>10s}")
    _log(f"{'':24s} {'':>14s} {'':>10s} "
         f"{'agree':>9s} {'frames':>10s} {'frames':>10s}")
    for name, metrics in report["models"].items():
        _log(f"{name[:24]:24s} "
             f"{metrics['total_log_likelihood']:>14,.1f} "
             f"{metrics['log_likelihood_per_frame']:>10.2f} "
             f"{metrics['voicing_agreement']:>9.3f} "
             f"{metrics['duration_mae_frames']:>10.2f} "
             f"{metrics['backoff_frames']:>10d}")
    _log("")
    _log("metrics are reported separately on purpose: there is no aggregate "
         "quality score.")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True),
                                   encoding="utf-8")
        _log(f"wrote {args.json}")
    return 0


def cmd_inspect_model(args) -> int:
    model = HMSModel.load(args.model)
    if args.json:
        payload = {
            "name": model.name,
            "stats": model.stats.to_dict(),
            "parameter_budget": model.n_free_params,
            "parameter_breakdown": {
                "phoneme": model.phoneme_n_free_params,
                "context": model.context_n_free_params,
                "class_backoff": model.backoff_n_free_params,
                "global_backoff": model.global_backoff_n_free_params,
            },
            "feature_spec": model.spec.to_dict(),
            "phonemes": model.phoneme_set.to_dict(),
            "duration_model": model.duration_model.to_dict(),
            "pitch_model": model.pitch_model.to_dict(),
            "hmms": {phone: {"n_states": hmm.n_states,
                             "n_components": hmm.states[0].gmm.n_components,
                             "n_free_params": hmm.n_free_params}
                     for phone, hmm in sorted(model.hmms.items())},
            "context_models": {
                key: {**(model.context_index.get(key) or {}),
                      "n_free_params": hmm.n_free_params}
                for key, hmm in sorted(model.contexts.items())},
            "global_backoff": (
                {"n_states": model.global_backoff.n_states,
                 "n_components":
                     model.global_backoff.states[0].gmm.n_components,
                 "n_free_params": model.global_backoff.n_free_params}
                if model.global_backoff is not None else None),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(model.summary())
    if args.phoneme:
        phone = args.phoneme
        hmm = model.get_hmm(phone)
        if hmm is None:
            print(f"\nno model for phoneme {phone!r}")
            return 2
        print(f"\nphoneme {phone}: {hmm.n_states} states")
        for index, state in enumerate(hmm.states):
            gmm = state.gmm
            print(f"  state {index}: K={gmm.n_components} "
                  f"duration={hmm.duration_mean_frames()[index]:.2f} frames "
                  f"(log mean {state.duration.mean:+.3f}, "
                  f"var {state.duration.variance:.3f}, "
                  f"n={state.duration.count}) "
                  f"p(self)={hmm.self_loops[index]:.3f} "
                  f"p(voiced)={state.voiced_prob:.2f}")
            print(f"    mean[0] (semitones re. note) = {gmm.means[0, 0]:+.3f}")
        stats = model.pitch_model.stats.get(phone)
        if stats:
            print("  pitch stats (semitones re. note): "
                  + ", ".join(f"{s.mean:+.2f}±{s.variance ** 0.5:.2f}"
                              for s in stats))
        voiced_prior = model.pitch_model.voiced_prior.get(phone)
        if voiced_prior is not None:
            print(f"  voiced prior: {voiced_prior:.2f}")
        duration = model.duration_model.stats.get(phone)
        if duration:
            print(f"  duration: mean {duration.frames:.2f} frames "
                  f"(log {duration.mean:+.3f}, var {duration.variance:.3f}, "
                  f"n={duration.count})")
    return 0


def cmd_demo(args) -> int:
    """End-to-end: generate the example corpus, train, synthesise."""
    from hms.data.demo_singer import make_dataset

    out_dir = Path(args.out)
    corpus_dir = out_dir / "corpus"
    _log("1/3  generating the example corpus (synthetic singer -> WAV + labels)")
    info = make_dataset(corpus_dir, fs=args.fs, label_jitter_ms=args.label_jitter,
                        seed=args.seed)
    _log(f"  {info['utterances']} phrases, {info['segments']} segments, "
         f"{info['seconds']:.1f} s of audio")

    parameters = load_parameters(getattr(args, "config", None))
    config = training_config_from_parameters(parameters)
    config.label_file = info["labels"]
    config.wav_dir = info["wav_dir"]
    config.fs = args.fs
    # The demo corpus generator writes timestamps in seconds regardless of the
    # user's corpus-label time_unit setting.
    config.time_unit = "seconds"
    if args.iterations is not None:
        config.n_iterations = args.iterations
    if args.vocoder:
        config.vocoder = args.vocoder
    config.vibrato_enabled = args.vibrato

    _log("2/3  training")
    phonemes = load_phoneme_set(getattr(args, "phonemes", None))
    model = Trainer(config, phonemes, log=_log).train()
    model.name = "hms-demo"
    model_dir = out_dir / "model"
    model.save(model_dir)
    _log(f"  model written to {model_dir} "
         f"({model.n_free_params:,} free parameters)")

    if args.score_only:
        _log(f"3/3  skipped synthesis (--score-only); labels: {info['labels']}")
        return 0

    _log("3/3  synthesising the corpus back")
    synth_config = synthesis_config_from_parameters(parameters)
    synth_config.seed = args.seed
    if args.vibrato:
        synth_config.vibrato = True
    synthesizer = Synthesizer(model, synth_config, log=_log)
    score = labels_module.load(info["score"])
    out_wav = out_dir / "demo.wav"
    # render every utterance with a small gap between them (none at the end)
    import numpy as np
    silence = np.zeros(int(0.25 * args.fs))
    pieces = [synthesizer.synthesize(labels_module.Score([utterance])).audio
              for utterance in score]
    joined: List[np.ndarray] = []
    for index, piece in enumerate(pieces):
        if index:
            joined.append(silence)
        joined.append(piece)
    wavio.write_wav(out_wav, np.concatenate(joined), args.fs)
    _log(f"  wrote {out_wav}")
    _log("")
    _log("done. Try:")
    _log(f"  hms inspect-model --model {model_dir}")
    _log(f"  hms synth --model {model_dir} --score {info['score']} --out /tmp/out.wav")
    _log(f"  (training labels are in {info['labels']})")
    return 0


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hms", description="Hidden Markov Singer - a lightweight, "
        "non-neural singing synthesizer (HMM/GMM + WORLD)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", action="version",
                        version=f"hms {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config(argument_parser):
        argument_parser.add_argument(
            "--config", default=None,
            help="parameters.yaml to use (default: the bundled one)")
        argument_parser.add_argument(
            "--phonemes", default=None,
            help="phonemes.yaml to use (default: the bundled one)")

    # -- doctor ------------------------------------------------------------
    doctor = sub.add_parser("doctor", help="report available backends")
    doctor.add_argument("--vocoder", default=None,
                        help="backend to report on (auto|native|pyworld|builtin)")
    doctor.set_defaults(func=cmd_doctor)

    # -- extract -----------------------------------------------------------
    extract = sub.add_parser("extract", help="analyse a corpus into WORLD "
                                             "parameters")
    add_config(extract)
    extract.add_argument("--labels", required=True, help="label file")
    extract.add_argument("--wav-dir", required=True, help="directory of WAVs")
    extract.add_argument("--out", required=True, help="output directory")
    extract.add_argument("--features", action="store_true",
                         help="also write the acoustic feature matrices")
    extract.add_argument("--fs", type=int, default=None,
                         help="sample rate of the corpus (default: from "
                              "parameters.yaml)")
    extract.add_argument("--vocoder", default=None)
    extract.set_defaults(func=cmd_extract)

    # -- train -------------------------------------------------------------
    train = sub.add_parser("train", help="train a voice model")
    add_config(train)
    train.add_argument("--labels", required=True, help="label file")
    train.add_argument("--wav-dir", required=True, help="directory of WAVs")
    train.add_argument("--out", required=True, help="model directory")
    train.add_argument("--name", default=None, help="model name")
    train.add_argument("--iterations", type=int, default=None,
                       help="embedded-Viterbi iterations")
    train.add_argument("--method", default=None,
                       choices=["viterbi", "baum_welch"])
    train.add_argument("--covariance", default=None,
                       choices=["diag", "tied"],
                       help="GMM covariance type ('tied' shares one covariance "
                            "per state - useful for very small corpora)")
    train.add_argument("--n-mcep", type=int, default=None,
                       help="mel-cepstral coefficients (spectral resolution)")
    train.add_argument("--delta", dest="delta", action="store_true",
                       default=None, help="use delta features (default on)")
    train.add_argument("--no-delta", dest="delta", action="store_false",
                       help="disable delta features (not recommended)")
    train.add_argument("--delta2", dest="delta2", action="store_true",
                       default=None, help="also use delta-delta features")
    train.add_argument("--vibrato", dest="vibrato", action="store_true",
                       default=None, help="enable the vibrato component")
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--vocoder", default=None)
    train.add_argument("--fs", type=int, default=None,
                       help="sample rate of the corpus (default: from "
                            "parameters.yaml). HMS does not resample; every "
                            "WAV must already be at this rate.")
    train.add_argument("--f0-estimation", default=None, choices=["dio", "harvest"])
    train.add_argument("--pitch-variation", type=float, default=None)
    train.add_argument("--context", dest="context", action="store_true",
                       default=None,
                       help="learn sparse phoneme-context HMMs for the "
                            "contexts observed in the corpus")
    train.add_argument("--no-context", dest="context", action="store_false",
                       help="do not learn phoneme-context HMMs "
                            "(the default; context.enabled in parameters.yaml)")
    train.add_argument("--evaluate", action="store_true",
                       help="report the training-set log likelihood")
    train.add_argument("--resume", action="store_true",
                       help="(reserved) continue training an existing model")
    train.set_defaults(func=cmd_train)

    # -- synth -------------------------------------------------------------
    synth = sub.add_parser("synth", help="render a score to a WAV file")
    add_config(synth)
    synth.add_argument("--model", required=True, help="model directory")
    synth.add_argument("--score", required=True,
                       help="score file (same format as labels)")
    synth.add_argument("--out", required=True, help="output WAV file")
    synth.add_argument("--utterance", default=None,
                       help="render only this utterance id")
    synth.add_argument("--variance-scale", type=float, default=None,
                       help="scale on the delta variances: >1 follows the "
                            "frame means more literally (livelier), <1 smooths")
    synth.add_argument("--pitch-variation", type=float, default=None,
                       help="scale of the learned deviation from the note")
    synth.add_argument("--f0-source", default=None,
                       choices=["score", "acoustic", "state_means"],
                       help="score F0 (default), optional acoustic deviation, "
                            "or separate pitch-model state means")
    synth.add_argument("--f0-file", default=None,
                       help="external F0 trajectory (.npy, or text with one "
                            "value per line): Hz, one value per synthesis "
                            "frame, 0.0 for unvoiced. Overrides the "
                            "generated contour, which is never resized, so "
                            "the file must hold one value per frame")
    synth.add_argument("--duration-mode", default=None,
                       choices=["score", "model"],
                       help="use the score's durations or predict them")
    synth.add_argument("--mixture", default=None,
                       choices=["dominant", "marginal"])
    synth.add_argument("--tempo", type=float, default=None,
                       help="speed factor when predicting durations")
    synth.add_argument("--transpose", type=float, default=None,
                       help="semitones to add to every note")
    synth.add_argument("--pitch-smoothing", type=int, default=None,
                       help="moving average width on F0, in frames")
    synth.add_argument("--vibrato", dest="vibrato", action="store_true",
                       default=None, help="force vibrato on")
    synth.add_argument("--no-vibrato", dest="vibrato", action="store_false",
                       help="force vibrato off")
    synth.add_argument("--vibrato-depth", type=float, default=None)
    synth.add_argument("--vibrato-rate", type=float, default=None)
    synth.add_argument("--seed", type=int, default=None)
    synth.add_argument("--vocoder", default=None)
    synth.add_argument("--float-wav", action="store_true",
                       help="write 32-bit float WAV instead of 16-bit PCM")
    synth.add_argument("--trace", default=None,
                       help="write a per-frame state/F0 trace (TSV)")
    synth.add_argument("--save-params", default=None,
                       help="also save the WORLD parameters (.npz)")
    synth.set_defaults(func=cmd_synth)

    # -- evaluate ----------------------------------------------------------
    evaluate = sub.add_parser(
        "evaluate",
        help="objectively compare models on an evaluation corpus "
             "(reports separate metrics; no aggregate quality score)")
    add_config(evaluate)
    evaluate.add_argument("--labels", required=True,
                          help="evaluation label file")
    evaluate.add_argument("--wav-dir", required=True,
                          help="directory of the evaluation WAVs")
    evaluate.add_argument("--model", action="append", required=True,
                          help="model directory (repeat the flag to compare "
                               "several models)")
    evaluate.add_argument("--json", default=None,
                          help="also write the full report as JSON")
    evaluate.add_argument("--vocoder", default=None)
    evaluate.set_defaults(func=cmd_evaluate)

    # -- inspect-model -----------------------------------------------------
    inspect = sub.add_parser("inspect-model",
                             help="print what a trained model contains")
    inspect.add_argument("--model", required=True, help="model directory")
    inspect.add_argument("--phoneme", default=None,
                         help="detail one phoneme's states")
    inspect.add_argument("--json", action="store_true",
                         help="dump the whole model as JSON")
    inspect.set_defaults(func=cmd_inspect_model)

    # -- demo --------------------------------------------------------------
    demo = sub.add_parser("demo", help="generate the example corpus, train, "
                                       "and synthesise it")
    add_config(demo)
    demo.add_argument("--out", default="hms-demo",
                      help="output directory for corpus, model and audio")
    demo.add_argument("--fs", type=int, default=44100)
    demo.add_argument("--iterations", type=int, default=None)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--vocoder", default=None)
    demo.add_argument("--vibrato", action="store_true",
                      help="enable vibrato in the demo")
    demo.add_argument("--label-jitter", type=float, default=0.0,
                      help="perturb training labels by N ms (robustness demo)")
    demo.add_argument("--score-only", action="store_true",
                      help="stop after training")
    demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        # bad arguments / malformed input files (e.g. --f0-file) are user
        # errors, not crashes: report them the same way as a missing file
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
