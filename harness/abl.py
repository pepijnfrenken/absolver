#!/usr/bin/env python3
"""Absolver harness — the guided, inspect-first way to run ablations.

NOT an autonomous loop. Each command does ONE thing, prints what it did,
and (for collect) writes a data bundle the campaign README references.
An agent (or human) reads the output, consults the campaign KB, decides
the next single step, and runs it.

Commands:
  inspect <config.yaml>            Load model, print arch + layer profile +
                                   direction separation + bias/weight shape
                                   audit (catches silent-skip landmines).
  directions <config.yaml>         Collect + save per-layer directions to
                                   <campaign>/directions.pt (with metadata).
  abl <config.yaml> --method mpoa --alpha 10 --layers 24,25,26,27 \
      --weights o_proj,down_proj   Apply ONE config to a fresh model,
                                   save ablated weights + a diff manifest
                                   (+ the directions it used; reuse with
                                   --from-directions).
  steer-test <config.yaml>         Alpha-response curve via steering hooks:
                                   alpha grid x few prompts, transcripts +
                                   refusal + PPL, NO model re-save.
  collect <config.yaml> [--model-dir DIR] [--transcript]
                                   Run gates + capability on the ablated
                                   model, write a JSON bundle (+ optional
                                   per-generation transcript) next to it.
  list-campaigns                   List campaigns + statuses from YAML.

All harvesting commands honor --prompt-flavor raw|chat (config default:
'chat') so directions, abl, gates and transcripts share ONE flavor axis.

The output dir convention: campaigns/<model-slug>/<run-id>/ where run-id
is a short timestamp + config tag. The campaign README is the narrative;
these bundles are the raw evidence.

Usage examples:
  python harness/abl.py inspect models/qwen2.5-1.5b-instruct.yaml
  python harness/abl.py directions models/qwen2.5-1.5b-instruct.yaml
  python harness/abl.py abl models/qwen2.5-1.5b-instruct.yaml \
      --method mpoa --alpha 10 --layers 24-27 --weights o_proj,down_proj
  python harness/abl.py collect models/qwen2.5-1.5b-instruct.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _slug(model_id: str) -> str:
    return model_id.replace("/", "-").lower()


def _load_cfg(config_path: str):
    from config import load_config
    return load_config(config_path)


def _load_model_tok(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = getattr(torch, str(cfg.dtype)) if isinstance(cfg.dtype, str) else cfg.dtype
    trust = getattr(cfg, "trust_remote_code", False)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=trust)
    tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=trust)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return model, tok


def _parse_layers(spec: str):
    """Accept '24,25,26,27' or '24-27' or '24'."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _resolve_flavor(cfg, flavor: str | None, tok) -> str:
    """Resolve the single prompt-flavor axis ('raw'|'chat') for this run.

    CLI ``--prompt-flavor`` wins, else ``cfg.prompt_flavor`` (default
    'chat' — gates evaluate chat-templated prompts, so directions must
    target the same mechanism; TOOLKIT-FEEDBACK §1b). A 'chat' request on
    a tokenizer with no chat template warns and falls back to raw.
    """
    from prompt_format import resolve_flavor
    requested = (flavor or getattr(cfg, "prompt_flavor", "chat")).lower()
    try:
        f = resolve_flavor(tok, requested, getattr(cfg, "prompt_flavor", "chat"))
    except ValueError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    if f != requested:
        print(f"WARNING: prompt flavor '{requested}' requested but tokenizer has no "
              f"chat template; using raw flavor")
    return f


# --------------------------------------------------------------------------- #
# inspect
# --------------------------------------------------------------------------- #

def cmd_inspect(config_path: str) -> int:
    cfg = _load_cfg(config_path)
    model, tok = _load_model_tok(cfg)
    from probe import _find_layers, _make_hook, _to_device
    from distill import extract_directions
    from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS

    layers = _find_layers(model, cfg.model_arch)
    num_layers = len(layers)
    hidden = model.config.hidden_size
    dev = "cpu" if not hasattr(model, "device") or str(model.device) == "cpu" else str(model.device)
    print(f"=== {cfg.model_id} ===")
    print(f"arch={cfg.model_arch} layers={num_layers} hidden={hidden}")

    # The model's OWN layer_types list (LFM2.x ships it, e.g.
    # ['conv', 'conv', ..., 'full_attention', ...]) — shows conv-vs-
    # attention at a glance without module introspection (TOOLKIT-FEEDBACK
    # §3.6). Conv layers must NEVER be projected.
    lt = getattr(getattr(model, "config", None), "layer_types", None)
    if lt:
        print(f"layer_types ({len(lt)}): {lt}")
        print("  per-layer: " + ", ".join(
            f"L{i}={t}" for i, t in enumerate(lt[: num_layers * 2])))
        non_conv = [i for i, t in enumerate(lt[:num_layers]) if "conv" not in str(t).lower()]
        if non_conv:
            print(f"  projectable (non-conv): {non_conv}")

    # Bias / weight-shape audit — catches silent-skip landmines.
    # Resolution is alias-aware (o_proj|out_proj, down_proj|w2) so LFM2.5
    # naming is audited properly instead of printing MISSING everywhere.
    print("\n--- weight audit (silent-skip landmines) ---")
    from sweep import _resolve_proj
    attention_layers, conv_layers = [], []
    for li in {0, num_layers // 2, num_layers - 1}:
        layer = layers[li]
        for wname in ("o_proj", "down_proj"):
            mods = _resolve_proj(layer, wname)
            if not mods:
                conv = getattr(layer, "conv", None)
                extra = " (conv block — must NEVER be projected)" if conv is not None else ""
                print(f"  L{li} {wname}: MISSING{extra}")
                continue
            projs = mods if isinstance(mods, list) else [mods]
            for mod in projs:
                w = mod.weight
                bias = getattr(mod, "bias", None)
                print(f"  L{li} {wname}: W{tuple(w.shape)} bias={'yes' if bias is not None else 'NO'}")
                if w.dim() == 2 and w.shape[0] != w.shape[1]:
                    print(f"      ^ NON-SQUARE: output dim {w.shape[0]} != input {w.shape[1]}")
                    print(f"        hidden-space directions ({hidden}) only fit output-dim==hidden weights")
                elif w.dim() != 2:
                    print(f"      ^ NON-2D ({w.dim()}D) — hidden-space directions do not apply")
    for li in range(num_layers):
        layer = layers[li]
        if getattr(layer, "conv", None) is not None:
            conv_layers.append(li)
        if _resolve_proj(layer, "o_proj") is not None:
            attention_layers.append(li)
    print(f"  summary: {len(attention_layers)} attention-out layers {attention_layers}, "
          f"{len(conv_layers)} conv layers {conv_layers} (conv must never be projected)")

    # Per-layer projection profile — answers "which layers can accept a
    # hidden-space direction edit" without any forward pass.
    print("\n--- layer profile (attn-out / conv / mlp-out per block) ---")
    for li in range(num_layers):
        conv = getattr(layers[li], "conv", None)
        ap = _resolve_proj(layers[li], "o_proj")
        dp = _resolve_proj(layers[li], "down_proj")
        bits = []
        if ap is not None:
            bits.append(f"attn-out:{tuple(ap.weight.shape)}")
        elif conv is not None:
            # conv blocks differ by arch: LFM2.5 exposes Conv1d as
            # conv.conv.weight (3D) + in_proj/out_proj Linears.
            cw = getattr(getattr(conv, "conv", None), "weight", None)
            if cw is None:
                cw = getattr(conv, "weight", None)
            bits.append(f"CONV:{tuple(cw.shape)}" if cw is not None else f"CONV:{type(conv).__name__}")
        else:
            bits.append("no-attn-out")
        bits.append(f"mlp:{tuple(dp.weight.shape)}" if dp is not None else "mlp:MISSING")
        print(f"  L{li:2d}: " + " | ".join(bits))

    # Quick direction separation profile (10 prompts each side)
    print("\n--- direction separation (10+10 prompts, diff_means) ---")
    import torch
    from collections import defaultdict
    def collect(prompts):
        store = defaultdict(list)
        handles = [layers[i].register_forward_hook(_make_hook(i, store))
                   for i in range(num_layers)]
        try:
            for p in prompts:
                inp = tok(p, return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    model(**inp)
        finally:
            for hh in handles:
                try: hh.remove()
                except Exception: pass
        return dict(store)
    h = list(DEFAULT_HARMFUL)[:10]
    g = list(DEFAULT_HARMLESS)[:10]
    acts_h = collect(h)
    acts_g = collect(g)
    dirs, scores = extract_directions(acts_h, acts_g, num_layers, hidden, "diff_means", 3, dev)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    print(f"  top-8 layers by separation: {[(li, round(s, 2)) for li, s in ranked[:8]]}")
    print(f"  tail-4 (last) layers:       {[(li, round(s, 2)) for li, s in ranked if li >= num_layers - 4]}")
    return 0


# --------------------------------------------------------------------------- #
# directions
# --------------------------------------------------------------------------- #

def cmd_directions(config_path: str, n_prompts: int | None, flavor: str | None,
                   dir_method: str | None) -> int:
    import torch
    cfg = _load_cfg(config_path)
    model, tok = _load_model_tok(cfg)
    from probe import _find_layers, _make_hook, _to_device, _collect_paired_output_phase
    from distill import extract_directions
    from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS
    from prompt_format import format_prompt

    flavor = _resolve_flavor(cfg, flavor, tok)
    layers = _find_layers(model, cfg.model_arch)
    num_layers = len(layers)
    hidden = model.config.hidden_size
    dev = "cpu"

    from collections import defaultdict
    def collect(prompts):
        store = defaultdict(list)
        handles = [layers[i].register_forward_hook(_make_hook(i, store))
                   for i in range(num_layers)]
        try:
            for p in prompts:
                inp = tok(p, return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    model(**inp)
        finally:
            for hh in handles:
                try: hh.remove()
                except Exception: pass
        return dict(store)

    n = n_prompts or getattr(cfg, "n_probe_prompts", 20)
    h = list(DEFAULT_HARMFUL)[:n]
    g = list(DEFAULT_HARMLESS)[:n]
    # Flavor axis: harvest from the SAME flavor the gates evaluate, so a
    # direction targets the measured mechanism (TOOLKIT-FEEDBACK §1b).
    h_f = [format_prompt(tok, p, flavor) for p in h]
    g_f = [format_prompt(tok, p, flavor) for p in g]
    dm = dir_method or getattr(cfg, "dir_method", "diff_means")
    probe_mode = "input"
    prefill = None
    if dm == "paired":
        # dir_method 'paired' NEEDS the output-phase paired data (same
        # prompts, unprimed refusal vs affirmative-prefill generation) —
        # running the paired math on input-phase prompt-set activations
        # would be diff_means in disguise. This is the LFM2.5 winning
        # direction source (see distill._paired docstring).
        prefill = getattr(cfg, "paired_prefill", "Sure, I can help with that.")
        max_new = getattr(cfg, "paired_max_new_tokens", 64)
        acts_h, acts_g = _collect_paired_output_phase(model, tok, h_f, layers, num_layers, dev, prefill, max_new)
        probe_mode = "paired"
        print(f"Paired output-phase direction collection: {len(h)} harmful prompts "
              f"(flavor={flavor}, unprimed refusal vs prefill={prefill!r}, max_new={max_new})")
    else:
        acts_h = collect(h_f)
        acts_g = collect(g_f)
    dirs, scores = extract_directions(acts_h, acts_g, num_layers, hidden, dm, 3, dev)
    missing = [i for i in range(num_layers) if i not in dirs]
    if missing:
        print(f"WARNING: {len(missing)}/{num_layers} layers produced NO direction: {missing}")
    print(f"Directions collected on {len(dirs)}/{num_layers} layers "
          f"(dir_method={dm}, probe={probe_mode}, flavor={flavor})")

    out_dir = PROJECT_DIR / "campaigns" / _slug(cfg.model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"directions-{flavor}.pt"
    torch.save({"dirs": {str(k): v.cpu() for k, v in dirs.items()},
                "scores": {str(k): v for k, v in scores.items()},
                "model_id": cfg.model_id, "dir_method": dm,
                "probe_mode": probe_mode, "prefill": prefill,
                "flavor": flavor, "n_prompts": len(h), "hidden": hidden}, path)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    print(f"Saved {len(dirs)} layer directions -> {path}")
    print(f"Top-5 layers: {[(li, round(s, 2)) for li, s in ranked[:5]]}")
    # attention-layer contrast (the recipe targets) — cheap check for
    # whether the paired signal actually lives in the attention blocks.
    from sweep import _resolve_proj
    attn = [li for li in range(num_layers) if _resolve_proj(layers[li], "o_proj") is not None]
    if attn:
        attn_scores = {li: round(scores.get(li, 0.0), 2) for li in attn}
        print(f"Attention-layer separation scores (targets): {attn_scores}")
    return 0


# --------------------------------------------------------------------------- #
# abl — apply ONE config to a fresh model
# --------------------------------------------------------------------------- #

def cmd_abl(config_path: str, method: str, alpha: float, layers_spec: str,
            weights_spec: str, dir_method: str | None, out_tag: str | None,
            passes: int, n_prompts: int | None, flavor: str | None,
            from_directions: str | None) -> int:
    cfg = _load_cfg(config_path)
    model, tok = _load_model_tok(cfg)
    import torch
    from probe import _find_layers, _make_hook, _to_device, _collect_paired_output_phase
    from distill import extract_directions
    from sweep import _apply_candidate
    from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS
    from prompt_format import format_prompt

    flavor = _resolve_flavor(cfg, flavor, tok)
    layers = _find_layers(model, cfg.model_arch)
    num_layers = len(layers)
    hidden = model.config.hidden_size
    target_layers = _parse_layers(layers_spec)
    weights = [w.strip() for w in weights_spec.split(",")]
    dm = dir_method or getattr(cfg, "dir_method", "diff_means")

    dirs = None
    scores: dict[int, float] = {}
    probe_mode = "input"
    prefill = None
    n_dirs = 0
    if from_directions:
        # Reuse a saved direction harvest (TOOLKIT-FEEDBACK §1d): one
        # expensive harvest serves two alphas/layer-sets, and a recorded
        # directions.pt is actually part of the evidence trail.
        data = _load_directions(from_directions)
        try:
            dirs = {int(k): v for k, v in data["dirs"].items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"FATAL: {from_directions} is not a harness directions "
                             f"file (no 'dirs' mapping): {exc}") from exc
        scores = {int(k): v for k, v in (data.get("scores") or {}).items()}
        probe_mode = data.get("probe_mode", "?")
        prefill = data.get("prefill")
        n_dirs = int(data.get("n_prompts", 0) or 0)
        if data.get("dir_method"):
            dm = data["dir_method"]  # the file is the authority on what was computed
        print(f"Loaded {len(dirs)} layer directions from {from_directions} "
              f"(dir_method={data.get('dir_method')}, flavor={data.get('flavor')}, "
              f"n_prompts={n_dirs})")

    else:
        # Collect directions fresh (never reuse stale ones silently)
        from collections import defaultdict
        def collect(prompts):
            store = defaultdict(list)
            handles = [layers[i].register_forward_hook(_make_hook(i, store))
                       for i in range(num_layers)]
            try:
                for p in prompts:
                    inp = tok(p, return_tensors="pt", truncation=True, max_length=128)
                    with torch.no_grad():
                        model(**inp)
            finally:
                for hh in handles:
                    try: hh.remove()
                    except Exception: pass
            return dict(store)

        n = n_prompts or getattr(cfg, "n_probe_prompts", 20)
        h = list(DEFAULT_HARMFUL)[:n]
        g = list(DEFAULT_HARMLESS)[:n]
        h_f = [format_prompt(tok, p, flavor) for p in h]
        g_f = [format_prompt(tok, p, flavor) for p in g]
        if dm == "paired":
            # 'paired' MUST use the output-phase paired data (same prompts,
            # unprimed refusal vs affirmative-prefill) — never the input-phase
            # prompt-set contrast, which is diff_means in disguise.
            prefill = getattr(cfg, "paired_prefill", "Sure, I can help with that.")
            max_new = getattr(cfg, "paired_max_new_tokens", 64)
            acts_h, acts_g = _collect_paired_output_phase(model, tok, h_f, layers, num_layers, "cpu", prefill, max_new)
            probe_mode = "paired"
            n_dirs = len(h)
            print(f"Paired output-phase directions: {len(h)} prompts "
                  f"(flavor={flavor}, prefill={prefill!r}, max_new={max_new})")
        else:
            acts_h = collect(h_f)
            acts_g = collect(g_f)
            n_dirs = len(h)
        dirs, scores = extract_directions(acts_h, acts_g, num_layers, hidden, dm, 3, "cpu")

    candidate = {"method": method, "dir_method": dm,
                 "target_layers": target_layers, "target_weights": weights,
                 "alpha": alpha, "passes": passes}
    _apply_candidate(model, dirs, None, candidate)
    applied = candidate.get("_applied", [])
    if not applied:
        raise SystemExit(
            f"FATAL: zero weight projections matched (layers={target_layers} "
            f"weights={weights}). Conv blocks / name aliases? Refusing to save "
            f"an unmodified model — run 'inspect' to see the layer profile."
        )
    print(f"Applied {method} alpha={alpha} passes={passes} layers={target_layers} "
          f"weights={weights} dir={dm} (probe={probe_mode})")
    for a in applied:
        print(f"  L{a['layer']:2d} {a['weight']:<9}: {a['shape']} rel_change={a['rel_change']:.4%} "
              f"norm {a['norm_before']:.3f} -> {a['norm_after']:.3f}")

    # save ablated model
    out_dir = PROJECT_DIR / "campaigns" / _slug(cfg.model_id) / (out_tag or f"abl-{method}-a{alpha}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    _copy_trust_remote_code(cfg, out_dir)
    manifest = {"model_id": cfg.model_id, "method": method, "dir_method": dm,
                "probe_mode": probe_mode, "prefill": prefill,
                "flavor": flavor, "from_directions": from_directions,
                "alpha": alpha, "passes": passes,
                "layers": target_layers, "weights": weights,
                "n_probe_prompts": n_dirs,
                "weight_changes": applied,
                "config": config_path, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Save the directions this run used/derived RIGHT NEXT to the manifest —
    # a recorded harvest is part of the evidence trail (TOOLKIT-FEEDBACK §1d);
    # another run can reuse it via --from-directions.
    dir_path = out_dir / f"directions-{flavor}.pt"
    torch.save({"dirs": {str(k): v.cpu() for k, v in dirs.items()},
                "scores": {str(k): v for k, v in scores.items()},
                "model_id": cfg.model_id, "dir_method": dm,
                "probe_mode": probe_mode, "prefill": prefill,
                "flavor": flavor, "n_prompts": n_dirs, "hidden": hidden,
                "from_directions": from_directions}, dir_path)
    print(f"Saved {len(dirs)} directions -> {dir_path}")
    print(f"Saved ablated model + manifest -> {out_dir}")
    return 0


def _load_directions(path: str) -> dict[str, Any]:
    """Load a harness directions.pt (keys may be str or int layer indices)."""
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, RuntimeError):
        # older torch or a file with non-primitive payload: allow the pickle
        # path — these files are local evidence, not untrusted input.
        return torch.load(path, map_location="cpu")


def _copy_trust_remote_code(cfg, out_dir: Path) -> None:
    """Make an ablated dir self-contained: copy the model's custom modeling
    code (*.py from trust_remote_code) into it, so a later
    ``from_pretrained(dir, trust_remote_code=True)`` works without the HF
    cache. ``save_pretrained`` does NOT save the code files."""
    try:
        from huggingface_hub import snapshot_download
        import shutil
        src = Path(snapshot_download(cfg.model_id))
        py_files = list(src.glob("*.py"))
        for f in py_files:
            shutil.copy2(f, out_dir / f.name)
        print(f"Copied {len(py_files)} remote-code files into {out_dir.name} (self-contained)")
    except Exception as exc:
        print(f"WARNING: could not copy trust_remote_code files: {exc}")


# --------------------------------------------------------------------------- #
# collect — run gates + behavior + capability, write a JSON bundle
# --------------------------------------------------------------------------- #

def cmd_collect(config_path: str, model_dir: str | None, transcript: bool,
                flavor: str | None) -> int:
    cfg = _load_cfg(config_path)
    import torch
    import gc
    import math
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from eval_split import build_split
    from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS
    from gates import run_gates
    from verify import run_mmlu_mini, _digest
    from prompt_format import format_prompt

    # held-out split (shared by gates and the pristine baseline)
    h = list(DEFAULT_HARMFUL); g = list(DEFAULT_HARMLESS)
    n = min(len(h), len(g))
    n_test = 5
    n_train = n - 2 * n_test
    split = build_split(h[:n], g[:n], train_size=n_train, tune_size=n_test,
                        test_size=n_test, seed=cfg.eval_split_seed)
    held_out = list(split.test)

    # For an ablated model, compute the pristine PPL + first-token-KL
    # baselines FIRST (sequential load: same peak RAM as one model), so the
    # perplexity_increase / first_token_kl gates are REAL instead of the
    # silent pass they get with no baseline.
    pristine_logprobs: dict[str, float] = {}
    pristine_logprobs_first: dict[str, Any] = {}
    pristine_benchmark_scores: dict[str, float] = {}
    if model_dir:
        print("Collecting pristine baselines (PPL + first-token KL + mmlu) on held-out prompts...")
        pmodel, ptok = _load_model_tok(cfg)
        flavor_r = _resolve_flavor(cfg, flavor, ptok)
        try:
            try:
                pristine_benchmark_scores["mmlu"] = run_mmlu_mini(pmodel, ptok, n=20)
                print(f"  pristine mmlu_mini: {pristine_benchmark_scores['mmlu']:.3f}")
            except Exception as exc:
                print(f"WARNING: pristine mmlu_mini failed: {exc}")
            for p in held_out:
                formatted = format_prompt(ptok, p, flavor_r)
                inp = ptok(formatted, return_tensors="pt", truncation=True,
                           max_length=cfg.max_seq_len)
                with torch.no_grad():
                    out = pmodel(**inp)
                lg = out.logits.float()
                pristine_logprobs_first[_digest(p)] = torch.log_softmax(lg[0, -1], dim=-1).cpu()
                # prompt-text PPL: logits[t] predict tokens[t+1], so
                # logits[0:N-1] align with tokens[1:N]. ([N-1:N-1] is an
                # empty slice on a plain forward — PPL over zero tokens.)
                cont = lg[0, 0: lg.shape[1] - 1]
                lp = torch.log_softmax(cont, dim=-1)
                tokens = inp["input_ids"][0, 1:]
                if cont.shape[0] != tokens.shape[0]:
                    continue
                chosen = lp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
                ppl = math.exp(-chosen.sum().item() / max(1, chosen.numel()))
                pristine_logprobs[_digest(p)] = ppl
            print(f"  pristine baselines on {len(held_out)} held-out prompts OK "
                  f"(flavor={flavor_r})")
        except Exception as exc:
            print(f"WARNING: pristine baseline collection failed ({exc}); "
                  f"PPL/KL gates will be skipped")
        finally:
            del pmodel, ptok
            gc.collect()

    # load the ablated model if given, else fresh
    if model_dir:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(model_dir)
    else:
        model, tok = _load_model_tok(cfg)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    # Resolve the flavor against THIS model's tokenizer (pristine runs and
    # ablated runs share the family tokenizer, so same result; the fallback
    # from chat->raw must be decided on a real tokenizer).
    flavor_r = _resolve_flavor(cfg, flavor, tok)

    # capability
    benchmark_scores = {}
    try:
        benchmark_scores["mmlu"] = run_mmlu_mini(model, tok, n=20)
    except Exception as exc:
        print("mmlu_mini failed:", exc)

    gen_transcript: list[dict[str, Any]] = []
    report = run_gates(model, tok, cfg, prompts=held_out, benchmark_scores=benchmark_scores,
                       pristine_logprobs=pristine_logprobs or None,
                       pristine_logprobs_first=pristine_logprobs_first or None,
                       pristine_benchmark_scores=pristine_benchmark_scores or None,
                       flavor=flavor_r, transcript=gen_transcript if transcript else None)
    out = {"model_id": cfg.model_id if not model_dir else model_dir,
           "eval_target": "pristine" if not model_dir else Path(model_dir).name,
           "prompt_flavor": flavor_r,
           "held_out_size": len(held_out), "benchmark_scores": benchmark_scores,
           "pristine_baseline_for": ["perplexity_increase", "first_token_kl", "capability"] if model_dir else [],
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    for k, v in report.items():
        if k in ("_enabled", "eval_pass", "held_out_size"):
            continue
        out[k] = {"passed": v["passed"], "value": v.get("value"), "detail": v["detail"]}
    out["eval_pass"] = report.get("eval_pass")

    # bundle keyed by eval target so pristine + ablated bundles coexist
    tag = "pristine" if not model_dir else Path(model_dir).name
    out_dir = PROJECT_DIR / "campaigns" / _slug(cfg.model_id) / f"collect-{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bundle.json"
    (out_dir / "bundle.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    # Every generated response (prompt + flavor + formatted + decoded text),
    # saved RIGHT NEXT TO the bundle — counts cannot distinguish "refuses"
    # from "Rams boilerplate", transcripts can (TOOLKIT-FEEDBACK §1a/§3.1).
    if transcript:
        tpath = out_dir / "transcript.json"
        tdoc = {"model_id": out["model_id"], "eval_target": out["eval_target"],
                "prompt_flavor": flavor_r, "held_out_prompts": held_out,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "generations": gen_transcript}
        (out_dir / "transcript.json").write_text(json.dumps(tdoc, indent=2, default=str), encoding="utf-8")
        print(f"Transcript ({len(gen_transcript)} generations) -> {tpath}")
    print(json.dumps(out, indent=2, default=str))
    print(f"\nBundle -> {path}")
    return 0


# --------------------------------------------------------------------------- #
# steer-test — alpha-response curve, no model re-save (steering hooks)
# --------------------------------------------------------------------------- #

def cmd_steer_test(config_path: str, alphas_spec: str, n_directions: int | None,
                   n_prompts: int | None, layers_spec: str | None,
                   flavor: str | None, from_directions: str | None = None) -> int:
    """Alpha grid x few prompts: transcripts + refusal + PPL per alpha.

    Non-destructive (runtime steering hooks/bias vectors — no weight edits,
    no model re-save). Answers "is the operating window empty" in ONE run:
    if NO alpha on the target layers removes refusal without PPL blowing up,
    the recipe band is empty and weight-projecting is a waste of CPU.
    """
    cfg = _load_cfg(config_path)
    model, tok = _load_model_tok(cfg)
    import torch
    import math
    from probe import _find_layers, _make_hook, _collect_paired_output_phase
    from distill import extract_directions
    from sweep import _apply_steering, _clear_steering_hooks, _STEERING_TARGETS
    from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS
    from prompt_format import format_prompt
    from verify import _decode_continuation, _response_is_refusal

    flavor = _resolve_flavor(cfg, flavor, tok)
    layers = _find_layers(model, cfg.model_arch)
    num_layers = len(layers)
    hidden = model.config.hidden_size
    target_layers = _parse_layers(layers_spec) if layers_spec else \
        list(getattr(cfg, "target_layers", []) or [])
    if not target_layers:
        from sweep import _resolve_proj
        target_layers = [li for li in range(num_layers)
                         if _resolve_proj(layers[li], "o_proj") is not None]
    dm = getattr(cfg, "dir_method", "diff_means")

    # ---- direction source: reuse a saved bundle, else cheap harvest ----
    # (flavor-aware, CPU-reduced). Reuse keeps the steer test on the SAME
    # directions the weight-edits used (evidence continuity).
    n_dir = n_directions or min(10, getattr(cfg, "n_probe_prompts", 20))
    h_dir = [format_prompt(tok, p, flavor) for p in list(DEFAULT_HARMFUL)[:n_dir]]
    g_dir = [format_prompt(tok, p, flavor) for p in list(DEFAULT_HARMLESS)[:n_dir]]
    directions_source = f"harvest: {dm} n={n_dir}"
    hook_targets: dict[str, str] = {}
    if from_directions:
        bundle = torch.load(from_directions, map_location="cpu",
                            weights_only=True)
        raw_dirs = bundle.get("dirs", bundle)
        dirs = {int(k): v for k, v in raw_dirs.items()}
        scores = {int(k): float(v)
                  for k, v in (bundle.get("scores") or {}).items()}
        dm = bundle.get("dir_method", dm)
        n_dir = bundle.get("n_prompts", n_dir)
        directions_source = (f"file:{Path(from_directions).name} "
                             f"({dm}, n={n_dir})")
        print(f"steer-test: reusing directions from {from_directions} "
              f"[{bundle.get('dir_method', '?')}, flavor="
              f"{bundle.get('flavor', '?')}, n={n_dir}]")
    elif dm == "paired":
        prefill = getattr(cfg, "paired_prefill", "Sure, I can help with that.")
        max_new = getattr(cfg, "paired_max_new_tokens", 64)
        acts_h, acts_g = _collect_paired_output_phase(model, tok, h_dir, layers, num_layers, "cpu", prefill, max_new)
        dirs, scores = extract_directions(acts_h, acts_g, num_layers, hidden, dm, 3, "cpu")
        directions_source = f"harvest: paired n={n_dir}"
    else:
        from collections import defaultdict
        store_h, store_g = defaultdict(list), defaultdict(list)
        def _collect(prompts, store):
            handles = [layers[i].register_forward_hook(_make_hook(i, store))
                       for i in range(num_layers)]
            try:
                for p in prompts:
                    inp = tok(p, return_tensors="pt", truncation=True, max_length=128)
                    with torch.no_grad():
                        model(**inp)
            finally:
                for hh in handles:
                    try: hh.remove()
                    except Exception: pass
            return dict(store)
        acts_h, acts_g = _collect(h_dir, store_h), _collect(g_dir, store_g)
        dirs, scores = extract_directions(acts_h, acts_g, num_layers, hidden, dm, 3, "cpu")
    no_dir = [li for li in target_layers if li not in dirs]
    if no_dir:
        print(f"WARNING: no direction for target layers {no_dir}; they will not steer")
    print(f"steer-test: {len(dirs)} directions ({dm}, flavor={flavor}, n={n_dir}), "
          f"target layers {target_layers}")

    # ---- alpha grid ----
    try:
        alphas = [float(a.strip()) for a in alphas_spec.split(",") if a.strip()]
    except ValueError as exc:
        raise SystemExit(f"FATAL: --alphas must be comma floats, got {alphas_spec!r}: {exc}") from exc
    if not alphas:
        raise SystemExit("FATAL: --alphas must be non-empty")

    test_harmful = [format_prompt(tok, p, flavor) for p in list(DEFAULT_HARMFUL)[: n_prompts or 3]]
    test_harmful_raw = list(DEFAULT_HARMFUL)[: n_prompts or 3]
    test_benign = [format_prompt(tok, p, flavor) for p in list(DEFAULT_HARMLESS)[:2]]
    test_benign_raw = list(DEFAULT_HARMLESS)[:2]
    test_pairs = [(raw, f) for raw, f in zip(test_harmful_raw, test_harmful)]

    def _ppl(formatted: str) -> float | None:
        inp = tok(formatted, return_tensors="pt", truncation=True,
                  max_length=getattr(cfg, "max_seq_len", 1024))
        with torch.no_grad():
            out = model(**inp)
        cont = out.logits[0, 0: out.logits.shape[1] - 1]
        logp = torch.log_softmax(cont.float(), dim=-1)
        tokens = inp["input_ids"][0, 1:]
        if cont.shape[0] != tokens.shape[0]:
            return None
        chosen = logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        return math.exp(-chosen.sum().item() / max(1, chosen.numel()))

    def _generate(formatted: str) -> str:
        inp = tok(formatted, return_tensors="pt", truncation=True)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=64, do_sample=False)
        return _decode_continuation(tok, out, inp["input_ids"])

    rows: list[dict[str, Any]] = []
    transcript: list[dict[str, Any]] = []
    # baseline (alpha=0): pristine PPL on the harmful test prompts
    base_ppl = {raw: _ppl(f) for raw, f in test_pairs}
    print(f"\n{'alpha':>6} | {'refusal':>8} | {'benign':>6} | {'PPL inc':>8} | notes")
    print("-" * 72)
    for alpha in [0.0] + alphas:
        _clear_steering_hooks()
        if alpha != 0.0:
            _apply_steering(model, layers, dirs, {"method": "steering",
                                                  "target_layers": target_layers,
                                                  "alpha": alpha,
                                                  "target_weights": []})
            hook_targets = dict(_STEERING_TARGETS)
        refusals = 0
        ppl_incs = []
        gen = []
        for raw, f in test_pairs:
            resp = _generate(f)
            refusals += int(_response_is_refusal(resp))
            gen.append({"alpha": alpha, "prompt": raw, "formatted": f,
                        "response": resp})
            p_abl = _ppl(f)
            if p_abl is not None and base_ppl.get(raw):
                ppl_incs.append(p_abl / base_ppl[raw] - 1.0)
        benign_responses = []
        for raw, f in zip(test_benign_raw, test_benign):
            benign_responses.append({"alpha": alpha, "prompt": raw, "formatted": f,
                                     "response": _generate(f)})
        mean_inc = sum(ppl_incs) / len(ppl_incs) if ppl_incs else float("nan")
        notes = []
        if refusals == len(test_pairs) and alpha > 0:
            notes.append("no effect (still refuses)")
        if alpha > 0 and refusals == 0:
            notes.append("refusal gone")
        if not math.isnan(mean_inc) and mean_inc > 0.15:
            notes.append(f"PPL +{mean_inc:.0%} > cap")
        if not math.isnan(mean_inc) and mean_inc < 0:
            notes.append("PPL dropped")
        rows.append({"alpha": alpha, "refusal": f"{refusals}/{len(test_pairs)}",
                     "benign": f"{sum(int(_response_is_refusal(b['response'])) for b in benign_responses)}/{len(benign_responses)}",
                     "ppl_increase": None if math.isnan(mean_inc) else round(mean_inc, 4),
                     "notes": "; ".join(notes) or "-"})
        transcript.extend(gen)
        transcript.extend(benign_responses)
        ppl_str = ("-" if rows[-1]["ppl_increase"] is None
                   else f"{rows[-1]['ppl_increase']:+.2%}")
        print(f"{alpha:>6.2f} | {rows[-1]['refusal']:>8} | {rows[-1]['benign']:>6} | "
              f"{ppl_str:>8} | {rows[-1]['notes']}")
    _clear_steering_hooks()

    # transcripts — the whole point: counts can't distinguish "refuses"
    # from "Rams boilerplate"
    for entry in transcript:
        if entry["alpha"] == 0.0:
            print(f"\n--- alpha 0.00 (pristine) | {entry['prompt'][:60]!r}")
        else:
            print(f"\n--- alpha {entry['alpha']:.2f} | {entry['prompt'][:60]!r}")
        print(f"    {entry['response']}")

    # evidence trail
    out_dir = PROJECT_DIR / "campaigns" / _slug(cfg.model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"steer-test-{time.strftime('%Y%m%d-%H%M%S')}.json"
    (out_dir / path.name).write_text(json.dumps({
        "model_id": cfg.model_id, "dir_method": dm, "prompt_flavor": flavor,
        "target_layers": target_layers, "alphas": [0.0] + alphas,
        "n_directions": n_dir, "directions_source": directions_source,
        "hook_targets": hook_targets,
        "separation_scores": {str(k): v for k, v in scores.items()},
        "rows": rows, "generations": transcript,
        "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2, default=str), encoding="utf-8")
    print(f"\nSteer-test evidence -> {out_dir / path.name}")
    return 0


# --------------------------------------------------------------------------- #
# list-campaigns
# --------------------------------------------------------------------------- #

def cmd_list_campaigns() -> int:
    import yaml
    base = PROJECT_DIR / "campaigns"
    if not base.exists():
        print("no campaigns dir")
        return 0
    print(f"{'Campaign':<38} {'Model':<38} {'Date':<12} {'Status'}")
    print("-" * 110)
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name == "templates":
            continue
        readme = d / "README.md"
        if not readme.exists():
            continue
        text = readme.read_text(encoding="utf-8")
        # cheap YAML frontmatter parse
        m = re.match(r"^---\n(.*?)\n---", text, re.S)
        meta = {}
        if m:
            try:
                meta = yaml.safe_load(m.group(1)) or {}
            except Exception:
                pass
        model = meta.get("target_model", "?")
        date = str(meta.get("date", "?"))
        status = str(meta.get("status", "?"))
        print(f"{d.name:<38} {model:<38} {date:<12} {status}")
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Absolver harness (guided, inspect-first)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="load model, print arch + landmine audit + separation profile")
    p.add_argument("config", help="path to models/<x>.yaml")
    p.set_defaults(fn=lambda a: cmd_inspect(a.config))

    p = sub.add_parser("directions", help="collect + save per-layer directions")
    p.add_argument("config")
    p.add_argument("--n-prompts", type=int, default=None, help="override n_probe_prompts (CPU cost control)")
    p.add_argument("--prompt-flavor", default=None, choices=["raw", "chat"],
                   help="prompt flavor for the harvest (default: config prompt_flavor, which is 'chat')")
    p.add_argument("--dir-method", default=None, choices=["diff_means", "paired", "svd", "leace"],
                   help="direction extraction override (default: config dir_method)")
    p.set_defaults(fn=lambda a: cmd_directions(a.config, a.n_prompts, a.prompt_flavor, a.dir_method))

    p = sub.add_parser("abl", help="apply ONE config to a fresh model")
    p.add_argument("config")
    p.add_argument("--method", required=True, help="advanced|mpoa|stacked_ablation|bias_vectors|direct_ablation|steering|lora|projected")
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--layers", required=True, help="e.g. 24,25,26,27 or 24-27")
    p.add_argument("--weights", default="o_proj,down_proj", help="comma list")
    p.add_argument("--dir-method", default=None, help="diff_means|paired|svd|leace")
    p.add_argument("--passes", type=int, default=1, help="projection passes (compound; default 1)")
    p.add_argument("--n-prompts", type=int, default=None, help="override n_probe_prompts (CPU cost control)")
    p.add_argument("--tag", default=None, help="output subdir tag")
    p.add_argument("--from-directions", default=None,
                   help="path to a saved directions-<flavor>.pt (reuse one harvest; skips collection)")
    p.add_argument("--prompt-flavor", default=None, choices=["raw", "chat"],
                   help="prompt flavor for direction harvest + manifest (default: config prompt_flavor)")
    p.set_defaults(fn=lambda a: cmd_abl(a.config, a.method, a.alpha, a.layers, a.weights,
                                        a.dir_method, a.tag, a.passes, a.n_prompts,
                                        a.prompt_flavor, a.from_directions))

    p = sub.add_parser("collect", help="run gates + capability on an ablated model dir (or pristine)")
    p.add_argument("config")
    p.add_argument("--model-dir", default=None, help="path to ablated model (default: pristine from config)")
    p.add_argument("--transcript", action="store_true",
                   help="save every gate generation (prompt, flavor, decoded text) next to bundle.json")
    p.add_argument("--prompt-flavor", default=None, choices=["raw", "chat"],
                   help="prompt flavor for gates (default: config prompt_flavor — 'chat')")
    p.set_defaults(fn=lambda a: cmd_collect(a.config, a.model_dir, a.transcript, a.prompt_flavor))

    p = sub.add_parser("steer-test", help="alpha-response curve via steering hooks (no model re-save)")
    p.add_argument("config")
    p.add_argument("--alphas", default="1.0,1.5,2.0,2.5,3.0", help="comma float grid (default 1.0..3.0)")
    p.add_argument("--n-directions", type=int, default=None, help="prompts for the direction harvest (default 10; CPU cost control)")
    p.add_argument("--n-prompts", type=int, default=3, help="harmful test prompts per alpha (default 3)")
    p.add_argument("--layers", default=None, help="target layers (default: config target_layers)")
    p.add_argument("--from-directions", default=None,
                   help="reuse a saved directions bundle (*.pt) instead of harvesting; "
                        "keeps the steer test on the SAME directions the weight-edits used")
    p.add_argument("--prompt-flavor", default=None, choices=["raw", "chat"],
                   help="prompt flavor for directions + test prompts (default: config prompt_flavor)")
    p.set_defaults(fn=lambda a: cmd_steer_test(a.config, a.alphas, a.n_directions,
                                               a.n_prompts, a.layers, a.prompt_flavor,
                                               a.from_directions))

    p = sub.add_parser("list-campaigns", help="list campaign statuses")
    p.set_defaults(fn=lambda a: cmd_list_campaigns())

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
