#!/usr/bin/env python3
# qdg_v0.5.0.py — Quick Dataset Generator (Quixoticode) — Optimized
# Version: 0.5.0
# - HF tokenizer/model are loaded once and cached (no reload per sample)
# - Progress updates and logs are throttled
# - Atomic write to output file
# - Better error handling and UI responsiveness
#
# Usage: python qdg_v0.5.0.py
#
# Optional requirements for model generation:
#   pip install customtkinter transformers torch
# Ollama usage requires the `ollama` CLI installed and available in PATH.

import os
import sys
import time
import json
import threading
import queue
import random
import subprocess
import traceback
import webbrowser
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox

# try optional UI lib
try:
    import customtkinter as ctk
    CTK = True
except Exception:
    CTK = False

# optional heavy deps
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

ROOT = os.path.abspath(os.path.dirname(__file__))
OUT_DIR = os.path.join(ROOT, "qdg_output")
EXAMPLES_DIR = os.path.join(ROOT, "examples")
LIBS_DIR = os.path.join(ROOT, "libs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EXAMPLES_DIR, exist_ok=True)
os.makedirs(LIBS_DIR, exist_ok=True)

LOG_PATH = os.path.join(ROOT, "qdg.log")

TEMPLATES = {
    "german_conversational": [
        ("User", "Wie geht's dir?"),
        ("Assistant", "Mir geht's gut, danke! Wie kann ich dir helfen?")
    ],
    "math_basic": [
        ("User", "Was ist 1+1?"),
        ("Assistant", "1+1 = 2")
    ],
}

# UI queue for communicating logs / progress to UI thread
_ui_q = queue.Queue()

def log(msg: str, to_ui: bool = True):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line)
    if to_ui:
        try:
            _ui_q.put(("log", line))
        except Exception:
            pass

def sanitize_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-","_") else "_" for c in s).strip("_") or "out"

def smart_read_json_lines(path):
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    out.append(obj)
                except Exception:
                    out.append({"text": line})
    except Exception as e:
        log(f"smart_read_json_lines failed: {e}")
    return out

# libs.json helpers
def load_libs(libdir=LIBS_DIR):
    libs_path = os.path.join(libdir, "libs.json")
    if not os.path.exists(libs_path):
        default = {
            "GREETINGS": ["Hallo", "Hi", "Servus", "Guten Tag"],
            "CLOSINGS": ["Viele Grüße", "LG", "Beste Grüße"],
            "MATH_SIMPLE": ["1+1", "2+2", "12*12"],
            "NAMES": ["Anna", "Ben", "Chris"]
        }
        try:
            with open(libs_path, "w", encoding="utf-8") as f:
                json.dump(default, f, indent=2, ensure_ascii=False)
            log(f"Default libs.json created at {libs_path}")
        except Exception as e:
            log(f"Could not create default libs.json: {e}")
    try:
        with open(libs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {k.upper(): v for k, v in data.items()}
    except Exception as e:
        log(f"Failed to load libs.json: {e}")
        return {}

# placeholder replacement (supports {KEY} and {{KEY}})
import re
_PLACEHOLDER_RE = re.compile(r"\{\{?\s*([A-Za-z0-9_]+)\s*\}?\}")

def substitute_placeholders(text: str, libs: dict):
    if not text or not libs:
        return text
    def repl(m):
        key = m.group(1).upper()
        arr = libs.get(key)
        if not arr:
            return m.group(0)
        return str(random.choice(arr))
    return _PLACEHOLDER_RE.sub(repl, text)

# -------------------------
# Model caching and helpers
# -------------------------
_MODEL_CACHE = {}   # HF models cached: name -> (tokenizer, model, device)
_OLLAMA_SEEN = set()  # for quieter logging

def get_cached_hf_model(model_name: str):
    """
    Load and cache HF tokenizer/model once per process.
    Raises RuntimeError if transformers not available.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not available (pip install transformers)")
    log(f"[HF Cache] Loading tokenizer/model '{model_name}' (only once)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    if TORCH_AVAILABLE and device == "cuda":
        model.to("cuda")
    _MODEL_CACHE[model_name] = (tokenizer, model, device)
    log(f"[HF Cache] Model '{model_name}' loaded and cached (device={device}).")
    return _MODEL_CACHE[model_name]

# small dummy context manager for no-torch paths
class dummy_context:
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False

def generate_with_hf_cached(model_name: str, prompt: str, opts: dict):
    """
    Generate with HF model (cached). opts must contain:
      - max_new_tokens, temperature, top_p, top_k
    """
    tokenizer, model, device = get_cached_hf_model(model_name)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    if TORCH_AVAILABLE and device == "cuda":
        inputs = {k:v.to("cuda") for k,v in inputs.items()}
    try:
        with (torch.no_grad() if TORCH_AVAILABLE else dummy_context()):
            gen = model.generate(
                **inputs,
                max_new_tokens=opts.get("max_new_tokens", 128),
                do_sample=True,
                temperature=opts.get("temperature", 0.9),
                top_p=opts.get("top_p", 0.95),
                top_k=opts.get("top_k", 50),
                pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        return text
    except Exception as e:
        raise

def generate_with_ollama_cached(model_name: str, prompt: str, timeout=120):
    """
    Call ollama for each prompt but only log "first time" message once for that model.
    (Depending on your ollama installation, the server may still need to load the model.)
    """
    if model_name not in _OLLAMA_SEEN:
        log(f"[Ollama] calling model '{model_name}'. First call may boot model; subsequent calls will be quieter.")
        _OLLAMA_SEEN.add(model_name)
    try:
        proc = subprocess.run(["ollama", "run", model_name], input=prompt, text=True, capture_output=True, timeout=timeout)
        out = proc.stdout.strip() or proc.stderr.strip()
        return out
    except subprocess.TimeoutExpired:
        raise RuntimeError("ollama run timed out")
    except FileNotFoundError:
        raise RuntimeError("ollama CLI not found (install & add to PATH)")
    except Exception as e:
        raise

# -------------------------
# Dataset builders (use cached HF)
# -------------------------
def build_from_examples_file(examples_path: str, out_path: str, count: int, libs: dict,
                             model_mode=None, model_name=None, model_opts=None,
                             stop_event=None, ui_cb=None):
    items = smart_read_json_lines(examples_path)
    if not items:
        raise RuntimeError("No examples read")
    total = count
    written = 0
    tmp_path = out_path + ".tmp"
    progress_step = max(1, total // 200)  # update about every 0.5%
    with open(tmp_path, "w", encoding="utf-8") as out_f:
        i = 0
        while written < total:
            if stop_event and stop_event.is_set():
                log("Generation stopped by user")
                break
            seed = random.choice(items)
            prompt_template = seed.get("prompt") or seed.get("input") or seed.get("text") or ""
            prompt = substitute_placeholders(prompt_template, libs)
            completion = seed.get("completion") or seed.get("output") or ""
            completion = substitute_placeholders(completion, libs)
            final_completion = completion or ""
            if model_mode in ("hf", "ollama") and model_name:
                try:
                    full_prompt = (model_opts.get("system_prompt","") + "\n" + prompt).strip()
                    if model_mode == "hf":
                        gen = generate_with_hf_cached(model_name, full_prompt, model_opts)
                    else:
                        gen = generate_with_ollama_cached(model_name, full_prompt, timeout=model_opts.get("timeout", 120))
                    final_completion = gen.strip() or final_completion
                except Exception as e:
                    log(f"Model generation failed for prompt (continuing): {e}")
                    final_completion = final_completion or f"[model_error] {e}"
            else:
                final_completion = final_completion or "(keine Completion im Beispiel)"
            entry = {"prompt": prompt, "completion": final_completion,
                     "meta": {"seed_idx": i, "source": os.path.basename(examples_path),
                              "mode": model_mode or "templates"}}
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1
            i += 1
            if ui_cb and (written % progress_step == 0 or written == total):
                ui_cb("progress", int(100 * (written / total)))
    # atomic replace
    try:
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            os.remove(out_path)
        except Exception:
            pass
        os.replace(tmp_path, out_path)
    return out_path

def build_from_templates(kind: str, out_path: str, count: int, libs: dict,
                         stop_event=None, ui_cb=None):
    available = list(TEMPLATES.keys())
    if kind not in available and kind != "random":
        kind = "random"
    written = 0
    tmp_path = out_path + ".tmp"
    progress_step = max(1, count // 200)
    with open(tmp_path, "w", encoding="utf-8") as out_f:
        while written < count:
            if stop_event and stop_event.is_set():
                log("Template generation stopped by user")
                break
            k = random.choice(available) if kind == "random" else kind
            conv = TEMPLATES.get(k, [])
            user_msgs = [m for t,m in conv if t == "User"]
            assistant_msgs = [m for t,m in conv if t == "Assistant"]
            prompt = substitute_placeholders(random.choice(user_msgs) if user_msgs else "Frage: ...", libs)
            completion = substitute_placeholders(random.choice(assistant_msgs) if assistant_msgs else "Antwort: ...", libs)
            entry = {"prompt": prompt, "completion": completion, "meta": {"kind": k, "source": "builtin_templates"}}
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1
            if ui_cb and (written % progress_step == 0 or written == count):
                ui_cb("progress", int(100 * (written / count)))
    os.replace(tmp_path, out_path)
    return out_path

def build_using_model_seeded(seeds, out_path, count, libs, model_mode, model_name, model_opts, stop_event=None, ui_cb=None):
    total = count
    written = 0
    tmp_path = out_path + ".tmp"
    progress_step = max(1, total // 200)
    with open(tmp_path, "w", encoding="utf-8") as out_f:
        while written < total:
            if stop_event and stop_event.is_set():
                log("Model-seeded generation stopped by user")
                break
            seed_prompt = substitute_placeholders(random.choice(seeds), libs)
            try:
                full_prompt = (model_opts.get("system_prompt","") + "\n" + seed_prompt).strip()
                if model_mode == "hf":
                    gen = generate_with_hf_cached(model_name, full_prompt, model_opts)
                else:
                    gen = generate_with_ollama_cached(model_name, full_prompt, timeout=model_opts.get("timeout", 120))
                completion = gen.strip()
            except Exception as e:
                completion = f"[model_error] {e}"
                log(f"Model generation failed (continuing): {e}")
            entry = {"prompt": seed_prompt, "completion": completion, "meta": {"source": f"{model_mode}:{model_name}"}}
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1
            if ui_cb and (written % progress_step == 0 or written == total):
                ui_cb("progress", int(100 * (written / total)))
    os.replace(tmp_path, out_path)
    return out_path

# -------------------------
# UI (customtkinter)
# -------------------------
class QDGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("QDG — Quick Dataset Generator (Quixoticode)")
        try:
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("blue")
        except Exception:
            pass
        self.ui_queue = queue.Queue()
        self.stop_event = threading.Event()
        self._build_ui()
        self._start_ui_poller()
        log("QDG UI started")

    def _build_ui(self):
        self.root.geometry("1100x720")
        main = ctk.CTkFrame(self.root, corner_radius=8)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        title_fr = ctk.CTkFrame(main, corner_radius=6)
        title_fr.pack(fill="x", padx=8, pady=(8,8))
        lbl = ctk.CTkLabel(title_fr, text="QDG — Quick Dataset Generator", font=("Roboto", 16, "bold"))
        lbl.pack(side="left", padx=(8,10))
        v_lbl = ctk.CTkLabel(title_fr, text="Version 0.5.0")
        v_lbl.pack(side="left")

        content = ctk.CTkFrame(main, corner_radius=6)
        content.pack(fill="both", expand=True, padx=8, pady=4)
        left = ctk.CTkFrame(content, corner_radius=6)
        left.pack(side="left", fill="both", expand=True, padx=(8,6), pady=6)
        right = ctk.CTkFrame(content, width=360, corner_radius=6)
        right.pack(side="right", fill="y", padx=(6,8), pady=6)

        opt_fr = ctk.CTkFrame(left, corner_radius=6)
        opt_fr.pack(fill="both", expand=True, padx=6, pady=6)

        ctk.CTkLabel(opt_fr, text="Mode:", anchor="w").pack(fill="x", padx=8, pady=(8,4))
        self.mode_cb = ctk.CTkComboBox(opt_fr, values=["templates","examples","hf_model","ollama_model"], command=self._on_mode_change)
        self.mode_cb.set("examples")
        self.mode_cb.pack(fill="x", padx=8)

        ctk.CTkLabel(opt_fr, text="Beispieldatei (examples.jsonl):", anchor="w").pack(fill="x", padx=8, pady=(8,4))
        frm_examples = ctk.CTkFrame(opt_fr, corner_radius=6)
        frm_examples.pack(fill="x", padx=8)
        self.ent_examples = ctk.CTkEntry(frm_examples, placeholder_text="z.B. examples/example.jsonl")
        self.ent_examples.pack(side="left", fill="x", expand=True, padx=(6,6))
        btn_examples = ctk.CTkButton(frm_examples, text="Öffnen", width=80, command=self._pick_examples_file)
        btn_examples.pack(side="right", padx=(0,6))

        ctk.CTkLabel(opt_fr, text="libs-Ordner (für libs.json):", anchor="w").pack(fill="x", padx=8, pady=(8,4))
        frm_libs = ctk.CTkFrame(opt_fr)
        frm_libs.pack(fill="x", padx=8)
        self.ent_libs = ctk.CTkEntry(frm_libs, placeholder_text="libs (enthält libs.json)")
        self.ent_libs.pack(side="left", fill="x", expand=True, padx=(6,6))
        btn_libs = ctk.CTkButton(frm_libs, text="Öffnen", width=80, command=self._pick_libs_folder)
        btn_libs.pack(side="right", padx=(0,6))

        ctk.CTkLabel(opt_fr, text="Modell (HF Repo oder OLLAMA Modell):", anchor="w").pack(fill="x", padx=8, pady=(8,4))
        model_fr = ctk.CTkFrame(opt_fr, fg_color="transparent")
        model_fr.pack(fill="x", padx=8)

        default_models = ["microsoft/phi-3-mini-4k-instruct", "distilgpt2"]
        # try to detect local ollama models (non-blocking short timeout)
        try:
            ollama_models = []
            proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=1)
            if proc.returncode == 0:
                lines = [l.strip() for l in (proc.stdout or proc.stderr).splitlines() if l.strip()]
                if len(lines) > 1:
                    ollama_models = [line.split()[0] for line in lines[1:]]
            default_models += [f"ollama:{m}" for m in ollama_models]
        except Exception:
            pass

        self.model_selector = ctk.CTkComboBox(model_fr, values=default_models)
        self.model_selector.pack(side="left", fill="x", expand=True, padx=(0, 6))
        if default_models:
            self.model_selector.set(default_models[0])
        refresh_btn = ctk.CTkButton(model_fr, text="🔄", width=35, command=self._refresh_models)
        refresh_btn.pack(side="right")

        ctk.CTkLabel(opt_fr, text="System-Prompt (optional):", anchor="w").pack(fill="x", padx=8, pady=(8,4))
        self.txt_sys = ctk.CTkTextbox(opt_fr, height=80)
        self.txt_sys.pack(fill="x", padx=8)

        params_fr = ctk.CTkFrame(opt_fr)
        params_fr.pack(fill="x", padx=8, pady=(8,6))
        ctk.CTkLabel(params_fr, text="Samples:").grid(row=0, column=0, sticky="w", padx=4)
        self.spin_samples = ctk.CTkEntry(params_fr, width=90)
        self.spin_samples.grid(row=0, column=1, sticky="w", padx=4)
        self.spin_samples.insert(0, "200")

        ctk.CTkLabel(params_fr, text="Max New Tokens:").grid(row=1, column=0, sticky="w", padx=4, pady=(6,0))
        self.ent_max_tokens = ctk.CTkEntry(params_fr, width=90)
        self.ent_max_tokens.grid(row=1, column=1, sticky="w", padx=4, pady=(6,0))
        self.ent_max_tokens.insert(0, "128")

        ctk.CTkLabel(params_fr, text="Temperature:").grid(row=1, column=2, sticky="w", padx=4, pady=(6,0))
        self.ent_temp = ctk.CTkEntry(params_fr, width=90)
        self.ent_temp.grid(row=1, column=3, sticky="w", padx=4, pady=(6,0))
        self.ent_temp.insert(0, "0.9")

        ctk.CTkLabel(params_fr, text="Top-p:").grid(row=2, column=0, sticky="w", padx=4, pady=(6,0))
        self.ent_topp = ctk.CTkEntry(params_fr, width=90)
        self.ent_topp.grid(row=2, column=1, sticky="w", padx=4, pady=(6,0))
        self.ent_topp.insert(0, "0.95")

        ctk.CTkLabel(params_fr, text="Top-k:").grid(row=2, column=2, sticky="w", padx=4, pady=(6,0))
        self.ent_topk = ctk.CTkEntry(params_fr, width=90)
        self.ent_topk.grid(row=2, column=3, sticky="w", padx=4, pady=(6,0))
        self.ent_topk.insert(0, "50")

        ctk.CTkLabel(opt_fr, text="Ausgabedatei (JSONL):", anchor="w").pack(fill="x", padx=8, pady=(8,4))
        out_row = ctk.CTkFrame(opt_fr)
        out_row.pack(fill="x", padx=8)
        self.ent_out = ctk.CTkEntry(out_row)
        self.ent_out.pack(side="left", fill="x", expand=True, padx=(0,6))
        self.ent_out.insert(0, f"qdg_{int(time.time())}.jsonl")
        btn_out = ctk.CTkButton(out_row, text="Ordner öffnen", width=110, command=lambda: os.startfile(OUT_DIR) if sys.platform == "win32" else webbrowser.open(OUT_DIR))
        btn_out.pack(side="right")

        action_row = ctk.CTkFrame(opt_fr)
        action_row.pack(fill="x", padx=8, pady=(8,10))
        self.btn_generate = ctk.CTkButton(action_row, text="Generieren", fg_color="#16a34a", hover_color="#15803d", command=self._on_generate)
        self.btn_generate.pack(side="left", expand=True, fill="x", padx=(0,6))
        self.btn_cancel = ctk.CTkButton(action_row, text="Abbrechen", fg_color="#ef4444", hover_color="#dc2626", command=self._on_cancel, state="disabled")
        self.btn_cancel.pack(side="left", expand=True, fill="x")

        ctk.CTkLabel(right, text="Fortschritt", anchor="w", font=("Roboto", 12, "bold")).pack(fill="x", padx=8, pady=(8,4))
        self.pb = ctk.CTkProgressBar(right)
        self.pb.set(0.0)
        self.pb.pack(fill="x", padx=8, pady=(0,8))

        ctk.CTkLabel(right, text="Log", anchor="w", font=("Roboto", 12, "bold")).pack(fill="x", padx=8, pady=(8,4))
        self.txt_log = ctk.CTkTextbox(right, height=20)
        self.txt_log.pack(fill="both", expand=True, padx=8, pady=(0,8))

        self.ent_examples.insert(0, os.path.join(EXAMPLES_DIR, "example.jsonl"))
        self.ent_libs.insert(0, LIBS_DIR)
        self._on_mode_change(self.mode_cb.get())

    def _start_ui_poller(self):
        self.root.after(200, self._ui_poller)

    def _ui_poller(self):
        try:
            while True:
                typ, payload = _ui_q.get_nowait()
                if typ == "log":
                    self._append_log(payload, internal=True)
                elif typ == "progress":
                    self._set_progress(payload)
                elif typ == "done":
                    self.btn_generate.configure(state="normal")
                    self.btn_cancel.configure(state="disabled")
                    messagebox.showinfo("Fertig", f"Datensatz gespeichert:\n{payload}")
                elif typ == "error":
                    self.btn_generate.configure(state="normal")
                    self.btn_cancel.configure(state="disabled")
                    messagebox.showerror("Fehler", str(payload))
        except queue.Empty:
            pass
        self.root.after(200, self._ui_poller)

    def _append_log(self, txt, internal=False):
        try:
            self.txt_log.insert("end", f"{txt}\n")
            self.txt_log.see("end")
        except Exception:
            pass
        if not internal:
            print(txt)

    def _set_progress(self, pct):
        try:
            val = float(pct) / 100.0
            if val < 0: val = 0.0
            if val > 1: val = 1.0
            self.pb.set(val)
        except Exception:
            pass

    def _pick_examples_file(self):
        p = filedialog.askopenfilename(title="Beispieldatei auswählen", filetypes=[("JSONL", "*.jsonl"), ("Alle Dateien", "*.*")], initialdir=EXAMPLES_DIR)
        if p:
            self.ent_examples.delete(0, "end"); self.ent_examples.insert(0, p)

    def _pick_libs_folder(self):
        d = filedialog.askdirectory(title="libs-Ordner auswählen", initialdir=LIBS_DIR)
        if d:
            self.ent_libs.delete(0,"end"); self.ent_libs.insert(0, d)

    def _on_mode_change(self, val):
        is_model_mode = val in ("hf_model", "ollama_model")
        self.model_selector.configure(state="normal" if is_model_mode else "disabled")
        self.txt_sys.configure(state="normal" if is_model_mode else "disabled")

    def _refresh_models(self):
        log("Aktualisiere Modell-Liste...")
        try:
            ollama_models = []
            proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=2)
            if proc.returncode == 0:
                lines = [l.strip() for l in (proc.stdout or proc.stderr).splitlines() if l.strip()]
                if len(lines) > 1:
                    ollama_models = [line.split()[0] for line in lines[1:]]
            models = ["microsoft/phi-3-mini-4k-instruct", "distilgpt2"] + [f"ollama:{m}" for m in ollama_models]
            self.model_selector.configure(values=models)
            if models:
                self.model_selector.set(models[0])
            self._append_log("Modell-Liste aktualisiert.")
        except Exception as e:
            self._append_log(f"Modell-Refresh fehlgeschlagen: {e}")

    def _on_generate(self):
        mode = self.mode_cb.get()
        out_fname = self.ent_out.get().strip() or f"qdg_{int(time.time())}.jsonl"
        out_path = os.path.join(OUT_DIR, sanitize_name(out_fname))
        libs = load_libs(self.ent_libs.get().strip() or LIBS_DIR)
        try:
            samples = int(self.spin_samples.get())
            max_new_tokens = int(self.ent_max_tokens.get())
            temperature = float(self.ent_temp.get())
            top_p = float(self.ent_topp.get())
            top_k = int(self.ent_topk.get())
        except Exception as e:
            messagebox.showerror("Parameter-Fehler", f"Ungültiger numerischer Parameter: {e}")
            return

        model_identifier = self.model_selector.get().strip()
        model_mode, model_name = None, None
        if mode == "hf_model":
            model_mode, model_name = "hf", model_identifier
        elif mode == "ollama_model":
            model_mode = "ollama"
            model_name = model_identifier.replace("ollama:", "")

        if mode in ("hf_model", "ollama_model") and not model_name:
            messagebox.showerror("Modell fehlt", "Bitte ein Modell auswählen oder eingeben.")
            return

        examples_path = self.ent_examples.get().strip()
        if mode == "examples" and not os.path.exists(examples_path):
            messagebox.showerror("Datei fehlt", f"Beispieldatei nicht gefunden: {examples_path}")
            return

        model_opts = {
            "system_prompt": self.txt_sys.get("0.0", "end").strip(),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "timeout": 180
        }

        # Preload HF in background to reduce initial latency
        if model_mode == "hf":
            def _preload():
                try:
                    get_cached_hf_model(model_name)
                except Exception as e:
                    log(f"Preload failed: {e}")
            threading.Thread(target=_preload, daemon=True).start()

        self.btn_generate.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.stop_event.clear()
        thread = threading.Thread(target=self._generate_bg, args=(mode, examples_path, out_path, samples, libs, model_mode, model_name, model_opts))
        thread.daemon = True
        thread.start()

    def _on_cancel(self):
        self.stop_event.set()
        self._append_log("Stopp-Anfrage gesendet. Prozess wird nach dem aktuellen Element beendet.")
        self.btn_cancel.configure(state="disabled")

    def _generate_bg(self, mode, examples_path, out_path, samples, libs, model_mode, model_name, model_opts):
        try:
            self._ui_cb("log", f"Starte Generierung: mode={mode}, samples={samples}, model={model_mode}:{model_name}")
            if mode == "templates":
                res = build_from_templates("random", out_path, samples, libs, self.stop_event, self._ui_cb)
            elif mode == "examples":
                res = build_from_examples_file(examples_path, out_path, samples, libs,
                                               model_mode=None, model_name=None, model_opts=None,
                                               stop_event=self.stop_event, ui_cb=self._ui_cb)
            elif mode in ("hf_model", "ollama_model"):
                seeds = []
                if examples_path and os.path.exists(examples_path):
                    exs = smart_read_json_lines(examples_path)
                    for e in exs:
                        p = e.get("prompt") or e.get("input") or e.get("text")
                        if p:
                            seeds.append(p)
                if not seeds:
                    seeds = ["Stelle eine kurze Frage und gib eine passende Antwort auf Deutsch."]
                res = build_using_model_seeded(seeds, out_path, samples, libs, model_mode, model_name, model_opts,
                                              stop_event=self.stop_event, ui_cb=self._ui_cb)
            else:
                res = build_from_templates("random", out_path, samples, libs, self.stop_event, self._ui_cb)
            _ui_q.put(("done", res))
        except Exception as e:
            tb = traceback.format_exc()
            log(f"Generierung fehlgeschlagen: {e}\n{tb}")
            _ui_q.put(("error", f"{e}"))

    def _ui_cb(self, typ, payload):
        try:
            _ui_q.put((typ, payload))
        except Exception:
            pass

# Entrypoint
def main():
    if not CTK:
        print("customtkinter nicht installiert. Bitte 'pip install customtkinter' und starte erneut.")
        root = tk.Tk()
        root.withdraw()
        res = messagebox.askyesno("customtkinter fehlt", "customtkinter ist nicht installiert. Möchtest du jetzt die einfache UI nutzen?")
        if not res:
            return
    root = ctk.CTk() if CTK else tk.Tk()
    app = QDGApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
