#!/usr/bin/env python3
# qdg.py — Quick Dataset Generator (QDG) — Quixoticode
# UI + Engine to generate high-quality, dynamic datasets (multilingual + multi-user)
#
# Features:
# - Templates with placeholders (e.g. "Hallo {name}, ...")
# - Import personal records (CSV/JSON/JSONL) to create personalized examples
# - Paraphrase optionally with transformers if available, otherwise rule-based augmentation
# - Multi-turn conversation generation, style variations, formal/informal (DE) switching
# - Weighting of categories/templates, splits (train/val/test), progress UI, preview, cancel
# - Exposes QDGEngine API for programmatic use (e.g. integrate into QMT)
#
# Usage: python qdg.py
#
# Author: Quixoticode

import os
import sys
import time
import json
import csv
import math
import random
import threading
import queue
import re
import datetime
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox

# Optional heavy deps for paraphrasing
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# -----------------------
# Utilities
# -----------------------
APP_CREDIT = "Quixoticode"
BASE_OUT = os.path.join(os.path.dirname(__file__), "qdg_output")
LOG_PATH = os.path.join(os.path.dirname(__file__), "qdg.log")
ensure_dir = lambda p: os.makedirs(p, exist_ok=True) or p

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line)

def sanitize_name(s: str):
    s = s.strip()
    s = re.sub(r"[^\w\-\_\. ]+", "_", s)
    return s or f"dataset_{int(time.time())}"

# Safe dict for formatting templates
class SafeDict(dict):
    def __missing__(self, key):
        return f"<{key}>"

# -----------------------
# Simple built-in synonyms (expandable via import)
# -----------------------
# Small curated synonyms to generate variation (German / English)
SYNONYM_DICT = {
    "de": {
        "hallo": ["hi", "servus", "grüß dich"],
        "wie geht's": ["wie läuft's", "wie ist es bei dir", "wie geht es dir"],
        "bitte": ["sei so nett", "bitte sehr", "wär nett"],
        "danke": ["vielen Dank", "merci", "dankeschön"],
    },
    "en": {
        "hello": ["hi", "hey", "greetings"],
        "how are you": ["how's it going", "how do you do", "how are you doing"],
        "please": ["kindly", "please do", "do me the favor"],
        "thanks": ["thank you", "cheers"],
    }
}

def synonym_substitute(text: str, lang="de", strength=0.2):
    # Replace occasional tokens with synonyms using the small dict
    if lang not in SYNONYM_DICT:
        return text
    words = re.split(r"(\W+)", text)  # keep punctuation
    for i, w in enumerate(words):
        if random.random() > strength: 
            continue
        lw = w.lower()
        if lw in SYNONYM_DICT[lang]:
            subs = SYNONYM_DICT[lang][lw]
            choice = random.choice(subs)
            # preserve capitalization
            if w and w[0].isupper():
                choice = choice.capitalize()
            words[i] = choice
    return "".join(words)

# -----------------------
# Paraphraser wrapper (optional transformers)
# -----------------------
class Paraphraser:
    def __init__(self, model_name=None, lang="de"):
        self.model_name = model_name
        self.lang = lang
        self.use_transformers = False
        self.model = None
        self.tokenizer = None
        if TRANSFORMERS_AVAILABLE and model_name:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.use_transformers = True
                log(f"[Paraphraser] using transformers model {model_name}")
            except Exception as e:
                log(f"[Paraphraser] failed to load transformers model {model_name}: {e}")
                self.use_transformers = False

    def paraphrase(self, text: str, num_return=1, strength=0.6):
        """
        strength: 0..1, lower = less change (we use heuristics)
        """
        if not text or not text.strip():
            return text
        if self.use_transformers and self.model and self.tokenizer:
            try:
                # basic seq2seq paraphrase using provided model
                # Note: user-supplied model should be a paraphraser (e.g., t5-small-finetuned-paraphrase)
                inputs = self.tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
                out = self.model.generate(inputs, num_beams=4, num_return_sequences=1, max_length=512, early_stopping=True)
                cand = self.tokenizer.decode(out[0], skip_special_tokens=True)
                return cand
            except Exception as e:
                log(f"[Paraphraser] transformers paraphrase error: {e}")
                # fallback
        # fallback heuristic paraphrase
        # operations: synonym_subst, minor reordering, punctuation tweaks
        p = text
        # small probabilistic suffix/prefix reordering
        if random.random() < strength * 0.35:
            # split on comma or 'und' etc.
            parts = re.split(r"([,;:-])", p)
            if len(parts) > 1:
                random.shuffle(parts)
                p = "".join(parts)
        # synonym substitution
        p = synonym_substitute(p, lang=self.lang[:2], strength=min(0.5, strength))
        # swap simple phrases (greetings)
        if random.random() < 0.2 * strength:
            p = p.replace("?", ".")
        return p

# -----------------------
# Template & Record classes
# -----------------------
class Template:
    def __init__(self, name="tpl", prompt_tpl="{user}", completion_tpl="{assistant}", weight=1, lang="de", style="neutral", multi_turn=False):
        self.name = name
        self.prompt_tpl = prompt_tpl
        self.completion_tpl = completion_tpl
        self.weight = weight
        self.lang = lang  # "de" or "en"
        self.style = style
        self.multi_turn = multi_turn

class RecordStore:
    def __init__(self):
        self.records = []  # list of dicts
        self.fields = set()

    def load_csv(self, path, delimiter=",", encoding="utf-8"):
        try:
            with open(path, newline="", encoding=encoding) as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for r in reader:
                    self.add_record(r)
            log(f"[RecordStore] loaded CSV: {path} ({len(self.records)} records)")
            return True, None
        except Exception as e:
            return False, str(e)

    def load_json(self, path, encoding="utf-8"):
        try:
            with open(path, "r", encoding=encoding) as f:
                data = json.load(f)
                if isinstance(data, list):
                    for it in data:
                        if isinstance(it, dict):
                            self.add_record(it)
                elif isinstance(data, dict):
                    self.add_record(data)
            log(f"[RecordStore] loaded JSON: {path} ({len(self.records)} records)")
            return True, None
        except Exception as e:
            return False, str(e)

    def load_jsonl(self, path, encoding="utf-8"):
        try:
            with open(path, "r", encoding=encoding) as f:
                for line in f:
                    line=line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            self.add_record(obj)
                    except Exception:
                        # skip
                        continue
            log(f"[RecordStore] loaded JSONL: {path} ({len(self.records)} records)")
            return True, None
        except Exception as e:
            return False, str(e)

    def add_record(self, rec: dict):
        if not isinstance(rec, dict):
            return
        self.records.append(rec)
        self.fields.update(rec.keys())

    def sample(self):
        if not self.records:
            return {}
        return random.choice(self.records)

# -----------------------
# QDGEngine: generation logic (programmatic)
# -----------------------
class QDGEngine:
    def __init__(self, templates=None, records: RecordStore=None, paraphraser: Paraphraser=None):
        self.templates = templates or []
        self.records = records or RecordStore()
        self.paraphraser = paraphraser or Paraphraser()  # may be non-functional
        # default categories for convenience
        self.default_templates_if_empty()

    def default_templates_if_empty(self):
        if self.templates:
            return
        # small defaults for bootstrapping (DE + EN)
        self.templates = [
            Template(name="smalltalk_de", prompt_tpl="Hallo {name}, wie geht's dir?", completion_tpl="Mir geht es gut, danke! Und dir?", weight=3, lang="de"),
            Template(name="identity_de", prompt_tpl="Wer bist du?", completion_tpl="Ich bin Redlight 1.0 Mini, erstellt von Quixoticode.", weight=1, lang="de"),
            Template(name="smalltalk_en", prompt_tpl="Hello, how are you?", completion_tpl="I'm fine, thanks! How about you?", weight=2, lang="en"),
            Template(name="math_en", prompt_tpl="What is {a} plus {b}?", completion_tpl="{sum}", weight=2, lang="en"),
        ]

    def add_template(self, tpl: Template):
        self.templates.append(tpl)

    def list_template_names(self):
        return [t.name for t in self.templates]

    def _choose_template(self, lang=None):
        # weighted random choose among templates filtered by lang if provided
        tpls = [t for t in self.templates if (lang is None or t.lang.startswith(lang))]
        if not tpls:
            tpls = self.templates[:]
        weights = [max(0.001, t.weight) for t in tpls]
        chosen = random.choices(tpls, weights=weights, k=1)[0]
        return chosen

    def _fill_template(self, tpl: Template, record: dict=None):
        # prepare mapping (SafeDict) with record + some dynamic variables
        ctx = SafeDict()
        if record:
            for k, v in record.items():
                # flatten types to strings
                ctx[k] = str(v)
        # add dynamic fields
        ctx["now"] = datetime.datetime.utcnow().isoformat()
        # simple numeric fields for math templates if placeholders found
        if "{a}" in tpl.prompt_tpl or "{b}" in tpl.prompt_tpl or "{sum}" in tpl.completion_tpl:
            a = random.randint(1, 200)
            b = random.randint(1, 200)
            ctx["a"] = a
            ctx["b"] = b
            ctx["sum"] = a + b
        # various names if name not provided
        if "name" not in ctx or not ctx.get("name"):
            ctx["name"] = record.get("name") if record and record.get("name") else random.choice(["Anna","Lukas","Mia","Tim","Sophie","Paul"])
        # format strings safe
        try:
            prompt = tpl.prompt_tpl.format_map(ctx)
        except Exception:
            prompt = tpl.prompt_tpl
        try:
            completion = tpl.completion_tpl.format_map(ctx)
        except Exception:
            completion = tpl.completion_tpl
        return prompt, completion

    def _apply_style(self, text: str, style="neutral", lang="de", informal=True):
        # minor stylistic modifications
        t = text
        if lang.startswith("de"):
            if informal:
                # ensure "du" forms, maybe add emoji sometimes
                if random.random() < 0.1:
                    t = t + " 🙂"
            else:
                # more formal: replace contractions (basic)
                t = t.replace("wie geht's", "wie geht es")
        else:
            if random.random() < 0.05:
                t = t + " 🙂"
        return t

    def generate_sample(self, schema="prompt_completion", lang="de", paraphrase_level=0.0, style="neutral", informal=True, multi_turn=False, max_turns=3):
        """
        Generates one sample according to schema:
        schema: "prompt_completion" | "user_assistant" | "instruction_io" | "chat_messages"
        paraphrase_level: 0..1 (higher -> more paraphrase)
        """
        tpl = self._choose_template(lang)
        record = self.records.sample() if self.records.records else {}
        prompt, completion = self._fill_template(tpl, record)
        # style
        prompt = self._apply_style(prompt, style=tpl.style, lang=tpl.lang, informal=informal)
        completion = self._apply_style(completion, style=tpl.style, lang=tpl.lang, informal=informal)

        # optionally paraphrase both prompt+completion based on level
        if paraphrase_level > 0.01:
            prompt = self.paraphraser.paraphrase(prompt, strength=paraphrase_level)
            completion = self.paraphraser.paraphrase(completion, strength=paraphrase_level)

        # multi-turn: create small dialog
        if multi_turn or tpl.multi_turn:
            turns = []
            user_turn = {"role":"user", "text": prompt}
            turns.append(user_turn)
            # assistant reply
            assistant_turn = {"role":"assistant", "text": completion}
            turns.append(assistant_turn)
            # create follow-ups
            n_follow = random.randint(0, max(0, max_turns-1))
            for i in range(n_follow):
                # create a follow-up question (paraphrase of user or a clarifying Q)
                if random.random() < 0.5:
                    follow_q = "Könntest du das genauer erklären?" if tpl.lang.startswith("de") else "Could you explain that further?"
                else:
                    follow_q = "Und was bedeutet das genau?" if tpl.lang.startswith("de") else "And what exactly does that mean?"
                follow_q = self.paraphraser.paraphrase(follow_q, strength=paraphrase_level*0.5) if paraphrase_level>0.1 else follow_q
                turns.append({"role":"user", "text": follow_q})
                # assistant short reply
                short_ans = self.paraphraser.paraphrase("Natürlich, gern.", strength=paraphrase_level*0.4) if paraphrase_level>0.1 else ("Natürlich, gern." if tpl.lang.startswith("de") else "Sure, happy to.")
                turns.append({"role":"assistant", "text": short_ans})

            # produce output as chat_messages schema or flatten
            if schema == "chat_messages":
                return {"messages": turns}
            else:
                # flatten to single prompt/completion: join user messages as prompt, join assistant messages as completion
                all_user = "\n".join([t["text"] for t in turns if t["role"]=="user"])
                all_assistant = "\n".join([t["text"] for t in turns if t["role"]=="assistant"])
                prompt = all_user
                completion = all_assistant

        # format according to schema
        if schema == "prompt_completion":
            return {"prompt": prompt, "completion": completion}
        elif schema == "user_assistant":
            return {"user": prompt, "assistant": completion}
        elif schema == "instruction_io":
            return {"instruction": "Beantworte die Anfrage.", "input": prompt, "output": completion}
        elif schema == "chat_messages":
            # create minimal chat array
            return {"messages": [{"role":"user","text":prompt}, {"role":"assistant","text":completion}]}
        else:
            return {"user": prompt, "assistant": completion}

    def generate_bulk(self, n_samples:int, schema="prompt_completion", lang="de", paraphrase_level=0.0, style="neutral", informal=True, multi_turn=False, max_turns=3, progress_callback=None, stop_event=None):
        """
        Yields samples (or returns list) and calls progress_callback(pct, written) occasionally.
        stop_event: threading.Event to cancel
        """
        written = 0
        results = []
        for i in range(n_samples):
            if stop_event and stop_event.is_set():
                break
            s = self.generate_sample(schema=schema, lang=lang, paraphrase_level=paraphrase_level, style=style, informal=informal, multi_turn=multi_turn, max_turns=max_turns)
            results.append(s)
            written += 1
            if progress_callback and (i % max(1, n_samples//200) == 0 or i==n_samples-1):
                pct = int(100.0 * written / max(1, n_samples))
                progress_callback(pct, written)
        return results

# -----------------------
# GUI (Tkinter) — QDGApp
# -----------------------
class QDGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("QDG — Quick Dataset Generator (Quixoticode)")
        self.uiq = queue.Queue()
        self.eng = QDGEngine()
        self.records = self.eng.records
        self.paraphraser = self.eng.paraphraser
        self.templates = self.eng.templates
        self.gen_thread = None
        self.stop_event = threading.Event()
        self._build_ui()
        self._start_poller()
        log("QDG UI started")

    def _build_ui(self):
        pad = 6
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        # Top: metadata
        meta = ttk.LabelFrame(main, text="Dataset Metadata", padding=8)
        meta.pack(fill="x", pady=(0,pad))
        ttk.Label(meta, text="Name:").grid(row=0, column=0, sticky="w")
        self.ent_name = ttk.Entry(meta, width=40)
        self.ent_name.grid(row=0, column=1, padx=6)
        self.ent_name.insert(0, f"qdg_{int(time.time())}")

        ttk.Label(meta, text="Author:").grid(row=0, column=2, sticky="w")
        self.ent_author = ttk.Entry(meta, width=30)
        self.ent_author.grid(row=0, column=3, padx=6)

        ttk.Label(meta, text="Language:").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.cmb_lang = ttk.Combobox(meta, values=["de","en"], width=6, state="readonly")
        self.cmb_lang.set("de")
        self.cmb_lang.grid(row=1, column=1, sticky="w", padx=6, pady=(6,0))

        ttk.Label(meta, text="Description:").grid(row=1, column=2, sticky="w", pady=(6,0))
        self.ent_desc = ttk.Entry(meta, width=40)
        self.ent_desc.grid(row=1, column=3, padx=6, pady=(6,0))

        # Middle: Templates (left) / Records (center) / Generation controls (right)
        middle = ttk.Frame(main)
        middle.pack(fill="both", expand=True, pady=(0,pad))

        # Templates panel
        tpl_fr = ttk.LabelFrame(middle, text="Templates (prompt / completion)", padding=6)
        tpl_fr.pack(side="left", fill="both", expand=True, padx=(0,pad))

        self.lst_templates = tk.Listbox(tpl_fr, height=12)
        self.lst_templates.pack(fill="both", expand=True, side="left")
        tpl_scroll = ttk.Scrollbar(tpl_fr, orient="vertical", command=self.lst_templates.yview)
        tpl_scroll.pack(side="left", fill="y")
        self.lst_templates.config(yscrollcommand=tpl_scroll.set)

        tbtn_fr = ttk.Frame(tpl_fr)
        tbtn_fr.pack(side="right", fill="y", padx=6)
        ttk.Button(tbtn_fr, text="Neu", command=self._add_template).pack(fill="x", pady=(0,6))
        ttk.Button(tbtn_fr, text="Bearbeiten", command=self._edit_template).pack(fill="x", pady=(0,6))
        ttk.Button(tbtn_fr, text="Löschen", command=self._delete_template).pack(fill="x", pady=(0,6))
        ttk.Button(tbtn_fr, text="Import (JSON)", command=self._import_templates).pack(fill="x", pady=(0,6))
        ttk.Button(tbtn_fr, text="Export (JSON)", command=self._export_templates).pack(fill="x", pady=(0,6))

        # Records panel
        rec_fr = ttk.LabelFrame(middle, text="Records / Personal Data", padding=6)
        rec_fr.pack(side="left", fill="both", expand=True, padx=(0,pad))
        self.lst_records = tk.Listbox(rec_fr, height=12)
        self.lst_records.pack(fill="both", expand=True, side="left")
        rec_scroll = ttk.Scrollbar(rec_fr, orient="vertical", command=self.lst_records.yview)
        rec_scroll.pack(side="left", fill="y")
        self.lst_records.config(yscrollcommand=rec_scroll.set)

        rbtn_fr = ttk.Frame(rec_fr)
        rbtn_fr.pack(side="right", fill="y", padx=6)
        ttk.Button(rbtn_fr, text="CSV/JSON laden", command=self._load_records_file).pack(fill="x", pady=(0,6))
        ttk.Button(rbtn_fr, text="Neu (manuell)", command=self._add_record_manual).pack(fill="x", pady=(0,6))
        ttk.Button(rbtn_fr, text="Felder anzeigen", command=self._show_record_fields).pack(fill="x", pady=(0,6))
        ttk.Button(rbtn_fr, text="Leeren", command=self._clear_records).pack(fill="x", pady=(0,6))

        # Generation controls
        gen_fr = ttk.LabelFrame(middle, text="Generate", padding=6)
        gen_fr.pack(side="right", fill="y")

        ttk.Label(gen_fr, text="Schema:").grid(row=0, column=0, sticky="w")
        self.cmb_schema = ttk.Combobox(gen_fr, values=["prompt_completion","user_assistant","instruction_io","chat_messages"], state="readonly", width=18)
        self.cmb_schema.set("prompt_completion")
        self.cmb_schema.grid(row=0, column=1, padx=6, pady=(0,6))

        ttk.Label(gen_fr, text="Samples:").grid(row=1, column=0, sticky="w")
        self.spin_count = ttk.Spinbox(gen_fr, from_=1, to=1000000, width=12)
        self.spin_count.set(1000)
        self.spin_count.grid(row=1, column=1, padx=6, pady=(0,6))

        ttk.Label(gen_fr, text="Paraphrase level:").grid(row=2, column=0, sticky="w")
        self.scale_para = ttk.Scale(gen_fr, from_=0.0, to=1.0, orient="horizontal")
        self.scale_para.set(0.25)
        self.scale_para.grid(row=2, column=1, padx=6, pady=(0,6))

        self.var_use_transformer = tk.BooleanVar(value=False)
        ttk.Checkbutton(gen_fr, text="use transformers paraphraser (optional)", variable=self.var_use_transformer).grid(row=3, column=0, columnspan=2, sticky="w", pady=(0,6))

        ttk.Label(gen_fr, text="Paraphrase model (HF):").grid(row=4, column=0, sticky="w")
        self.ent_para_model = ttk.Entry(gen_fr, width=22)
        self.ent_para_model.insert(0, "")  # e.g., "Vamsi/T5_Paraphrase_Paws"
        self.ent_para_model.grid(row=4, column=1, padx=6, pady=(0,6))

        ttk.Label(gen_fr, text="Multi-turn (max turns):").grid(row=5, column=0, sticky="w")
        self.spin_turns = ttk.Spinbox(gen_fr, from_=1, to=10, width=8)
        self.spin_turns.set(2)
        self.spin_turns.grid(row=5, column=1, padx=6, pady=(0,6))

        ttk.Label(gen_fr, text="Split % (train/val/test):").grid(row=6, column=0, sticky="w")
        self.ent_split = ttk.Entry(gen_fr, width=20)
        self.ent_split.insert(0, "80,10,10")
        self.ent_split.grid(row=6, column=1, padx=6, pady=(0,6))

        ttk.Label(gen_fr, text="Out folder:").grid(row=7, column=0, sticky="w")
        self.ent_out = ttk.Entry(gen_fr, width=22)
        self.ent_out.insert(0, os.path.abspath(BASE_OUT))
        self.ent_out.grid(row=7, column=1, padx=6, pady=(0,6))
        ttk.Button(gen_fr, text="Choose...", command=self._choose_out).grid(row=7, column=2, padx=6)

        # Buttons
        ttk.Button(gen_fr, text="Preview (5)", command=self._preview).grid(row=8, column=0, columnspan=3, sticky="we", pady=(6,3))
        self.btn_start = ttk.Button(gen_fr, text="Start Generate", command=self._start_generate)
        self.btn_start.grid(row=9, column=0, columnspan=3, sticky="we", pady=(3,3))
        self.btn_stop = ttk.Button(gen_fr, text="Stop", command=self._stop_generate, state="disabled")
        self.btn_stop.grid(row=10, column=0, columnspan=3, sticky="we", pady=(3,3))

        # Bottom: progress & log
        bottom = ttk.Frame(main)
        bottom.pack(fill="both", expand=True, pady=(pad,0))

        self.pb_total = ttk.Progressbar(bottom, orient="horizontal", mode="determinate")
        self.pb_total.pack(fill="x", padx=6, pady=(0,6))
        self.txt_log = tk.Text(bottom, height=12, wrap="word", state="disabled")
        self.txt_log.pack(fill="both", expand=True, side="left", padx=(6,0))
        log_scroll = ttk.Scrollbar(bottom, orient="vertical", command=self.txt_log.yview)
        log_scroll.pack(side="left", fill="y")
        self.txt_log.config(yscrollcommand=log_scroll.set)

        # populate templates list
        self._refresh_template_list()
        self._refresh_records_list()

    # -----------------------
    # Templates UI handlers
    # -----------------------
    def _refresh_template_list(self):
        self.lst_templates.delete(0, "end")
        for t in self.templates:
            self.lst_templates.insert("end", f"{t.name} [{t.lang}] w={t.weight} multi={t.multi_turn}")

    def _add_template(self):
        win = tk.Toplevel(self.root)
        win.title("Neue Template erstellen")
        ttk.Label(win, text="Name:").grid(row=0,column=0, sticky="w")
        ent_name = ttk.Entry(win, width=40); ent_name.grid(row=0,column=1, padx=6, pady=4)
        ttk.Label(win, text="Language (de/en):").grid(row=1,column=0, sticky="w")
        cmb_lang = ttk.Combobox(win, values=["de","en"], width=6); cmb_lang.set("de"); cmb_lang.grid(row=1,column=1, sticky="w")
        ttk.Label(win, text="Weight:").grid(row=2,column=0, sticky="w")
        ent_weight = ttk.Entry(win, width=8); ent_weight.insert(0, "1"); ent_weight.grid(row=2,column=1, sticky="w")
        ttk.Label(win, text="Prompt template:").grid(row=3,column=0, sticky="nw")
        txt_prompt = tk.Text(win, width=70, height=4); txt_prompt.grid(row=3,column=1, pady=4)
        ttk.Label(win, text="Completion template:").grid(row=4,column=0, sticky="nw")
        txt_comp = tk.Text(win, width=70, height=4); txt_comp.grid(row=4,column=1, pady=4)
        chk_multi = tk.BooleanVar(value=False)
        ttk.Checkbutton(win, text="Multi-turn template", variable=chk_multi).grid(row=5,column=1, sticky="w", pady=(4,0))

        def do_add():
            name = ent_name.get().strip() or f"tpl_{int(time.time())}"
            lang = cmb_lang.get().strip()
            try:
                weight = float(ent_weight.get())
            except Exception:
                weight = 1.0
            p = txt_prompt.get("1.0","end").strip()
            c = txt_comp.get("1.0","end").strip()
            tpl = Template(name=name, prompt_tpl=p or "{user}", completion_tpl=c or "{assistant}", weight=weight, lang=lang, multi_turn=chk_multi.get())
            self.templates.append(tpl)
            self._refresh_template_list()
            win.destroy()
            log(f"Template added: {name}")

        ttk.Button(win, text="OK", command=do_add).grid(row=6,column=1, sticky="e", pady=8)

    def _edit_template(self):
        sel = self.lst_templates.curselection()
        if not sel:
            messagebox.showinfo("Auswahl", "Bitte Template wählen.")
            return
        idx = sel[0]
        tpl = self.templates[idx]
        win = tk.Toplevel(self.root)
        win.title("Template bearbeiten")
        ttk.Label(win, text="Name:").grid(row=0,column=0, sticky="w")
        ent_name = ttk.Entry(win, width=40); ent_name.grid(row=0,column=1, padx=6, pady=4); ent_name.insert(0, tpl.name)
        ttk.Label(win, text="Language (de/en):").grid(row=1,column=0, sticky="w")
        cmb_lang = ttk.Combobox(win, values=["de","en"], width=6); cmb_lang.set(tpl.lang); cmb_lang.grid(row=1,column=1, sticky="w")
        ttk.Label(win, text="Weight:").grid(row=2,column=0, sticky="w")
        ent_weight = ttk.Entry(win, width=8); ent_weight.insert(0, str(tpl.weight)); ent_weight.grid(row=2,column=1, sticky="w")
        ttk.Label(win, text="Prompt template:").grid(row=3,column=0, sticky="nw")
        txt_prompt = tk.Text(win, width=70, height=4); txt_prompt.grid(row=3,column=1, pady=4); txt_prompt.insert("1.0", tpl.prompt_tpl)
        ttk.Label(win, text="Completion template:").grid(row=4,column=0, sticky="nw")
        txt_comp = tk.Text(win, width=70, height=4); txt_comp.grid(row=4,column=1, pady=4); txt_comp.insert("1.0", tpl.completion_tpl)
        chk_multi = tk.BooleanVar(value=tpl.multi_turn)
        ttk.Checkbutton(win, text="Multi-turn template", variable=chk_multi).grid(row=5,column=1, sticky="w", pady=(4,0))

        def do_save():
            tpl.name = ent_name.get().strip() or tpl.name
            tpl.lang = cmb_lang.get().strip() or tpl.lang
            try:
                tpl.weight = float(ent_weight.get())
            except Exception:
                tpl.weight = tpl.weight
            tpl.prompt_tpl = txt_prompt.get("1.0","end").strip() or tpl.prompt_tpl
            tpl.completion_tpl = txt_comp.get("1.0","end").strip() or tpl.completion_tpl
            tpl.multi_turn = chk_multi.get()
            self._refresh_template_list()
            win.destroy()
            log(f"Template edited: {tpl.name}")

        ttk.Button(win, text="Save", command=do_save).grid(row=6,column=1, sticky="e", pady=8)

    def _delete_template(self):
        sel = self.lst_templates.curselection()
        if not sel:
            return
        idx = sel[0]
        tpl = self.templates.pop(idx)
        self._refresh_template_list()
        log(f"Template deleted: {tpl.name}")

    def _import_templates(self):
        f = filedialog.askopenfilename(title="Templates JSON", filetypes=[("JSON", "*.json"), ("All files","*.*")])
        if not f:
            return
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                # expect list of dicts with keys: name,prompt_tpl,completion_tpl,weight,lang,multi_turn
                count=0
                for it in data:
                    name = it.get("name") or f"tpl_{int(time.time()*1000)}"
                    t = Template(name=name, prompt_tpl=it.get("prompt_tpl","{user}"), completion_tpl=it.get("completion_tpl","{assistant}"), weight=it.get("weight",1), lang=it.get("lang","de"), multi_turn=it.get("multi_turn",False))
                    self.templates.append(t)
                    count+=1
                self._refresh_template_list()
                log(f"Imported {count} templates from {f}")
        except Exception as e:
            messagebox.showerror("Import error", str(e))
            log(f"Template import failed: {e}")

    def _export_templates(self):
        f = filedialog.asksaveasfilename(title="Export Templates JSON", defaultextension=".json", filetypes=[("JSON","*.json")])
        if not f:
            return
        data=[]
        for t in self.templates:
            data.append({"name":t.name,"prompt_tpl":t.prompt_tpl,"completion_tpl":t.completion_tpl,"weight":t.weight,"lang":t.lang,"multi_turn":t.multi_turn})
        try:
            with open(f,"w",encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            log(f"Templates exported to {f}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))
            log(f"Template export failed: {e}")

    # -----------------------
    # Records UI handlers
    # -----------------------
    def _load_records_file(self):
        f = filedialog.askopenfilename(title="Load records (CSV/JSON/JSONL)", filetypes=[("CSV/JSON/JSONL","*.csv *.json *.jsonl *.ndjson"), ("All files","*.*")])
        if not f:
            return
        ext = os.path.splitext(f)[1].lower()
        ok=False; msg=None
        if ext in (".csv",):
            ok,msg = self.records.load_csv(f)
        elif ext in (".json",):
            ok,msg = self.records.load_json(f)
        elif ext in (".jsonl", ".ndjson"):
            ok,msg = self.records.load_jsonl(f)
        else:
            # try all
            ok,msg = self.records.load_jsonl(f)
            if not ok:
                ok,msg = self.records.load_json(f)
            if not ok:
                ok,msg = self.records.load_csv(f)
        if ok:
            self._refresh_records_list()
            messagebox.showinfo("Done", f"Loaded records from {f}. Total records: {len(self.records.records)}")
        else:
            messagebox.showerror("Load failed", msg or "Unknown error")

    def _refresh_records_list(self):
        self.lst_records.delete(0,"end")
        # show up to 200 records summary
        for i, r in enumerate(self.records.records[:200]):
            summary = ", ".join(f"{k}={str(v)[:20]}" for k,v in list(r.items())[:3])
            self.lst_records.insert("end", f"{i+1}: {summary}")

    def _add_record_manual(self):
        win = tk.Toplevel(self.root)
        win.title("Add manual record (JSON)")
        txt = tk.Text(win, width=80, height=12)
        txt.pack(fill="both", expand=True, padx=6, pady=6)
        txt.insert("1.0", json.dumps({"name":"Anna","city":"Berlin"}, ensure_ascii=False, indent=2))
        def do_add():
            try:
                obj = json.loads(txt.get("1.0","end"))
                if isinstance(obj, dict):
                    self.records.add_record(obj)
                    self._refresh_records_list()
                    win.destroy()
                    log("Manual record added")
                else:
                    messagebox.showerror("Format", "JSON Objekt erwartet")
            except Exception as e:
                messagebox.showerror("JSON error", str(e))
        ttk.Button(win, text="Add", command=do_add).pack(pady=6)

    def _show_record_fields(self):
        fields = sorted(list(self.records.fields))
        messagebox.showinfo("Fields", f"Detected fields ({len(fields)}):\n\n" + ", ".join(fields[:200]))

    def _clear_records(self):
        if messagebox.askyesno("Clear records", "All records will be removed. Continue?"):
            self.records = RecordStore()
            self.eng.records = self.records
            self._refresh_records_list()
            log("Records cleared")

    # -----------------------
    # Generation control
    # -----------------------
    def _choose_out(self):
        d = filedialog.askdirectory(title="Choose output folder", initialdir=self.ent_out.get())
        if d:
            self.ent_out.delete(0,"end"); self.ent_out.insert(0,d)

    def _preview(self):
        cfg = self._gather_config()
        # use engine to generate 5 samples quickly (no heavy paraphrase models)
        self._append_log("Previewing 5 samples...")
        preview = self.eng.generate_bulk(5, schema=cfg["schema"], lang=cfg["lang"], paraphrase_level=min(0.25, cfg["paraphrase_level"]), style="neutral", informal=True, multi_turn=cfg["multi_turn"], max_turns=cfg["max_turns"])
        # show in window
        win = tk.Toplevel(self.root)
        win.title("Preview")
        txt = tk.Text(win, wrap="word", width=100, height=30)
        txt.pack(fill="both", expand=True)
        for s in preview:
            txt.insert("end", json.dumps(s, ensure_ascii=False) + "\n")
        txt.config(state="disabled")

    def _append_log(self, msg):
        ts=time.strftime("%Y-%m-%d %H:%M:%S")
        line=f"[{ts}] {msg}\n"
        try:
            self.txt_log.config(state="normal")
            self.txt_log.insert("end", line)
            self.txt_log.see("end")
            self.txt_log.config(state="disabled")
        except Exception:
            pass
        log(msg)

    def _gather_config(self):
        name = sanitize_name(self.ent_name.get().strip() or f"qdg_{int(time.time())}")
        lang = self.cmb_lang.get()
        try:
            count = int(self.spin_count.get())
        except Exception:
            count = 1000
        paraphrase_level = float(self.scale_para.get())
        schema = self.cmb_schema.get()
        try:
            split_text = self.ent_split.get().split(",")
            split = [int(x.strip()) for x in split_text]
            if len(split) != 3 or sum(split) <= 0:
                split = [80,10,10]
        except Exception:
            split = [80,10,10]
        outdir = os.path.abspath(self.ent_out.get() or BASE_OUT)
        multi_turn = int(self.spin_turns.get())>1
        max_turns = int(self.spin_turns.get())
        use_transformer = self.var_use_transformer.get()
        para_model = self.ent_para_model.get().strip() or None
        return {
            "name": name,
            "author": self.ent_author.get().strip(),
            "desc": self.ent_desc.get().strip(),
            "lang": lang,
            "count": count,
            "paraphrase_level": paraphrase_level,
            "schema": schema,
            "split": split,
            "outdir": outdir,
            "multi_turn": multi_turn,
            "max_turns": max_turns,
            "use_transformer": use_transformer,
            "para_model": para_model
        }

    def _start_generate(self):
        if self.gen_thread and self.gen_thread.is_alive():
            messagebox.showinfo("Running", "Generation is already running.")
            return
        cfg = self._gather_config()
        outdir = ensure_dir(os.path.join(cfg["outdir"], cfg["name"]))
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        base_fname = f"{cfg['name']}_{timestamp}"
        meta = {
            "name": cfg["name"],
            "author": cfg["author"],
            "desc": cfg["desc"],
            "lang": cfg["lang"],
            "schema": cfg["schema"],
            "count": cfg["count"],
            "paraphrase_level": cfg["paraphrase_level"],
            "split": cfg["split"],
            "created_by": APP_CREDIT,
            "timestamp": timestamp,
            "templates": [ {"name":t.name,"lang":t.lang,"weight":t.weight,"multi_turn":t.multi_turn} for t in self.templates ],
            "record_count": len(self.records.records)
        }
        # write metadata
        try:
            with open(os.path.join(outdir, "metadata.json"), "w", encoding="utf-8") as mf:
                json.dump(meta, mf, indent=2, ensure_ascii=False)
        except Exception as e:
            self._append_log(f"Failed to save metadata: {e}")

        # possibly init paraphraser with model
        if cfg["use_transformer"] and cfg["para_model"] and TRANSFORMERS_AVAILABLE:
            self.paraphraser = Paraphraser(model_name=cfg["para_model"], lang=cfg["lang"])
            self.eng.paraphraser = self.paraphraser
        elif cfg["use_transformer"] and cfg["para_model"] and not TRANSFORMERS_AVAILABLE:
            self._append_log("transformers not available — falling back to rule-based paraphraser")

        # start background thread
        self.stop_event.clear()
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.pb_total['value'] = 0
        self.gen_thread = threading.Thread(target=self._generate_bg, args=(cfg, outdir, base_fname), daemon=True)
        self.gen_thread.start()
        self._append_log(f"Generation started: {cfg['count']} samples -> {outdir}")

    def _stop_generate(self):
        if not (self.gen_thread and self.gen_thread.is_alive()):
            return
        self.stop_event.set()
        self.btn_stop.config(state="disabled")
        self._append_log("Stop requested — will finalize and save partial files.")

    def _generate_bg(self, cfg, outdir, base_fname):
        count = cfg["count"]
        schema = cfg["schema"]
        lang = cfg["lang"]
        para_level = cfg["paraphrase_level"]
        multi_turn = cfg["multi_turn"]
        max_turns = cfg["max_turns"]
        split = cfg["split"]  # percentages
        # compute splits
        ssum = sum(split)
        train_n = int(count * (split[0]/ssum))
        val_n = int(count * (split[1]/ssum))
        test_n = count - train_n - val_n

        # open files
        train_path = os.path.join(outdir, f"{base_fname}_train.jsonl")
        val_path = os.path.join(outdir, f"{base_fname}_val.jsonl")
        test_path = os.path.join(outdir, f"{base_fname}_test.jsonl")
        writers = {
            "train": open(train_path, "w", encoding="utf-8"),
            "val": open(val_path, "w", encoding="utf-8"),
            "test": open(test_path, "w", encoding="utf-8")
        }

        # generate iteratively, save periodically
        written = {"train":0,"val":0,"test":0}
        total_written = 0
        def progress_cb(pct, w):
            # called from engine
            self.uiq.put(("progress", {"pct":pct, "written":w}))

        # We'll generate in order and assign to splits
        try:
            for i in range(count):
                if self.stop_event.is_set():
                    break
                # choose which split to write to based on counters
                if written["train"] < train_n:
                    bucket = "train"
                elif written["val"] < val_n:
                    bucket = "val"
                else:
                    bucket = "test"
                sample = self.eng.generate_sample(schema=schema, lang=lang, paraphrase_level=para_level, multi_turn=multi_turn, max_turns=max_turns)
                # write
                writers[bucket].write(json.dumps(sample, ensure_ascii=False) + "\n")
                written[bucket] += 1
                total_written += 1
                # update progress occasionally
                if i % max(1, count//200) == 0 or i==count-1:
                    pct = int(100.0 * total_written / max(1, count))
                    self.uiq.put(("progress", {"pct": pct, "written": total_written}))
                # flush every 500 lines
                if total_written % 500 == 0:
                    for w in writers.values():
                        w.flush()
            # finalize
            for w in writers.values():
                try:
                    w.flush()
                    w.close()
                except:
                    pass

            # metadata update
            meta_path = os.path.join(outdir, "metadata.json")
            try:
                meta = {}
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                meta.update({"final_counts": written, "generated_at": datetime.datetime.utcnow().isoformat()})
                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump(meta, mf, indent=2, ensure_ascii=False)
            except Exception as e:
                self.uiq.put(("log", f"Failed to update metadata: {e}"))

            if self.stop_event.is_set():
                self.uiq.put(("log", f"Generation stopped. Partial counts: {written}"))
                self._append_log("Generation stopped by user — partial files saved.")
            else:
                self.uiq.put(("log", f"Generation finished. Files: {train_path}, {val_path}, {test_path}"))
                self._append_log(f"Generation finished. Files saved to {outdir}")

        except Exception as e:
            self._append_log(f"Generation error: {e}")
            log(f"Exception in generation thread: {e}")
        finally:
            # ensure buttons reset
            self.uiq.put(("done", {"outdir": outdir}))
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")

    # -----------------------
    # UI Poller
    # -----------------------
    def _start_poller(self):
        self.root.after(150, self._ui_poller)

    def _ui_poller(self):
        try:
            while True:
                item = self.uiq.get_nowait()
                if not item:
                    continue
                typ = item[0]
                payload = item[1] if len(item)>1 else None
                if typ == "progress":
                    pct = payload.get("pct",0)
                    written = payload.get("written",0)
                    self.pb_total['value'] = pct
                    self._append_log(f"Progress: {pct}% ({written})")
                elif typ == "log":
                    self._append_log(payload)
                elif typ == "done":
                    self._append_log("Generation done.")
                    messagebox.showinfo("Done", f"Generation finished. Output: {payload.get('outdir')}")
                    self.pb_total['value'] = 100
        except queue.Empty:
            pass
        self.root.after(150, self._ui_poller)

# -----------------------
# Entrypoint
# -----------------------
def main():
    ensure_dir(BASE_OUT)
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        style.theme_use("clam")
    except Exception:
        pass
    app = QDGApp(root)
    root.geometry("1100x720")
    root.mainloop()

if __name__ == "__main__":
    main()
