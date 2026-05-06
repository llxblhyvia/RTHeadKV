"""Build a multilingual LongBench-style benchmark.

The generated jsonl files share the same schema as the original LongBench
(`_id`, `dataset`, `language`, `length`, `context`, `input`, `answers`,
`all_classes`) so that `run_longbench.py` can consume them directly.

Two sources are supported out of the box:

  * XQuAD (extractive QA in EN, DE, ZH, plus more) — the gold passage is
    inserted into a long distractor context built from other XQuAD passages
    in the same language, simulating a long-context retrieval setting.
  * MGSM (multilingual grade-school math word problems) — the target problem
    is appended to a long preamble of distractor problems in the same
    language; the model must locate and solve only the queried problem.

Usage
-----

    python data/build_multilingual_longbench.py \\
        --tasks xquad_en xquad_de xquad_zh mgsm_en mgsm_de mgsm_sw mgsm_zh \\
        --target_length 6000 \\
        --num_examples 150 \\
        --out_dir data/MultilingualLongBench

If the HuggingFace `datasets` package is unavailable / offline, you can pass
`--source_dir <dir>` pointing at a directory that contains pre-downloaded
parquet/json files named `xquad.<lang>.json` or `mgsm_<lang>.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable

# XQuAD configs distributed by Google. Each config has 1190 (q, p, a) triples.
XQUAD_LANGS = {
    'xquad_en': ('xquad.en', 'en'),
    'xquad_de': ('xquad.de', 'de'),
    'xquad_zh': ('xquad.zh', 'zh'),
    'xquad_es': ('xquad.es', 'es'),
    'xquad_ar': ('xquad.ar', 'ar'),
    'xquad_hi': ('xquad.hi', 'hi'),
    'xquad_vi': ('xquad.vi', 'vi'),
    'xquad_ru': ('xquad.ru', 'ru'),
    'xquad_th': ('xquad.th', 'th'),
    'xquad_tr': ('xquad.tr', 'tr'),
}

# MGSM languages (juletxara/mgsm).
MGSM_LANGS = {
    'mgsm_en': 'en',
    'mgsm_de': 'de',
    'mgsm_sw': 'sw',
    'mgsm_zh': 'zh',
    'mgsm_es': 'es',
    'mgsm_fr': 'fr',
    'mgsm_ja': 'ja',
    'mgsm_ru': 'ru',
    'mgsm_te': 'te',
    'mgsm_th': 'th',
    'mgsm_bn': 'bn',
}


def _approx_word_count(text: str, lang: str) -> int:
    """Crude length proxy used to control padded-context size.

    For CJK languages we use character count, otherwise whitespace tokens.
    """
    if lang in ('zh', 'ja'):
        return len(text)
    return len(text.split())


def _load_xquad(lang_cfg: str, source_dir: str | None):
    """Yield {'context', 'question', 'answer'} dicts."""
    if source_dir:
        path = Path(source_dir) / f'{lang_cfg}.json'
        if not path.exists():
            raise FileNotFoundError(f'Missing local XQuAD file: {path}')
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        for article in raw['data']:
            for para in article['paragraphs']:
                ctx = para['context']
                for qa in para['qas']:
                    if not qa['answers']:
                        continue
                    yield {
                        'context': ctx,
                        'question': qa['question'],
                        'answers': [a['text'] for a in qa['answers']],
                    }
        return

    from datasets import load_dataset  # imported lazily

    ds = load_dataset('google/xquad', lang_cfg, split='validation')
    for row in ds:
        ans = row['answers']['text'] if isinstance(row['answers'], dict) else row['answers']
        if not ans:
            continue
        yield {
            'context': row['context'],
            'question': row['question'],
            'answers': list(ans),
        }


def _load_mgsm(lang: str, source_dir: str | None):
    if source_dir:
        path = Path(source_dir) / f'mgsm_{lang}.json'
        if not path.exists():
            raise FileNotFoundError(f'Missing local MGSM file: {path}')
        with open(path, 'r', encoding='utf-8') as f:
            for row in json.load(f):
                yield {
                    'question': row['question'],
                    'answer': str(row.get('answer_number', row.get('answer', ''))),
                }
        return

    from datasets import load_dataset

    ds = load_dataset('juletxara/mgsm', lang, split='test')
    for row in ds:
        yield {
            'question': row['question'],
            'answer': str(row.get('answer_number', row.get('answer', ''))),
        }


def _build_xquad_examples(task: str, items, target_length: int, num_examples: int, lang: str, seed: int):
    rng = random.Random(seed)
    items = list(items)
    if not items:
        return []
    rng.shuffle(items)

    sampled = items[:num_examples]
    contexts_pool = [it['context'] for it in items]
    out = []
    for idx, it in enumerate(sampled):
        gold_ctx = it['context']
        distractors: list[str] = []
        # accumulate distractor passages until we reach the target length
        cur = _approx_word_count(gold_ctx, lang)
        local_pool = [p for p in contexts_pool if p != gold_ctx]
        rng.shuffle(local_pool)
        cursor = 0
        while cur < target_length and cursor < len(local_pool):
            p = local_pool[cursor]
            distractors.append(p)
            cur += _approx_word_count(p, lang)
            cursor += 1
        # randomly insert the gold passage among the distractors
        insert_pos = rng.randrange(0, len(distractors) + 1)
        passages = distractors[:insert_pos] + [gold_ctx] + distractors[insert_pos:]
        context = '\n\n'.join(passages)
        out.append({
            '_id': f'{task}_{idx}',
            'dataset': task,
            'language': lang,
            'length': _approx_word_count(context, lang),
            'context': context,
            'input': it['question'],
            'answers': it['answers'],
            'all_classes': [],
        })
    return out


def _build_mgsm_examples(task: str, items, target_length: int, num_examples: int, lang: str, seed: int):
    rng = random.Random(seed)
    items = list(items)
    if not items:
        return []
    rng.shuffle(items)

    sampled = items[:num_examples]
    out = []
    for idx, it in enumerate(sampled):
        target_q = it['question']
        distractors = []
        cur = _approx_word_count(target_q, lang)
        pool = [other['question'] for other in items if other['question'] != target_q]
        rng.shuffle(pool)
        cursor = 0
        while cur < target_length and cursor < len(pool):
            p = pool[cursor]
            distractors.append(p)
            cur += _approx_word_count(p, lang)
            cursor += 1
        # mark the question of interest with a stable header so the model can
        # distinguish it from the distractors
        header = {
            'en': 'TARGET PROBLEM:',
            'de': 'ZIELAUFGABE:',
            'zh': '目标题目：',
            'sw': 'SWALI LENGO:',
        }.get(lang, 'TARGET PROBLEM:')
        passages = ['DISTRACTOR PROBLEMS:'] + distractors + [header, target_q]
        context = '\n\n'.join(passages)
        out.append({
            '_id': f'{task}_{idx}',
            'dataset': task,
            'language': lang,
            'length': _approx_word_count(context, lang),
            'context': context,
            'input': target_q,
            'answers': [it['answer']],
            'all_classes': [],
        })
    return out


def build_task(task: str, *, target_length: int, num_examples: int, source_dir: str | None, seed: int):
    if task in XQUAD_LANGS:
        cfg, lang = XQUAD_LANGS[task]
        items = _load_xquad(cfg, source_dir)
        return _build_xquad_examples(task, items, target_length, num_examples, lang, seed)
    if task in MGSM_LANGS:
        lang = MGSM_LANGS[task]
        items = _load_mgsm(lang, source_dir)
        return _build_mgsm_examples(task, items, target_length, num_examples, lang, seed)
    raise ValueError(f'Unknown task: {task}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', required=True,
                        help=f'Subset of: {sorted(list(XQUAD_LANGS) + list(MGSM_LANGS))}')
    parser.add_argument('--target_length', type=int, default=6000,
                        help='Approximate target context length (words / chars).')
    parser.add_argument('--num_examples', type=int, default=150)
    parser.add_argument('--out_dir', type=str, default='data/MultilingualLongBench')
    parser.add_argument('--source_dir', type=str, default=None,
                        help='Optional offline directory with raw json files.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for task in args.tasks:
        examples = build_task(
            task,
            target_length=args.target_length,
            num_examples=args.num_examples,
            source_dir=args.source_dir,
            seed=args.seed,
        )
        out_path = Path(args.out_dir) / f'{task}.jsonl'
        with open(out_path, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        print(f'[build_multilingual_longbench] wrote {len(examples)} examples -> {out_path}',
              file=sys.stderr)


if __name__ == '__main__':
    main()
