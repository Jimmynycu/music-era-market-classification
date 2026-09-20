"""Build the seven-page English submission report from verified results."""
import argparse
import csv
import html
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parent
W, H = 960, 540
INK, MUTED, LINE = '#111111', '#444444', '#AAAAAA'
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
CJK_FONT = Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
DATASETS = ('dataset_A', 'dataset_B')
TITLES = {'dataset_A': 'Release decade', 'dataset_B': 'Release market'}


def esc(value):
    return html.escape(str(value))


def pct(value):
    return f'{100 * value:.1f}%'


def load_json(path):
    with open(path) as stream:
        return json.load(stream)


def read_manifest(data_dir, dataset):
    with open(data_dir / dataset / 'manifest.csv', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len({r['sample_id'] for r in rows}) == len(rows)
    return rows


def validate_metrics(result, classes, n):
    cm = np.asarray(result['confusion_matrix'])
    assert cm.shape == (6, 6) and np.all(cm >= 0) and np.all(cm == cm.astype(int))
    assert int(cm.sum()) == n, f'Incomplete evaluation: {cm.sum()} of {n}'
    assert len(classes) == 6 and len(set(classes)) == 6
    assert abs(float(result['top1']) - np.trace(cm) / n) < 1e-7
    assert 0 <= result['top1'] <= result['top3'] <= 1
    if 'n_samples' in result:
        assert result['n_samples'] == n
    return cm.astype(int)


def error_pairs(cm, labels, count=3):
    pairs = [(int(cm[i, j] + cm[j, i]), labels[i], labels[j])
             for i in range(len(labels)) for j in range(i + 1, len(labels))]
    return sorted(pairs, key=lambda v: (-v[0], v[1], v[2]))[:count]


def missing_top1_labels(cm, labels):
    return [label for label, count in zip(labels, np.asarray(cm).sum(axis=0)) if count == 0]


def decade_errors(cm):
    distance = np.abs(np.arange(len(cm))[:, None] - np.arange(len(cm))[None, :])
    hist = np.array([cm[distance == k].sum() for k in range(1, len(cm))], dtype=int)
    total = int(hist.sum())
    return hist, int(hist[0]), total


class Report:
    def __init__(self, output, student_id, draft=False, student_name=None, draft_label='Submission details pending'):
        for name, filename in [('Body', 'DejaVuSans.ttf'), ('Bold', 'DejaVuSans-Bold.ttf'), ('Mono', 'DejaVuSansMono.ttf')]:
            pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / filename)))
        pdfmetrics.registerFont(TTFont('CJK', str(CJK_FONT)))
        pdfmetrics.registerFontFamily('Body', normal='Body', bold='Bold', italic='Body', boldItalic='Bold')
        self.cjk_glyphs = pdfmetrics.getFont('CJK').face.charToGlyph
        self.canvas = canvas.Canvas(str(output), pagesize=(W, H))
        self.canvas.setTitle('Homework 1: Music Release Decade and Market Classification')
        self.canvas.setAuthor(' / '.join(value for value in (student_name, student_id) if value) or 'Student ID pending')
        self.student_id, self.page, self.draft = student_id, 0, draft
        self.draft_label = draft_label

    def text(self, text, x, y, width, size=14, color=INK, bold=False, leading=None):
        # Droid supplies embedded Chinese outlines; DejaVu supplies Latin/math.
        # Preserve ReportLab markup and XML entities while switching text runs.
        chunks = re.split(r'(<[^>]+>|&[^;]+;)', text)
        for index, chunk in enumerate(chunks):
            if chunk.startswith('<') or chunk.startswith('&'):
                continue
            chunks[index] = re.sub(r'[^\x00-\x7f]+', lambda m: ''.join(
                f'<font name="CJK">{char}</font>' if ord(char) in self.cjk_glyphs else char
                for char in m.group()), chunk)
        text = ''.join(chunks)
        style = ParagraphStyle('text', fontName='Bold' if bold else 'Body', fontSize=size,
                               leading=leading or size * 1.4, textColor=colors.HexColor(color), wordWrap='CJK')
        p = Paragraph(text, style)
        _, height = p.wrap(width, H)
        assert y + height <= (H - 6 if y >= 510 else H - 32), f'Text overflows page {self.page}: {text[:90]}'
        p.drawOn(self.canvas, x, H - y - height)
        return height

    def rule(self, y):
        self.canvas.setStrokeColor(colors.HexColor(LINE))
        self.canvas.line(44, H-y, W-44, H-y)

    def new_page(self, title, subtitle=None):
        if self.page:
            self.canvas.showPage()
        self.page += 1
        self.canvas.setFillColor(colors.white)
        self.canvas.rect(0, 0, W, H, fill=1, stroke=0)
        self.text(title, 44, 32, 880, size=23)
        if subtitle:
            self.text(subtitle, 44, 76, 872, size=11.5, color=MUTED)
        self.rule(505)
        self.text(f'Student ID: {esc(self.student_id or "pending")}  |  Homework 1',44,516,690,size=8,color=MUTED)
        self.canvas.setFont('Body', 8)
        self.canvas.setFillColor(colors.HexColor(MUTED))
        self.canvas.drawRightString(W-44, 19, f'{self.page:02d} / 7')

    def figure(self, fig, x, y, width, height, target):
        fig.savefig(target, dpi=180, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        self.canvas.drawImage(str(target), x, H-y-height, width=width, height=height,
                              preserveAspectRatio=True, anchor='c', mask='auto')

    def table(self, headers, rows, x, y, widths, rowheight=35, size=12):
        self.canvas.setStrokeColor(colors.HexColor(LINE))
        self.canvas.line(x,H-y,x+sum(widths),H-y)
        self.canvas.line(x,H-y-rowheight,x+sum(widths),H-y-rowheight)
        offset = x
        for value, width in zip(headers, widths):
            self.text(esc(value), offset+8, y+8, width-16, size=size)
            offset += width
        for i, row in enumerate(rows):
            top = y + (i+1) * rowheight
            offset = x
            for value, width in zip(row, widths):
                self.text(str(value), offset+8, top+8, width-16, size=size)
                offset += width
        bottom=y+(len(rows)+1)*rowheight
        self.canvas.line(x,H-bottom,x+sum(widths),H-bottom)
        return y + (len(rows)+1)*rowheight


def confusion_figure(cm, labels, title=None):
    fig, ax = plt.subplots(figsize=(5.5, 4.1))
    ax.imshow(cm, cmap='Greys', vmin=0, vmax=max(1, cm.sum(axis=1).max()))
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12,
                    color='white' if cm[i,j] > cm.sum(axis=1).max()*.55 else INK)
    ax.set_xticks(range(6), labels, rotation=25, ha='right', fontsize=10)
    ax.set_yticks(range(6), labels, fontsize=10)
    ax.set(xlabel='Predicted class', ylabel='True class', title=title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def clean_axes(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color(LINE)
    ax.tick_params(colors=MUTED)
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)


def source_links(items):
    return '; '.join(f'<link href="{esc(url)}" color="#005A9C">{esc(label)}</link>'
                     for label, url in items)


def write_sources(r):
    r.text('Sources (click the names)', 44, 365, 872, size=12)
    models = [('MERT paper', 'https://arxiv.org/abs/2306.00107'),
              ('MERT weights / code', 'https://huggingface.co/m-a-p/MERT-v1-95M/tree/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'),
              ('Qwen paper', 'https://arxiv.org/abs/2503.20215'),
              ('Qwen model', 'https://huggingface.co/Qwen/Qwen2.5-Omni-3B'),
              ('GGUF weights', 'https://huggingface.co/ggml-org/Qwen2.5-Omni-3B-GGUF/tree/75f1b73b657a50f5092502799457ccb4a4a1f9df'),
              ('llama.cpp', 'https://github.com/ggml-org/llama.cpp/tree/efa28e950'),
              ('audio runtime caveat', 'https://github.com/ggml-org/llama.cpp/discussions/13759')]
    code = [('PyTorch', 'https://github.com/pytorch/pytorch'),
            ('Transformers', 'https://github.com/huggingface/transformers'),
            ('Hugging Face Hub', 'https://github.com/huggingface/huggingface_hub'),
            ('librosa', 'https://github.com/librosa/librosa'),
            ('scikit-learn', 'https://scikit-learn.org/stable/about.html'),
            ('NumPy', 'https://numpy.org/citing-numpy/'),
            ('SciPy', 'https://github.com/scipy/scipy'),
            ('SoundFile', 'https://github.com/bastibe/python-soundfile'),
            ('joblib', 'https://github.com/joblib/joblib'),
            ('threadpoolctl', 'https://github.com/joblib/threadpoolctl')]
    other = [('Matplotlib', 'https://matplotlib.org/stable/project/citing.html'),
             ('ReportLab', 'https://www.reportlab.com/opensource/'),
             ('Course dataset / instructions', 'https://drive.google.com/drive/folders/1C8RymiLbr-EGmkxh2Ap5TIybnJYqNsb4')]
    r.text('Models and papers: ' + source_links(models),44,392,872,size=10.2)
    r.text('Software: ' + source_links(code),44,432,872,size=10.2)
    r.text('Figures, report and data: ' + source_links(other),44,476,872,size=10.2)


def build(args):
    supervised = load_json(args.supervised)
    finetuned = load_json(args.finetuned)
    alm = load_json(args.alm)
    manifests = {ds: read_manifest(args.data_dir, ds) for ds in DATASETS}
    audit = load_json(ROOT/'results'/'dataset_audit.json')
    reproduction = load_json(ROOT/'results'/'finetune_reproduction_check.json')
    assert reproduction.get('valid') is True and reproduction.get('exact_reproduction_checked') is True
    assert reproduction['test_samples'] == {
        ds: sum(row['split'] == 'test' for row in manifests[ds]) for ds in DATASETS}
    frozen_reproduction = load_json(ROOT/'results'/'reproduction_check.json')
    assert frozen_reproduction.get('valid') is True and frozen_reproduction.get('exact_reproduction_checked') is True
    assert frozen_reproduction['test_samples'] == reproduction['test_samples']
    assert all(audit[ds]['all_sha256_match'] and audit[ds]['all_audio_format_checks_passed']
               and audit[ds]['n_samples'] == len(manifests[ds])
               and audit[ds]['manifest_sha256'] == hashlib.sha256((args.data_dir/ds/'manifest.csv').read_bytes()).hexdigest()
               for ds in DATASETS)
    # Results must come from completed experiments, never a preview fixture.
    bdata, sdata, adata = supervised['datasets'], finetuned['datasets'], alm['datasets']
    matrices, counts = {}, {}
    for ds in DATASETS:
        counts[ds] = Counter(r['split'] for r in manifests[ds])
        labels = sorted({r['label'] for r in manifests[ds] if r['split']=='train'})
        for name, entry in [('frozen', bdata[ds]), ('supervised', sdata[ds]), ('alm', adata[ds])]:
            class_order = entry['classes']
            assert set(class_order) == set(labels), f'Unexpected classes: {name} {ds}'
            matrices[name, ds] = validate_metrics(entry['validation'], class_order, counts[ds]['validation'])
            supports = Counter(r['label'] for r in manifests[ds] if r['split'] == 'validation')
            assert matrices[name, ds].sum(axis=1).tolist() == [supports[label] for label in class_order]
        assert bdata[ds]['cv_results'], 'Frozen training CV results required'
        assert sdata[ds]['selected'], 'Fine-tuned model metadata required'
        assert sdata[ds]['checkpoint_roundtrip_exact'] is True
        audit_ft = sdata[ds]['weight_audit']
        assert audit_ft['frozen_weights_unchanged'] is True
        assert audit_ft['frozen_sha256_before'] == audit_ft['frozen_sha256_after']
        assert audit_ft['trainable_pretrained_parameters'] == 28351488
        assert set(audit_ft['blocks']) == {'8', '9', '10', '11'}
        assert all(block['changed_elements'] > 0 and block['delta_l2'] > 0 for block in audit_ft['blocks'].values())
        selection = sdata[ds]['selection']
        fit_ids, holdout_ids = set(selection['fit_ids']), set(selection['holdout_ids'])
        train_ids = {row['sample_id'] for row in manifests[ds] if row['split'] == 'train'}
        assert fit_ids.isdisjoint(holdout_ids) and fit_ids | holdout_ids == train_ids
        assert selection['fit_samples'] == len(fit_ids) and selection['holdout_samples'] == len(holdout_ids)
        assert sdata[ds]['final_training']['n_samples'] == len(train_ids)
        assert 1 <= sdata[ds]['selected']['epochs'] <= 6
        history = selection['history']
        assert [row['epoch'] for row in history] == list(range(1, 7))
        winner = max(history, key=lambda row: (row['selection_score'], row['holdout_top1'], -row['epoch']))
        assert sdata[ds]['selected']['epochs'] == winner['epoch']
        assert [row['epoch'] for row in sdata[ds]['final_training']['history']] == list(range(1, winner['epoch']+1))
        for phase in (selection, sdata[ds]['final_training']):
            proof = phase['encoder_gradient_proof']
            assert proof['measured_before_gradient_clipping'] is True
            assert set(proof['blocks']) == {'8', '9', '10', '11'}
            assert all(block['all_gradients_finite'] is True and block['gradient_l2'] > 0
                       and np.isfinite(block['gradient_l2']) and block['parameter_tensors_with_gradient'] > 0
                       for block in proof['blocks'].values())
        assert adata[ds]['invalid_outputs'] >= 0
        for comparison in bdata[ds]['comparisons'].values():
            validate_metrics(comparison['validation'], bdata[ds]['classes'], counts[ds]['validation'])
    meta = alm['metadata']
    for key in ('model', 'model_url', 'revision', 'quantization', 'engine', 'engine_url',
                'prompt_design', 'prompt_A', 'prompt_B', 'audio_processing', 'parsing', 'invalid_handling'):
        assert meta.get(key), f'Missing ALM method metadata: {key}'
    for url in (args.cloud_url, args.repo_url):
        if url:
            assert urlparse(url).scheme == 'https' and urlparse(url).netloc
    student_id = args.student_id if args.student_id != 'STUDENT_ID_PENDING' else None
    assert not student_id or '/' not in student_id
    output = args.output or ROOT/'output'/'pdf'/'report_en.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    figures = ROOT/'tmp'/'pdfs'
    figures.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont(str(CJK_FONT))
    plt.rcParams.update({'font.family': ['DejaVu Sans', 'Droid Sans Fallback'],
                         'font.size':11, 'axes.titleweight':'normal', 'axes.unicode_minus':False})
    draft_label = 'Course cloud link pending' if student_id and not args.cloud_url else 'Submission details pending'
    r = Report(output, student_id, args.draft or not student_id or not args.cloud_url,
               student_name=args.student_name, draft_label=draft_label)
    va, vb = bdata['dataset_A']['validation'], bdata['dataset_B']['validation']

    # 1. Assignment, data, and submission links.
    r.new_page('Homework 1: Music Release Decade and Market')
    identity = ('Name: '+esc(args.student_name)+'    ' if args.student_name else '')+'Student ID: '+esc(student_id or 'pending')
    r.text(identity,44,86,872,size=14)
    r.text('This experiment classifies release decade and release market from the supplied 30-second audio excerpts. It first trains classifiers on frozen MERT features, then fine-tunes the last four MERT blocks. Qwen2.5-Omni provides an audio language model comparison. All three methods are evaluated on the same official validation recordings.',44,129,872,size=13)
    rows=[]
    for ds,description in [('dataset_A','US music, six decades: 1960s to 2010s'),('dataset_B','1980s music, six release markets')]:
        n=counts[ds];rows.append([ds,description,n['train'],n['validation'],n['test']])
    r.table(['Dataset','Classification task','Train','Validation','Test'],rows,44,212,[122,346,134,135,135],rowheight=35,size=12)
    r.text('Audio is mono, 24 kHz, PCM 16-bit WAV. Each class has 171 training and 22 validation recordings in A, and 133 training and 17 validation recordings in B. The assignment states that the official splits are artist-disjoint. No additional labeled audio was used.',44,338,872,size=12)
    r.text('Test labels are hidden. Accuracy refers to official validation, except for the internal training holdout on page 4. Top-3 is correct when the true label is among three distinct ranked predictions.',44,398,872,size=12)
    r.text('Code: <link href="'+esc(args.repo_url)+'">'+esc(args.repo_url)+'</link>',44,450,872,size=10.5)
    cloud=('<link href="'+esc(args.cloud_url)+'">'+esc(args.cloud_url)+'</link>') if args.cloud_url else 'pending; the GitHub code link is separate.'
    r.text('Public course submission folder: '+cloud,44,477,872,size=10.5)

    # 2. Frozen-feature workflow and training-only model selection.
    r.new_page('Frozen MERT features and classifiers', 'train.py learns six-class linear weights from fixed features extracted by features.py; MERT stays unchanged.')
    r.text('features.py splits each 30-second recording into six non-overlapping 5-second windows. Each window is centered and divided by sqrt(variance + 1e-7). Frozen MERT-v1-95M returns 13 hidden-state levels with 768 channels each. The mean and population standard deviation over all frames in all six windows give 1,536 features per level.',44,121,872,size=12)
    r.text('The program also extracts 244 acoustic features from the full excerpt: time means and standard deviations of 64 log-mel bands, 20 MFCCs, 20 MFCC deltas, 12 chroma bins, and centroid, bandwidth, rolloff, flatness, RMS and zero-crossing rate. STFT uses 2,048 samples and a hop of 512. Features are cached for train.py.',44,204,872,size=12)
    r.text('Model selection uses stratified three-fold CV within the training split (seed 2026). StandardScaler is fitted on the training part of each fold, followed by per-recording L2 normalization. Five fixed feature recipes each use Ridge with alpha = 0.01, 0.1, 1, 10 or RBF SVC with C = 0.1, 1, 10, 100 (gamma=scale): 40 candidates per task.',44,287,872,size=11.7)
    rows=[]
    recipes={'dataset_A':'Mean levels 1–4, 5–8 and 9–12, concatenated with acoustic features','dataset_B':'Mean of levels 5–8'}
    for ds in DATASETS:
        v=bdata[ds]['selected']
        rows.append([TITLES[ds],recipes[ds],f"{v['feature_dimensions']:,}",'Ridge alpha=1'])
    r.table(['Task','Selected features','Size','Classifier'],rows,44,361,[120,510,94,148],rowheight=34,size=10)
    r.text('Selection maximizes Top-1 + 0.5×Top-3; ties use Top-1, then fixed candidate order. The scaler and classifier are refitted on all official training data before validation. No augmentation is used.',44,473,872,size=10,leading=13)

    # 3. Both frozen classifiers' validation matrices and error discussion.
    r.new_page('Frozen classifier validation results', 'All confusion matrices show counts. Rows are true classes; columns are predicted classes.')
    for k,ds in enumerate(DATASETS):
        x=44+k*446; val=bdata[ds]['validation'];cm=matrices['frozen',ds]
        r.text(f"{TITLES[ds]}: Top-1 {pct(val['top1'])}, Top-3 {pct(val['top3'])} (n={counts[ds]['validation']})",x,121,423,size=11.6)
        r.figure(confusion_figure(cm,bdata[ds]['classes']),x,152,420,239,figures/f'en-{ds}-frozen.png')
        if ds=='dataset_A':
            hist,adjacent,total=decade_errors(cm)
            message=f'{adjacent}/{total} errors ({pct(adjacent/total)}) are between adjacent decades. The 2000s and 2010s are confused 15 times. Recall is 17/22 for the 1960s and 4/22 for the 2000s. Recording style may carry decade cues, but this experiment does not establish which cues the classifier uses.'
        else:
            message='US and UK account for 10/60 errors (16.7%). Recall is 12/17 for Brazil and 4/17 for Spain. These labels describe release markets, not artist nationality or vocal language. Different markets may share musical styles and production practices.'
        r.text(message,x,403,421,size=10.2,leading=13.3)
    a=bdata['dataset_A']['comparisons']; b=bdata['dataset_B']['comparisons']
    r.text(f"Acoustic baseline Top-1 / Top-3: A {pct(a['acoustic']['validation']['top1'])} / {pct(a['acoustic']['validation']['top3'])}; B {pct(b['acoustic']['validation']['top1'])} / {pct(b['acoustic']['validation']['top3'])}. MERT-only A: {pct(a['mert']['validation']['top1'])} / {pct(a['mert']['validation']['top3'])}. The CV-selected A model has higher Top-3, but not the highest Top-1.",44,478,872,size=9.3,leading=12)

    # 4. Partial fine-tuning protocol, selection, and refitting.
    r.new_page('Partial fine-tuning of MERT', 'finetune_model.py defines the model; finetune.py handles data splits, training and checkpoints.')
    r.text('The same pretrained MERT base is loaded. Transformer blocks 8–11 (zero-based) are unfrozen: 28,351,488 pretrained parameters. Other weights stay fixed. Final-layer frame means and sqrt(population variance + 1e-5) form a 1,536-dimensional vector, followed by LayerNorm(1536), Dropout(0.1) and Linear(1536,6).',44,119,872,size=12)
    r.text('Each training recording contributes one random 5-second crop per epoch, with the same waveform normalization as the frozen method. Training uses FP32 cross-entropy and AdamW: backbone learning rate 1e-5, head rate 1e-3, weight decay 0.01 and gradient norm cap 1. Microbatch size is 4, accumulated to 16. Validation and test average logits from six fixed 5-second windows.',44,200,872,size=11.7)
    r.text('A stratified 20% holdout is drawn only from official training data (seed 2026). Epochs 1–6 are compared by Top-1 + 0.5×Top-3, breaking ties by Top-1 then earlier epoch. Holdout evaluation also covers all 30 seconds.',44,281,872,size=12)
    rows=[]
    for ds in DATASETS:
        info=sdata[ds]; c=info['selected'];split=info['selection']
        rows.append([TITLES[ds],f"{split['fit_samples']} / {split['holdout_samples']}",c['epochs'],pct(c['holdout_top1']),pct(c['holdout_top3'])])
    r.table(['Task','Internal fit / holdout','Epochs','Holdout Top-1','Holdout Top-3'],rows,44,349,[120,262,142,174,174],rowheight=32,size=10.5)
    r.text('After epoch selection, the original pretrained weights and a fresh head are reloaded, then fitted on all official training data. Records confirm nonzero gradients and weight updates in all four blocks, with frozen weights unchanged. Each task stores its updated blocks and head.',44,461,872,size=10.5,leading=13.5)

    # 5. Fine-tuned results, including class collapse.
    r.new_page('Fine-tuned model validation results', 'Top-1 and Top-3 are lower than the frozen classifiers for both tasks.')
    for k,ds in enumerate(DATASETS):
        x=44+k*446;val=sdata[ds]['validation'];cm=matrices['supervised',ds]
        r.text(f"{TITLES[ds]}: Top-1 {pct(val['top1'])}, Top-3 {pct(val['top3'])} (n={counts[ds]['validation']})",x,121,423,size=11.6)
        r.figure(confusion_figure(cm,sdata[ds]['classes']),x,151,420,236,figures/f'en-{ds}-finetuned.png')
        if ds=='dataset_A':
            hist,adjacent,total=decade_errors(cm)
            message=f'Adjacent-decade errors: {adjacent}/{total} ({pct(adjacent/total)}). Recall for the 2000s is 0/22; only one of 132 recordings is predicted as 2000s.'
        else:
            message='Brazil, Spain and Germany are never predicted as Top-1, so recall is zero for all three. Predictions are confined to US, UK and Italy.'
        r.text(message,x,400,421,size=11)
    r.text('This fine-tuning experiment was defined after observing official validation results of the frozen model. Fitting and epoch selection use only official training data. The methods also differ in pooling and classifier head, and frozen A includes acoustic features, so this comparison cannot isolate the effect of unfreezing.',44,457,872,size=10.5,leading=13.5)

    # 6. Full ALM experiment, method and both matrices.
    r.new_page('Audio language model comparison', 'Qwen2.5-Omni-3B Thinker, Q4_K_M language model and Q8_0 audio projector; llama.cpp b11049 / Vulkan.')
    prompt_url=args.repo_url+'/blob/codex/music-era-market/alm_eval.py'
    left='alm_eval.py sends the full 30-second clip to a local server. It is resampled to 16 kHz, padded with 30 seconds of silence, and processed as two 3,000-frame chunks (1,500 audio tokens). This experimental path has not been shown equivalent to original Transformers preprocessing.<br/>The fixed prompt asks for six ranked labels using instruments, rhythm, voice and production cues from the whole clip. B explicitly separates release market from nationality and language.'
    r.text(left,44,113,421,size=9.6,leading=12.6)
    right='One preset prompt was chosen to limit inference cost and parsing ambiguity; it was not changed after validation. <link href="'+esc(prompt_url)+'">Exact prompts and label mappings: alm_eval.py</link>.<br/>GBNF requires each of six codes once; parsing takes the top three. Scores 6 to 1 are ranks. Temperature=0, seed=42. An invalid answer is retried once with the same input, then replaced by a flagged fixed ranking. Server errors stop the run.'
    r.text(right,495,113,421,size=9.6,leading=12.6)
    for k,ds in enumerate(DATASETS):
        x=44+k*446;val=adata[ds]['validation'];cm=matrices['alm',ds]
        r.text(f"{TITLES[ds]}: Top-1 {pct(val['top1'])}, Top-3 {pct(val['top3'])} (n={counts[ds]['validation']})",x,230,423,size=11)
        r.figure(confusion_figure(cm,adata[ds]['classes']),x+10,253,405,203,figures/f'en-{ds}-alm.png')
        absent=missing_top1_labels(cm,adata[ds]['classes']);pred=cm.sum(axis=0);i=int(np.argmax(pred))
        r.text('Never Top-1: '+esc(', '.join(absent))+f". Most common: {adata[ds]['classes'][i]} ({pred[i]}/{pred.sum()}).",x,463,421,size=9.2)
    r.text('Every validation recording was evaluated: A 132, B 102. All 234 outputs were valid; no fallback rankings were used.',44,483,872,size=9.5)

    # 7. Usage, reproducibility, limitations and source links.
    r.new_page('Running the code and sources')
    r.text('features.py extracts features; train.py fits the frozen-feature classifiers; finetune.py trains the fine-tuned models. Follow README to install packages and set the A/B test paths, then run inference.py. README contains the full commands and model paths.',44,101,872,size=12)
    r.text('Write D15921B14.json with three distinct classes per recording, ordered by descending model score. The default is the fine-tuned model; README also gives the frozen-model command. verify_submission.py checks the output format and reproduced predictions.',44,171,872,size=12)
    r.text('Both methods were rerun from all 234 test audio files and exactly reproduced the saved predictions. Test labels are hidden, so this verifies reproducibility, not test accuracy.',44,240,872,size=12)
    r.text('Pretraining overlap with these recordings is unknown. Anonymized data has no artist names, so internal training splits cannot be grouped by artist. The causes of class concentration in the audio language model were not separately tested.',44,299,872,size=11,color=MUTED)
    write_sources(r)
    assert r.page == 7, r.page
    r.canvas.save()
    print(output.resolve())


def self_check():
    cm=np.eye(6,dtype=int)*2
    cm[0,1]=2; cm[0,3]=1
    histogram,adjacent,total=decade_errors(cm)
    assert histogram.tolist()==[2,0,1,0,0] and (adjacent,total)==(2,3)
    assert error_pairs(cm,list('abcdef'),1)==[(2,'a','b')]
    metrics={'top1':12/15,'top3':1,'confusion_matrix':cm.tolist()}
    assert validate_metrics(metrics,list('abcdef'),15).sum()==15
    order = [5, 2, 0, 4, 1, 3]
    reordered = dict(metrics, confusion_matrix=cm[np.ix_(order, order)].tolist())
    result = validate_metrics(reordered, [list('abcdef')[i] for i in order], 15)
    assert result.sum(axis=1).tolist() == cm.sum(axis=1)[order].tolist()
    assert np.trace(result) == np.trace(cm)
    collapsed = np.zeros((6, 6), dtype=int)
    collapsed[:, :3] = 1
    assert missing_top1_labels(collapsed, list('abcdef')) == list('def')
    assert set(missing_top1_labels(collapsed[np.ix_(order, order)], [list('abcdef')[i] for i in order])) == set('def')
    try:
        validate_metrics(metrics,list('abcdef'),16)
    except AssertionError:
        pass
    else:
        raise AssertionError('Incomplete evaluations must fail')
    print('Report result validation and error-analysis self-check passed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student-id', help='Optional; omitted identity is visibly marked pending')
    parser.add_argument('--student-name', help='Optional student name for the cover and PDF author metadata')
    parser.add_argument('--repo-url', default='https://github.com/Jimmynycu/music-era-market-classification')
    parser.add_argument('--cloud-url')
    parser.add_argument('--supervised',type=Path,default=ROOT/'results'/'metrics.json')
    parser.add_argument('--finetuned',type=Path,default=ROOT/'results'/'finetune_metrics.json')
    parser.add_argument('--alm',type=Path,default=ROOT/'results'/'alm_metrics.json')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--draft',action='store_true',help='Mark submission details as pending')
    parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:
        self_check()
    else:
        build(args)
