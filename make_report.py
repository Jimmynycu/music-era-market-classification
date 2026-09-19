"""Build a ten-page Traditional Chinese 16:9 report from verified results."""
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
INK, MUTED = '#182C37', '#586A74'
TEAL, ORANGE, PAPER, LINE = '#087E8B', '#CB6C40', '#F5F7F6', '#D9E3E3'
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
CJK_FONT = Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
DATASETS = ('dataset_A', 'dataset_B')
TITLES = {'dataset_A': '發行年代', 'dataset_B': '發行市場'}


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
    def __init__(self, output, student_id, draft=False):
        for name, filename in [('Body', 'DejaVuSans.ttf'), ('Bold', 'DejaVuSans-Bold.ttf')]:
            pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / filename)))
        pdfmetrics.registerFont(TTFont('CJK', str(CJK_FONT)))
        pdfmetrics.registerFontFamily('Body', normal='Body', bold='Bold', italic='Body', boldItalic='Bold')
        self.cjk_glyphs = pdfmetrics.getFont('CJK').face.charToGlyph
        self.canvas = canvas.Canvas(str(output), pagesize=(W, H))
        self.canvas.setTitle('音樂發行年代與市場分類：實驗報告')
        self.canvas.setAuthor(student_id or '學號待補')
        self.student_id, self.page, self.draft = student_id, 0, draft

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

    def rect(self, x, y, width, height, fill=PAPER, radius=10):
        self.canvas.setFillColor(colors.HexColor(fill))
        self.canvas.roundRect(x, H - y - height, width, height, radius, fill=1, stroke=0)

    def rule(self, y):
        self.canvas.setStrokeColor(colors.HexColor(LINE))
        self.canvas.line(44, H-y, W-44, H-y)

    def new_page(self, title, subtitle=None, section='實驗報告'):
        if self.page:
            self.canvas.showPage()
        self.page += 1
        self.canvas.setFillColor(colors.white)
        self.canvas.rect(0, 0, W, H, fill=1, stroke=0)
        self.canvas.setFillColor(colors.HexColor(TEAL))
        self.canvas.rect(0, H-7, W, 7, fill=1, stroke=0)
        self.text(section, 44, 22, 700, size=9, color=TEAL, bold=True)
        self.text(title, 44, 48, 880, size=28, bold=True)
        if subtitle:
            self.text(subtitle, 44, 94, 872, size=12, color=MUTED)
        self.rule(505)
        self.text(f'學號：{esc(self.student_id or "待補")}　｜　作業一　｜　僅使用本機 CPU／GPU',44,516,690,size=8,color=MUTED)
        self.canvas.setFont('Body', 8)
        self.canvas.setFillColor(colors.HexColor(MUTED))
        self.canvas.drawRightString(W-44, 19, f'{self.page:02d} / 10')
        if self.draft:
            self.text('提交資訊待補',794,23,122,size=9,color=ORANGE)

    def card(self, x, y, width, title, value, detail, accent=TEAL):
        self.rect(x, y, width, 116)
        self.text(title, x+18, y+15, width-36, size=11, color=MUTED)
        self.text(value, x+18, y+37, width-36, size=30, color=accent, bold=True)
        self.text(detail, x+18, y+84, width-36, size=10, color=MUTED)

    def notes(self, entries, x, y, width, size=14, gap=15):
        for title, body in entries:
            height = self.text(f'<b>{esc(title)}</b><br/>{body}', x, y, width, size=size)
            y += height + gap
        return y

    def figure(self, fig, x, y, width, height, target):
        fig.savefig(target, dpi=180, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        self.canvas.drawImage(str(target), x, H-y-height, width=width, height=height,
                              preserveAspectRatio=True, anchor='c', mask='auto')

    def table(self, headers, rows, x, y, widths, rowheight=35, size=12):
        self.rect(x, y, sum(widths), rowheight, TEAL, radius=4)
        offset = x
        for value, width in zip(headers, widths):
            self.text(esc(value), offset+10, y+8, width-18, size=size, color='#FFFFFF', bold=True)
            offset += width
        for i, row in enumerate(rows):
            top = y + (i+1) * rowheight
            if i % 2 == 0:
                self.rect(x, top, sum(widths), rowheight, PAPER, radius=0)
            offset = x
            for value, width in zip(row, widths):
                self.text(str(value), offset+10, top+8, width-18, size=size)
                offset += width
        return y + (len(rows)+1)*rowheight


def confusion_figure(cm, labels, title=None):
    fig, ax = plt.subplots(figsize=(5.5, 4.1))
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=max(1, cm.sum(axis=1).max()))
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12,
                    color='white' if cm[i,j] > cm.sum(axis=1).max()*.55 else INK)
    ax.set_xticks(range(6), labels, rotation=25, ha='right', fontsize=10)
    ax.set_yticks(range(6), labels, fontsize=10)
    ax.set(xlabel='預測類別', ylabel='真實類別', title=title)
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
    output = args.output or ROOT/'output'/'pdf'/'report_zh-TW.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    figures = ROOT/'tmp'/'pdfs'
    figures.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont(str(CJK_FONT))
    plt.rcParams.update({'font.family': ['DejaVu Sans', 'Droid Sans Fallback'],
                         'font.size':11, 'axes.titleweight':'normal', 'axes.unicode_minus':False})
    r = Report(output, student_id, args.draft or not student_id or not args.cloud_url)
    va, vb = bdata['dataset_A']['validation'], bdata['dataset_B']['validation']

    # 1. Lead with the better measured frozen-feature systems, without reselecting predictions.
    r.new_page('音樂發行年代與市場分類', '凍結特徵分類器、預訓練模型微調，以及本機音訊語言模型的比較', '作業一　／　繁體中文實驗報告')
    r.text('先看較佳表現：<br/>凍結 MERT 特徵分類器',48,155,520,size=28,bold=True)
    r.text('兩項任務皆以 30 秒音樂片段預測單一類別。凍結方案的官方驗證表現優於本次微調；以下先呈現凍結方案，再分析微調與音訊語言模型的結果及限制。',48,259,505,size=15)
    r.text('學號：'+esc(student_id or '待補'),48,352,510,size=13,color=MUTED)
    r.card(616,150,298,'凍結 A　發行年代／Top-1',pct(va['top1']),f"Top-3 {pct(va['top3'])}　｜　132 筆驗證資料")
    r.card(616,284,298,'凍結 B　發行市場／Top-1',pct(vb['top1']),f"Top-3 {pct(vb['top3'])}　｜　102 筆驗證資料")
    r.rect(44,424,872,67)
    r.text('<b>公開程式碼與重現說明：</b><link href="'+esc(args.repo_url)+'" color="'+TEAL+'">'+esc(args.repo_url)+'</link>',60,435,840,size=11)
    cloud=('<link href="'+esc(args.cloud_url)+'" color="'+TEAL+'">'+esc(args.cloud_url)+'</link>') if args.cloud_url else '待補；GitHub 程式碼網址不等同課程要求的公開雲端資料夾。'
    r.text('課程要求之公開雲端資料夾：'+cloud,60,461,840,size=10,color=MUTED)

    # 2. Data and separation of fitting, selection, and reporting.
    r.new_page('資料集與評估規則', '課程資料共 2,292 筆；官方分割依作業說明採歌手互斥。所有 SHA-256 與音訊格式檢查通過。')
    rows=[]
    for ds,description in [('dataset_A','美國音樂，1960s–2010s'),('dataset_B','1980s 音樂，六個發行市場')]:
        n=counts[ds]
        rows.append([ds,description,n['train'],n['validation'],n['test']])
    r.table(['資料集','任務範圍','訓練','驗證','測試'],rows,44,140,[122,346,134,135,135],rowheight=40,size=12)
    r.notes([
        ('音訊與類別', '每筆皆為 30 秒、單聲道、24 kHz、PCM 16-bit WAV。A 每類訓練 171 筆、驗證 22 筆；B 每類訓練 133 筆、驗證 17 筆。未加入額外標註音訊。'),
        ('指標定義', 'Top-1 檢查第一順位是否正確；Top-3 檢查正確答案是否位於三個不同類別的候選清單。混淆矩陣數字均為筆數，列為真實類別、欄為預測類別。'),
        ('市場標籤的意義', 'B 的 US、UK、Brazil、Spain、Germany、Italy 表示發行市場，不直接等於歌手國籍、演唱語言或音樂文化。')
    ],44,288,510,size=12,gap=13)
    r.rect(592,286,324,200)
    r.text('訓練、選擇與報告分開',611,300,286,size=14,color=TEAL,bold=True)
    r.text('凍結方案：只在訓練集做三折交叉驗證。<br/><br/>微調方案：只在訓練集內切出 20% 選擇訓練輪數，再以全部訓練資料重訓。<br/><br/>官方驗證僅供結果報告；微調前已看過凍結方案的驗證結果，因此不能稱為完全未見的確認性測試。測試標籤不可見。',611,332,286,size=11.5)

    # 3. Frozen features and fixed training-only selection.
    r.new_page('凍結模型：完整片段特徵與訓練內選模', 'MERT-v1-95M 保持預訓練權重不變；兩個任務各自訓練分類器。模型與公共程式來源見第 10 頁。')
    steps=[('01　音訊分段','保留完整 30 秒，切成六段不重疊的 5 秒窗；各窗去均值，除以 sqrt(變異數 + 1e-7)。'),
           ('02　MERT 表示','取 13 層隱藏表示，每層 768 維；跨六段所有影格計算平均與母體標準差，每層 1,536 維。'),
           ('03　聲學基準','244 維：64 log-mel、20 MFCC、20 差分、12 chroma 與六項描述子的時間平均／標準差。'),
           ('04　分類與選擇','StandardScaler 僅用當折訓練資料擬合，再逐筆 L2 正規化。三折分層交叉驗證，種子 2026。')]
    for i,(title,bodytext) in enumerate(steps):
        x=44+i*222
        r.rect(x,137,206,158)
        r.text(title,x+13,151,181,size=13,color=TEAL,bold=True)
        r.text(bodytext,x+13,183,180,size=11.5)
    rows=[]
    recipes={'dataset_A':'分層 MERT＋聲學；Ridge α=1','dataset_B':'MERT 第 5–8 層平均；Ridge α=1'}
    for ds in DATASETS:
        selected=bdata[ds]['selected']
        rows.append([TITLES[ds],recipes[ds],pct(selected['cv_top1']),pct(selected['cv_top3'])])
    r.table(['任務','訓練 CV 選定方案','CV Top-1','CV Top-3'],rows,44,315,[128,456,144,144],rowheight=34,size=11)
    r.text('固定 40 組候選：五種特徵配方 ×〔Ridge α = 0.01／0.1／1／10；RBF SVC C = 0.1／1／10／100，gamma=scale〕。依 Top-1 + 0.5×Top-3 最大者選擇，再以 Top-1、固定順序破同分。A 拼接第 1–4／5–8／9–12 層平均及聲學統計，共 4,852 維；B 為 1,536 維。',44,435,872,size=10.5)
    r.text('聲學設定：STFT 2,048 點、hop 512；描述子為頻譜重心、頻寬、rolloff、flatness、RMS、過零率。無資料增強。',44,483,872,size=9,color=MUTED)

    # 4-5. The selected frozen models' complete confusion matrices and errors.
    for ds in DATASETS:
        info=bdata[ds]; cm=matrices['frozen',ds]; classes=info['classes']; val=info['validation']
        r.new_page('凍結方案結果：'+TITLES[ds],f"完整官方驗證集，共 {counts[ds]['validation']} 筆。矩陣為計數；分類器與特徵由訓練集內 CV 決定。")
        r.figure(confusion_figure(cm,classes),35,134,531,355,figures/f'zh-{ds}-frozen.png')
        r.card(594,137,150,'Top-1',pct(val['top1']),f'{int(np.trace(cm))} 筆正確')
        r.card(762,137,154,'Top-3',pct(val['top3']),f"共 {counts[ds]['validation']} 筆")
        recalls=cm.diagonal()/cm.sum(axis=1)
        best=int(np.argmax(recalls)); worst=int(np.argmin(recalls))
        pairs=error_pairs(cm,classes)
        pairstr='；'.join(f'{a}／{b}：{n}' for n,a,b in pairs)
        if ds=='dataset_A':
            hist,adjacent,total=decade_errors(cm)
            notes=[('多數錯誤落在相鄰年代',f'{adjacent}/{total}（{pct(adjacent/total)}）的 Top-1 錯誤相差 10 年。相差 1–5 個年代的錯誤數依序為 '+ '、'.join(map(str,hist))+ '。'),
                   ('類別與混淆',f'召回率最高 {classes[best]}：{cm[best,best]}/22；最低 {classes[worst]}：{cm[worst,worst]}/22。雙向混淆前三名：{pairstr}。'),
                   ('表示有用，不等於因果解釋','音色、節奏、錄音技術可能帶有年代線索，但目前未做控制實驗，不能由混淆矩陣斷言模型使用了哪一項線索。')]
        else:
            errors=int(cm.sum()-np.trace(cm)); biggest=pairs[0]
            notes=[('最高與最低召回率',f'{classes[best]}：{cm[best,best]}/17（{pct(recalls[best])}）；{classes[worst]}：{cm[worst,worst]}/17（{pct(recalls[worst])}）。'),
                   ('混淆分散於多個市場',f'{biggest[1]}／{biggest[2]} 有 {biggest[0]} 筆雙向混淆，占 {errors} 筆錯誤的 {pct(biggest[0]/errors)}，並非多數。前三名：{pairstr}。'),
                   ('解讀範圍','發行市場本來就可能共享語言、曲風及製作方式。這些是可能的解釋，尚未以額外標註或消融實驗驗證。')]
        r.notes(notes,594,276,322,size=11.5,gap=12)

    # 6. Fine-tuning is real and its selection protocol remains explicit.
    r.new_page('部分微調：實際更新預訓練編碼器', '每個任務各自訓練；解凍第 8–11 個 transformer 區塊（索引從 0 起），共 28,351,488 個預訓練參數。')
    r.notes([
        ('模型與聲音', '其餘編碼器固定。最後一層影格平均與 sqrt(母體變異數 + 1e-5) 拼成 1,536 維；分類頭為 LayerNorm → Dropout(0.1) → Linear(6)。每輪每首歌隨機裁切 5 秒；推論平均完整六段 logits。'),
        ('最佳化',f"FP32、交叉熵、AdamW。主幹 LR 1e-5／分類頭 LR 1e-3、weight decay 0.01、梯度範數上限 1；microbatch {finetuned['protocol']['micro_batch_size']}，累積至有效 batch 16。")
    ],44,137,421,size=11.5,gap=13)
    r.notes([
        ('僅用訓練資料選輪數', '種子 2026，分層保留 20% 作內部驗證。在第 1–6 輪最大化 Top-1 + 0.5×Top-3，再按 Top-1、較早輪數破同分。選定後重載原始預訓練權重與全新分類頭，用全部訓練資料重訓。'),
        ('權重更新的直接證據','兩階段皆量得四個區塊的有限非零梯度；四區塊權重確實改變，凍結參數 SHA-256 前後一致。儲存後重新載入的參數完全相同。')
    ],495,137,421,size=11.5,gap=13)
    rows=[]
    for ds in DATASETS:
        x=sdata[ds]; c=x['selected']; split=x['selection']
        deltas=' / '.join(f"{x['weight_audit']['blocks'][str(i)]['delta_l2']:.3f}" for i in range(8,12))
        rows.append([TITLES[ds],str(c['epochs']),f"{split['fit_samples']} / {split['holdout_samples']}",pct(c['holdout_top1'])+' / '+pct(c['holdout_top3']),deltas])
    r.table(['任務','輪數','內部擬合／保留','保留 Top-1／3','區塊 8／9／10／11 權重 ΔL2'],rows,44,329,[104,56,153,172,387],rowheight=33,size=10.5)
    r.rect(44,443,872,49,fill='#FFF2E9')
    r.text('驗證曝光揭露：微調方案是在看過凍結方案驗證分數後制定；配方固定後才執行，僅以內部訓練保留集選輪數。官方驗證是描述性比較，不能視為完全未見的確認性測試。',58,453,844,size=10.5,color=ORANGE)

    # 7. Both fine-tuned confusion matrices, including poor results.
    r.new_page('微調結果：未超越凍結分類器', '完整官方驗證；兩個任務分別訓練 6 輪與 3 輪。先呈現凍結方案，不代表依驗證結果重新選擇提交模型。')
    for k,ds in enumerate(DATASETS):
        x=44+k*446; val=sdata[ds]['validation']; cm=matrices['supervised',ds]
        r.text(TITLES[ds],x,136,423,size=16,color=TEAL,bold=True)
        r.text(f"Top-1 {pct(val['top1'])}　｜　Top-3 {pct(val['top3'])}　｜　n={counts[ds]['validation']}",x,166,423,size=12)
        r.figure(confusion_figure(cm,sdata[ds]['classes']),x,192,420,239,figures/f'zh-{ds}-finetuned.png')
        if ds=='dataset_A':
            hist,adjacent,total=decade_errors(cm)
            i=sdata[ds]['classes'].index('2000s')
            message=f'相鄰年代錯誤 {adjacent}/{total}（{pct(adjacent/total)}）；2000s 召回 {cm[i,i]}/{cm[i].sum()}，僅 {cm[:,i].sum()}/{cm.sum()} 筆預測為 2000s。'
        else:
            absent=missing_top1_labels(cm,sdata[ds]['classes'])
            message='、'.join(absent)+' 完全未被預測為 Top-1，三類召回率皆為零。結果顯示明顯類別集中，不能僅看整體分數。'
        r.text(esc(message),x,439,421,size=11,color=ORANGE)
    r.text('微調與凍結方案同時改變了池化方式、分類頭，且 A 的凍結方案使用額外聲學特徵，因此此比較無法單獨識別「解凍權重」的因果效果。',44,484,872,size=9.5,color=MUTED)

    # 8. Local ALM protocol and full coverage results; exact original prompt is linked.
    r.new_page('音訊語言模型：完整驗證與固定提示詞', 'Qwen2.5-Omni-3B Thinker；Q4_K_M 語言模型＋Q8_0 音訊投影器；llama.cpp b11049／Vulkan，僅本機運算。')
    left='輸入完整 30 秒 WAV；內部轉為 16 kHz，補 30 秒靜音，以兩個 3,000-frame 區塊產生 1,500 audio tokens。這是實驗性相容路徑，尚未證明等同原始 Transformers。<br/>固定一種提示：聽完整片段，參考可聽見的樂器、節奏、聲線與錄音製作，依可能性排列六類；B 明示市場不等於國籍或語言。未給範例、未用驗證回饋調整。'
    r.text(left,44,133,421,size=10.1,leading=14.4)
    right='單一預先制定提示可降低本機成本及解析歧義。<link href="'+esc(args.repo_url+'/blob/main/alm_eval.py')+'" color="'+TEAL+'">完整原文與標籤映射見 alm_eval.py</link>。<br/>GBNF 限制為六代碼的 720 種排列，嚴格檢查六個代碼各一次，取前三名；6–1 分僅為順序，不是機率。temperature=0、seed=42。無效答案以同輸入重試一次，再失敗用固定排序並標記；伺服器故障則停止。全數 234 筆均有效，無替代排序。'
    r.text(right,495,133,421,size=10.1,leading=14.4)
    for k,ds in enumerate(DATASETS):
        x=44+k*446; val=adata[ds]['validation']; cm=matrices['alm',ds]
        r.text(f"{TITLES[ds]}：Top-1 {pct(val['top1'])}／Top-3 {pct(val['top3'])}（{counts[ds]['validation']} 筆）",x,235,423,size=11.5,color=TEAL,bold=True)
        r.figure(confusion_figure(cm,adata[ds]['classes']),x+10,259,405,209,figures/f'zh-{ds}-alm.png')
        absent=missing_top1_labels(cm,adata[ds]['classes']); predicted=cm.sum(axis=0); most=int(np.argmax(predicted))
        message='未預測：'+ '、'.join(absent)+f"；最常選 {adata[ds]['classes'][most]}（{predicted[most]}/{predicted.sum()}）。"
        r.text(esc(message),x,477,423,size=9.5,color=MUTED)

    # 9. Complete score comparison and reproducibility limitations.
    r.new_page('整體比較、可重現性與限制', '以下為官方驗證上的描述性結果；均衡六分類的隨機 Top-1／Top-3 期望值為 16.7%／50.0%。')
    rows=[]
    method_rows=[('純聲學基準',lambda ds:bdata[ds]['comparisons']['acoustic']['validation']),
                 ('僅 MERT 凍結特徵',lambda ds:bdata[ds]['comparisons']['mert']['validation']),
                 ('凍結方案（訓練 CV 選定）',lambda ds:bdata[ds]['validation']),
                 ('部分微調 MERT',lambda ds:sdata[ds]['validation']),
                 ('音訊語言模型',lambda ds:adata[ds]['validation'])]
    for name,get in method_rows:
        a,b=(get(ds) for ds in DATASETS)
        rows.append([name,pct(a['top1']),pct(a['top3']),pct(b['top1']),pct(b['top3'])])
    r.table(['方法','A Top-1','A Top-3','B Top-1','B Top-3'],rows,44,134,[344,132,132,132,132],rowheight=30,size=11)
    r.notes([
        ('本機與重現',f"i7-4790K、16 GB RAM、GTX 1060 6 GB。微調使用 FP32 累積梯度；凍結特徵以 FP64 池化，分類器使用 1 個 BLAS 執行緒。重新從 WAV 推論，234 筆微調測試排名完全一致（132 A＋102 B）；凍結方案亦通過完整重現。"),
        ('預測檔分開保留','微調：finetune_predictions.json；凍結：predictions.json。報告排序沒有改變既有提交方法或預測。每筆含三個不同合法標籤；測試標籤隱藏，無法報測試準確率。')
    ],44,335,421,size=10.5,gap=12)
    r.notes([
        ('比較的限制','A 的凍結選定方案以 Top-3 換取部分 Top-1；並非每個指標都最佳。微調前已看過凍結驗證結果；兩種方案也不同時控制分類頭、池化及聲學特徵，不能宣稱單一因素造成差異。'),
        ('未知與未驗證因素','歌手資料匿名，訓練內分割無法再按歌手分組；預訓練語料是否包含這些歌曲未知。ALM 類別集中可能與提示順序、量化或實作有關，但尚未隔離原因；不能推論為原始 Qwen 的一般能力。')
    ],495,335,421,size=10.5,gap=12)

    # 10. Clickable citations, including pretrained licences and implementation sources.
    r.new_page('參考文獻與公開實作來源', '以下項目均可點擊；程式、版本、原始輸出、模型檢查與完整提示詞保留於公開儲存庫。')
    refs=[
      ('MERT：Li 等，ICLR 2024，音樂自監督表示學習論文。','https://arxiv.org/abs/2306.00107'),
      ('MERT-v1-95M 權重與程式；固定 revision 12af15fef9d0…；授權 CC BY-NC 4.0，完整版本見連結。','https://huggingface.co/m-a-p/MERT-v1-95M/tree/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'),
      ('librosa：聲學特徵與音訊分析實作。','https://github.com/librosa/librosa'),
      ('scikit-learn：標準化、分類器與交叉驗證；Pedregosa 等，JMLR 2011。','https://scikit-learn.org/stable/about.html'),
      ('Qwen2.5-Omni：模型論文與原始設計。','https://arxiv.org/abs/2503.20215'),
      ('Qwen2.5-Omni-3B GGUF；固定 revision 75f1b73b657a…；完整版本見連結。','https://huggingface.co/ggml-org/Qwen2.5-Omni-3B-GGUF/tree/75f1b73b657a50f5092502799457ccb4a4a1f9df'),
      ('llama.cpp：b11049／efa28e950，本機量化音訊推論。','https://github.com/ggml-org/llama.cpp/tree/efa28e950'),
      ('llama.cpp 音訊相容性與實驗性品質限制說明。','https://github.com/ggml-org/llama.cpp/discussions/13759'),
      ('PyTorch：編碼器載入、梯度訓練及推論。','https://github.com/pytorch/pytorch'),
      ('Hugging Face Transformers：預訓練模型介面。','https://github.com/huggingface/transformers'),
      ('NumPy 與 SciPy：數值運算；各套件版本見 requirements.txt。','https://numpy.org/citing-numpy/'),
      ('SciPy 公開實作。','https://github.com/scipy/scipy'),
      ('SoundFile：WAV 解碼與輸出。','https://github.com/bastibe/python-soundfile'),
      ('joblib：凍結分類器與標準化參數儲存。','https://github.com/joblib/joblib'),
      ('Matplotlib：結果圖表與混淆矩陣。','https://matplotlib.org/stable/project/citing.html'),
      ('課程作業一、Dataset A／B 清單與資料說明。','https://drive.google.com/drive/folders/1C8RymiLbr-EGmkxh2Ap5TIybnJYqNsb4'),
    ]
    for i,(title,url) in enumerate(refs):
        x=44+(i//8)*446; y=135+(i%8)*42
        r.text(f'<font color="{TEAL}">[{i+1}]</font> <link href="{esc(url)}" color="{INK}">{esc(title)}</link>',x,y,420,size=9.5,leading=12.5)
    r.text('<link href="https://www.reportlab.com/opensource/" color="'+TEAL+'">ReportLab 報告產生器</link>　｜　<link href="https://github.com/joblib/threadpoolctl" color="'+TEAL+'">threadpoolctl 執行緒控制</link>　｜　<link href="https://huggingface.co/Qwen/Qwen2.5-Omni-3B" color="'+TEAL+'">Qwen 原始模型</link>　｜　<link href="'+esc(args.repo_url)+'" color="'+TEAL+'">本作業公開程式碼</link>',44,481,872,size=9)
    assert r.page == 10, r.page
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
