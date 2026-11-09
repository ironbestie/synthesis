"""Create the 2-page report from measured aggregates and exported listening notes."""
import argparse,csv,json,textwrap
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from prepare import ROOT, MODELS, load

LANG={'indonesian':'Индонезийский','belarusian':'Белорусский'}
NAMES={'mms':'MMS','supertonic':'Supertonic','omnivoice':'OmniVoice','silero':'Silero','outetts':'OuteTTS'}
CATS=['Скороговорки','Слоговые ряды','Повторы (k = 8)','Сложные слова','Числительные','Предложения']
LABEL={'chars':'числом символов','words':'числом слов','phoneme_count':'числом фонем','word_repeat':'долей повторных слов','vowels':'долей гласных','cons_run':'длиной цепочки согласных','bigram_repeat':'долей повторных пар фонем','k':'числом повторов'}
def register():
    for name,mac,linux in [('Body','Arial.ttf','DejaVuSans.ttf'),('Bold','Arial Bold.ttf','DejaVuSans-Bold.ttf')]:
        fonts=[Path('/System/Library/Fonts/Supplemental')/mac,Path('/usr/share/fonts/truetype/dejavu')/linux]
        pdfmetrics.registerFont(TTFont(name,str(next(p for p in fonts if p.exists()))))
def render(notes_file=None):
    register()
    stats=load(ROOT/'summary.csv')
    samples=load(ROOT/'listening_sample.csv')
    notes=json.loads(Path(notes_file).read_text()) if notes_file else {}
    complete=all(str(notes.get(r['model']+'/'+r['id'],'')).strip() for r in samples)
    name='report.pdf' if complete else 'report_draft.pdf'
    c=Canvas(str(ROOT/name),pagesize=A4)
    c.setTitle('Синтез речи и анализ ошибок')
    c.setAuthor('Дина Ермилова')
    w,h=A4;teal='#176174';ink='#192f38'
    def line(y,title,fontsize=13):
        c.setFont('Bold',fontsize);c.setFillColor(colors.HexColor(teal));c.drawString(35,y,title)
    def para(y,s,sz=10,leading=14,width=525):
        p=Paragraph(escape(s),ParagraphStyle('p',fontName='Body',fontSize=sz,leading=leading,textColor=colors.HexColor(ink)))
        _,ht=p.wrap(width,1000);p.drawOn(c,35,y-ht);return y-ht
    def footer(n):
        c.setFont('Body',8);c.setFillColor(colors.HexColor('#49606a'));c.drawString(35,22,'Дина Ермилова  /  МТИ2531  /  Вариант 28')
        c.drawRightString(w-35,22,f'{n} / 2')
    def table(y,matrix,widths):
        t=Table(matrix,colWidths=widths)
        t.setStyle(TableStyle([('FONT',(0,0),(-1,-1),'Body',9),('FONT',(0,0),(-1,0),'Bold',9),
             ('TEXTCOLOR',(0,0),(-1,0),colors.white),('BACKGROUND',(0,0),(-1,0),colors.HexColor(teal)),
             ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#eff4f5')]),
             ('ALIGN',(1,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
             ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        _,ht=t.wrap(530,1000);t.drawOn(c,35,y-ht);return y-ht
    def get(l,m,scope,key):
        return next(r for r in stats if (r['language'],r['model'],r['scope'],r['group'])==(l,m,scope,key))
    def pct(v):return f'{100*float(v):.1f}'.replace('.',',') if str(v).strip() else 'н/д'
    line(793,'Синтез речи и анализ ошибок',22)
    y=para(776,'Индонезийский и белорусский. По три синтезатора и один распознаватель omniASR_CTC_300M_v2. Аудио подаётся в ASR на 16 кГц, декодирование greedy CTC без языковой подсказки.')
    y=para(y-7,'Вариант включает 1 800 входов. В основном сравнении у каждой модели по 150 примеров: 25 на категорию, для повторов k = 8. В отдельном эксперименте по 25 базовых выражений при каждом k.',9)
    for l,models in MODELS.items():
        y-=21;line(y,LANG[l]+' язык');cats=sorted(r['group'] for r in stats if r['language']==l and r['model']==models[0] and r['scope']=='category')
        matrix=[['Micro CER / WER, %']+[NAMES[m] for m in models]]
        for label,cat in list(zip(CATS,cats))+[('Все категории','all')]:
            rows=[get(l,m,'overall' if cat=='all' else 'category',cat) for m in models]
            matrix.append([label]+[pct(r['micro_cer'])+' / '+pct(r['micro_wer']) for r in rows])
        y=table(y-8,matrix,[155]+[366/len(models)]*len(models))
    y=para(y-12,'MMS использует встроенный голос, Supertonic F1 работает в 5 шагов, Silero использует bel_larisa. У OmniVoice 16 шагов, скорость 1. У OuteTTS температура 0,4. Фиксированные голоса FLEURS для клонирования, seed 28. Полные настройки приведены в README.',9)
    y=para(y-10,'Micro складывает ошибки и длины эталонов. Полные совпадения, macro-метрики и технические сбои доступны в summary.csv. CER/WER измеряют всю цепочку TTS и ASR.',9)
    cor_table=load(ROOT/'figures/correlations.csv')
    y-=22;line(y,'Какие свойства текста связаны с CER')
    matrix=[['Язык / модель','Символы','Повторные слова','Цепочка согласных']]
    for l,models in MODELS.items():
        for m in models:
            vals=[]
            for feat in ['chars','word_repeat','cons_run']:
                r=next(v for v in cor_table if v['language']==l and v['model']==m and v['metric']=='cer' and v['feature']==feat)
                vals.append(f"{float(r['spearman']):.2f}".replace('.',',') if r['spearman'] else 'н/д')
            matrix.append([LANG[l]+' / '+NAMES[m]]+vals)
    y=table(y-6,matrix,[185,90,122,122])
    para(y-5,'Корреляции Спирмена на основной выборке. Последний столбец использует самую длинную последовательность согласных внутри слова.',8,10)
    footer(1);c.showPage()
    line(797,'Повторы, свойства текста и разбор ошибок',18)
    # Two thumbnail charts, each with a separate scale per metric and explicit model legends.
    for j,l in enumerate(MODELS):
        c.drawImage(str(ROOT/'figures'/(l+'_repetitions.png')),46,630-j*154,width=500,height=154,preserveAspectRatio=True,anchor='c')
    y=463;line(y,'Результаты');y-=8
    for l,models in MODELS.items():
        rows=[get(l,m,'overall','all') for m in models]
        eligible=[r for r in rows if r['micro_cer']]
        if not eligible:continue
        lo=min(eligible,key=lambda r:float(r['micro_cer']));hi=max(eligible,key=lambda r:float(r['micro_cer']))
        y=para(y,f"{LANG[l]}. Наименьшая micro CER у {NAMES[lo['model']]} ({pct(lo['micro_cer'])}%), наибольшая у {NAMES[hi['model']]} ({pct(hi['micro_cer'])}%). Полных совпадений у {NAMES[lo['model']]}: {pct(lo['exact_match'])}%.",9,12)-5
    rep=[r for r in stats if r['scope']=='repetition' and r['group']=='64' and r['micro_wer']]
    if rep:
        def delta(r):return float(r['micro_wer'])-float(get(r['language'],r['model'],'repetition','1')['micro_wer'])
        worst=max(rep,key=lambda r:abs(delta(r)));base=get(worst['language'],worst['model'],'repetition','1')
        y=para(y,f"Наибольшее по модулю изменение WER между k = 1 и k = 64 у {NAMES[worst['model']]} ({LANG[worst['language']].lower()}): {pct(base['micro_wer'])}% и {pct(worst['micro_wer'])}% соответственно. Между точками сравниваются одни и те же 25 базовых выражений.",9,12)-5
    correlations=[r for r in load(ROOT/'figures/correlations.csv') if r['spearman'] and r['scope']=='main_150' and r['metric']=='cer']
    if correlations:
        best=max(correlations,key=lambda r:abs(float(r['spearman'])))
        y=para(y,f"Среди проверенных связей с CER в основной выборке наибольшее по модулю значение с {LABEL[best['feature']]} у {NAMES[best['model']]} ({LANG[best['language']].lower()}, ρ = {float(best['spearman']):.2f}, n = {best['n']}). Категории и признаки могут быть связаны между собой.",9,12)-5
    y-=15;line(y,'Примеры распознавания');y-=8
    for l in MODELS:
        rs=[r for r in samples if r['language']==l and not r['k'] and float(r['cer'])>0]
        if rs:
            r=min(rs,key=lambda r:len(r['text'])+len(r['recognized']))
            snip=lambda text:textwrap.shorten(text,width=130,placeholder='…') if text else '[пусто]'
            y=para(y,f"{NAMES[r['model']]} / {r['id']}. Эталон «{snip(r['text'])}». ASR «{snip(r['recognized'])}». CER {pct(r['cer'])}%, WER {pct(r['wer'])}%.",9,12)-6
    y-=14;line(y,'Проверка на слух');y-=8
    if complete:
        y=para(y,'Прослушано по 10 записей на язык. Полные субъективные наблюдения сохранены в listening_notes.json. На слух оценивались разборчивость, темп, шумы и естественность голоса.',9,12)-5
        y=para(y,'Индонезийский. Повторы Supertonic оценены очень хорошо. У MMS высокий темп затруднял разбор длинного повтора. У OmniVoice повтор к концу переходил в шум, а числительное in_05_08 было неразборчиво. Обычная фраза OmniVoice воспринималась как естественная речь с плохим качеством записи.',9,12)-5
        y=para(y,'Белорусский. У Silero заметен синтетический голос, но слова различимы. Длинный повтор OuteTTS неразборчив, в коротких записях слышны шумы. У OmniVoice в длинном повторе слышны другие слова, при этом обычная фраза и числительное звучат очень чётко.',9,12)-5
        r=next(r for r in samples if r['model']=='omnivoice' and r['id']=='be_05_07')
        y=para(y,f"Оценка на слух и метрики расходятся на отдельных примерах. Чёткое по оценке слушателя «трыццаць тры» (OmniVoice, be_05_07) получило CER {pct(r['cer'])}% и WER {pct(r['wer'])}%: ASR выдал символы другого алфавита. Слоговой ряд Supertonic также оценён хорошо при WER 100%. Шум, темп и естественность не измеряются CER/WER напрямую. Ошибки всей цепочки требуют проверки на слух.",9,12)-5
        assert y > 40, f'Listening section overlaps footer: {y}'
    else:
        para(y,f'Ожидаются заметки слушателя. Сейчас в листе {len(samples)} записей с исходным текстом и распознаванием.',9,12)
    footer(2);c.save();print(ROOT/name)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--notes');render(p.parse_args().notes)
