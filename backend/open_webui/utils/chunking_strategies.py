"""
Multi-strategy document chunking for RAG knowledge bases.
Adapted from reference RAG-Chunking project. Provides 8 chunking methods
plus keyword extraction and question generation.
"""

import re

try:
    import jieba.analyse
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

CHUNKING_METHODS = {
    'naive': 'Fixed-size Chunking',
    'general': 'General (Paragraph-aware)',
    'book': 'Book (Chapter Recognition)',
    'paper': 'Paper (Section Recognition)',
    'resume': 'Resume (Module Recognition)',
    'table': 'Table (CSV/Markdown->KV)',
    'qa': 'Q&A Pair Recognition',
    'auto': 'Auto (Content-Routing + Genre)',
}


def chunk_document(content: str, method: str, params: dict | None = None):
    if params is None:
        params = {}
    methods = {
        'naive': naive_chunking,
        'general': general_chunking,
        'book': book_chunking,
        'paper': paper_chunking,
        'resume': resume_chunking,
        'table': table_chunking,
        'qa': qa_chunking,
        'auto': auto_chunking,
    }
    fn = methods.get(method, general_chunking)
    return fn(content, params)


# Chinese "container / discourse" words (方法/问题/内容…) that raw frequency
# ranking on one short chunk over-weights even though they carry little topical
# signal. Dropping them pushes the focus onto the real subject words.
KEYWORD_STOPWORDS = {
    '一个', '一种', '这个', '那个', '这些', '那些', '这样', '那样', '其中', '然后',
    '就是', '还是', '什么', '主要', '没有', '进行', '以及', '还有', '如果', '因为',
    '所以', '但是', '已经', '通过', '使用', '可以', '内容', '方式', '方法', '问题',
    '情况', '方面', '部分', '里面', '时候', '东西', '当前', '目前', '整个', '相关',
    '一定', '重要', '非常', '比较', '由于', '比如', '例如', '需要', '可能', '应该',
    '对于', '关于', '来说', '是否', '之间', '以上', '以下', '同时', '其实', '容易',
    '等等', '我们', '他们', '你们', '自己', '一个', '一些', '这种', '那段', '这段',
    '本', '中', '上', '下', '内', '外', '后', '前', '和', '与', '或', '并', '及',
    '的', '了', '在', '是', '为', '等',
}

# Small English stop list: lets code tokens / acronyms (RAG, file_id) survive
# as keywords while function words in English prose do not.
_EN_STOPWORDS = {
    'the', 'and', 'for', 'with', 'from', 'this', 'that', 'these', 'those',
    'are', 'was', 'were', 'you', 'your', 'not', 'can', 'have', 'has', 'had',
    'into', 'use', 'uses', 'using', 'used', 'after', 'before', 'over',
    'under', 'about', 'between', 'which', 'where', 'when', 'what', 'how',
    'why', 'its', 'their', 'then', 'also', 'will', 'would', 'could', 'should',
    'may', 'might', 'each', 'both', 'one', 'two', 'all', 'any', 'per', 'who',
    'whom', 'whose', 'does', 'did', 'of', 'to', 'in', 'on', 'at', 'as', 'or',
    'a', 'an', 'it', 'by',
}

# Latin / code-ish tokens (RAG, ChromaDB, file_id, Top-K) are high-signal in a
# Chinese dev-knowledge base, so pull them out before Chinese word ranking.
_LATIN_TOKEN_RE = re.compile(r'[A-Za-z][A-Za-z0-9_\-]*(?:\.[A-Za-z0-9_\-]+)*')


def extract_keywords(text: str, topK: int = 5) -> list[str]:
    """Rank a chunk's topical terms with no corpus available.

    Two signal streams are merged:
      1. Latin / code-ish tokens (RAG, ChromaDB, file_id, Top-K) — concrete
         and unambiguous in a Chinese dev-knowledge base.
      2. Chinese content words via jieba TextRank restricted to noun/verb POS.
         (jieba's default TF-IDF needs a document collection and over-weights
         frequent fluff on a single short text, so it is used only as a
         fallback when TextRank yields nothing.)
    """
    if not text or not text.strip():
        return []

    latin = []
    for tok in _LATIN_TOKEN_RE.findall(text):
        t = tok.strip('._-')
        if len(t) < 2 or t.lower() in _EN_STOPWORDS:
            continue
        latin.append(t)
    if latin:

        def _strength(t: str) -> tuple:
            # Identifiers / versions / proper-case terms rank before plain
            # prose words; ties broken by frequency then alphabetically.
            return (
                -(2 if re.search(r'[_\-0-9]', t) else 0)
                - (1 if re.search(r'[A-Z]', t) else 0)
                - (1 if len(t) >= 5 else 0),
                -latin.count(t),
                t,
            )

        latin = sorted(set(latin), key=_strength)

    chinese: list[str] = []
    if JIEBA_AVAILABLE:
        sample = text[:8000]
        words = []
        try:
            words = jieba.analyse.textrank(
                sample,
                topK=topK * 3,
                withWeight=False,
                allowPOS=('n', 'nz', 'vn', 'v', 'nt', 'ns', 'nr'),
            ) or []
        except Exception:
            words = []
        if not words:
            try:
                words = jieba.analyse.extract_tags(sample[:5000], topK=topK * 3) or []
            except Exception:
                words = []
        chinese = [w for w in words if len(w) >= 2 and w not in KEYWORD_STOPWORDS]
    if not chinese:
        # Regex fallback when jieba is unavailable: rank 2-4 char Chinese
        # n-grams by frequency instead of treating whole clauses as "words".
        word_freq: dict[str, int] = {}
        for run in re.findall(r'[一-龥]+', text):
            for size in (2, 3, 4):
                for i in range(len(run) - size + 1):
                    gram = run[i:i + size]
                    if gram in KEYWORD_STOPWORDS:
                        continue
                    word_freq[gram] = word_freq.get(gram, 0) + 1
        ranked = sorted(word_freq.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
        chinese = [w for w, _ in ranked[: topK * 3]]

    out: list[str] = []
    seen: set[str] = set()
    for w in latin + chinese:
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= topK:
            break
    return out


def _split_sentences(text: str) -> list[str]:
    # Sentence enders are CJK stops, plus English full stops followed by
    # whitespace or a CJK char. Dots inside versions/code (v0.10.2, file.py)
    # are not boundaries because they are not followed by whitespace/CJK.
    parts = re.split(r'(?<=[。！？；!?])\s*|[.!?](?=\s|[一-龥])|[\r\n]+', text)
    return [p.strip() for p in parts if p and p.strip()]


def _pick_question(topic: str, frames: list[str], used: set[str]) -> str | None:
    t = f'「{topic}」'
    for f in frames:
        q = f.format(t=t)
        if q not in used:
            return q
    return None


# "用 X 来 做…" / "用 X 做…" instrumental sentences deserve a precise question
# about the tool X (e.g. 「ChromaDB」是用来做什么的？) instead of a generic one.
_INSTRUMENT_RE = re.compile(
    r'用\s*([^，。！？；、\s“”「」]{2,20}?)(?:\s*(?:来|去))?\s*'
    r'(存|放|记|写|读|查|取|做|构建|实现|查询|检索|展示|生成|管理|提取|拼接|记录|训练)'
)


def _instrument_question(sentence: str) -> tuple[str, str] | None:
    m = _INSTRUMENT_RE.search(sentence)
    if not m:
        return None
    tool = m.group(1).strip('的_')
    if not tool:
        return None
    return tool, f'「{tool}」是用来做什么的？'


# (cue regexes, frame pool) pairs — a definition/role/comparison frame is only
# chosen when one of its cues appears in the SAME sentence, so the question is
# approximately answerable from that sentence instead of being glued on at
# random. The `None` entry is the generic fallback used when no cue matched.
_QUESTION_CUES = (
    ((r'如何', r'怎么', r'怎样', r'通过.{0,12}(实现|完成|做到)', r'步骤'),
     ('{t}具体是如何实现的？', '实现{t}一般要分哪些步骤？')),
    ((r'因为', r'由于', r'之所以', r'原因在于', r'是为了', r'为了', r'目的是', r'为什么'),
     ('{t}是为了解决什么问题？', '为什么要用到{t}？')),
    ((r'作用是', r'用于', r'用来', r'承担', r'负责', r'起到', r'充当'),
     ('{t}起到什么作用？', '{t}主要用来做什么？')),
    ((r'包括', r'分为', r'包含', r'构成', r'由.{0,10}组成'),
     ('{t}包含哪些内容？', '{t}由哪几部分组成？')),
    ((r'相比', r'对比', r'比较', r'优于', r'强于', r'区别于', r'不同于', r'差异', r'区别'),
     ('{t}与常见做法有什么不同？', '{t}的优势体现在哪里？')),
    ((r'特点', r'特性', r'优势', r'亮点', r'具备', r'支持', r'能够'),
     ('{t}有哪些特点？', '{t}支持哪些能力？')),
    ((r'是指', r'指的是', r'称之为', r'称为', r'叫做', r'定义为', r'简称', r'^.{0,16}是'),
     ('什么是{t}？', '{t}具体指什么？')),
    (None, ('关于{t}，这段内容讲了什么？', '{t}具体是什么情况？')),
)

# Deterministic last-resort summary bank (also reachable from the exception
# path, so even an error never emits the identical filler three times).
_SUMMARY_QUESTIONS = (
    '这段内容主要介绍了什么？',
    '这段内容想说明的核心观点是什么？',
    '阅读这段内容后可以提炼出哪些要点？',
)


def generate_questions(text: str, count: int = 3) -> list[str]:
    """Deterministic, content-anchored sample questions (no LLM).

    Per sentence: the topic is the chunk's own strongest keyword that appears
    in that sentence, and the interrogative frame is picked by a cue that also
    appears in the sentence (how / why / role / composition / compare /
    feature / definition). This keeps questions related to the actual text
    instead of glueing an arbitrary word into one of two fixed templates, and
    varied frames avoid the old "three copies of the same generic sentence".
    """
    try:
        if len(text) > 3000:
            text = text[:3000]
        sentences = _split_sentences(text)
        keywords = extract_keywords(text, topK=min(8, max(count * 2, 4)))
        candidates = [
            s for s in sentences
            if len(s) >= 8
            and (
                len(re.findall(r'[一-龥]', s)) >= 4
                or bool(re.search(r'[A-Za-z][A-Za-z0-9_\-]{2,}', s))
            )
        ]

        questions: list[str] = []
        used: set[str] = set()
        used_topics: set[str] = set()
        for sentence in candidates:
            if len(questions) >= count:
                break
            topic = next((k for k in keywords if k in sentence), None)
            # Prefer a specific cue-matched frame (定义/作用/组成/对比…);
            # otherwise an instrument question ("用 X 来做…"); only then the
            # generic "关于 X，这段内容讲了什么" frame.
            specific = None
            for cues, pool in _QUESTION_CUES:
                if cues is None:
                    break
                if any(re.search(c, sentence) for c in cues):
                    specific = pool
                    break
            q = None
            qtopic = None
            if topic and specific:
                q = _pick_question(topic, specific, used)
                qtopic = topic
            if q is None:
                inst = _instrument_question(sentence)
                if inst and inst[0] not in used_topics and inst[1] not in used:
                    qtopic, q = inst
            if q is None and topic:
                q = _pick_question(topic, _QUESTION_CUES[-1][1], used)
                qtopic = topic
            if q:
                used.add(q)
                if qtopic:
                    used_topics.add(qtopic)
                questions.append(q)

        # Spread: if more sentences produced questions than needed, keep an
        # even sample across the text instead of only the head.
        if len(questions) > count:
            span = len(questions) - 1
            step = max(count - 1, 1)
            questions = [questions[min(int(i * span / step), len(questions) - 1)] for i in range(count)]

        # Top up with keyword-anchored frames, so filler never is the same
        # sentence three times, still points at real content, and never repeats
        # a topic that already produced a real question.
        for kw in keywords:
            if len(questions) >= count:
                break
            if kw in used_topics:
                continue
            q = _pick_question(kw, _QUESTION_CUES[-1][1], used)
            if q:
                used.add(q)
                used_topics.add(kw)
                questions.append(q)

        # Last resort: short, distinct summary questions.
        n = 1
        while len(questions) < count:
            if n <= len(_SUMMARY_QUESTIONS):
                q = _SUMMARY_QUESTIONS[n - 1]
            else:
                q = f'这段内容还能提炼出第{n - len(_SUMMARY_QUESTIONS)}个要点是什么？'
            n += 1
            if q not in used:
                used.add(q)
                questions.append(q)
        return questions[:count]
    except Exception:
        return [_SUMMARY_QUESTIONS[i % len(_SUMMARY_QUESTIONS)] for i in range(count)]


# ── Naive ──

def naive_chunking(content: str, params: dict) -> list[dict]:
    chunk_size = params.get('chunk_size', 500)
    overlap = params.get('overlap', 50)
    if chunk_size <= 0:
        chunk_size = 500
    step = chunk_size - overlap
    if step <= 0:
        # overlap >= chunk_size would make `start` stop advancing (infinite
        # loop); clamp overlap so every iteration makes forward progress.
        step = max(1, chunk_size // 2)
        overlap = chunk_size - step
    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        chunks.append({
            'content': content[start:end],
            'metadata': {'method': 'naive', 'start': start, 'end': end}
        })
        start += step
    return chunks


# ── General (paragraph-aware) ──

def _split_oversized_para(para: str, chunk_size: int) -> list[str]:
    """Split a single paragraph larger than chunk_size at sentence boundaries."""
    pieces = []
    sentences = [s for s in re.split(r'(?<=[。！？.!?])', para) if s.strip()]
    current = ''
    for sentence in sentences:
        if current and len(current) + len(sentence) > chunk_size:
            pieces.append(current.strip())
            current = sentence
        else:
            current += sentence
    if current.strip():
        pieces.append(current.strip())
    return pieces or [para[:chunk_size]]


def general_chunking(content: str, params: dict) -> list[dict]:
    chunk_size = params.get('chunk_size', 1000)
    overlap = params.get('overlap', 100)
    paragraphs = re.split(r'\n\n+', content)
    chunks = []
    current_chunk = []
    current_length = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > chunk_size:
            # A single paragraph that alone exceeds the target size would blow
            # past chunk_size once flushed; split it internally at sentence
            # boundaries instead.
            if current_chunk:
                chunks.append({'content': '\n\n'.join(current_chunk), 'metadata': {'method': 'general', 'paragraphs': len(current_chunk)}})
                current_chunk = []
                current_length = 0
            for piece in _split_oversized_para(para, chunk_size):
                chunks.append({'content': piece, 'metadata': {'method': 'general', 'paragraphs': 1}})
            continue
        para_length = len(para)
        if current_length + para_length > chunk_size and current_chunk:
            chunks.append({'content': '\n\n'.join(current_chunk), 'metadata': {'method': 'general', 'paragraphs': len(current_chunk)}})
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-1]
                current_chunk = [overlap_text, para]
                current_length = len(overlap_text) + para_length
            else:
                current_chunk = [para]
                current_length = para_length
        else:
            current_chunk.append(para)
            current_length += para_length
    if current_chunk:
        chunks.append({'content': '\n\n'.join(current_chunk), 'metadata': {'method': 'general', 'paragraphs': len(current_chunk)}})
    return chunks


# ── Book (chapter recognition) ──

BOOK_CHAPTER_PATTERN = re.compile(
    r'^(?:第\s*[一二三四五六七八九十百千0-9]+\s*[章节部篇]|Chapter\s+\d+|CHAPTER\s+\d+)',
    re.IGNORECASE,
)


def book_chunking(content: str, params: dict) -> list[dict]:
    chunks = []
    lines = content.split('\n')
    current_chapter = None
    current_section = []
    # Only real chapter markers open a new chunk. Treating any short
    # non-punctuation line as a heading used to fragment ordinary prose into
    # one-line "chapters" (and then silently fall back to general).
    max_size = params.get('chunk_size', 3000)
    flush_threshold = params.get('min_chunk_length', 20)
    chapter_found = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if BOOK_CHAPTER_PATTERN.match(line):
            chapter_found = True
            if current_section:
                chunks.append({'content': '\n'.join(current_section), 'metadata': {'method': 'book', 'title': current_chapter or '未命名章节'}})
            current_chapter = line
            current_section = [line]
        else:
            current_section.append(line)
            if sum(len(l) for l in current_section) > max_size:
                chunks.append({'content': '\n'.join(current_section), 'metadata': {'method': 'book', 'title': current_chapter or '未命名章节'}})
                current_section = []
    if current_section and (chapter_found or len('\n'.join(current_section)) > flush_threshold):
        chunks.append({'content': '\n'.join(current_section), 'metadata': {'method': 'book', 'title': current_chapter or '未命名章节'}})
    return chunks if chunks else general_chunking(content, params)


# ── Paper / Resume: shared section-title matching ──
# Titles must match a whole short line (optionally numbered), never a
# substring of a body sentence. Substring matching turned every sentence
# containing "方法"/"结果"/"经验"/"技能" etc. into a bogus section header.

PAPER_SECTION_TITLES = {
    'abstract', '摘要', 'introduction', '引言', '绪论', 'related work', '相关工作',
    'methodology', '方法', '方法论', 'experiment', 'experiments', '实验', '数据与实验',
    'result', 'results', '结果', '实验结果', 'discussion', '讨论', 'conclusion',
    'conclusions', '结论', 'reference', 'references', '参考文献', 'bibliography',
    'acknowledgment', 'acknowledgments', '致谢',
}

RESUME_SECTION_TITLES = {
    '个人信息', '基本信息', '个人资料', '个人简介', '个人总结', '关于我', '联系方式',
    'personal information', 'basic info', 'personal info', 'contact', 'about me',
    '教育背景', '教育经历', '学历', 'education', 'academic', 'academics',
    '工作经历', '工作经验', '职业经历', '工作履历', 'employment', 'experience', 'work experience',
    '项目经验', '项目经历', '项目实践', '项目列表', 'project', 'projects',
    '技能', '专业技能', '技术栈', 'skills', 'technical skills',
    '证书', '资格证书', '荣誉证书', 'certificate', 'certificates', 'license',
    '自我评价', 'summary', 'objective',
}


def _normalize_heading(line: str) -> str:
    """Strip leading numbering ('1.2 ', '一、', '（一）') and lowercase."""
    low = line.lower().strip()
    low = re.sub(
        r'^(?:\d+(?:\.\d+)*[\.、．\s]+|[一二三四五六七八九十]+[、．.\s]+|（[一二三四五六七八九十]+）|\(\s*\d+\s*\))',
        '', low,
    )
    return low.strip()


def _is_section_title(line: str, titles: set) -> bool:
    line = line.strip()
    if not line or len(line) > 40:
        return False
    return _normalize_heading(line) in titles


# ── Paper (section recognition) ──

def paper_chunking(content: str, params: dict) -> list[dict]:
    chunks = []
    lines = content.split('\n')
    current_section = None
    current_content = []
    section_found = False
    for line in lines:
        if _is_section_title(line, PAPER_SECTION_TITLES):
            section_found = True
            if current_content:
                chunks.append({'content': '\n'.join(current_content), 'metadata': {'method': 'paper', 'section': current_section or '未命名'}})
            current_section = line.strip()
            current_content = [line.strip()]
        else:
            if line.strip():
                current_content.append(line.strip())
    # Once any section is recognized, flush short sections too; the length
    # threshold only guards the no-section case so plain text falls back to
    # general instead of being emitted as a single "未命名" chunk.
    if current_content and (section_found or len('\n'.join(current_content)) > 50):
        chunks.append({'content': '\n'.join(current_content), 'metadata': {'method': 'paper', 'section': current_section or '未命名'}})
    return chunks if chunks else general_chunking(content, params)


# ── Resume (module recognition) ──

def resume_chunking(content: str, params: dict) -> list[dict]:
    chunks = []
    lines = content.split('\n')
    current_section = None
    current_content = []
    section_found = False
    for line in lines:
        if _is_section_title(line, RESUME_SECTION_TITLES):
            section_found = True
            if current_content:
                chunks.append({'content': '\n'.join(current_content), 'metadata': {'method': 'resume', 'section': current_section or '未命名'}})
            current_section = line.strip()
            current_content = [line.strip()]
        else:
            if line.strip():
                current_content.append(line.strip())
    if current_content and (section_found or len('\n'.join(current_content)) > 20):
        chunks.append({'content': '\n'.join(current_content), 'metadata': {'method': 'resume', 'section': current_section or '未命名'}})
    return chunks if chunks else general_chunking(content, params)


# ── Table (CSV/Markdown) ──

def _looks_like_csv(lines) -> bool:
    """A CSV table: >=2 comma-bearing lines with consistent-ish column counts."""
    data = [l.strip() for l in lines if l.strip() and not l.strip().startswith('#')]
    if len(data) < 2:
        return False
    comma_lines = [l for l in data if ',' in l]
    if len(comma_lines) < 2:
        return False
    counts = sorted({len(l.split(',')) for l in comma_lines})
    return counts[-1] - counts[0] <= 2


def table_chunking(content: str, params: dict) -> list[dict]:
    lines = content.split('\n')
    has_csv = _looks_like_csv(lines)
    has_md = any('|' in l for l in lines[:10])
    if has_csv:
        chunks = _chunk_csv_table(lines)
        return chunks if chunks else general_chunking(content, params)
    elif has_md:
        chunks = _chunk_markdown_table(lines)
        return chunks if chunks else general_chunking(content, params)
    return general_chunking(content, params)


def _chunk_csv_table(lines):
    chunks = []
    if not lines:
        return chunks
    headers = [h.strip() for h in lines[0].strip().split(',')]
    for i in range(1, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        values = [v.strip() for v in line.split(',')]
        if len(values) != len(headers):
            values = (values + [''] * len(headers))[:len(headers)]
        chunks.append({'content': '\n'.join(f"{h}: {v}" for h, v in zip(headers, values)),
                       'metadata': {'method': 'table', 'format': 'csv', 'row': i}})
    return chunks


def _chunk_markdown_table(lines):
    chunks = []
    in_table = False
    header = None
    rows = []
    title = None
    for line in lines:
        line = line.strip()
        if not line:
            if in_table and header and rows:
                chunks.extend(_convert_md_table(header, rows, title))
                header = None
                rows = []
                in_table = False
            continue
        if line.startswith('#'):
            if header and rows:
                chunks.extend(_convert_md_table(header, rows, title))
                header = None
                rows = []
            title = line.lstrip('#').strip()
            in_table = False
            continue
        if '|' in line:
            if all(c in '|-: \t' for c in line):
                continue
            if not in_table:
                in_table = True
                header = line
            else:
                rows.append(line)
        elif in_table and header and rows:
            chunks.extend(_convert_md_table(header, rows, title))
            header = None
            rows = []
            in_table = False
    if header and rows:
        chunks.extend(_convert_md_table(header, rows, title))
    return chunks


def _convert_md_table(header_line, data_rows, title=None):
    chunks = []
    headers = [h.strip() for h in header_line.split('|') if h.strip()]
    for i, row_line in enumerate(data_rows):
        parts = row_line.split('|')
        values = [v.strip() for v in parts[1:-1]] if len(parts) > 2 else []
        if len(values) != len(headers):
            values = (values + [''] * len(headers))[:len(headers)]
        kv = [f"表格: {title}", ""] if title else []
        kv.extend(f"{h}: {v}" for h, v in zip(headers, values))
        chunks.append({'content': '\n'.join(kv), 'metadata': {'method': 'table', 'format': 'markdown', 'row': i + 1, 'title': title or '未命名'}})
    return chunks


# ── QA Pairs ──

QA_PATTERN = r'(?:问题?[:：]|Q[:：])\s*(.+?)\s*(?:答案?[:：]|A[:：])\s*(.+?)(?=(?:问题?[:：]|Q[:：])|$)'


def qa_chunking(content: str, params: dict) -> list[dict]:
    matches = re.findall(QA_PATTERN, content, re.DOTALL | re.IGNORECASE)
    chunks = []
    for question, answer in matches:
        chunks.append({
            'content': f"问题: {question.strip()}\n答案: {answer.strip()}",
            'metadata': {'method': 'qa', 'question': question.strip(), 'answer': answer.strip()}
        })
    return chunks if chunks else general_chunking(content, params)


# ── Auto (content-routing + whole-document genre pre-detection) ──

# 'auto' splits one document into ordered, verbatim runs (markdown table /
# CSV table / Q&A / prose) and routes each run to the strategy that
# understands it: tables go to the table helpers, Q&A goes to qa_chunking,
# prose goes to whichever of book/paper/resume/general best fits the whole
# document's genre (book chapters > paper sections > resume modules > none).
#
# Invariants:
#   * Every run is routed as its ORIGINAL character slice (never re-serialized
#     and never merged across runs), so a single-type document fed to 'auto'
#     is chunked exactly as if that type's own strategy had been chosen —
#     the tests assert that equality per chunk.
#   * Classification is deliberately conservative: anything ambiguous stays
#     prose, and any run whose handler returns nothing falls back to
#     general_chunking — content is never dropped.
#   * Each chunk's metadata['method'] records the *inner* strategy that truly
#     produced it (book/table/qa/general/...), not 'auto' — so a mixed
#     document's preview shows the real per-run methods.

def _count_title_hits(lines, kind: str) -> int:
    """How many lines look like structural titles of `kind`."""
    if kind == 'book':
        return sum(1 for ln in lines if BOOK_CHAPTER_PATTERN.match(ln.strip()))
    titles = PAPER_SECTION_TITLES if kind == 'paper' else RESUME_SECTION_TITLES
    return sum(1 for ln in lines if _is_section_title(ln, titles))


def _detect_genre(lines) -> str:
    """Whole-document genre over prose lines; 'general' when nothing matches.
    Ties on hit count are broken by book > paper > resume for determinism."""
    priority = {'book': 3, 'paper': 2, 'resume': 1}
    hits = {k: _count_title_hits(lines, k) for k in priority}
    candidates = [k for k in priority if hits[k] > 0]
    if not candidates:
        return 'general'
    return max(candidates, key=lambda k: (hits[k], priority[k]))


def _is_sep_row(line: str) -> bool:
    """A markdown table separator line: only '|', '-', ':', spaces/tabs."""
    return bool(line) and '|' in line and all(c in '|-: \t' for c in line)


def _trim_qa_span(content: str, match) -> int:
    """Where a Q&A match's answer runs to EOF it swallows any following prose
    (qa_chunking's regex does this). 'auto' is more conservative: stop the run
    at the first blank line inside the answer so unrelated paragraphs stay
    prose instead of being mislabeled as the answer."""
    body = content[match.start(2):match.end()]
    gap = re.search(r'\n[ \t]*\n', body)
    if gap:
        return match.start(2) + gap.start()
    return match.end()


def _split_by_type(content: str):
    """Single forward pass over `content` → ordered [(kind, verbatim_slice)].
    Priority is md_table > csv > qa > prose; every character belongs to exactly
    one run (whitespace-only prose gaps are dropped)."""
    lines = content.split('\n')
    n = len(lines)
    starts = []
    acc = 0
    for ln in lines:
        starts.append(acc)
        acc += len(ln) + 1

    def span(i, j):
        # Verbatim char slice of lines[i:j] (keeps the '\n' separators inside).
        if j <= i:
            return None
        return starts[i], starts[j - 1] + len(lines[j - 1])

    events = []  # (kind, lo, hi) — non-overlapping, sorted later

    def add_event(kind, lo, hi):
        for _, elo, ehi in events:
            if lo < ehi and elo < hi:
                return  # overlapping an earlier island: leave it to prose
        events.append((kind, lo, hi))

    i = 0
    while i < n:
        s = lines[i].strip()
        if not s:
            i += 1
            continue

        # 1) Markdown table: a '|' header line whose next non-blank line is a
        #    '|-: \t' separator. Data rows are contiguous non-blank '|' lines.
        if '|' in s:
            k = i + 1
            while k < n and k <= i + 4 and not lines[k].strip():
                k += 1
            if k < n and _is_sep_row(lines[k].strip()):
                m = k + 1
                while m < n and lines[m].strip() and '|' in lines[m] and not _is_sep_row(lines[m].strip()):
                    m += 1
                sp = span(i, m)
                if sp:
                    add_event('md_table', sp[0], sp[1])
                i = m
                continue

        # 2) CSV block: consecutive non-blank ','-bearing lines. Blanks DO end
        #    the block (a blank paragraph boundary is how prose paragraphs are
        #    separated; a real CSV rarely has interior blanks). Stricter than
        #    _looks_like_csv: header + >=2 data rows, so comma-heavy prose
        #    rarely false-positives into a table.
        if ',' in s:
            k = i
            while k < n and lines[k].strip() and ',' in lines[k]:
                k += 1
            block = [ln.strip() for ln in lines[i:k]]
            if len(block) >= 3 and _looks_like_csv(block):
                sp = span(i, k)
                if sp:
                    add_event('csv', sp[0], sp[1])
                i = k
                continue

        i += 1

    # 3) Q&A runs: the SAME regex qa_chunking uses, so boundaries align and a
    #    pure Q&A document chunks identically. Spans that would cross a
    #    table/csv island are dropped (their text already belongs to prose).
    for mt in re.finditer(QA_PATTERN, content, re.DOTALL | re.IGNORECASE):
        lo, hi = mt.span()
        if any(lo < ehi and elo < hi for _, elo, ehi in events):
            continue
        add_event('qa', lo, _trim_qa_span(content, mt))

    events.sort(key=lambda e: e[1])

    runs = []
    pos = 0
    for kind, lo, hi in events:
        if lo > pos:
            prose = content[pos:lo]
            if prose.strip():
                runs.append(('prose', prose))
        runs.append((kind, content[lo:hi]))
        pos = max(pos, hi)
    if pos < len(content):
        tail = content[pos:]
        if tail.strip():
            runs.append(('prose', tail))
    return runs


def auto_chunking(content: str, params: dict) -> list[dict]:
    if not content.strip():
        return []
    runs = _split_by_type(content)
    if not runs:
        return []

    # Whole-document genre pre-detection over the prose body only (tables and
    # Q&A islands are routed by type, not by genre).
    prose_lines = []
    for kind, sl in runs:
        if kind == 'prose':
            prose_lines.extend(sl.split('\n'))
    genre = _detect_genre(prose_lines) if prose_lines else 'general'
    handler = {
        'book': book_chunking,
        'paper': paper_chunking,
        'resume': resume_chunking,
        'general': general_chunking,
    }[genre]

    out = []
    for kind, sl in runs:  # runs are already in original document order
        if kind == 'prose':
            out.extend(handler(sl, params) or general_chunking(sl, params))
        elif kind == 'md_table':
            out.extend(_chunk_markdown_table(sl.split('\n')) or general_chunking(sl, params))
        elif kind == 'csv':
            out.extend(_chunk_csv_table(sl.split('\n')) or general_chunking(sl, params))
        else:  # qa
            out.extend(qa_chunking(sl, params) or general_chunking(sl, params))
    return out
