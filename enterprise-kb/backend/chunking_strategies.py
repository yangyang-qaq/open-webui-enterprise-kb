"""
Multi-strategy document chunking for RAG knowledge bases.
Adapted from reference RAG-Chunking project. Provides 7 chunking methods
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
    }
    fn = methods.get(method, general_chunking)
    return fn(content, params)


def extract_keywords(text: str, topK: int = 5) -> list[str]:
    if not text:
        return []
    try:
        if JIEBA_AVAILABLE:
            if len(text) > 5000:
                text = text[:5000]
            return jieba.analyse.extract_tags(text, topK=topK, withWeight=False) or []
    except Exception:
        pass
    try:
        # Regex fallback when jieba is unavailable: rank 2-4 char Chinese
        # n-grams by frequency instead of treating whole clauses as "words".
        stopwords = {'一个', '这个', '可以', '进行', '以及', '还有', '其中', '然后',
                     '就是', '这样', '什么', '主要', '没有', '我们', '你们', '他们',
                     '如果', '因为', '所以', '但是', '已经', '通过', '使用'}
        words = re.findall(r'[一-龥]+', text)
        word_freq = {}
        for run in words:
            for size in (2, 3, 4):
                for i in range(len(run) - size + 1):
                    gram = run[i:i + size]
                    if gram in stopwords:
                        continue
                    word_freq[gram] = word_freq.get(gram, 0) + 1
        if not word_freq:
            return []
        ranked = sorted(word_freq.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
        return [w for w, _ in ranked[:topK]]
    except Exception:
        return []


def generate_questions(text: str, count: int = 3) -> list[str]:
    try:
        questions = []
        if len(text) > 2000:
            text = text[:2000]
        sentences = re.split(r'[。！？.!?]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        for sentence in sentences[:count]:
            # Pick a concrete topic (keyword) so the question stays grammatical;
            # never splice a truncated sentence into a template.
            topic = None
            try:
                topic = (extract_keywords(sentence, topK=1) or [None])[0]
            except Exception:
                topic = None
            if not topic:
                continue
            if '如何' in sentence or '怎么' in sentence:
                questions.append(f"「{topic}」具体是如何实现的？")
            elif '是' in sentence or '为' in sentence:
                questions.append(f"「{topic}」指的是什么？")
            else:
                questions.append(f"关于「{topic}」，这段内容说了什么？")
        while len(questions) < count:
            questions.append("这段内容的主要观点是什么？")
        return questions[:count]
    except Exception:
        return ["这段内容的主要观点是什么？"] * count


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

def book_chunking(content: str, params: dict) -> list[dict]:
    chunks = []
    lines = content.split('\n')
    current_chapter = None
    current_section = []
    # Only real chapter markers open a new chunk. Treating any short
    # non-punctuation line as a heading used to fragment ordinary prose into
    # one-line "chapters" (and then silently fall back to general).
    chapter_pattern = re.compile(
        r'^(?:第\s*[一二三四五六七八九十百千0-9]+\s*[章节部篇]|Chapter\s+\d+|CHAPTER\s+\d+)',
        re.IGNORECASE,
    )
    max_size = params.get('chunk_size', 3000)
    flush_threshold = params.get('min_chunk_length', 20)
    chapter_found = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if chapter_pattern.match(line):
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

def qa_chunking(content: str, params: dict) -> list[dict]:
    qa_pattern = r'(?:问题?[:：]|Q[:：])\s*(.+?)\s*(?:答案?[:：]|A[:：])\s*(.+?)(?=(?:问题?[:：]|Q[:：])|$)'
    matches = re.findall(qa_pattern, content, re.DOTALL | re.IGNORECASE)
    chunks = []
    for question, answer in matches:
        chunks.append({
            'content': f"问题: {question.strip()}\n答案: {answer.strip()}",
            'metadata': {'method': 'qa', 'question': question.strip(), 'answer': answer.strip()}
        })
    return chunks if chunks else general_chunking(content, params)
