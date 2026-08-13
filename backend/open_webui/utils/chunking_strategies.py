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
        words = re.findall(r'[一-龥]+', text[:1000])
        word_freq = {}
        for word in words:
            if len(word) >= 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        return [w for w, _ in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:topK]]
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
            if '是' in sentence or '为' in sentence:
                questions.append(f"关于这段内容，{sentence[:20]}...的具体情况是什么？")
            elif '如何' in sentence or '怎么' in sentence:
                questions.append(f"{sentence[:30]}...？")
            else:
                questions.append(f"这段内容中提到的{sentence[:15]}...是什么意思？")
            if len(questions) >= count:
                break
        while len(questions) < count:
            questions.append("这段内容的主要观点是什么？")
        return questions[:count]
    except Exception:
        return ["这段内容的主要观点是什么？"] * count


# ── Naive ──

def naive_chunking(content: str, params: dict) -> list[dict]:
    chunk_size = params.get('chunk_size', 500)
    overlap = params.get('overlap', 50)
    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        chunks.append({
            'content': content[start:end],
            'metadata': {'method': 'naive', 'start': start, 'end': end}
        })
        start = end - overlap
    return chunks


# ── General (paragraph-aware) ──

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
    chapter_pattern = re.compile(r'^(第[一二三四五六七八九十百千\d]+[章节部篇]|Chapter\s+\d+|CHAPTER\s+\d+)', re.IGNORECASE)
    max_size = params.get('chunk_size', 3000)
    for line in lines:
        line = line.strip()
        is_chapter = chapter_pattern.match(line)
        is_short_title = len(line) < 50 and line and not line.endswith(('。', '！', '？', '.', '!', '?', '，', ','))
        if is_chapter or is_short_title:
            if current_section and len('\n'.join(current_section)) > 100:
                chunks.append({'content': '\n'.join(current_section), 'metadata': {'method': 'book', 'title': current_chapter or '未命名章节'}})
            current_chapter = line
            current_section = [line]
        else:
            if line:
                current_section.append(line)
            if sum(len(l) for l in current_section) > max_size:
                chunks.append({'content': '\n'.join(current_section), 'metadata': {'method': 'book', 'title': current_chapter or '未命名章节'}})
                current_section = []
    if current_section and len('\n'.join(current_section)) > 100:
        chunks.append({'content': '\n'.join(current_section), 'metadata': {'method': 'book', 'title': current_chapter or '未命名章节'}})
    return chunks if chunks else general_chunking(content, params)


# ── Paper (section recognition) ──

def paper_chunking(content: str, params: dict) -> list[dict]:
    chunks = []
    lines = content.split('\n')
    keywords = ['abstract', '摘要', 'introduction', '引言', '绪论', 'related work', '相关工作',
                'methodology', '方法', '方法论', 'experiment', '实验', 'result', '结果',
                'discussion', '讨论', 'conclusion', '结论', 'reference', '参考文献', 'acknowledgment', '致谢']
    current_section = None
    current_content = []
    for line in lines:
        line_lower = line.lower().strip()
        is_section = any(kw in line_lower for kw in keywords) and len(line.strip()) < 100
        if is_section:
            if current_content and len('\n'.join(current_content)) > 50:
                chunks.append({'content': '\n'.join(current_content), 'metadata': {'method': 'paper', 'section': current_section or '未命名'}})
            current_section = line.strip()
            current_content = [line.strip()]
        else:
            if line.strip():
                current_content.append(line.strip())
    if current_content and len('\n'.join(current_content)) > 50:
        chunks.append({'content': '\n'.join(current_content), 'metadata': {'method': 'paper', 'section': current_section or '未命名'}})
    return chunks if chunks else general_chunking(content, params)


# ── Resume (module recognition) ──

def resume_chunking(content: str, params: dict) -> list[dict]:
    chunks = []
    lines = content.split('\n')
    keywords = ['个人信息', '基本信息', 'personal', 'contact', '教育背景', '教育经历', 'education',
                '工作经历', '工作经验', 'experience', 'employment', '项目经验', '项目经历', 'project',
                '技能', '专业技能', 'skill', '证书', '资格证书', 'certificate', '自我评价', 'summary', 'objective']
    current_section = None
    current_content = []
    for line in lines:
        line_lower = line.lower().strip()
        is_section = any(kw in line_lower for kw in keywords) and len(line.strip()) < 50
        if is_section:
            if current_content and len('\n'.join(current_content)) > 20:
                chunks.append({'content': '\n'.join(current_content), 'metadata': {'method': 'resume', 'section': current_section or '未命名'}})
            current_section = line.strip()
            current_content = [line.strip()]
        else:
            if line.strip():
                current_content.append(line.strip())
    if current_content and len('\n'.join(current_content)) > 20:
        chunks.append({'content': '\n'.join(current_content), 'metadata': {'method': 'resume', 'section': current_section or '未命名'}})
    return chunks if chunks else general_chunking(content, params)


# ── Table (CSV/Markdown) ──

def table_chunking(content: str, params: dict) -> list[dict]:
    lines = content.split('\n')
    has_csv = any(',' in l and not l.startswith('#') for l in lines[:10])
    has_md = any('|' in l for l in lines[:10])
    if has_csv:
        return _chunk_csv_table(lines)
    elif has_md:
        return _chunk_markdown_table(lines)
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
