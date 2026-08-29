"""Unit tests for chunking_strategies.py (pure functions, no DB dependency)."""

from open_webui.utils.chunking_strategies import (
    CHUNKING_METHODS,
    book_chunking,
    chunk_document,
    extract_keywords,
    general_chunking,
    generate_questions,
    naive_chunking,
    paper_chunking,
    qa_chunking,
    resume_chunking,
    table_chunking,
)


class TestChunkDocument:
    def test_dispatches_to_naive(self):
        chunks = chunk_document('a' * 100, 'naive')
        assert chunks[0]['metadata']['method'] == 'naive'

    def test_dispatches_to_general(self):
        chunks = chunk_document('段落一\n\n段落二', 'general')
        assert chunks[0]['metadata']['method'] == 'general'

    def test_unknown_method_falls_back_to_general(self):
        chunks = chunk_document('任意文本', 'nonsense')
        assert all(c['metadata']['method'] == 'general' for c in chunks)

    def test_none_params_uses_defaults(self):
        assert chunk_document('内容', 'naive', None)

    def test_all_methods_produce_chunks(self):
        for method in CHUNKING_METHODS:
            assert chunk_document('测试内容 ' * 20, method), method


class TestNaiveChunking:
    def test_empty_content(self):
        assert naive_chunking('', {}) == []

    def test_smaller_than_chunk_size_single_chunk(self):
        chunks = naive_chunking('hello', {'chunk_size': 100, 'overlap': 0})
        assert len(chunks) == 1
        assert chunks[0]['content'] == 'hello'

    def test_fixed_size_split(self):
        chunks = naive_chunking('abcdefghij', {'chunk_size': 4, 'overlap': 0})
        assert [c['content'] for c in chunks] == ['abcd', 'efgh', 'ij']

    def test_overlap(self):
        chunks = naive_chunking('abcdefghij', {'chunk_size': 4, 'overlap': 2})
        assert chunks[0]['content'] == 'abcd'
        assert chunks[1]['content'] == 'cdef'

    def test_metadata_start_end(self):
        chunks = naive_chunking('abcdefgh', {'chunk_size': 4, 'overlap': 0})
        assert chunks[0]['metadata']['start'] == 0
        assert chunks[0]['metadata']['end'] == 4


class TestGeneralChunking:
    def test_empty_content(self):
        assert general_chunking('', {}) == []

    def test_single_paragraph(self):
        chunks = general_chunking('只有一段内容', {})
        assert len(chunks) == 1

    def test_merges_short_paragraphs(self):
        chunks = general_chunking('短段落\n\n更短', {})
        assert len(chunks) == 1
        assert '短段落' in chunks[0]['content']
        assert '更短' in chunks[0]['content']

    def test_splits_when_cumulative_exceeds_size(self):
        p1 = 'a' * 600
        p2 = 'b' * 600
        chunks = general_chunking(f'{p1}\n\n{p2}', {'chunk_size': 1000, 'overlap': 0})
        assert len(chunks) == 2
        assert chunks[0]['content'] == p1
        assert chunks[1]['content'] == p2

    def test_overlap_keeps_last_paragraph(self):
        p1 = 'a' * 600
        p2 = 'b' * 600
        chunks = general_chunking(f'{p1}\n\n{p2}', {'chunk_size': 1000, 'overlap': 100})
        assert len(chunks) == 2
        assert p1 in chunks[1]['content']
        assert p2 in chunks[1]['content']

    def test_metadata_records_method(self):
        chunks = general_chunking('一段\n\n两段', {})
        assert chunks[0]['metadata']['method'] == 'general'


class TestBookChunking:
    def test_recognizes_chinese_chapters(self):
        content = '第一章 引言\n' + '内容' * 60 + '\n第二章 方法\n' + '细节' * 60
        chunks = book_chunking(content, {})
        titles = {c['metadata'].get('title') for c in chunks}
        assert '第一章 引言' in titles
        assert '第二章 方法' in titles

    def test_falls_back_to_general(self):
        chunks = book_chunking('没有章节的普通文本', {})
        assert all(c['metadata']['method'] == 'general' for c in chunks)


class TestPaperChunking:
    def test_recognizes_sections(self):
        content = '摘要\n' + '本文主要研究了自然语言处理领域的相关问题。' * 4 + '\n结论\n' + '研究表明该方向具有广阔的应用前景。' * 4
        chunks = paper_chunking(content, {})
        sections = {c['metadata'].get('section') for c in chunks}
        assert '摘要' in sections
        assert '结论' in sections

    def test_falls_back_to_general(self):
        chunks = paper_chunking('普通文本没有学术章节', {})
        assert all(c['metadata']['method'] == 'general' for c in chunks)


class TestResumeChunking:
    def test_recognizes_modules(self):
        content = '教育背景\n在某知名大学完成了四年的本科学习\n工作经历\n在某科技公司担任高级工程师五年有余'
        chunks = resume_chunking(content, {})
        sections = {c['metadata'].get('section') for c in chunks}
        assert '教育背景' in sections
        assert '工作经历' in sections

    def test_falls_back_to_general(self):
        chunks = resume_chunking('普通简历内容没有模块关键词', {})
        assert all(c['metadata']['method'] == 'general' for c in chunks)


class TestTableChunking:
    def test_csv_table(self):
        chunks = table_chunking('name,age\nalice,30\nbob,25', {})
        assert chunks[0]['metadata']['format'] == 'csv'
        assert 'name: alice' in chunks[0]['content']

    def test_csv_aligns_mismatched_columns(self):
        chunks = table_chunking('name,age\nalice,30,extra', {})
        assert 'age: 30' in chunks[0]['content']

    def test_markdown_table(self):
        content = '| name | age |\n|---|---|\n| alice | 30 |'
        chunks = table_chunking(content, {})
        assert chunks[0]['metadata']['format'] == 'markdown'
        assert 'name: alice' in chunks[0]['content']

    def test_falls_back_to_general(self):
        chunks = table_chunking('没有表格的普通文本', {})
        assert all(c['metadata']['method'] == 'general' for c in chunks)


class TestQaChunking:
    def test_recognizes_qa_pairs(self):
        content = '问题：什么是RAG？答案：检索增强生成。\n问题：什么是LLM？答案：大语言模型。'
        chunks = qa_chunking(content, {})
        assert len(chunks) == 2
        assert chunks[0]['metadata']['method'] == 'qa'
        assert '检索增强生成' in chunks[0]['content']

    def test_falls_back_to_general(self):
        chunks = qa_chunking('普通文本没有问答', {})
        assert all(c['metadata']['method'] == 'general' for c in chunks)


class TestExtractKeywords:
    def test_empty_text(self):
        assert extract_keywords('') == []

    def test_returns_strings_within_limit(self):
        kws = extract_keywords('机器学习是人工智能的一个重要分支，人工智能正在改变世界。')
        assert isinstance(kws, list)
        assert all(isinstance(k, str) for k in kws)
        assert len(kws) <= 5


class TestGenerateQuestions:
    def test_returns_exact_count(self):
        qs = generate_questions('这是一个测试句子，用来验证问题生成功能是否正常工作。', count=3)
        assert len(qs) == 3

    def test_all_non_empty_strings(self):
        qs = generate_questions('内容', count=2)
        assert all(isinstance(q, str) and q for q in qs)