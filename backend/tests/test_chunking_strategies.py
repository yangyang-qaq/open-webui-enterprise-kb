"""Unit tests for chunking_strategies.py (pure functions, no DB dependency)."""

import re

from open_webui.utils.chunking_strategies import (
    CHUNKING_METHODS,
    auto_chunking,
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


class TestAutoRouting:
    """'auto' splits a doc into typed runs (md table / csv / qa / prose) and
    routes each run verbatim to the matching strategy. The verbatim-slice
    invariant means a SINGLE-TYPE document fed to 'auto' chunks exactly like
    that type's own strategy — asserted here as full-list equality."""

    @staticmethod
    def _methods(chunks):
        return [c['metadata']['method'] for c in chunks]

    def test_empty_and_blank_input(self):
        assert auto_chunking('', {}) == []
        assert auto_chunking('\n  \n', {}) == []

    def test_pure_prose_equals_general(self):
        doc = ('第一段讨论分块与检索的关系，补充若干实现细节。'
               '\n\n第二段讨论向量化入库，以及状态与向量库不一致的排查思路。')
        chunks = auto_chunking(doc, {})
        assert chunks == general_chunking(doc, {})
        assert self._methods(chunks) == ['general']

    def test_pure_book_equals_book(self):
        doc = '第一章 引言\n' + ('本书讨论知识库的构建方法与实践。' * 30) + \
            '\n第二章 系统设计\n' + ('我们设计了分块与检索的主线流程。' * 30)
        chunks = auto_chunking(doc, {})
        assert chunks == book_chunking(doc, {})
        assert all(m == 'book' for m in self._methods(chunks))

    def test_pure_paper_equals_paper(self):
        doc = ('摘要\n' + ('本文研究检索增强生成的质量评估。' * 5) +
               '\n方法\n' + ('我们提出基于黄金集的评测框架。' * 5) +
               '\n结论\n' + ('实验表明评测门禁能拦截回归。' * 5))
        chunks = auto_chunking(doc, {})
        assert chunks == paper_chunking(doc, {})
        assert all(m == 'paper' for m in self._methods(chunks))

    def test_pure_resume_equals_resume(self):
        doc = ('教育背景\n在某知名大学完成了四年的本科学习\n'
               '工作经历\n在某科技公司担任高级工程师五年有余')
        chunks = auto_chunking(doc, {})
        assert chunks == resume_chunking(doc, {})
        assert all(m == 'resume' for m in self._methods(chunks))

    def test_pure_markdown_table(self):
        doc = '| name | age |\n|---|---|\n| alice | 30 |\n| bob | 25 |'
        chunks = auto_chunking(doc, {})
        assert chunks == table_chunking(doc, {})
        assert all(c['metadata']['format'] == 'markdown' for c in chunks)
        assert any('name: alice' in c['content'] for c in chunks)

    def test_pure_csv_table(self):
        doc = 'name,age\nalice,30\nbob,25\ncarol,28'
        chunks = auto_chunking(doc, {})
        assert chunks == table_chunking(doc, {})
        assert all(c['metadata']['format'] == 'csv' for c in chunks)
        assert any('name: bob' in c['content'] for c in chunks)

    def test_pure_qa_equals_qa(self):
        doc = ('问题：什么是RAG？答案：检索增强生成。\n'
               '问题：什么是向量？答案：把文本映射成数值向量。')
        chunks = auto_chunking(doc, {})
        assert chunks == qa_chunking(doc, {})
        assert all(m == 'qa' for m in self._methods(chunks))

    def test_mixed_order_and_typed_islands(self):
        doc = ('这是开头的普通说明文字，介绍知识库的用途与范围。\n\n'
               '| 名称 | 数量 |\n|---|---|\n| 苹果 | 3 |\n\n'
               '问题：如何重建索引？\n答案：重跑上传流程。\n\n'
               '结尾还有一段普通文字作为收束。')
        chunks = auto_chunking(doc, {})
        methods = self._methods(chunks)
        assert methods[0] == 'general'
        assert methods[-1] == 'general'
        assert 'table' in methods and 'qa' in methods
        assert methods.index('table') < methods.index('qa')
        assert any('苹果' in c['content'] for c in chunks if c['metadata']['method'] == 'table')
        assert any('重跑上传' in c['content'] for c in chunks if c['metadata']['method'] == 'qa')
        # 全文正文（去掉分隔符后）既不丢字也不重复
        joined = ''.join(c['content'] for c in chunks)
        joined = re.sub(r'\s+', '', joined)
        raw = re.sub(r'\s+', '', doc)
        for token in ('开头', '苹果', '重跑上传', '收束'):
            assert token in joined
        assert raw.count('苹果') == joined.count('苹果')

    def test_resume_genre_with_table_and_qa(self):
        doc = ('教育背景\n2019-2023 就读某大学计算机专业。\n'
               '项目经历\n主导了知识库检索系统的开发。\n\n'
               '| 技能 | 熟练度 |\n|---|---|\n| Python | 高 |\n\n'
               '问题：这个系统用了什么架构？\n答案：前后端分离加 RAG。')
        chunks = auto_chunking(doc, {})
        assert self._methods(chunks) == ['resume', 'resume', 'table', 'qa']
        assert any('教育背景' in c['content'] for c in chunks)
        assert any('Python' in c['content'] for c in chunks)
        assert any('RAG' in c['content'] for c in chunks)

    def test_stray_question_without_answer_stays_prose(self):
        doc = '说明文字开头。\n问题：这条没有答案\n继续讲别的内容。'
        chunks = auto_chunking(doc, {})
        assert all(m == 'general' for m in self._methods(chunks))
        assert '没有答案' in ''.join(c['content'] for c in chunks)

    def test_header_only_md_table_falls_back_to_general(self):
        doc = '| a | b |\n|---|---|'
        chunks = auto_chunking(doc, {})
        assert chunks == general_chunking(doc, {})
        assert any('a' in c['content'] for c in chunks)