<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import type { i18n as i18nType } from 'i18next';
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { toast } from 'svelte-sonner';

	const i18n = getContext<Writable<i18nType>>('i18n');
	import { user } from '$lib/stores';
	import {
		searchKnowledgeFilesById,
		getFileChunks,
		previewKnowledgeChunks,
		mergeKnowledgeChunks,
		splitKnowledgeChunk,
		reindexKnowledgeChunks,
		aiRefineKnowledgeChunks
	} from '$lib/apis/knowledge';

	import Spinner from '$lib/components/common/Spinner.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';

	let knowledgeId = '';
	let loaded = false;
	let loading = false;
	let reindexing = false;
	let aiRefining = false;
	let files: any[] = [];
	let selectedFileId = '';
	let chunkMethod = 'general';
	let chunks: any[] = [];
	let selectedChunks: Set<string> = new Set();
	let showMergeBtn = false;
	let splitChunkId = '';
	let splitAtValue = 200;
	let showSplitModal = false;
	let showChunkDetail = '';
	let chunkDetailContent = '';

	$: if ($page.params.id) { knowledgeId = $page.params.id; }
	$: showMergeBtn = selectedChunks.size >= 2;

	const loadFiles = async () => {
		try {
			const result = await searchKnowledgeFilesById($user?.token ?? '', knowledgeId, null, null, null, null, 1);
			files = result?.items ?? [];
		} catch (e) { console.error(e); }
	};

	const loadChunks = async () => {
		if (!selectedFileId) return;
		loading = true;
		try {
			chunks = await getFileChunks($user?.token ?? '', knowledgeId, selectedFileId);
			if (!chunks || chunks.length === 0) {
				chunks = await previewKnowledgeChunks($user?.token ?? '', knowledgeId, selectedFileId, chunkMethod);
			}
		} catch (e: any) { toast.error(e?.detail ?? 'Failed to load chunks'); }
		finally { loading = false; }
	};

	const handlePreview = async () => {
		if (!selectedFileId) return;
		loading = true;
		try {
			chunks = await previewKnowledgeChunks($user?.token ?? '', knowledgeId, selectedFileId, chunkMethod);
			toast.success(`Generated ${chunks.length} chunks`);
		} catch (e: any) { toast.error(e?.detail ?? 'Failed to preview chunks'); }
		finally { loading = false; }
	};

	const handleMerge = async () => {
		if (selectedChunks.size < 2) return;
		const dbIndices = [...selectedChunks].map(id => chunks.find(c => c.id === id)?.chunk_index).filter(x => x != null).sort((a, b) => a - b);
		for (let i = 1; i < dbIndices.length; i++) { if (dbIndices[i] !== dbIndices[i - 1] + 1) { toast.error('Only adjacent chunks can be merged.'); return; } }
		try {
			await mergeKnowledgeChunks($user?.token ?? '', knowledgeId, selectedFileId, Math.min(...dbIndices), Math.max(...dbIndices));
			selectedChunks.clear();
			await loadChunks();
			toast.success('Chunks merged');
		} catch (e: any) { toast.error(e?.detail ?? 'Merge failed'); }
	};

	const handleSplit = async () => {
		if (!splitChunkId || splitAtValue <= 0) return;
		try {
			await splitKnowledgeChunk($user?.token ?? '', knowledgeId, splitChunkId, splitAtValue);
			showSplitModal = false; splitChunkId = '';
			await loadChunks();
			toast.success('Chunk split');
		} catch (e: any) { toast.error(e?.detail ?? 'Split failed'); }
	};

	const handleReindex = async () => {
		reindexing = true;
		try { const result = await reindexKnowledgeChunks($user?.token ?? '', knowledgeId); toast.success(`Reindexed ${result?.chunks_processed ?? 0} chunks`); }
		catch (e: any) { toast.error(e?.detail ?? 'Reindex failed'); }
		finally { reindexing = false; }
	};

	const handleAIRefine = async () => {
		if (!selectedFileId) return;
		aiRefining = true;
		try {
			const result = await aiRefineKnowledgeChunks($user?.token ?? '', knowledgeId, selectedFileId);
			await loadChunks(); // 刷新 Keywords/Questions 列
			if ((result?.ai_fallback ?? 0) > 0)
				toast.warning(`Refined ${result?.ai_ok ?? 0}/${result?.chunks_processed ?? 0}, ${result.ai_fallback} fell back to heuristic`);
			else
				toast.success(`AI Refine done: ${result?.ai_ok ?? result?.chunks_processed ?? 0} chunks`);
		} catch (e: any) { toast.error(e?.detail ?? 'AI Refine failed'); }
		finally { aiRefining = false; }
	};

	const toggleChunk = (chunkId: string) => {
		if (selectedChunks.has(chunkId)) selectedChunks.delete(chunkId); else selectedChunks.add(chunkId);
		selectedChunks = new Set(selectedChunks);
	};

	const openDetail = (chunkId: string) => {
		const chunk = chunks.find(c => c.id === chunkId);
		if (chunk) { showChunkDetail = chunkId; chunkDetailContent = chunk.content; }
	};

	onMount(() => { loadFiles(); loaded = true; });
</script>

<div class="flex flex-col h-full">
	<div class="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
		<div class="flex items-center gap-4">
			<button on:click={() => goto(`/workspace/knowledge/${knowledgeId}`)} class="text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">← Back to Knowledge Base</button>
			<h1 class="text-xl font-semibold">Chunk Manager</h1>
		</div>
		<div class="flex gap-2">
			<button on:click={handlePreview} disabled={!selectedFileId || loading} class="px-3 py-1.5 text-sm rounded-lg bg-gray-100 hover:bg-gray-200 dark:bg-gray-700 dark:hover:bg-gray-600 disabled:opacity-50">Preview / Refresh</button>
			<button on:click={handleReindex} disabled={reindexing || chunks.length === 0} class="px-3 py-1.5 text-sm rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50">
				{#if reindexing}<Spinner /> Reindexing...{:else}Reindex Vectors{/if}
			</button>
			<button on:click={handleAIRefine} disabled={aiRefining || !selectedFileId || chunks.length === 0} class="px-3 py-1.5 text-sm rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 disabled:opacity-50">
				{#if aiRefining}<Spinner /> Refining...{:else}AI Refine{/if}
			</button>
		</div>
	</div>

	<div class="p-4 border-b border-gray-200 dark:border-gray-700 flex gap-4 items-end">
		<div class="flex-1">
			<label class="block text-sm font-medium mb-2">Select File</label>
			<select bind:value={selectedFileId} on:change={loadChunks} class="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm">
				<option value="">-- Choose a file --</option>
				{#each files as file}
					<option value={file.id}>{file.filename}</option>
				{/each}
			</select>
		</div>
		<div class="w-48">
			<label class="block text-sm font-medium mb-2">Method</label>
			<select bind:value={chunkMethod} class="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm">
				<option value="general">General</option>
				<option value="naive">Fixed-size</option>
				<option value="book">Book</option>
				<option value="paper">Paper</option>
				<option value="resume">Resume</option>
				<option value="table">Table</option>
				<option value="qa">Q&A</option>
				<option value="auto">Auto</option>
			</select>
		</div>
	</div>

	<div class="flex-1 overflow-auto p-4">
		{#if loading}
			<div class="flex items-center justify-center py-20"><Spinner /> <span class="ml-3 text-sm">Loading chunks...</span></div>
		{:else if chunks.length > 0}
			<div class="flex items-center gap-2 mb-3">
				<button on:click={handleMerge} disabled={!showMergeBtn} class="px-3 py-1 text-sm rounded bg-amber-100 hover:bg-amber-200 dark:bg-amber-900 dark:hover:bg-amber-800 disabled:opacity-50">Merge Selected ({selectedChunks.size})</button>
				<span class="text-xs text-gray-400">{chunks.length} chunks total</span>
			</div>
			<div class="rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
				<table class="w-full text-sm">
					<thead class="bg-gray-50 dark:bg-gray-800">
						<tr>
							<th class="w-10 px-2 py-2 text-left">#</th>
							<th class="w-10 px-2 py-2 text-left"><input type="checkbox" disabled /></th>
							<th class="px-3 py-2 text-left">Preview</th>
							<th class="w-24 px-3 py-2 text-right">Tokens</th>
							<th class="px-3 py-2 text-left text-xs">Keywords</th>
							<th class="px-3 py-2 text-left text-xs">Questions</th>
							<th class="w-24 px-3 py-2 text-center">Actions</th>
						</tr>
					</thead>
					<tbody>
						{#each chunks as chunk, idx}
							<tr class="border-t border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50" class:bg-blue-50={selectedChunks.has(chunk.id)}>
								<td class="px-2 py-2 text-gray-400 text-xs">{idx}</td>
								<td class="px-2 py-2"><input type="checkbox" checked={selectedChunks.has(chunk.id)} on:change={() => toggleChunk(chunk.id)} /></td>
								<td class="px-3 py-2 max-w-md">
									<div class="truncate text-gray-700 dark:text-gray-300">{chunk.content.substring(0, 200)}...</div>
									{#if chunk.meta?.section_header || chunk.meta?.page}
										<div class="text-xs text-gray-400 mt-0.5">{#if chunk.meta?.page}Page {chunk.meta.page}{/if}{#if chunk.meta?.section_header} · {chunk.meta.section_header}{/if}</div>
									{/if}
								</td>
								<td class="px-3 py-2 text-right text-gray-400 text-xs">~{chunk.token_count ?? '-'}</td>
								<td class="px-3 py-2 text-xs text-gray-500">
									<div class="flex flex-wrap gap-1">
										{#each (chunk.meta?.keywords || []) as kw}
											<span class="px-1 py-0.5 bg-blue-50 dark:bg-blue-900/30 rounded text-blue-600 dark:text-blue-400 text-xs">{kw}</span>
										{/each}
									</div>
								</td>
								<td class="px-3 py-2 text-xs text-gray-500 max-w-[12rem] truncate">{chunk.meta?.questions?.[0] || '-'}</td>
								<td class="px-3 py-2 text-center">
									<div class="flex items-center justify-center gap-1">
										<button on:click={() => openDetail(chunk.id)} class="text-xs text-blue-600 hover:underline">View</button>
										<button on:click={() => { splitChunkId = chunk.id; splitAtValue = Math.floor((chunk.content?.length ?? 0) / 2); showSplitModal = true; }} class="text-xs text-gray-500 hover:underline">Split</button>
									</div>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else}
			<div class="text-center py-20 text-gray-400">
				{#if selectedFileId}
					<p class="mb-2">No chunks yet.</p>
					<button on:click={handlePreview} class="px-4 py-2 rounded-lg bg-blue-600 text-white text-sm hover:bg-blue-700">Generate Chunk Preview</button>
				{:else}
					Select a file to preview its chunks.
				{/if}
			</div>
		{/if}
	</div>

	{#if showChunkDetail}
		<div class="fixed inset-0 z-50 flex items-center justify-center bg-black/50" on:click={() => showChunkDetail = ''} on:keydown={(e) => { if (e.key === 'Escape') showChunkDetail = ''; }} role="dialog" tabindex="-1">
			<div class="bg-white dark:bg-gray-900 rounded-xl shadow-xl max-w-2xl w-full mx-4 max-h-[80vh] flex flex-col" on:click|stopPropagation>
				<div class="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
					<h3 class="font-semibold">Chunk Detail</h3>
					<button on:click={() => showChunkDetail = ''} class="text-gray-400 hover:text-gray-600">✕</button>
				</div>
				<div class="p-4 overflow-auto flex-1">
					{#if chunks.find(c => c.id === showChunkDetail)?.meta?.keywords?.length}
						<div class="mb-3">
							<div class="text-xs font-medium text-gray-500 mb-1">Keywords</div>
							<div class="flex flex-wrap gap-1">
								{#each chunks.find(c => c.id === showChunkDetail)?.meta?.keywords || [] as kw}
									<span class="px-2 py-1 bg-blue-100 dark:bg-blue-900/40 rounded text-blue-700 dark:text-blue-300 text-sm font-medium">{kw}</span>
								{/each}
							</div>
						</div>
					{/if}
					{#if chunks.find(c => c.id === showChunkDetail)?.meta?.questions?.length}
						<div class="mb-3 p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg">
							<div class="text-xs font-medium text-amber-700 dark:text-amber-300 mb-1">Sample Questions</div>
							{#each chunks.find(c => c.id === showChunkDetail)?.meta?.questions || [] as q}
								<div class="text-sm text-amber-800 dark:text-amber-200 mb-1">• {q}</div>
							{/each}
						</div>
					{/if}
					<pre class="whitespace-pre-wrap text-sm text-gray-700 dark:text-gray-300 font-mono">{chunkDetailContent}</pre>
				</div>
			</div>
		</div>
	{/if}

	{#if showSplitModal}
		<div class="fixed inset-0 z-50 flex items-center justify-center bg-black/50" on:click={() => showSplitModal = false} on:keydown={(e) => { if (e.key === 'Escape') showSplitModal = false; }} role="dialog" tabindex="-1">
			<div class="bg-white dark:bg-gray-900 rounded-xl shadow-xl max-w-md w-full mx-4" on:click|stopPropagation>
				<div class="p-4 border-b border-gray-200 dark:border-gray-700"><h3 class="font-semibold">Split Chunk</h3></div>
				<div class="p-4 space-y-3">
					<div>
						<label class="block text-sm mb-1">Split at character offset</label>
						<input type="number" bind:value={splitAtValue} min="1" class="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm" />
						<p class="text-xs text-gray-400 mt-1">The chunk will be split into two at this character position.</p>
					</div>
				</div>
				<div class="flex justify-end gap-2 p-4 border-t border-gray-200 dark:border-gray-700">
					<button on:click={() => showSplitModal = false} class="px-4 py-1.5 text-sm rounded-lg bg-gray-100 hover:bg-gray-200 dark:bg-gray-700">Cancel</button>
					<button on:click={handleSplit} class="px-4 py-1.5 text-sm rounded-lg bg-blue-600 text-white hover:bg-blue-700">Split</button>
				</div>
			</div>
		</div>
	{/if}
</div>
