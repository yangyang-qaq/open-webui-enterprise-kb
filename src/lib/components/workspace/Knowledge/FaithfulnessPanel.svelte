<script lang="ts">
	import { getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { toast } from 'svelte-sonner';

	const i18n = getContext<Writable<any>>('i18n');
	import { user } from '$lib/stores';
	import { WEBUI_API_BASE_URL } from '$lib/constants';

	import Spinner from '$lib/components/common/Spinner.svelte';

	// ── Types ──
	type Verdict = 'yes' | 'no' | 'insufficient';

	interface Claim {
		text: string;
		supported: Verdict;
		evidence: string;
	}

	interface ResultItem {
		query: string;
		answer: string;
		faithfulness: number;
		supported: number;
		total: number;
		refusal: boolean;
		claims: Claim[];
		context_chunks: any[];
	}

	// ── State ──
	let knowledgeId = '';
	$: if ($page.params.id) knowledgeId = $page.params.id;

	let query = '';
	let k = 10;
	let running = false;

	let results: ResultItem[] = [];
	let aggregate: number | null = null;
	let threshold = 0.8;
	let model = '';

	// ── Run faithfulness evaluation ──
	async function runEval() {
		const questions = query
			.split('\n')
			.map((s) => s.trim())
			.filter(Boolean);
		if (questions.length === 0) return;

		running = true;
		results = [];
		aggregate = null;

		try {
			const res = await fetch(`${WEBUI_API_BASE_URL}/knowledge/${knowledgeId}/evaluate/faithfulness`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					authorization: `Bearer ${$user?.token}`
				},
				body: JSON.stringify({ queries: questions, k })
			});
			if (!res.ok) throw await res.json();
			const data = await res.json();
			results = data.results ?? [];
			aggregate = data.aggregate_faithfulness ?? null;
			threshold = data.threshold ?? 0.8;
			model = data.model ?? '';
		} catch (e: any) {
			toast.error(e?.detail ?? 'Faithfulness evaluation failed');
		} finally {
			running = false;
		}
	}

	// ── Score color ──
	function scoreColor(val: number): string {
		if (val > 0.8) return 'text-green-600';
		if (val > 0.5) return 'text-amber-600';
		return 'text-red-500';
	}

	function verdictBadge(supported: Verdict): { label: string; cls: string } {
		if (supported === 'yes')
			return { label: '✓ 支持', cls: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300' };
		if (supported === 'no')
			return { label: '✗ 矛盾', cls: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300' };
		return { label: '? 不足', cls: 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300' };
	}
</script>

<div class="flex flex-col h-full">
	<!-- Header -->
	<div class="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
		<div>
			<h1 class="text-xl font-semibold">生成质量评测 · Faithfulness</h1>
			<p class="text-sm text-gray-500 mt-1">
				检索 → 生成 → LLM-as-judge 忠实度判定，逐条拆解主张判断是否被检索片段支持
			</p>
		</div>
		<button
			on:click={() => goto(`/workspace/knowledge/${knowledgeId}`)}
			class="text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
		>
			← Back to Files
		</button>
	</div>

	<!-- Query Input -->
	<div class="p-4 border-b border-gray-100 dark:border-gray-800 bg-gray-50 dark:bg-gray-900">
		<textarea
			bind:value={query}
			rows="2"
			placeholder="输入测试问题，一行一个（支持多条批量评测）..."
			class="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm mb-2 resize-none"
		></textarea>
		<div class="flex items-center justify-end gap-2">
			<select
				bind:value={k}
				class="w-20 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-2 text-sm"
			>
				<option value="5">K=5</option>
				<option value="10">K=10</option>
				<option value="15">K=15</option>
				<option value="20">K=20</option>
			</select>
			<button
				on:click={runEval}
				disabled={running || !query.trim()}
				class="px-4 py-2 text-sm rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 transition"
			>
				{#if running}<Spinner />{:else}评测{/if}
			</button>
		</div>
	</div>

	<div class="flex-1 overflow-auto">
		{#if running}
			<div class="flex items-center justify-center py-20">
				<Spinner /><span class="ml-3 text-sm text-gray-400">检索 → 生成 → 判分中...</span>
			</div>
		{:else if results.length > 0}
			<div class="p-4 space-y-4">
				<!-- Aggregate Panel -->
				<div class="rounded-lg border border-gray-200 dark:border-gray-700 p-4 bg-gray-50 dark:bg-gray-900">
					<div class="flex items-center justify-between">
						<div>
							<div class="text-xs text-gray-500 uppercase tracking-wide">聚合 Faithfulness</div>
							<div class="text-3xl font-bold {scoreColor(aggregate ?? 0)}">
								{aggregate?.toFixed(4) ?? 'N/A'}
							</div>
						</div>
						<div class="text-right text-xs text-gray-500 space-y-1">
							<div>题目数: {results.length}</div>
							<div>模型: {model || 'deepseek-chat'}</div>
							<div>
								门槛: {threshold}{' '}
								{#if aggregate !== null}
									{#if aggregate >= threshold}
										<span class="text-green-600 font-medium">PASS</span>
									{:else}
										<span class="text-red-500 font-medium">FAIL</span>
									{/if}
								{/if}
							</div>
						</div>
					</div>
				</div>

				<!-- Per-question results -->
				{#each results as result, i}
					<div class="rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
						<!-- Question header -->
						<div class="flex items-start justify-between gap-3 p-3 bg-gray-50 dark:bg-gray-900 border-b border-gray-100 dark:border-gray-800">
							<div class="flex-1 min-w-0">
								<div class="text-xs text-gray-400 mb-1">#{i + 1}</div>
								<div class="text-sm font-medium text-gray-800 dark:text-gray-200">{result.query}</div>
							</div>
							<div class="shrink-0 text-right">
								<div class="text-xl font-bold {scoreColor(result.faithfulness)}">{result.faithfulness?.toFixed(4)}</div>
								<div class="text-xs text-gray-400">
									{#if result.refusal}
										refusal
									{:else}
										{result.supported}/{result.total} claims
									{/if}
								</div>
							</div>
						</div>

						<div class="p-3 space-y-3">
							<!-- Generated answer -->
							<div>
								<div class="text-xs font-medium text-gray-500 mb-1">生成答案</div>
								<div class="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap leading-relaxed">
									{result.answer || '(空)'}
								</div>
							</div>

							<!-- Claims breakdown -->
							{#if result.claims.length > 0}
								<div>
									<div class="text-xs font-medium text-gray-500 mb-1">主张拆解（LLM-as-judge）</div>
									<ul class="space-y-2">
										{#each result.claims as claim}
											<li class="rounded-md border border-gray-100 dark:border-gray-800 p-2">
												<div class="flex items-start gap-2">
													<span class="shrink-0 text-xs font-medium px-2 py-0.5 rounded {verdictBadge(claim.supported).cls}">
														{verdictBadge(claim.supported).label}
													</span>
													<span class="text-sm text-gray-700 dark:text-gray-300">{claim.text}</span>
												</div>
												{#if claim.evidence}
													<div class="mt-1 ml-2 text-xs text-gray-400 italic">证据: {claim.evidence}</div>
												{/if}
											</li>
										{/each}
									</ul>
								</div>
							{/if}

							<!-- Retrieved chunks -->
							{#if result.context_chunks.length > 0}
								<details class="text-sm">
									<summary class="cursor-pointer text-xs font-medium text-gray-500 hover:text-gray-700">
										检索片段 ({result.context_chunks.length})
									</summary>
									<ul class="mt-2 space-y-1">
										{#each result.context_chunks as chunk}
											<li class="text-xs text-gray-500 dark:text-gray-400 border-l-2 border-gray-200 dark:border-gray-700 pl-2">
												<span class="font-mono text-gray-400">#{chunk.rank}</span>
												{#if chunk.source}
													<span class="text-gray-400"> · {chunk.source}</span>
												{/if}
												<span class="ml-1 line-clamp-2">{chunk.text}</span>
											</li>
										{/each}
									</ul>
								</details>
							{/if}
						</div>
					</div>
				{/each}
			</div>
		{:else}
			<div class="text-center py-20 text-gray-400">
				<p class="mb-2">输入问题评测生成质量。</p>
				<p class="text-xs">系统将真实检索 → 生成答案 → 用 LLM 逐条判断答案主张是否被检索片段支持（忠实度）。</p>
			</div>
		{/if}
	</div>
</div>
