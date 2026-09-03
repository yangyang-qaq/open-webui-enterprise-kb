<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import { toast } from 'svelte-sonner';
	import { user } from '$lib/stores';
	import { WEBUI_API_BASE_URL } from '$lib/constants';
	import Spinner from '$lib/components/common/Spinner.svelte';

	interface AgentRole { id: string; name: string; icon: string; default_prompt: string; }
	interface Round { index: number; tool: string; tool_name: string; icon: string; args: string; obs: string; }

	let knowledgeId = '';
	$: if ($page.params.id) knowledgeId = $page.params.id;

	let roles: AgentRole[] = [];
	let query = '';
	let running = false;
	let rounds: Round[] = [];
	let answer = '';
	let error = '';

	function roleInfo(id: string): AgentRole | undefined {
		return roles.find((r) => r.id === id);
	}

	async function loadRoles() {
		try {
			const res = await fetch(`${WEBUI_API_BASE_URL}/knowledge/_workflows/roles`, {
				headers: { authorization: `Bearer ${$user?.token}` }
			});
			if (res.ok) roles = await res.json();
		} catch (e) {
			// ignore
		}
	}

	function handleEvent(ev: any) {
		switch (ev.type) {
			case 'round': {
				const info = roleInfo(ev.tool);
				rounds = [
					...rounds,
					{
						index: ev.index,
						tool: ev.tool,
						tool_name: info?.name || ev.tool,
						icon: info?.icon || '🤖',
						args: ev.args ?? '',
						obs: ''
					}
				];
				break;
			}
			case 'observation':
				rounds = rounds.map((r) => (r.index === ev.index ? { ...r, obs: ev.preview ?? '' } : r));
				break;
			case 'answer':
				answer = ev.content ?? '';
				break;
			case 'error':
				error = ev.message ?? '推理出错';
				break;
			case 'done':
				if (ev.status === 'error' && !error) error = '推理被中止';
				break;
		}
	}

	async function runAutonomous() {
		if (!query.trim() || running) return;
		running = true;
		rounds = [];
		answer = '';
		error = '';

		let failed = '';
		try {
			const res = await fetch(`${WEBUI_API_BASE_URL}/knowledge/_agents/autonomous/exec`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					authorization: `Bearer ${$user?.token}`
				},
				body: JSON.stringify({ query: query.trim(), knowledge_id: knowledgeId, max_steps: 8 })
			});
			if (!res.ok) {
				const j = await res.json().catch(() => ({}));
				throw new Error(j.detail || `HTTP ${res.status}`);
			}
			const reader = res.body?.getReader();
			const decoder = new TextDecoder();
			if (!reader) throw new Error('无法读取响应流');
			while (true) {
				const { done, value } = await reader.read();
				if (done) break;
				for (const line of decoder.decode(value).split('\n')) {
					if (!line.startsWith('data: ')) continue;
					let ev: any;
					try {
						ev = JSON.parse(line.slice(6));
					} catch (e) {
						continue;
					}
					handleEvent(ev);
				}
			}
			if (error) failed = error;
		} catch (e: any) {
			error = e?.message || String(e);
			failed = error;
		} finally {
			running = false;
		}

		if (failed) toast.error('推理中止');
		else toast.success('推理完成');
	}

	onMount(() => loadRoles());
</script>

<div class="flex flex-col h-full">
	<!-- 控制区 -->
	<div class="p-4 border-b border-gray-200 dark:border-gray-700 space-y-3">
		<div>
			<label class="text-sm font-medium text-gray-700 dark:text-gray-300">你的问题</label>
			<textarea
				bind:value={query}
				rows="2"
				placeholder="例如：这个知识库主要讲了什么？Agent 会自主决定检索几轮、是否需要分析/汇报/校验/翻译。"
				class="w-full mt-1 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm"
			></textarea>
		</div>
		<div class="flex items-center justify-between gap-3 flex-wrap">
			<div class="flex flex-wrap gap-1">
				{#each roles as r}
					<span class="text-xs px-2 py-0.5 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300">
						{r.icon} {r.name}
					</span>
				{/each}
			</div>
			<button
				on:click={runAutonomous}
				disabled={running || !query.trim()}
				class="px-3 py-1.5 text-sm rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 flex items-center gap-1"
			>
				{#if running}<Spinner />{:else}▶{/if} 开始自主推理
			</button>
		</div>
	</div>

	<!-- 轨迹区 -->
	<div class="flex-1 overflow-auto p-4 space-y-3">
		{#if running && rounds.length === 0 && !error}
			<div class="flex items-center gap-2 text-sm text-gray-500 py-4">
				<Spinner /><span>Agent 正在思考第一步…</span>
			</div>
		{/if}

		{#each rounds as round (round.index)}
			<div class="rounded-lg border border-gray-200 dark:border-gray-700 p-3">
				<div class="flex items-center gap-2">
					<span class="text-xs font-mono px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700 text-gray-500">#{round.index}</span>
					<span class="text-sm font-medium">{round.icon} 调用 {round.tool_name}</span>
				</div>
				{#if round.args}
					<div class="text-xs text-gray-500 mt-1 whitespace-pre-wrap">
						<span class="text-gray-400">参数：</span>{round.args}
					</div>
				{/if}
				{#if round.obs}
					<div class="text-xs text-gray-600 dark:text-gray-300 mt-1 whitespace-pre-wrap">
						<span class="text-gray-400">→ 返回：</span>{round.obs}
					</div>
				{/if}
			</div>
		{/each}

		{#if error}
			<div class="rounded-lg border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-900/20 p-3 text-sm text-red-600 dark:text-red-400 whitespace-pre-wrap">
				⚠ {error}
			</div>
		{/if}

		{#if answer}
			<div class="rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 p-4">
				<h3 class="text-sm font-semibold text-gray-700 dark:text-gray-200 mb-2">最终结论</h3>
				<div class="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">{answer}</div>
			</div>
		{/if}

		{#if !running && rounds.length === 0 && !answer && !error}
			<div class="text-center py-16 text-gray-400">
				<p class="text-sm">输入问题后点击「开始自主推理」。</p>
				<p class="text-xs mt-1">Agent 会自主决定：先检索知识库，需要时再调 分析/汇报/校验/翻译 工具，最后给出结论。</p>
			</div>
		{/if}
	</div>
</div>
