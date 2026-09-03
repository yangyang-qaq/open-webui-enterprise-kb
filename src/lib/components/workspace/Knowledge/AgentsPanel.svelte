<script lang="ts">
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import AgentWorkflowEditor from '$lib/components/workspace/Knowledge/AgentWorkflowEditor.svelte';
	import AutonomousAgentPanel from '$lib/components/workspace/Knowledge/AutonomousAgentPanel.svelte';

	export let initialMode: 'orchestration' | 'autonomous' = 'orchestration';

	let knowledgeId = '';
	$: if ($page.params.id) knowledgeId = $page.params.id;

	let mode: 'orchestration' | 'autonomous' = initialMode;

	function switchMode(m: 'orchestration' | 'autonomous') {
		mode = m;
	}
</script>

<div class="flex flex-col h-full">
	<div class="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
		<div>
			<h1 class="text-xl font-semibold">多 Agent 协作</h1>
			<p class="text-sm text-gray-500 mt-1">编排式流水线 · 自主式 LangChain Agent</p>
		</div>
		<div class="flex items-center gap-3">
			<div class="inline-flex items-center rounded-lg bg-gray-100 dark:bg-gray-800 p-1">
				<button
					on:click={() => switchMode('orchestration')}
					class="px-3 py-1.5 text-sm rounded-md transition {mode === 'orchestration'
						? 'bg-white dark:bg-gray-900 text-blue-600 dark:text-blue-400 shadow-sm'
						: 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'}"
				>
					编排式
				</button>
				<button
					on:click={() => switchMode('autonomous')}
					class="px-3 py-1.5 text-sm rounded-md transition {mode === 'autonomous'
						? 'bg-white dark:bg-gray-900 text-blue-600 dark:text-blue-400 shadow-sm'
						: 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'}"
				>
					自主式
				</button>
			</div>
			<button
				on:click={() => goto(`/workspace/knowledge/${knowledgeId}`)}
				class="text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
			>
				← 返回
			</button>
		</div>
	</div>

	<div class="flex-1 overflow-hidden">
		{#if mode === 'orchestration'}
			<AgentWorkflowEditor embedded />
		{:else}
			<AutonomousAgentPanel />
		{/if}
	</div>
</div>
