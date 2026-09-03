<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { toast } from 'svelte-sonner';
	import { user } from '$lib/stores';
	import { WEBUI_API_BASE_URL } from '$lib/constants';
	import Spinner from '$lib/components/common/Spinner.svelte';

	// Types
	interface AgentRole { id: string; name: string; icon: string; default_prompt: string; }
	interface WorkflowStep { order_index: number; agent_role: string; knowledge_id?: string; prompt_template?: string; input_var?: string; output_var?: string; }
	interface Workflow { id: string; name: string; description?: string; steps: WorkflowStep[]; created_at: number; }

	let knowledgeId = '';
	$: if ($page.params.id) knowledgeId = $page.params.id;

	let roles: AgentRole[] = [];
	let workflows: Workflow[] = [];
	let loaded = false;
	let executing = false;
	let executionLog: any[] = [];
	let execRunId = '';
	let execWfName = '';
	let execQuery = '';
	let showDownload = false;

	// Create/edit state
	let showEditor = false;
	let wfName = '';
	let wfDesc = '';
	let wfSteps: WorkflowStep[] = [];

	function addStep() {
		wfSteps = [...wfSteps, { order_index: wfSteps.length, agent_role: 'retriever', knowledge_id: '', prompt_template: '', input_var: '', output_var: '' }];
	}
	function removeStep(idx: number) { wfSteps = wfSteps.filter((_, i) => i !== idx).map((s, i) => ({ ...s, order_index: i })); }

	// Load
	async function loadWorkflows() {
		try {
			const res = await fetch(`${WEBUI_API_BASE_URL}/knowledge/_workflows`, { headers: { authorization: `Bearer ${$user?.token}` } });
			if (res.ok) workflows = await res.json();
		} catch (e) { console.error(e); }
	}
	async function loadRoles() {
		try {
			const res = await fetch(`${WEBUI_API_BASE_URL}/knowledge/_workflows/roles`, { headers: { authorization: `Bearer ${$user?.token}` } });
			if (res.ok) roles = await res.json();
		} catch (e) { console.error(e); }
	}

	// Create
	async function createWorkflow() {
		try {
			const body = { name: wfName, description: wfDesc, steps: wfSteps.map(s => ({ ...s, knowledge_id: s.knowledge_id || null })) };
			const res = await fetch(`${WEBUI_API_BASE_URL}/knowledge/_workflows`, {
				method: 'POST', headers: { 'Content-Type': 'application/json', authorization: `Bearer ${$user?.token}` }, body: JSON.stringify(body)
			});
			if (!res.ok) throw await res.json();
			showEditor = false; wfName = ''; wfDesc = ''; wfSteps = [];
			toast.success('工作流已创建');
			await loadWorkflows();
		} catch (e: any) { toast.error(e?.detail ?? '创建失败'); }
	}

	// Delete
	async function deleteWorkflow(id: string) {
		if (!confirm('删除此工作流？')) return;
		await fetch(`${WEBUI_API_BASE_URL}/knowledge/_workflows/${id}`, { method: 'DELETE', headers: { authorization: `Bearer ${$user?.token}` } });
		toast.success('已删除');
		await loadWorkflows();
	}

	// Execute
	async function executeWorkflow(wfId: string, wfName: string) {
		executing = true; executionLog = []; execRunId = ''; execWfName = wfName; showDownload = false;
		const query = prompt('输入查询问题：', '什么是微服务架构？');
		if (!query) { executing = false; return; }
		execQuery = query;
		try {
			const res = await fetch(`${WEBUI_API_BASE_URL}/knowledge/_workflows/exec`, {
				method: 'POST', headers: { 'Content-Type': 'application/json', authorization: `Bearer ${$user?.token}` },
				body: JSON.stringify({ query, workflow_id: wfId })
			});
			if (!res.ok) throw await res.json();
			const reader = res.body?.getReader();
			const decoder = new TextDecoder();
			if (reader) {
				while (true) {
					const { done, value } = await reader.read();
					if (done) break;
					for (const line of decoder.decode(value).split('\n')) {
						if (line.startsWith('data: ')) {
							try {
								const ev = JSON.parse(line.slice(6));
								if (ev.status === 'done' && ev.results) {
									execRunId = ev.run_id || '';
									executionLog = ev.results.map((r: any) => ({ role: r.role, status: r.status, output: r.output }));
									showDownload = true;
								} else if (ev.step !== undefined) {
									executionLog = [...executionLog, ev];
								}
							} catch (e) {}
						}
					}
				}
			}
			toast.success('执行完成');
		} catch (e: any) { toast.error(e?.detail ?? '执行失败'); }
		executing = false;
	}

	// Download Word document
	async function downloadDocx() {
		try {
			const body: any = { run_id: execRunId, wf_name: execWfName, query: execQuery, steps: executionLog };
			const res = await fetch(`${WEBUI_API_BASE_URL}/knowledge/_workflows/exec/download`, {
				method: 'POST', headers: { 'Content-Type': 'application/json', authorization: `Bearer ${$user?.token}` },
				body: JSON.stringify(body)
			});
			if (!res.ok) throw await res.json();
			const blob = await res.blob();
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = `${execWfName}_report.docx`;
			a.click();
			URL.revokeObjectURL(url);
			toast.success('Word 文档下载完成');
		} catch (e: any) { toast.error(e?.detail ?? '下载失败'); }
	}

	onMount(() => { loadRoles(); loadWorkflows(); loaded = true; });
</script>

<div class="flex flex-col h-full">
	<div class="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
		<div>
			<h1 class="text-xl font-semibold">Agent 工作流</h1>
			<p class="text-sm text-gray-500 mt-1">多 Agent 协作编排</p>
		</div>
		<div class="flex gap-2">
			<button on:click={() => { showEditor = true; wfName = ''; wfDesc = ''; wfSteps = [{ order_index: 0, agent_role: 'retriever', knowledge_id: knowledgeId, prompt_template: '', input_var: '', output_var: 'search_results' }]; }}
				class="px-3 py-1.5 text-sm rounded-lg bg-blue-600 text-white hover:bg-blue-700">+ 创建工作流</button>
			<button on:click={() => goto(`/workspace/knowledge/${knowledgeId}`)} class="text-sm text-gray-500 hover:text-gray-700">← 返回</button>
		</div>
	</div>

	<div class="flex-1 overflow-auto p-4">
		<!-- Workflow List -->
		{#if workflows.length === 0 && loaded}
			<div class="text-center py-20 text-gray-400">
				<p class="mb-2">暂无工作流。</p>
				<p class="text-xs">创建 Agent 工作流来编排多步检索与分析。</p>
			</div>
		{:else}
			<div class="space-y-3">
				{#each workflows as wf (wf.id)}
					<div class="rounded-lg border border-gray-200 dark:border-gray-700 p-4">
						<div class="flex items-start justify-between">
							<div>
								<div class="font-medium">{wf.name}</div>
								{#if wf.description}<div class="text-xs text-gray-500 mt-1">{wf.description}</div>{/if}
								<div class="flex items-center gap-1 mt-2">
									{#each wf.steps as step}
										<span class="text-xs px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700">
											{roles.find(r => r.id === step.agent_role)?.icon || '🤖'}
											{roles.find(r => r.id === step.agent_role)?.name || step.agent_role}
										</span>
										{#if step.order_index < wf.steps.length - 1}<span class="text-gray-300">→</span>{/if}
									{/each}
								</div>
							</div>
							<div class="flex gap-1">
								<button on:click={() => executeWorkflow(wf.id, wf.name)} disabled={executing}
									class="px-2 py-1 text-xs rounded bg-green-100 hover:bg-green-200 dark:bg-green-900 dark:hover:bg-green-800 text-green-700">▶ 执行</button>
								<button on:click={() => deleteWorkflow(wf.id)}
									class="px-2 py-1 text-xs rounded bg-red-50 hover:bg-red-100 dark:bg-red-900/30 text-red-500">删除</button>
							</div>
						</div>
					</div>
				{/each}
			</div>
		{/if}

		<!-- Execution Log -->
		{#if executionLog.length > 0}
			<div class="mt-4 p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
				<div class="flex items-center justify-between mb-2">
					<h3 class="text-sm font-medium">执行日志</h3>
					{#if showDownload}
						<button on:click={downloadDocx}
							class="px-3 py-1 text-xs rounded bg-blue-600 text-white hover:bg-blue-700 flex items-center gap-1">
							📥 下载 Word 文档
						</button>
					{/if}
				</div>
				<div class="space-y-2">
					{#each executionLog as log}
						<div class="text-xs p-2 rounded bg-white dark:bg-gray-800">
							<span class="font-medium">{log.role || log.status}</span>
							{#if log.output}<div class="text-gray-500 mt-1 whitespace-pre-wrap">{log.output}</div>{/if}
						</div>
					{/each}
				</div>
			</div>
		{/if}
	</div>

	<!-- Create/Edit Modal -->
	{#if showEditor}
		<div class="fixed inset-0 z-50 flex items-start justify-center bg-black/50 overflow-auto py-8" on:click={() => showEditor = false} role="dialog">
			<div class="bg-white dark:bg-gray-900 rounded-xl shadow-xl max-w-2xl w-full mx-4" on:click|stopPropagation>
				<div class="p-4 border-b"><h3 class="font-semibold">创建工作流</h3></div>
				<div class="p-4 space-y-3">
					<input type="text" bind:value={wfName} placeholder="工作流名称" class="w-full rounded-lg border px-3 py-2 text-sm dark:bg-gray-800 dark:border-gray-600" />
					<textarea bind:value={wfDesc} rows="2" placeholder="描述（可选）" class="w-full rounded-lg border px-3 py-2 text-sm dark:bg-gray-800 dark:border-gray-600"></textarea>

					<div class="space-y-2">
						<div class="flex items-center justify-between">
							<span class="text-sm font-medium">步骤</span>
							<button on:click={addStep} class="text-xs px-2 py-1 rounded bg-blue-100 text-blue-600 hover:bg-blue-200">+ 添加步骤</button>
						</div>
						{#each wfSteps as step, i}
							<div class="flex items-center gap-2 p-2 rounded bg-gray-50 dark:bg-gray-800">
								<span class="text-xs text-gray-400 w-6">{i + 1}</span>
								<select bind:value={step.agent_role} class="flex-1 rounded border px-2 py-1 text-xs dark:bg-gray-700 dark:border-gray-600">
									{#each roles as r}
										<option value={r.id}>{r.icon} {r.name}</option>
									{/each}
								</select>
								<input type="text" bind:value={step.output_var} placeholder="输出变量" class="w-24 rounded border px-2 py-1 text-xs dark:bg-gray-700 dark:border-gray-600" />
								<button on:click={() => removeStep(i)} class="text-xs text-red-400 hover:text-red-600">✕</button>
							</div>
						{/each}
					</div>
				</div>
				<div class="flex justify-end gap-2 p-4 border-t">
					<button on:click={() => showEditor = false} class="px-4 py-1.5 text-sm rounded-lg bg-gray-100 hover:bg-gray-200 dark:bg-gray-700">取消</button>
					<button on:click={createWorkflow} disabled={!wfName} class="px-4 py-1.5 text-sm rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50">创建</button>
				</div>
			</div>
		</div>
	{/if}
</div>
