<script lang="ts">
	import { onMount } from 'svelte';
	import StatTile from '$lib/components/StatTile.svelte';
	import TrendChart from '$lib/components/TrendChart.svelte';
	import {
		getStatus,
		getLatency,
		getBenchmarkHistory,
		runDuplexBench,
		type StatusResponse,
		type LatencyRow,
		type BenchmarkEntry
	} from '$lib/api';

	let status = $state<StatusResponse | null>(null);
	let latencyRows = $state<LatencyRow[]>([]);
	let history = $state<BenchmarkEntry[]>([]);
	let error = $state('');
	let runningDuplex = $state(false);

	async function loadAll() {
		error = '';
		try {
			const [statusRes, latencyRes, historyRes] = await Promise.all([
				getStatus(),
				getLatency(),
				getBenchmarkHistory()
			]);
			status = statusRes;
			latencyRows = latencyRes.rows;
			history = historyRes.entries;
		} catch (e) {
			error = `Không gọi được gateway (${(e as Error).message}). Kiểm tra Gateway đang chạy ` +
				`(python -m gateway.main) và cấu hình "Kết nối gateway" ở trên.`;
		}
	}

	async function handleRunDuplex() {
		runningDuplex = true;
		try {
			await runDuplexBench();
			await loadAll();
		} catch (e) {
			error = (e as Error).message;
		} finally {
			runningDuplex = false;
		}
	}

	onMount(loadAll);

	let duplexPoints = $derived(
		history
			.filter((e) => e.duplex_bench)
			.map((e, i) => ({ label: `#${i + 1}`, value: e.duplex_bench!.accuracy * 100 }))
	);

	let werPoints = $derived(
		history
			.filter((e) => e.asr_wer)
			.map((e, i) => ({ label: `#${i + 1}`, value: e.asr_wer!.overall_wer * 100 }))
	);

	let ttfaPoints = $derived(
		history
			.filter((e) => e.latency_vs_target?.measured_p50_ms != null)
			.map((e, i) => ({ label: `#${i + 1}`, value: e.latency_vs_target!.measured_p50_ms! }))
	);
	let ttfaTargetBand = $derived.by((): [number, number] | undefined => {
		const withTarget = history.find((e) => e.latency_vs_target?.target_range_ms);
		return withTarget?.latency_vs_target?.target_range_ms;
	});
</script>

<h1>Dashboard</h1>

{#if error}
	<div class="card error">{error}</div>
{/if}

<section class="stat-grid card">
	<StatTile label="Phiên đang mở" value={status ? String(status.active_sessions) : '—'} />
	<StatTile
		label="Uptime gateway"
		value={status ? `${Math.round(status.uptime_s / 60)} phút` : '—'}
	/>
	<StatTile
		label="Số lần benchmark đã chạy"
		value={String(history.length)}
		sub="python -m eval.run_benchmarks"
	/>
</section>

<section class="grid">
	<TrendChart
		title="Vietnamese Turn-Taking Bench — accuracy (%)"
		points={duplexPoints}
		yFormat={(v) => `${v.toFixed(1)}%`}
	/>
	<TrendChart
		title="ASR WER tổng (%) — chỉ có khi chạy --with-asr"
		points={werPoints}
		yFormat={(v) => `${v.toFixed(1)}%`}
	/>
	<TrendChart
		title="TTFA p50 (ms) so với mục tiêu roadmap"
		points={ttfaPoints}
		targetBand={ttfaTargetBand}
		targetLabel="Mục tiêu 600–950ms"
		yFormat={(v) => `${v.toFixed(0)}ms`}
	/>
</section>

<section class="card">
	<div class="section-head">
		<h3>Latency theo backend (từ hội thoại thật)</h3>
		<button onclick={handleRunDuplex} disabled={runningDuplex}>
			{runningDuplex ? 'Đang chạy...' : 'Chạy Turn-Taking Bench ngay'}
		</button>
	</div>
	{#if latencyRows.length === 0}
		<p class="empty">Chưa có log — cần một phiên hội thoại voice thật qua gateway (bot.py ghi log này).</p>
	{:else}
		<table>
			<thead>
				<tr>
					<th>Metric</th><th>STT</th><th>LLM</th><th>TTS</th><th>n</th><th>p50</th><th>p95</th>
				</tr>
			</thead>
			<tbody>
				{#each latencyRows as row (row.metric + row.stt_backend + row.llm_backend + row.tts_backend)}
					<tr>
						<td>{row.metric}</td>
						<td>{row.stt_backend}</td>
						<td>{row.llm_backend}</td>
						<td>{row.tts_backend}</td>
						<td>{row.n}</td>
						<td>{row.p50_ms.toFixed(0)}ms</td>
						<td>{row.p95_ms.toFixed(0)}ms</td>
					</tr>
				{/each}
			</tbody>
		</table>
	{/if}
</section>

<style>
	h1 {
		font-size: 1.3rem;
		margin-bottom: 1rem;
	}
	.error {
		color: var(--status-critical);
		margin-bottom: 1rem;
		font-size: 0.9rem;
	}
	.stat-grid {
		display: flex;
		gap: 2rem;
		margin-bottom: 1.5rem;
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
		gap: 1rem;
		margin-bottom: 1.5rem;
	}
	.section-head {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.75rem;
	}
	.section-head h3 {
		margin: 0;
		font-size: 0.95rem;
	}
	.section-head button {
		background: var(--series-1);
		color: white;
		border: none;
		border-radius: 6px;
		padding: 0.4rem 0.8rem;
		font-size: 0.8rem;
		cursor: pointer;
	}
	.section-head button:disabled {
		opacity: 0.6;
		cursor: default;
	}
	.empty {
		color: var(--text-muted);
		font-size: 0.85rem;
	}
</style>
