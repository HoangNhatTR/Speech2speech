<script lang="ts">
	// Line chart 1 chuỗi (đủ dùng cho benchmark theo thời gian: accuracy/WER/latency qua
	// các lần chạy) — 1 chuỗi không cần legend (tiêu đề đã gọi tên), nhưng luôn có bảng
	// dữ liệu thay thế (accessibility) và tooltip khi hover (dataviz skill).
	let {
		title,
		points,
		yFormat = (v: number) => v.toFixed(2),
		targetBand,
		targetLabel = 'Mục tiêu',
		height = 160
	}: {
		title: string;
		points: { label: string; value: number }[];
		yFormat?: (v: number) => string;
		targetBand?: [number, number];
		targetLabel?: string;
		height?: number;
	} = $props();

	let hovered = $state<number | null>(null);
	let showTable = $state(false);

	const width = 480;
	const padding = { top: 12, right: 12, bottom: 24, left: 12 };

	let plotWidth = $derived(width - padding.left - padding.right);
	let plotHeight = $derived(height - padding.top - padding.bottom);

	let domain = $derived.by(() => {
		const values = points.map((p) => p.value);
		if (targetBand) values.push(targetBand[0], targetBand[1]);
		const min = values.length ? Math.min(...values) : 0;
		const max = values.length ? Math.max(...values) : 1;
		const span = max - min || 1;
		// Đệm 10% mỗi bên để marker/line không dính mép.
		return { min: min - span * 0.1, max: max + span * 0.1 };
	});

	function yToPx(v: number): number {
		const { min, max } = domain;
		const ratio = (v - min) / (max - min || 1);
		return padding.top + plotHeight * (1 - ratio);
	}

	function xToPx(i: number): number {
		if (points.length <= 1) return padding.left + plotWidth / 2;
		return padding.left + (plotWidth * i) / (points.length - 1);
	}

	let linePath = $derived(
		points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${xToPx(i)} ${yToPx(p.value)}`).join(' ')
	);
</script>

<div class="chart-root card">
	<div class="chart-head">
		<h3>{title}</h3>
		<button class="link-btn" onclick={() => (showTable = !showTable)}>
			{showTable ? 'Xem biểu đồ' : 'Xem dạng bảng'}
		</button>
	</div>

	{#if points.length === 0}
		<p class="empty">Chưa có dữ liệu — chạy `python -m eval.run_benchmarks` để có điểm đầu tiên.</p>
	{:else if showTable}
		<table>
			<thead>
				<tr><th>Lần chạy</th><th>Giá trị</th></tr>
			</thead>
			<tbody>
				{#each points as p (p.label)}
					<tr><td>{p.label}</td><td>{yFormat(p.value)}</td></tr>
				{/each}
			</tbody>
		</table>
	{:else}
		<svg viewBox="0 0 {width} {height}" role="img" aria-label={title}>
			{#if targetBand}
				<rect
					x={padding.left}
					y={yToPx(targetBand[1])}
					width={plotWidth}
					height={yToPx(targetBand[0]) - yToPx(targetBand[1])}
					fill="var(--gridline)"
					opacity="0.6"
				/>
				<text x={padding.left + 4} y={yToPx(targetBand[1]) + 12} class="band-label"
					>{targetLabel}</text
				>
			{/if}

			<line
				x1={padding.left}
				y1={padding.top + plotHeight}
				x2={padding.left + plotWidth}
				y2={padding.top + plotHeight}
				stroke="var(--baseline)"
				stroke-width="1"
			/>

			<path d={linePath} fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round" />

			{#each points as p, i (p.label)}
				<circle
					cx={xToPx(i)}
					cy={yToPx(p.value)}
					r={hovered === i ? 5 : 3.5}
					fill="var(--series-1)"
					role="presentation"
					onmouseenter={() => (hovered = i)}
					onmouseleave={() => (hovered = null)}
				/>
			{/each}
		</svg>

		{#if hovered !== null}
			<div class="tooltip">
				<strong>{points[hovered].label}</strong>
				<span>{yFormat(points[hovered].value)}</span>
			</div>
		{/if}
	{/if}
</div>

<style>
	.chart-root {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.chart-head {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
	}
	h3 {
		margin: 0;
		font-size: 0.95rem;
		color: var(--text-primary);
	}
	.link-btn {
		background: none;
		border: none;
		color: var(--series-1);
		font-size: 0.8rem;
		cursor: pointer;
		padding: 0;
	}
	svg {
		width: 100%;
		height: auto;
		display: block;
	}
	circle {
		cursor: pointer;
	}
	.band-label {
		font-size: 9px;
		fill: var(--text-muted);
	}
	.tooltip {
		display: flex;
		gap: 0.5rem;
		font-size: 0.8rem;
		color: var(--text-secondary);
	}
	.empty {
		color: var(--text-muted);
		font-size: 0.85rem;
	}
</style>
