<script lang="ts">
	let {
		label,
		value,
		sub = '',
		status = 'neutral'
	}: {
		label: string;
		value: string;
		sub?: string;
		status?: 'neutral' | 'good' | 'warning' | 'critical';
	} = $props();
</script>

<div class="tile">
	<div class="label">{label}</div>
	<div
		class="value"
		class:good={status === 'good'}
		class:warning={status === 'warning'}
		class:critical={status === 'critical'}
	>
		{value}
	</div>
	{#if sub}<div class="sub">{sub}</div>{/if}
</div>

<style>
	.tile {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}
	.label {
		font-size: 0.8rem;
		color: var(--text-muted);
	}
	.value {
		font-size: 1.75rem;
		font-weight: 600;
		color: var(--text-primary);
		font-variant-numeric: tabular-nums;
	}
	.value.good {
		color: var(--status-good);
	}
	.value.warning {
		color: var(--status-warning);
	}
	.value.critical {
		color: var(--status-critical);
	}
	.sub {
		font-size: 0.8rem;
		color: var(--text-secondary);
	}
</style>
