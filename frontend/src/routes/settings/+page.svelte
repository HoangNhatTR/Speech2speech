<script lang="ts">
	import { onMount } from 'svelte';
	import { getConfig, putConfig, resetConfigKey } from '$lib/api';

	const SELECT_OPTIONS: Record<string, string[]> = {
		STT_BACKEND: ['cloud', 'local'],
		LLM_BACKEND: ['cloud', 'local'],
		TTS_BACKEND: ['cloud', 'local'],
		VIENEU_STREAMING: ['true', 'false'],
		DUPLEX_BACKCHANNEL_FILTER: ['true', 'false'],
		EMOTION_BACKEND: ['none', 'heuristic'],
		S2S_MODE: ['off', 'shadow', 'ack_only', 'speculative', 'primary'],
		S2S_SHADOW_BACKEND: ['probe', 'moshi_ws']
	};

	const LABELS: Record<string, string> = {
		STT_BACKEND: 'STT backend (Speech-to-Text)',
		LLM_BACKEND: 'LLM backend (hội thoại)',
		TTS_BACKEND: 'TTS backend (Text-to-Speech)',
		VIENEU_STREAMING: 'VieNeu true streaming',
		DUPLEX_BACKCHANNEL_FILTER: 'Lọc backchannel tiếng Việt (Giai đoạn 2)',
		DUPLEX_BARGE_IN_DELAY_MS: 'Độ trễ VAD trước khi barge-in (ms)',
		EMOTION_BACKEND: 'Kênh cảm xúc (heuristic, chưa phải SER thật)',
		S2S_MODE: 'Chế độ Dual-path Speech-to-Speech',
		S2S_SHADOW_BACKEND: 'Backend fast path (probe hoặc Moshi sidecar)',
		S2S_AUDIO_RING_MS: 'Audio ring buffer trong Gateway (ms)',
		S2S_SUBSCRIBER_QUEUE_FRAMES: 'Queue frame tối đa cho fast path',
		ANTHROPIC_MODEL: 'Model Anthropic (Claude) ghi đè',
		DEEPGRAM_LANGUAGE: 'Ngôn ngữ Deepgram',
		ELEVENLABS_MODEL: 'Model ElevenLabs',
		VLLM_MODEL: 'Model vLLM (backend local)'
	};

	let original = $state<Record<string, string>>({});
	let draft = $state<Record<string, string>>({});
	let overridden = $state<string[]>([]);
	let allowedKeys = $state<string[]>([]);
	let error = $state('');
	let saved = $state(false);

	async function load() {
		error = '';
		try {
			const res = await getConfig();
			original = { ...res.values };
			draft = { ...res.values };
			overridden = res.overridden;
			allowedKeys = res.allowed_keys;
		} catch (e) {
			error = `Không gọi được gateway (${(e as Error).message}).`;
		}
	}

	onMount(load);

	async function save() {
		saved = false;
		const diff: Record<string, string> = {};
		for (const key of allowedKeys) {
			if (draft[key] !== original[key]) diff[key] = draft[key];
		}
		if (Object.keys(diff).length === 0) return;
		try {
			await putConfig(diff);
			await load();
			saved = true;
		} catch (e) {
			error = (e as Error).message;
		}
	}

	async function resetKey(key: string) {
		try {
			await resetConfigKey(key);
			await load();
		} catch (e) {
			error = (e as Error).message;
		}
	}
</script>

<h1>Settings</h1>

{#if error}
	<div class="card error">{error}</div>
{/if}

<p class="note">
	Đổi ở đây áp dụng ngay cho <strong>phiên voice mới</strong> (không ảnh hưởng phiên đang mở),
	không cần restart Gateway — xem gateway/runtime_config.py. Không đổi được API key/secret ở
	đây (luôn chỉ đọc từ .env). <strong>SERVICES_LLM_BACKEND</strong> (7 service Vision/Text/Tool/
	Memory/Planning/Reasoning/Generation) chạy ở tiến trình runtime.dispatcher riêng, ngoài phạm
	vi trang này — vẫn phải sửa .env + restart dispatcher như trước.
</p>

<section class="card">
	{#each allowedKeys as key (key)}
		<div class="row">
			<div class="row-label">
				{LABELS[key] ?? key}
				{#if overridden.includes(key)}
					<span class="badge">đã override</span>
					<button class="link-btn" onclick={() => resetKey(key)}>reset về .env</button>
				{/if}
			</div>
			{#if SELECT_OPTIONS[key]}
				<select bind:value={draft[key]}>
					{#each SELECT_OPTIONS[key] as opt (opt)}
						<option value={opt}>{opt}</option>
					{/each}
				</select>
			{:else}
				<input bind:value={draft[key]} placeholder="(mặc định trong bot.py)" />
			{/if}
		</div>
	{/each}

	<div class="actions">
		<button class="save-btn" onclick={save}>Lưu thay đổi</button>
		{#if saved}<span class="saved-msg">Đã lưu.</span>{/if}
	</div>
</section>

<style>
	h1 {
		font-size: 1.3rem;
		margin-bottom: 0.75rem;
	}
	.error {
		color: var(--status-critical);
		margin-bottom: 1rem;
		font-size: 0.9rem;
	}
	.note {
		font-size: 0.85rem;
		color: var(--text-secondary);
		margin-bottom: 1.25rem;
		line-height: 1.5;
	}
	.row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.6rem 0;
		border-bottom: 1px solid var(--gridline);
		gap: 1rem;
	}
	.row:last-of-type {
		border-bottom: none;
	}
	.row-label {
		font-size: 0.875rem;
		color: var(--text-primary);
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
	.badge {
		font-size: 0.7rem;
		background: var(--series-1-soft);
		color: var(--series-1);
		padding: 0.1rem 0.4rem;
		border-radius: 4px;
	}
	.link-btn {
		background: none;
		border: none;
		color: var(--text-muted);
		font-size: 0.75rem;
		text-decoration: underline;
		cursor: pointer;
		padding: 0;
	}
	select,
	input {
		padding: 0.35rem 0.5rem;
		border: 1px solid var(--border);
		border-radius: 6px;
		background: var(--page-plane);
		color: var(--text-primary);
		min-width: 200px;
	}
	.actions {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		margin-top: 1rem;
	}
	.save-btn {
		background: var(--series-1);
		color: white;
		border: none;
		border-radius: 6px;
		padding: 0.5rem 1.2rem;
		cursor: pointer;
		font-size: 0.875rem;
	}
	.saved-msg {
		color: var(--status-good);
		font-size: 0.85rem;
	}
</style>
