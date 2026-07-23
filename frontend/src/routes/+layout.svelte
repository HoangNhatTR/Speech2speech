<script lang="ts">
	import favicon from '$lib/assets/favicon.svg';
	import '$lib/theme.css';
	import { page } from '$app/state';
	import { getBaseUrl, setBaseUrl, getToken, setToken } from '$lib/api';

	let { children } = $props();

	let baseUrl = $state(getBaseUrl());
	let token = $state(getToken());
	let showConnection = $state(false);

	function saveConnection() {
		setBaseUrl(baseUrl);
		setToken(token);
		showConnection = false;
		location.reload();
	}
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>Speech2Speech — Dashboard</title>
</svelte:head>

<div class="shell">
	<header>
		<div class="brand">Speech2Speech</div>
		<nav>
			<a href="/" class:active={page.url.pathname === '/'}>Dashboard</a>
			<a href="/settings" class:active={page.url.pathname === '/settings'}>Settings</a>
		</nav>
		<button class="conn-toggle" onclick={() => (showConnection = !showConnection)}>
			Kết nối gateway
		</button>
	</header>

	{#if showConnection}
		<div class="conn-panel card">
			<label>
				Gateway URL
				<input bind:value={baseUrl} placeholder="http://localhost:7860" />
			</label>
			<label>
				API key (GATEWAY_API_KEYS, để trống nếu GATEWAY_AUTH_DISABLED=true)
				<input bind:value={token} type="password" placeholder="changeme-dev-key" />
			</label>
			<button onclick={saveConnection}>Lưu</button>
		</div>
	{/if}

	<main>
		{@render children()}
	</main>
</div>

<style>
	.shell {
		max-width: 960px;
		margin: 0 auto;
		padding: 1.5rem;
	}
	header {
		display: flex;
		align-items: center;
		gap: 1.5rem;
		margin-bottom: 1.5rem;
	}
	.brand {
		font-weight: 700;
		font-size: 1.1rem;
	}
	nav {
		display: flex;
		gap: 1rem;
		flex: 1;
	}
	nav a {
		color: var(--text-secondary);
		text-decoration: none;
		font-size: 0.9rem;
		padding-bottom: 2px;
		border-bottom: 2px solid transparent;
	}
	nav a.active {
		color: var(--text-primary);
		border-bottom-color: var(--series-1);
	}
	.conn-toggle {
		background: none;
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 0.35rem 0.7rem;
		font-size: 0.8rem;
		color: var(--text-secondary);
		cursor: pointer;
	}
	.conn-panel {
		display: flex;
		gap: 1rem;
		align-items: flex-end;
		margin-bottom: 1.5rem;
		flex-wrap: wrap;
	}
	.conn-panel label {
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
		font-size: 0.8rem;
		color: var(--text-secondary);
		flex: 1;
		min-width: 220px;
	}
	.conn-panel input {
		padding: 0.4rem 0.5rem;
		border: 1px solid var(--border);
		border-radius: 6px;
		background: var(--page-plane);
		color: var(--text-primary);
	}
	.conn-panel button {
		background: var(--series-1);
		color: white;
		border: none;
		border-radius: 6px;
		padding: 0.5rem 1rem;
		cursor: pointer;
		height: fit-content;
	}
</style>
