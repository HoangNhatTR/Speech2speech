// Client gọi Dashboard/Settings API của gateway (gateway/dashboard_api.py) — BE/FE tách
// rời độc lập: FE này KHÔNG import bất kỳ code Python nào, chỉ nói chuyện qua HTTP.
// Auth: cùng cơ chế Bearer token với /api/offer (gateway/auth.py) — khi
// GATEWAY_AUTH_DISABLED=true (mặc định dev) token rỗng vẫn được chấp nhận.

const URL_STORAGE_KEY = 'speech2speech.gateway_url';
const TOKEN_STORAGE_KEY = 'speech2speech.gateway_api_key';
const DEFAULT_BASE_URL = import.meta.env.VITE_GATEWAY_URL ?? 'http://localhost:7860';

export function getBaseUrl(): string {
	if (typeof localStorage === 'undefined') return DEFAULT_BASE_URL;
	return localStorage.getItem(URL_STORAGE_KEY) ?? DEFAULT_BASE_URL;
}

export function setBaseUrl(url: string): void {
	localStorage.setItem(URL_STORAGE_KEY, url);
}

export function getToken(): string {
	if (typeof localStorage === 'undefined') return '';
	return localStorage.getItem(TOKEN_STORAGE_KEY) ?? '';
}

export function setToken(token: string): void {
	localStorage.setItem(TOKEN_STORAGE_KEY, token);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const token = getToken();
	const headers = new Headers(init?.headers);
	headers.set('Content-Type', 'application/json');
	if (token) headers.set('Authorization', `Bearer ${token}`);

	const res = await fetch(`${getBaseUrl()}${path}`, { ...init, headers });
	if (!res.ok) {
		const detail = await res.text().catch(() => '');
		throw new Error(`${res.status} ${res.statusText}${detail ? `: ${detail}` : ''}`);
	}
	return (await res.json()) as T;
}

export interface SessionInfo {
	session_id: string;
	kind: string;
	created_at: number;
}

export interface StatusResponse {
	uptime_s: number;
	active_sessions: number;
	sessions: SessionInfo[];
}

export function getStatus(): Promise<StatusResponse> {
	return request('/api/status');
}

export interface ConfigResponse {
	values: Record<string, string>;
	overridden: string[];
	allowed_keys: string[];
}

export function getConfig(): Promise<ConfigResponse> {
	return request('/api/config');
}

export function putConfig(overrides: Record<string, string>): Promise<ConfigResponse> {
	return request('/api/config', { method: 'PUT', body: JSON.stringify(overrides) });
}

export function resetConfigKey(key: string): Promise<{ values: Record<string, string> }> {
	return request(`/api/config/${encodeURIComponent(key)}`, { method: 'DELETE' });
}

export interface LatencyRow {
	metric: string;
	stt_backend: string;
	llm_backend: string;
	tts_backend: string;
	n: number;
	p50_ms: number;
	p95_ms: number;
	mean_ms: number;
}

export function getLatency(): Promise<{ rows: LatencyRow[] }> {
	return request('/api/metrics/latency');
}

export interface LatencyVsTarget {
	status: 'within_target' | 'outside_target' | 'no_data';
	measured_p50_ms?: number;
	target_range_ms?: [number, number];
}

export interface DuplexBenchSummary {
	accuracy: number;
	false_interrupt_rate: number | null;
	missed_interrupt_rate: number | null;
}

export interface BenchmarkEntry {
	ts: string;
	duplex_bench?: DuplexBenchSummary;
	latency?: LatencyRow[];
	latency_vs_target?: LatencyVsTarget;
	asr_wer?: {
		backend: string;
		overall_wer: number;
		by_domain_wer: Record<string, number>;
		by_noise_wer: Record<string, number>;
		code_switch_wer: number | null;
	};
	asr_wer_vs_sota?: { status: string; measured_pct?: number; sota_pct?: number; note?: string };
	tool_call_accuracy?: { accuracy: number; n: number };
}

export function getBenchmarkHistory(limit = 200): Promise<{ entries: BenchmarkEntry[] }> {
	return request(`/api/benchmarks/history?limit=${limit}`);
}

export function getBenchmarkLatest(): Promise<BenchmarkEntry> {
	return request('/api/benchmarks/latest');
}

export function runDuplexBench(): Promise<unknown> {
	return request('/api/benchmarks/duplex/run', { method: 'POST' });
}
