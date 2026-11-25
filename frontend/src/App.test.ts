import { describe, it, expect } from 'vitest';
import { resolveSystemStatus, resolveEffectiveWebSearchMode, BackendHealth } from './App';

describe('Application System Status Behavior', () => {
  it('resolves loading state correctly', () => {
    const status = resolveSystemStatus(null, true);
    expect(status.apiStatus).toBe('Checking...');
    expect(status.dbStatus).toBe('Docker / Local');
    expect(status.vectorStatus).toBe('Ready (768d)');
  });

  it('resolves healthy backend and database state with active vector extension', () => {
    const mockHealth: BackendHealth = {
      status: 'healthy',
      app_env: 'development',
      database: {
        connected: true,
        version: 'PostgreSQL 16.6',
        has_vector_extension: true,
        vector_version: '0.8.0',
        error: null,
      },
    };

    const status = resolveSystemStatus(mockHealth, false);
    expect(status.apiStatus).toBe('Online');
    expect(status.dbStatus).toBe('Connected');
    expect(status.vectorStatus).toBe('Active');
  });

  it('resolves degraded state when backend is reachable but database is disconnected', () => {
    const degradedHealth: BackendHealth = {
      status: 'degraded',
      app_env: 'development',
      database: {
        connected: false,
        version: null,
        has_vector_extension: false,
        vector_version: null,
        error: 'connection refused',
      },
    };

    const status = resolveSystemStatus(degradedHealth, false);
    expect(status.apiStatus).toBe('Online');
    expect(status.dbStatus).toBe('Docker / Local');
    expect(status.vectorStatus).toBe('Ready (768d)');
  });

  it('resolves disconnected state when backend is unreachable', () => {
    const status = resolveSystemStatus(null, false);
    expect(status.apiStatus).toBe('Disconnected');
    expect(status.dbStatus).toBe('Docker / Local');
    expect(status.vectorStatus).toBe('Ready (768d)');
  });

  it('normalizes and transforms in-text citations into interactive citation anchors', () => {
    const rawText = "Operating System Detection [Page 1] and another citation [Page\u202F1 (PDF p. 1)]";
    const normalized = rawText.replace(/[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]/g, ' ');
    const transformed = normalized.replace(/\[((?:[^\]\n]*\b)?(?:Page|p\.)\s*[0-9]+[^\]\n]*)\]/gi, (_match, label) => {
      const cleanLabel = label.trim();
      return `[${cleanLabel}](#cite-${encodeURIComponent(cleanLabel)})`;
    });

    expect(transformed).toContain('[Page 1](#cite-Page%201)');
    expect(transformed).toContain('[Page 1 (PDF p. 1)](#cite-Page%201%20(PDF%20p.%201))');
  });

  it('transforms inline web [Source N] references into clickable citation anchors', () => {
    const rawText = "FastAPI 0.165.0 was released [Source 1] and verified [Source 2].";
    const transformed = rawText.replace(/\[((?:[^\]\n]*\b)?(?:Page|p\.|Source)\s*[0-9]+[^\]\n]*)\]/gi, (_match, label) => {
      const cleanLabel = label.trim();
      return `[${cleanLabel}](#cite-${encodeURIComponent(cleanLabel)})`;
    });

    expect(transformed).toContain('[Source 1](#cite-Source%201)');
    expect(transformed).toContain('[Source 2](#cite-Source%202)');
  });

  it('enforces local-mode 0B egress and PDF RAG isolation on web search mode', () => {
    expect(resolveEffectiveWebSearchMode('local', 0, 'on')).toBe('off');
    expect(resolveEffectiveWebSearchMode('local', 0, 'auto')).toBe('off');
    expect(resolveEffectiveWebSearchMode('cloud', 2, 'on')).toBe('off');
    expect(resolveEffectiveWebSearchMode('cloud', 1, 'auto')).toBe('off');
    expect(resolveEffectiveWebSearchMode('cloud', 0, 'auto')).toBe('auto');
    expect(resolveEffectiveWebSearchMode('cloud', 0, 'on')).toBe('on');
    expect(resolveEffectiveWebSearchMode('cloud', 0, 'off')).toBe('off');
  });
});
