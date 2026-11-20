import { describe, it, expect } from 'vitest';
import { resolveSystemStatus, BackendHealth } from './App';

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
});
