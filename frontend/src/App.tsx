import { useState, useEffect, useRef, useCallback } from 'react';
import {
  MessageSquare,
  Plus,
  Zap,
  Paperclip,
  Send,
  Square,
  Sparkles,
  ExternalLink,
  Trash2,
  Copy,
  Check,
  Edit2,
  X,
  RotateCcw,
  Search,
  Archive,
  ArchiveRestore,
  Pencil,
  Loader2,
  AlertCircle,
  FileText,
  UploadCloud,
  CheckSquare,
  BookOpen,
  Cloud,
  Shield,
} from 'lucide-react';
import { MarkdownRenderer } from './components/MarkdownRenderer';
import { CitationDrawer } from './components/CitationDrawer';

export interface BackendHealth {
  status: string;
  app_env: string;
  database: {
    connected: boolean;
    version: string | null;
    has_vector_extension: boolean;
    vector_version: string | null;
    error: string | null;
  };
  groq_configured?: boolean;
  providers?: {
    groq_configured: boolean;
    gemini_configured: boolean;
    ollama_base_url: string;
  };
}

export interface CitationItem {
  document_id: string;
  document_name: string;
  viewer_page: number;
  book_page?: string | null;
  citation_label: string;
  heading?: string | null;
  excerpt?: string | null;
}

export interface DocumentItem {
  id: string;
  filename: string;
  file_type: string;
  file_size_bytes: number;
  page_count: number;
  chunk_count: number;
  embedding_model: string;
  status: string;
  error_message?: string | null;
  created_at: string;
}

export interface MemoryStatus {
  active: boolean;
  summary_status: 'Available' | 'Updated' | 'Not required';
  recent_tokens: number;
  summary_turns: number;
  recall_used: 'Used' | 'Not used';
  prompt_tokens: number;
}

export interface RateLimitStatus {
  status: 'waiting' | 'rate_limited' | 'ready';
  reason: string;
  wait_sec?: number;
  retry_after_sec?: number;
  remaining_tokens?: number;
  limit_tokens?: number;
}

export interface BackendTimings {
  retrieval_ms: number;
  rerank_ms: number;
  tok_ms: number;
  inf_ms: number;
  memory_ms?: number;
  queue_wait_ms?: number;
  prompt_tokens?: number;
  provider_ttft_ms?: number;
  llm_generation_ms: number;
  llm_total_ms?: number;
  backend_total_ms: number;
  rate_limit?: Record<string, any>;
}

export interface RagStatus {
  retrieval: string;
  reranker: 'OFF' | 'ONNX' | string;
  candidates_count: number;
  reranked_count: number;
  retrieval_latency_ms?: number;
  rerank_latency_ms: number;
  tok_ms?: number;
  inf_ms?: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  latencyMs?: number;
  ttftMs?: number;
  backendTimings?: BackendTimings;
  memoryStatus?: MemoryStatus;
  rateLimitStatus?: RateLimitStatus;
  citations?: CitationItem[];
  ragStatus?: RagStatus;
}

export interface ModelOption {
  id: string;
  name: string;
  provider: string;
}

export interface ConversationItem {
  id: string;
  title: string;
  model_id: string;
  provider: string;
  is_archived?: boolean;
  created_at: string;
  updated_at: string;
  message_count: number;
  last_message_preview?: string | null;
}

export const CLOUD_API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'https://tark-studio.onrender.com').replace(/\/+$/, '');
export const LOCAL_API_BASE_URL = 'http://127.0.0.1:8000';

const DEFAULT_CLOUD_MODELS: ModelOption[] = [
  { id: 'openai/gpt-oss-20b', name: 'GPT-OSS 20B (Groq Fast)', provider: 'groq' },
  { id: 'openai/gpt-oss-120b', name: 'GPT-OSS 120B (Groq)', provider: 'groq' },
  { id: 'qwen/qwen3.8-27b', name: 'Qwen 3.8 27B (Groq)', provider: 'groq' },
  { id: 'llama-3.3-70b-versatile', name: 'Llama 3.3 70B (Groq)', provider: 'groq' },
];

const DEFAULT_LOCAL_MODELS: ModelOption[] = [
  { id: 'llama3.2', name: 'Llama 3.2 (Local RTX 3050)', provider: 'ollama' },
  { id: 'qwen2.5', name: 'Qwen 2.5 (Local RTX 3050)', provider: 'ollama' },
];

export function resolveSystemStatus(health: BackendHealth | null, loading: boolean) {
  const isGroqReady = health?.providers?.groq_configured ?? health?.groq_configured;
  return {
    apiStatus: loading ? 'Checking...' : health ? 'Online' : 'Disconnected',
    dbStatus: health?.database?.connected ? 'Connected' : 'Docker / Local',
    vectorStatus: health?.database?.has_vector_extension ? 'Active' : 'Ready (768d)',
    groqStatus: isGroqReady ? 'Ready' : 'Missing Key',
  };
}

const STARTER_PROMPTS = [
  'Explain async generators in Python with a clean code example',
  'What are the key differences between HNSW and IVFFlat in pgvector?',
  'Draft a high-throughput streaming architecture with Server-Sent Events',
  'Write a Python script using AsyncGroq to stream chat completions',
];

export default function App() {
  // Dual-Engine Mode: 'cloud' (Groq LPU) vs 'local' (RTX 3050 GPU)
  const [engineMode, setEngineMode] = useState<'cloud' | 'local'>(() => {
    const saved = localStorage.getItem('tark_engine_mode');
    if (saved === 'cloud' || saved === 'local') return saved;
    if (typeof window !== 'undefined' && window.location.hostname.includes('pages.dev')) {
      return 'cloud';
    }
    return 'local';
  });

  const API_BASE_URL = engineMode === 'local' ? LOCAL_API_BASE_URL : CLOUD_API_BASE_URL;

  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [models, setModels] = useState<ModelOption[]>(engineMode === 'local' ? DEFAULT_LOCAL_MODELS : DEFAULT_CLOUD_MODELS);
  const [selectedModel, setSelectedModel] = useState<string>(engineMode === 'local' ? 'llama3.2' : 'openai/gpt-oss-20b');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [sessionTitle, setSessionTitle] = useState<string>('New Session');
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);

  // Persistent conversation states
  const [conversations, setConversations] = useState<ConversationItem[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(() => {
    return localStorage.getItem('tark_active_conversation_id');
  });
  const [editingConvId, setEditingConvId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');

  // Search, Archive, Loading & Error states
  const [searchQuery, setSearchQuery] = useState('');
  const [viewTab, setViewTab] = useState<'active' | 'archived'>('active');
  const [isLoadingConversations, setIsLoadingConversations] = useState(false);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [globalError, setGlobalError] = useState<string | null>(null);

  // Message inline edit states
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const [editingMessageContent, setEditingMessageContent] = useState('');

  // RAG Document Intelligence states
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [isDocDrawerOpen, setIsDocDrawerOpen] = useState(false);
  const [isUploadingDoc, setIsUploadingDoc] = useState(false);
  const [selectedReranker, setSelectedReranker] = useState<'onnx' | 'baseline'>('onnx');
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Citation Evidence Drawer states
  const [previewDrawerOpen, setPreviewDrawerOpen] = useState(false);
  const [previewCitations, setPreviewCitations] = useState<CitationItem[]>([]);
  const [previewInitialIndex, setPreviewInitialIndex] = useState(0);

  const handleOpenCitationPreview = useCallback(
    (targetLabel: string, messageCitations?: CitationItem[]) => {
      let citationsToUse =
        messageCitations && messageCitations.length > 0 ? [...messageCitations] : [];

      if (citationsToUse.length === 0) {
        // Look in other messages in the thread if available
        for (const m of messages) {
          if (m.citations && m.citations.length > 0) {
            citationsToUse = [...m.citations];
            break;
          }
        }
      }

      if (citationsToUse.length === 0) {
        // Fallback placeholder if citations metadata wasn't attached
        const pageMatch = targetLabel.match(/(\d+)/);
        const viewerPage = pageMatch ? parseInt(pageMatch[1], 10) : 1;
        citationsToUse = [
          {
            document_id: 'doc',
            document_name: documents[0]?.filename || 'Uploaded Document',
            viewer_page: viewerPage,
            book_page: pageMatch ? pageMatch[1] : null,
            citation_label: targetLabel,
            heading: 'Document Grounding Reference',
            excerpt: `Grounded reference for [${targetLabel}].\n\nThis citation is referencing the indexed knowledge base. Select a document or ask a specific question to view live verified passages.`,
          },
        ];
      }

      // Find matching citation by label or page number
      const normalizedTarget = targetLabel.toLowerCase().replace(/[^a-z0-9]/g, '');
      const pageMatch = targetLabel.match(/(\d+)/);
      const targetPage = pageMatch ? pageMatch[1] : null;

      let foundIdx = citationsToUse.findIndex((c) => {
        const normLabel = (c.citation_label || '').toLowerCase().replace(/[^a-z0-9]/g, '');
        if (normLabel && (normLabel.includes(normalizedTarget) || normalizedTarget.includes(normLabel))) {
          return true;
        }
        if (targetPage && (String(c.book_page) === targetPage || String(c.viewer_page) === targetPage)) {
          return true;
        }
        return false;
      });

      if (foundIdx === -1) {
        foundIdx = 0;
      }

      setPreviewCitations(citationsToUse);
      setPreviewInitialIndex(foundIdx);
      setPreviewDrawerOpen(true);
    },
    [messages, documents]
  );

  const sysStatus = resolveSystemStatus(health, loading);

  // Filter models based on active engine mode
  const availableModels = models.filter((m) => {
    if (engineMode === 'local') {
      return m.provider === 'ollama';
    }
    return m.provider !== 'ollama';
  });

  const displayModels = availableModels.length > 0
    ? availableModels
    : engineMode === 'local'
    ? DEFAULT_LOCAL_MODELS
    : DEFAULT_CLOUD_MODELS;

  const handleSwitchEngine = (mode: 'cloud' | 'local') => {
    if (mode === engineMode) return;
    setEngineMode(mode);
    localStorage.setItem('tark_engine_mode', mode);
    setGlobalError(null);
    if (mode === 'local') {
      setSelectedModel('llama3.2');
    } else {
      setSelectedModel('openai/gpt-oss-20b');
    }
  };

  const abortControllerRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Fetch ingested documents list
  const fetchDocuments = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/documents`);
      if (res.ok) {
        const data: DocumentItem[] = await res.json();
        setDocuments(data);
        setSelectedDocIds((prev) => {
          if (prev.length === 0 && data.length > 0) {
            return [data[0].id];
          }
          return prev;
        });
      }
    } catch (err) {
      console.error('Failed to load documents:', err);
    }
  }, [API_BASE_URL]);

  // Auto-poll documents when any document is still 'processing'
  useEffect(() => {
    const isAnyProcessing = documents.some((d) => d.status === 'processing');
    if (!isAnyProcessing) return;

    const interval = setInterval(() => {
      fetchDocuments();
    }, 1500);

    return () => clearInterval(interval);
  }, [documents, fetchDocuments]);

  const handleUploadFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsUploadingDoc(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/documents/upload`, {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        const uploadedDoc = await res.json();
        // Immediately register in documents list and attach to active chat grounding
        setDocuments((prev) => [uploadedDoc, ...prev.filter((d) => d.id !== uploadedDoc.id)]);
        setSelectedDocIds((prev) => [...prev.filter((id) => id !== uploadedDoc.id), uploadedDoc.id]);
        await fetchDocuments();
      } else {
        const err = await res.json();
        setGlobalError(`Upload failed: ${err.detail || res.statusText}`);
      }
    } catch (err: any) {
      setGlobalError(`Upload error: ${err.message}`);
    } finally {
      setIsUploadingDoc(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleDeleteDocument = async (docId: string) => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/documents/${docId}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        setSelectedDocIds((prev) => prev.filter((id) => id !== docId));
        await fetchDocuments();
      }
    } catch (err) {
      console.error('Failed to delete document:', err);
    }
  };

  const toggleDocSelection = (docId: string) => {
    setSelectedDocIds((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  };

  // Fetch conversations list from PostgreSQL with search & archive filter
  const fetchConversations = useCallback(
    async (query = searchQuery, tab = viewTab) => {
      setIsLoadingConversations(true);
      try {
        const isArchived = tab === 'archived';
        const url = new URL(`${API_BASE_URL}/api/v1/conversations`, window.location.origin);
        url.searchParams.set('is_archived', String(isArchived));
        if (query.trim()) {
          url.searchParams.set('q', query.trim());
        }
        const res = await fetch(url.toString());
        if (res.ok) {
          const data: ConversationItem[] = await res.json();
          setConversations(data);
          return data;
        }
      } catch (err) {
        console.error('Failed to load conversations:', err);
      } finally {
        setIsLoadingConversations(false);
      }
      return [];
    },
    [searchQuery, viewTab, API_BASE_URL]
  );

  // Load a specific conversation and all its messages
  const loadConversation = useCallback(async (id: string) => {
    // Abort any ongoing stream before switching threads
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsGenerating(false);
    }

    setIsLoadingMessages(true);
    setGlobalError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/conversations/${id}`);
      if (res.ok) {
        const data = await res.json();
        setActiveConversationId(data.id);
        localStorage.setItem('tark_active_conversation_id', data.id);
        setSessionTitle(data.title);
        if (data.model_id) {
          setSelectedModel(data.model_id);
        }
        setMessages(
          data.messages.map((m: any) => ({
            id: m.id,
            role: m.role,
            content: m.content,
            latencyMs: m.latency_ms,
            citations: m.citations || [],
          }))
        );
        return;
      }
    } catch (err) {
      console.error('Failed to load conversation messages:', err);
      setGlobalError('Failed to load conversation messages.');
    } finally {
      setIsLoadingMessages(false);
    }
    // Fallback if not found
    localStorage.removeItem('tark_active_conversation_id');
    setActiveConversationId(null);
    setMessages([]);
    setSessionTitle('New Session');
  }, [API_BASE_URL]);

  // Poll backend health & models list, restore saved conversation on engine switch or startup
  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE_URL}/api/v1/health`)
      .then((res) => res.json())
      .then((data: BackendHealth) => {
        setHealth(data);
        setLoading(false);
      })
      .catch(() => {
        setHealth(null);
        setLoading(false);
      });

    fetch(`${API_BASE_URL}/api/v1/models`)
      .then((res) => res.json())
      .then((data) => {
        if (data.models && data.models.length > 0) {
          setModels(data.models);
        }
      })
      .catch(() => {});

    fetchDocuments();

    // Restore active conversation after page refresh or switch
    fetchConversations().then((list) => {
      const savedId = localStorage.getItem('tark_active_conversation_id');
      if (savedId && list && list.some((c) => c.id === savedId)) {
        loadConversation(savedId);
      } else {
        handleNewChat();
      }
    });
  }, [engineMode, API_BASE_URL, fetchConversations, loadConversation, fetchDocuments]);

  // Trigger search & tab filter with debounce
  useEffect(() => {
    const timer = setTimeout(() => {
      fetchConversations(searchQuery, viewTab);
    }, 200);
    return () => clearTimeout(timer);
  }, [searchQuery, viewTab, fetchConversations]);

  // Auto-scroll: use instant scroll during active generation to prevent animation stutter
  useEffect(() => {
    if (isGenerating) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'auto' });
    } else {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isGenerating]);

  const handleTextareaInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 140)}px`;
    }
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsGenerating(false);
    }
  };

  const handleCopyResponse = async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageId(id);
      setTimeout(() => setCopiedMessageId(null), 2000);
    } catch {
      const textarea = document.createElement('textarea');
      textarea.value = text;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      setCopiedMessageId(id);
      setTimeout(() => setCopiedMessageId(null), 2000);
    }
  };

  const handleNewChat = () => {
    if (isGenerating) {
      handleStop();
    }
    setActiveConversationId(null);
    localStorage.removeItem('tark_active_conversation_id');
    setMessages([]);
    setSessionTitle('New Session');
  };

  const handleStartRename = (e: React.MouseEvent, conv: ConversationItem) => {
    e.stopPropagation();
    setEditingConvId(conv.id);
    setEditingTitle(conv.title);
  };

  const handleSaveRename = async (id: string) => {
    const trimmed = editingTitle.trim();
    if (!trimmed) {
      setEditingConvId(null);
      return;
    }

    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/conversations/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: trimmed }),
      });
      if (res.ok) {
        if (activeConversationId === id) {
          setSessionTitle(trimmed);
        }
        fetchConversations();
      }
    } catch (err) {
      console.error('Failed to rename conversation:', err);
    } finally {
      setEditingConvId(null);
    }
  };

  const handleDeleteConversation = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/conversations/${id}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        if (activeConversationId === id) {
          handleNewChat();
        }
        fetchConversations();
      }
    } catch (err) {
      console.error('Failed to delete conversation:', err);
    }
  };

  const handleArchiveConversation = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/conversations/${id}/archive`, {
        method: 'POST',
      });
      if (res.ok) {
        if (activeConversationId === id) {
          handleNewChat();
        }
        fetchConversations();
      }
    } catch (err) {
      console.error('Failed to archive conversation:', err);
    }
  };

  const handleUnarchiveConversation = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/conversations/${id}/unarchive`, {
        method: 'POST',
      });
      if (res.ok) {
        fetchConversations();
      }
    } catch (err) {
      console.error('Failed to unarchive conversation:', err);
    }
  };

  const handleGenerateTitle = async (convId = activeConversationId) => {
    if (!convId) return;
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/conversations/${convId}/generate-title`, {
        method: 'POST',
      });
      if (res.ok) {
        const data = await res.json();
        setSessionTitle(data.title);
        fetchConversations();
      }
    } catch (err) {
      console.error('Failed to generate title:', err);
    }
  };

  const executeStream = async (history: ChatMessage[], saveUserMessage: boolean = true) => {
    if (isGenerating || history.length === 0) return;

    const assistantMsgId = `assistant-${Date.now()}`;

    const newAssistantMsg: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
    };

    setMessages([...history, newAssistantMsg]);
    setIsGenerating(true);
    setGlobalError(null);

    abortControllerRef.current = new AbortController();

    let accumulatedText = '';
    let updateTimeout: any = null;

    try {
      const startTime = performance.now();
      const response = await fetch(`${API_BASE_URL}/api/v1/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          conversation_id: activeConversationId || undefined,
          messages: history.map((m) => ({
            role: m.role,
            content: m.content,
          })),
          model: selectedModel,
          temperature: 0.7,
          save_user_message: saveUserMessage,
          document_ids: selectedDocIds.length > 0 ? selectedDocIds : undefined,
          reranker_type: selectedDocIds.length > 0 ? selectedReranker : undefined,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(`Streaming failed with status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      accumulatedText = '';
      let buffer = '';
      updateTimeout = null;
      let hasPendingUpdate = false;

      const flushUpdate = () => {
        if (hasPendingUpdate) {
          hasPendingUpdate = false;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId ? { ...m, content: accumulatedText } : m
            )
          );
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          clearTimeout(updateTimeout);
          flushUpdate();
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split('\n\n');
        buffer = blocks.pop() || '';

        for (const block of blocks) {
          const lines = block.trim().split('\n');
          let eventType = '';
          let dataStr = '';

          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith('data: ')) {
              dataStr = line.slice(6).trim();
            }
          }

          if (eventType === 'meta' && dataStr) {
            try {
              const meta = JSON.parse(dataStr);
              if (meta.conversation_id && meta.conversation_id !== activeConversationId) {
                setActiveConversationId(meta.conversation_id);
                localStorage.setItem('tark_active_conversation_id', meta.conversation_id);
              }
            } catch {}
          } else if (eventType === 'token' && dataStr) {
            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.delta) {
                const isFirstToken = accumulatedText === '';
                if (isFirstToken) {
                  const measuredTtft = Math.round(performance.now() - startTime);
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMsgId ? { ...m, ttftMs: measuredTtft } : m
                    )
                  );
                }
                accumulatedText += parsed.delta;
                hasPendingUpdate = true;
                if (isFirstToken) {
                  flushUpdate();
                } else if (!updateTimeout) {
                  updateTimeout = setTimeout(() => {
                    flushUpdate();
                    updateTimeout = null;
                  }, 20);
                }
              }
            } catch {}
          } else if (eventType === 'memory_status' && dataStr) {
            try {
              const parsed = JSON.parse(dataStr);
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId ? { ...m, memoryStatus: parsed } : m
                )
              );
            } catch {}
          } else if (eventType === 'rate_limit' && dataStr) {
            try {
              const parsed: RateLimitStatus = JSON.parse(dataStr);
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId ? { ...m, rateLimitStatus: parsed } : m
                )
              );
            } catch {}
          } else if (eventType === 'sources' && dataStr) {
            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.citations && Array.isArray(parsed.citations)) {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMsgId
                      ? {
                          ...m,
                          citations: parsed.citations,
                          ragStatus: parsed.rag_status,
                        }
                      : m
                  )
                );
              }
            } catch {}
          } else if (eventType === 'done') {
            clearTimeout(updateTimeout);
            flushUpdate();
            const totalWallClock = Math.round(performance.now() - startTime);
            let parsedBackendTimings: BackendTimings | undefined;
            let parsedMemoryStatus: MemoryStatus | undefined;
            try {
              const donePayload = JSON.parse(dataStr);
              if (donePayload.timings) {
                parsedBackendTimings = donePayload.timings;
              }
              if (donePayload.memory_status) {
                parsedMemoryStatus = donePayload.memory_status;
              }
            } catch {}
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId
                  ? {
                      ...m,
                      latencyMs: totalWallClock,
                      backendTimings: parsedBackendTimings,
                      memoryStatus: parsedMemoryStatus || m.memoryStatus,
                    }
                  : m
              )
            );
            fetchConversations();
          } else if (eventType === 'error' && dataStr) {
            clearTimeout(updateTimeout);
            flushUpdate();
            try {
              const err = JSON.parse(dataStr);
              const errMsg = err.error || 'Failed to complete generation';
              accumulatedText = accumulatedText
                ? `${accumulatedText}\n\n*[Error: ${errMsg}]*`
                : `*[Error: ${errMsg}]*`;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId ? { ...m, content: accumulatedText } : m
                )
              );
            } catch {
              accumulatedText = accumulatedText || '*[Error: An unexpected error occurred]*';
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId ? { ...m, content: accumulatedText } : m
                )
              );
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name === 'AbortError' || err.name === 'DOMException' || String(err.message).toLowerCase().includes('abort')) {
        clearTimeout(updateTimeout);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? {
                  ...m,
                  content: accumulatedText
                    ? `${accumulatedText}\n\n*[Generation stopped]*`
                    : '*[Generation stopped]*',
                }
              : m
          )
        );
      } else {
        const errorText = engineMode === 'local'
          ? `*[Local Sovereign Engine is offline. To use your RTX 3050 GPU, start backend in terminal: \`cd backend && python run.py\`]*`
          : `*[Error connecting to TARK inference engine: ${err.message}]*`;
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? { ...m, content: errorText }
              : m
          )
        );
        setGlobalError(
          engineMode === 'local'
            ? 'Local engine offline — start backend: cd backend && python run.py'
            : `Inference connection error: ${err.message}`
        );
      }
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
      fetchConversations();
    }
  };

  const handleSend = (customPrompt?: string) => {
    const promptText = (customPrompt ?? input).trim();
    if (!promptText || isGenerating) return;

    // Update optimistic session title on first turn
    if (messages.length === 0) {
      const generatedTitle =
        promptText.length > 28 ? promptText.slice(0, 25) + '...' : promptText;
      setSessionTitle(generatedTitle);
    }

    const userMsgId = `user-${Date.now()}`;
    const newUserMsg: ChatMessage = {
      id: userMsgId,
      role: 'user',
      content: promptText,
    };

    const nextHistory = [...messages, newUserMsg];
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    executeStream(nextHistory, true);
  };

  const handleRegenerate = async (assistantMsgIdx: number) => {
    if (isGenerating) return;
    const assistantMsg = messages[assistantMsgIdx];
    if (!assistantMsg) return;

    const historyBefore = messages.slice(0, assistantMsgIdx);
    if (historyBefore.length === 0) return;

    // Truncate from this assistant message onwards in PostgreSQL
    if (activeConversationId) {
      try {
        await fetch(`${API_BASE_URL}/api/v1/conversations/${activeConversationId}/truncate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ from_message_id: assistantMsg.id }),
        });
      } catch (err) {
        console.error('Failed to truncate on regenerate:', err);
      }
    }

    executeStream(historyBefore, false);
  };

  const handleRetry = async (assistantMsgIdx: number) => {
    if (isGenerating) return;
    const historyBefore = messages.slice(0, assistantMsgIdx);
    if (historyBefore.length === 0) return;

    if (activeConversationId) {
      const assistantMsg = messages[assistantMsgIdx];
      try {
        await fetch(`${API_BASE_URL}/api/v1/conversations/${activeConversationId}/truncate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ from_message_id: assistantMsg.id }),
        });
      } catch {}
    }

    executeStream(historyBefore, false);
  };

  const handleSaveEditedMessage = async (msgId: string, newContent: string) => {
    const trimmed = newContent.trim();
    if (!trimmed || isGenerating) return;

    const msgIdx = messages.findIndex((m) => m.id === msgId);
    if (msgIdx === -1) return;

    if (activeConversationId) {
      try {
        await fetch(`${API_BASE_URL}/api/v1/conversations/${activeConversationId}/messages/${msgId}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: trimmed }),
        });
        await fetch(`${API_BASE_URL}/api/v1/conversations/${activeConversationId}/truncate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ after_message_id: msgId }),
        });
      } catch (err) {
        console.error('Failed to save edited message & truncate:', err);
      }
    }

    const updatedHistory = [
      ...messages.slice(0, msgIdx),
      { ...messages[msgIdx], content: trimmed },
    ];
    setEditingMessageId(null);
    executeStream(updatedHistory, false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-screen w-screen bg-[#10130F] text-[#F1EEE6] font-sans overflow-hidden p-2.5 sm:p-3.5 gap-3.5">
      {/* =========================================================================
          LEFT SIDEBAR (EMBER SAGE NEO-BRUTALISM)
      ========================================================================= */}
      <aside className="w-64 sm:w-72 flex-shrink-0 bg-[#171C17] border border-[#394336] rounded-xl flex flex-col justify-between p-4 shadow-xl relative select-none">
        {/* Corner tech notches */}
        <div className="absolute top-1.5 left-1.5 w-2 h-2 border-t border-l border-ember-copper/50 pointer-events-none" />
        <div className="absolute top-1.5 right-1.5 w-2 h-2 border-t border-r border-ember-copper/50 pointer-events-none" />
        <div className="absolute bottom-1.5 left-1.5 w-2 h-2 border-b border-l border-ember-copper/50 pointer-events-none" />
        <div className="absolute bottom-1.5 right-1.5 w-2 h-2 border-b border-r border-ember-copper/50 pointer-events-none" />

        <div className="space-y-4 flex flex-col min-h-0 flex-1">
          {/* Logo & Brand Header */}
          <div className="flex items-center gap-3 pt-1 px-1 flex-shrink-0">
            <div className="w-8 h-8 flex items-center justify-center flex-shrink-0">
              <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
                <path
                  d="M4 3L11 12L4 21H8.5L13.5 14.5L18.5 21H23L16 12L23 3H18.5L13.5 9.5L8.5 3H4Z"
                  fill="#E59A64"
                />
              </svg>
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <h1 className="font-sans font-bold text-lg tracking-wider text-[#F1EEE6]">
                  TARK <span className="font-light text-[#E59A64]">Studio</span>
                </h1>
              </div>
              <p className="font-mono text-[10px] text-[#A5B49A] tracking-tight">
                v0.1.0 • EMBER SAGE
              </p>
            </div>
          </div>

          {/* + New Chat CTA */}
          <button
            onClick={handleNewChat}
            className="w-full py-2.5 px-4 bg-[#E59A64] hover:bg-[#F0AB78] text-[#10130F] font-semibold text-sm rounded-lg flex items-center justify-center gap-2 shadow-[2px_2px_0px_0px_#A5B49A] active:translate-x-[1px] active:translate-y-[1px] transition-all cursor-pointer flex-shrink-0"
          >
            <Plus size={18} strokeWidth={2.5} />
            <span className="tracking-wide">New Chat</span>
          </button>

          {/* SECTION: WORKSPACE NAVIGATION */}
          <div className="space-y-1 pt-1 flex-shrink-0">
            <div className="flex items-center justify-between text-[11px] font-mono font-medium px-1.5">
              <span className="flex items-center gap-1 text-[#A5B49A]">
                <span className="text-ember-copper text-[10px]">▸</span> WORKSPACE
              </span>
              <span className="text-ember-copper font-mono font-bold tracking-tighter text-xs">
                /////
              </span>
            </div>

            <nav className="space-y-1 text-xs">
              <div className="w-full flex items-center gap-2.5 px-3 py-2 rounded-md bg-[#20271F] border border-[#A5B49A]/80 text-[#F1EEE6] font-medium shadow-sm">
                <MessageSquare size={15} className="text-ember-copper" />
                <span className="flex-1">AI Chat Workspace</span>
                <span className="text-[10px] font-mono text-[#A5B49A] bg-[#10130F] px-1.5 py-0.5 rounded border border-[#394336]">
                  Active
                </span>
              </div>

              {/* RAG Documents Drawer Button */}
              <button
                onClick={() => setIsDocDrawerOpen(true)}
                className="w-full flex items-center gap-2.5 px-3 py-2 rounded-md bg-[#171C17] border border-[#394336] hover:border-ember-copper text-[#B7BDAE] hover:text-[#F1EEE6] font-medium transition-colors cursor-pointer text-left"
              >
                <BookOpen size={15} className="text-[#A5B49A]" />
                <span className="flex-1">Documents (RAG)</span>
                <span className="text-[10px] font-mono text-[#E59A64] bg-[#20271F] px-1.5 py-0.5 rounded border border-[#394336]">
                  {documents.length} Doc{documents.length !== 1 ? 's' : ''}
                </span>
              </button>
            </nav>
          </div>

          {/* SECTION: RECENT CHATS (POSTGRESQL PERSISTENT THREADS) */}
          <div className="flex-1 flex flex-col min-h-0 pt-1 space-y-2">
            <div className="flex items-center justify-between text-[11px] font-mono font-medium px-1.5 flex-shrink-0">
              <span className="flex items-center gap-1 text-[#A5B49A]">
                <span className="text-ember-copper text-[10px]">▸</span> RECENT CHATS ({conversations.length})
              </span>
              <span className="text-ember-copper font-mono font-bold tracking-tighter text-xs">
                /////
              </span>
            </div>

            {/* View Tabs: Active vs Archived */}
            <div className="grid grid-cols-2 gap-1 bg-[#10130F] p-1 rounded-lg border border-[#394336] text-xs font-mono select-none flex-shrink-0">
              <button
                onClick={() => setViewTab('active')}
                className={`py-1 rounded text-center transition-colors cursor-pointer ${
                  viewTab === 'active'
                    ? 'bg-[#20271F] text-ember-copper font-semibold border border-ember-copper/50'
                    : 'text-[#929B89] hover:text-[#F1EEE6]'
                }`}
              >
                Active
              </button>
              <button
                onClick={() => setViewTab('archived')}
                className={`py-1 rounded text-center transition-colors cursor-pointer ${
                  viewTab === 'archived'
                    ? 'bg-[#20271F] text-ember-copper font-semibold border border-ember-copper/50'
                    : 'text-[#929B89] hover:text-[#F1EEE6]'
                }`}
              >
                Archived
              </button>
            </div>

            {/* PostgreSQL Search Input */}
            <div className="relative flex-shrink-0">
              <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[#929B89] pointer-events-none" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search chats & messages..."
                className="w-full bg-[#10130F] border border-[#394336] text-xs text-[#F1EEE6] placeholder:text-[#929B89] pl-7 pr-7 py-1.5 rounded-lg focus:outline-none focus:border-ember-copper font-sans select-text"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-[#929B89] hover:text-[#F1EEE6] cursor-pointer"
                  title="Clear search"
                >
                  <X size={13} />
                </button>
              )}
            </div>

            {/* Conversation List */}
            <div className="flex-1 overflow-y-auto space-y-1.5 pr-1 min-h-0 select-text">
              {isLoadingConversations ? (
                <div className="space-y-2 p-1">
                  <div className="h-9 rounded-md bg-[#20271F]/70 animate-pulse border border-[#394336]/40" />
                  <div className="h-9 rounded-md bg-[#20271F]/50 animate-pulse border border-[#394336]/30" />
                  <div className="h-9 rounded-md bg-[#20271F]/30 animate-pulse border border-[#394336]/20" />
                </div>
              ) : conversations.length === 0 ? (
                <div className="p-3 text-center rounded-lg border border-dashed border-[#394336] bg-[#10130F]/40 text-[#929B89] text-[11px] font-mono space-y-1">
                  {searchQuery ? (
                    <>
                      <p>No chats matching "{searchQuery}"</p>
                      <button
                        onClick={() => setSearchQuery('')}
                        className="text-ember-copper hover:underline cursor-pointer"
                      >
                        Clear search
                      </button>
                    </>
                  ) : viewTab === 'archived' ? (
                    <p>No archived conversations.</p>
                  ) : (
                    <p>
                      No saved chats yet.
                      <br />
                      Start a conversation!
                    </p>
                  )}
                </div>
              ) : (
                conversations.map((conv) => {
                  const isActive = conv.id === activeConversationId;
                  const isEditing = editingConvId === conv.id;

                  if (isEditing) {
                    return (
                      <div
                        key={conv.id}
                        className="bg-[#20271F] border border-ember-copper rounded-md p-1.5 flex items-center gap-1.5"
                      >
                        <input
                          type="text"
                          value={editingTitle}
                          onChange={(e) => setEditingTitle(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleSaveRename(conv.id);
                            if (e.key === 'Escape') setEditingConvId(null);
                          }}
                          autoFocus
                          className="flex-1 bg-[#10130F] border border-[#394336] text-[#F1EEE6] font-mono text-xs px-2 py-1 rounded focus:outline-none"
                        />
                        <button
                          onClick={() => handleSaveRename(conv.id)}
                          className="p-1 text-[#A5B49A] hover:text-[#F1EEE6] hover:bg-[#394336] rounded cursor-pointer"
                          title="Save"
                        >
                          <Check size={13} />
                        </button>
                        <button
                          onClick={() => setEditingConvId(null)}
                          className="p-1 text-[#929B89] hover:text-red-400 hover:bg-[#394336] rounded cursor-pointer"
                          title="Cancel"
                        >
                          <X size={13} />
                        </button>
                      </div>
                    );
                  }

                  return (
                    <div
                      key={conv.id}
                      onClick={() => loadConversation(conv.id)}
                      className={`group w-full text-left rounded-md px-2.5 py-2 text-xs flex items-center justify-between transition-all cursor-pointer border ${
                        isActive
                          ? 'bg-[#20271F] border-l-2 border-l-ember-copper border-t-[#394336] border-r-[#394336] border-b-[#394336] text-[#F1EEE6] shadow-sm font-medium'
                          : 'bg-[#10130F]/80 border-[#394336]/60 text-[#B7BDAE] hover:border-[#A5B49A]/60 hover:text-[#F1EEE6]'
                      }`}
                    >
                      <div className="flex items-center gap-2 truncate flex-1 min-w-0 pr-1">
                        <MessageSquare
                          size={13}
                          className={`flex-shrink-0 ${
                            isActive ? 'text-ember-copper' : 'text-[#A5B49A]'
                          }`}
                        />
                        <span className="truncate" title={conv.title}>
                          {conv.title}
                        </span>
                      </div>

                      <div className="flex items-center gap-1 flex-shrink-0">
                        {/* Message count badge */}
                        <span className="font-mono text-[10px] text-[#929B89] group-hover:hidden">
                          {conv.message_count}m
                        </span>

                        {/* Quick actions on hover */}
                        <button
                          onClick={(e) => handleStartRename(e, conv)}
                          className="hidden group-hover:flex p-1 text-[#929B89] hover:text-[#E59A64] hover:bg-[#20271F] rounded cursor-pointer"
                          title="Rename"
                        >
                          <Edit2 size={12} />
                        </button>

                        {/* Archive or Unarchive */}
                        {viewTab === 'active' ? (
                          <button
                            onClick={(e) => handleArchiveConversation(e, conv.id)}
                            className="hidden group-hover:flex p-1 text-[#929B89] hover:text-[#A5B49A] hover:bg-[#20271F] rounded cursor-pointer"
                            title="Archive conversation"
                          >
                            <Archive size={12} />
                          </button>
                        ) : (
                          <button
                            onClick={(e) => handleUnarchiveConversation(e, conv.id)}
                            className="hidden group-hover:flex p-1 text-[#929B89] hover:text-[#A5B49A] hover:bg-[#20271F] rounded cursor-pointer"
                            title="Unarchive conversation"
                          >
                            <ArchiveRestore size={12} />
                          </button>
                        )}

                        {/* Delete */}
                        <button
                          onClick={(e) => handleDeleteConversation(e, conv.id)}
                          className="hidden group-hover:flex p-1 text-[#929B89] hover:text-red-400 hover:bg-[#20271F] rounded cursor-pointer"
                          title="Delete"
                        >
                          <Trash2 size={12} />
                        </button>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>

        {/* SECTION: INFERENCE ENGINE & DATABASE PERSISTENCE STATUS */}
        <div className="pt-3 border-t border-[#394336] space-y-2 flex-shrink-0">
          <div className="flex items-center justify-between text-[11px] font-mono font-medium px-1">
            <span className="flex items-center gap-1 text-[#A5B49A]">
              <span className="text-ember-copper text-[10px]">▸</span> PERSISTENCE & INFERENCE
            </span>
            <span className="text-ember-copper font-mono font-bold tracking-tighter text-xs">
              /////
            </span>
          </div>

          <div className="bg-[#10130F] border border-[#394336] rounded-md p-2.5 font-mono text-xs space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-[#A5B49A] font-medium">
                <span
                  className={`inline-block w-2 h-2 rounded-full ${
                    health?.database.connected ? 'bg-emerald-500' : 'bg-red-500'
                  }`}
                />
                <span>PostgreSQL</span>
              </span>
              <span className="text-[10px] text-[#929B89]">
                {health?.database.connected ? 'Checkpointer Active' : 'Offline'}
              </span>
            </div>

            <div className="flex items-center justify-between border-t border-[#394336]/60 pt-1.5">
              <span className="flex items-center gap-1.5 text-[#A5B49A] font-medium">
                <span
                  className={`inline-block w-2 h-2 rounded-full ${
                    engineMode === 'local'
                      ? health?.database?.connected ? 'bg-emerald-500' : 'bg-amber-400'
                      : sysStatus.groqStatus === 'Ready' ? 'bg-emerald-500' : 'bg-ember-copper'
                  } animate-pulse`}
                />
                <span>{engineMode === 'local' ? 'NVIDIA RTX 3050' : 'Groq Cloud'}</span>
              </span>
              {engineMode === 'local' ? (
                <Shield size={12} className="text-emerald-400" />
              ) : (
                <ExternalLink size={12} className="text-[#A5B49A]" />
              )}
            </div>
            <div className="text-[11px] text-[#B7BDAE] truncate">
              {engineMode === 'local'
                ? 'CUDA Cores • Ollama Local'
                : displayModels.find((m) => m.id === selectedModel)?.name || selectedModel}
            </div>
          </div>
        </div>
      </aside>

      {/* =========================================================================
          MAIN CHAT WORKSPACE (CINEMATIC MOUNTAIN WALLPAPER + CLEAN RENDERING)
      ========================================================================= */}
      <section className="flex-1 flex flex-col justify-between h-full overflow-hidden border border-[#394336] rounded-xl relative bg-[#10130F] shadow-2xl">
        {/* Background Wallpaper: Mountains & Glowing Copper Sun */}
        <div
          className="absolute inset-0 bg-cover bg-center pointer-events-none opacity-65"
          style={{ backgroundImage: `url('/background.jpg')` }}
        />

        {/* Subtle Ambient Vignette Overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-[#10130F]/50 via-transparent to-[#10130F]/40 pointer-events-none" />

        {/* =========================================================================
            TOP CONTROL BAR (DUAL-ENGINE SWITCHER & SEGMENTED MODEL PICKER)
        ========================================================================= */}
        <header className="h-16 px-6 flex items-center justify-between z-10 border-b border-[#394336]/60 backdrop-blur-md flex-shrink-0 select-none">
          {/* Left: Dual Engine Switcher & Model Selector */}
          <div className="flex items-center gap-3">
            {/* Dual-Engine Mode Toggle */}
            <div className="flex items-center bg-[#171C17]/95 border border-[#394336] p-0.5 rounded-lg text-xs font-mono">
              <button
                type="button"
                onClick={() => handleSwitchEngine('cloud')}
                className={`px-2.5 py-1 rounded-md transition-all flex items-center gap-1.5 cursor-pointer ${
                  engineMode === 'cloud'
                    ? 'bg-ember-copper/20 text-ember-copper border border-ember-copper/40 font-semibold shadow-sm'
                    : 'text-[#929B89] hover:text-[#F1EEE6]'
                }`}
                title="Cloud Mode: High-throughput Groq LPU inference (Zero Setup)"
              >
                <Cloud size={12} />
                <span className="hidden sm:inline">Cloud LPU</span>
              </button>
              <button
                type="button"
                onClick={() => handleSwitchEngine('local')}
                className={`px-2.5 py-1 rounded-md transition-all flex items-center gap-1.5 cursor-pointer ${
                  engineMode === 'local'
                    ? 'bg-emerald-950/70 text-emerald-400 border border-emerald-500/50 font-semibold shadow-sm'
                    : 'text-[#929B89] hover:text-[#F1EEE6]'
                }`}
                title="Air-Gapped Private Mode: RTX 3050 Laptop GPU (0 Bytes Cloud Egress)"
              >
                <Shield size={12} className={engineMode === 'local' ? 'text-emerald-400' : ''} />
                <span>Local RTX 3050</span>
              </button>
            </div>

            {/* Model Selector filtered by active engine */}
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs font-semibold text-[#A5B49A] tracking-wider hidden lg:inline">
                MODEL:
              </span>
              <div className="relative">
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="bg-[#171C17]/95 border border-[#394336] text-[#F1EEE6] font-mono text-xs py-1.5 pl-3 pr-8 rounded-lg hover:border-[#A5B49A] transition-colors cursor-pointer focus:outline-none"
                >
                  {displayModels.map((m) => (
                    <option key={m.id} value={m.id} className="bg-[#171C17] text-[#F1EEE6]">
                      {m.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Active Session Badge & Smart Title Button */}
            <div className="hidden xl:flex items-center gap-2 px-3 py-1 bg-[#171C17]/80 border border-[#394336] rounded-md font-mono text-xs text-[#A5B49A]">
              <span className="w-1.5 h-1.5 rounded-full bg-ember-copper inline-block" />
              <span className="text-[#F1EEE6] font-medium truncate max-w-xs">{sessionTitle}</span>
              {activeConversationId && (
                <button
                  onClick={() => handleGenerateTitle()}
                  className="ml-1 text-[#929B89] hover:text-ember-copper transition-colors cursor-pointer"
                  title="Generate meaningful title via LangGraph"
                >
                  <Sparkles size={12} />
                </button>
              )}
            </div>
          </div>

          {/* Right: Telemetry & Engine Status Indicator */}
          <div className="flex items-center gap-3">
            <div className="hidden md:flex items-center gap-2 px-3 py-1 rounded-md bg-[#171C17]/80 border border-[#394336] font-mono text-xs text-[#A5B49A]">
              <Zap size={13} className="text-ember-copper" />
              <span>{engineMode === 'local' ? 'RTX 3050 CUDA' : 'LangGraph + PG Checkpoint'}</span>
            </div>

            <div
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md border text-xs font-mono transition-colors ${
                engineMode === 'local'
                  ? health?.status === 'healthy'
                    ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
                    : 'bg-amber-950/40 border-amber-500/40 text-amber-300'
                  : 'bg-[#20271F] border-[#394336] text-[#F1EEE6]'
              }`}
            >
              {engineMode === 'local' ? (
                health?.status === 'healthy' ? (
                  <>
                    <Shield size={12} className="text-emerald-400" />
                    <span>0B Egress (127.0.0.1:8000)</span>
                  </>
                ) : (
                  <>
                    <span className="w-2 h-2 rounded-full bg-amber-400 inline-block animate-pulse" />
                    <span>Local Offline (run python run.py)</span>
                  </>
                )
              ) : (
                <>
                  <span
                    className={`w-2 h-2 rounded-full inline-block ${
                      health ? 'bg-emerald-500' : 'bg-red-500'
                    }`}
                  />
                  <span>{sysStatus.apiStatus}</span>
                </>
              )}
            </div>
          </div>
        </header>

        {/* Global Error Alert Banner */}
        {globalError && (
          <div className="mx-6 mt-3 p-3 bg-red-950/80 border border-red-500/80 rounded-lg flex items-center justify-between text-xs text-red-200 z-20 shadow-lg">
            <div className="flex items-center gap-2">
              <AlertCircle size={15} className="text-red-400 flex-shrink-0" />
              <span>{globalError}</span>
            </div>
            <button
              onClick={() => setGlobalError(null)}
              className="text-red-300 hover:text-white cursor-pointer p-1"
            >
              <X size={14} />
            </button>
          </div>
        )}

        {/* =========================================================================
            CHAT MESSAGE STREAM SCROLLABLE VIEWPORT
        ========================================================================= */}
        <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-6 z-10 space-y-6">
          {isLoadingMessages ? (
            /* Loading Conversation Thread State */
            <div className="h-full flex items-center justify-center">
              <div className="flex items-center gap-2.5 px-4 py-2.5 rounded-lg bg-[#171C17]/90 border border-[#394336] font-mono text-xs text-[#A5B49A] shadow-md">
                <Loader2 size={16} className="animate-spin text-ember-copper" />
                <span>Restoring conversation thread from PostgreSQL...</span>
              </div>
            </div>
          ) : messages.length === 0 ? (
            /* Empty State Hero Banner */
            <div className="h-full flex flex-col items-center justify-center text-center max-w-xl mx-auto py-12">
              <div className="w-14 h-14 rounded-2xl bg-[#20271F]/90 border border-[#394336] flex items-center justify-center text-ember-copper mb-5 shadow-[4px_4px_0px_0px_#A5B49A]">
                <Sparkles size={28} />
              </div>
              <h2 className="font-serif text-3xl sm:text-4xl font-normal text-[#F1EEE6] tracking-wide mb-3">
                Industrial Intelligence Workspace
              </h2>
              <p className="font-sans text-sm text-[#B7BDAE] max-w-md mb-8 leading-relaxed">
                Persistent AI conversations powered by LangGraph, PostgreSQL checkpoints, and high-throughput Groq inference.
              </p>

              {/* Starter Prompt Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 w-full text-left">
                {STARTER_PROMPTS.map((prompt, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSend(prompt)}
                    className="p-3 rounded-lg bg-[#171C17]/90 hover:bg-[#20271F] border border-[#394336] hover:border-[#A5B49A] text-xs text-[#B7BDAE] hover:text-[#F1EEE6] transition-all flex items-start gap-2 group cursor-pointer shadow-sm"
                  >
                    <span className="text-ember-copper font-mono text-[11px] mt-0.5 group-hover:translate-x-0.5 transition-transform">
                      ▸
                    </span>
                    <span className="line-clamp-2">{prompt}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            /* Active Message List */
            <div className="max-w-4xl mx-auto space-y-6">
              {messages.map((msg, idx) => {
                const isUser = msg.role === 'user';
                const isAssistant = msg.role === 'assistant';

                return (
                  <div
                    key={msg.id}
                    className={`flex items-start gap-3.5 ${
                      isUser ? 'flex-row-reverse' : 'flex-row'
                    }`}
                  >
                    {/* Role Avatar */}
                    <div
                      className={`w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 font-mono text-xs font-bold border ${
                        isUser
                          ? 'bg-[#20271F] border-[#394336] text-ember-copper shadow-sm'
                          : 'bg-[#E59A64] border-[#E59A64] text-[#10130F] shadow-[2px_2px_0px_0px_#A5B49A]'
                      }`}
                    >
                      {isUser ? (
                        'U'
                      ) : (
                        <Sparkles size={16} fill="currentColor" strokeWidth={1} />
                      )}
                    </div>

                    {/* Message Bubble Container */}
                    <div
                      data-role={msg.role}
                      data-testid={`message-${msg.role}`}
                      className="flex-1 max-w-[85%] sm:max-w-[78%]"
                    >
                      <div
                        className={`p-4 sm:p-5 rounded-lg border text-sm leading-relaxed ${
                          isUser
                            ? 'bg-[#171C17]/95 border-[#394336] text-[#F1EEE6] shadow-[3px_3px_0px_0px_#A5B49A]'
                            : 'bg-[#171C17]/95 border-[#394336] text-[#F1EEE6] shadow-[4px_4px_0px_0px_#E59A64]'
                        }`}
                      >
                        {/* Message Content */}
                        {isAssistant ? (
                          msg.content ? (
                            <MarkdownRenderer
                              content={msg.content}
                              onCitationClick={(label) =>
                                handleOpenCitationPreview(label, msg.citations)
                              }
                            />
                          ) : (
                            /* Typing indicator pulse & rate-limit status */
                            <div className="flex flex-col gap-2">
                              {msg.rateLimitStatus && msg.rateLimitStatus.status !== 'ready' && (
                                <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-[#20271F] border border-[#E59A64]/40 text-xs font-mono text-[#E59A64] animate-pulse">
                                  <Zap size={13} className="text-[#E59A64]" />
                                  <span>
                                    {msg.rateLimitStatus.status === 'rate_limited'
                                      ? `Rate limit reached (TPM bucket refilling). Retrying in ${msg.rateLimitStatus.wait_sec || msg.rateLimitStatus.retry_after_sec}s...`
                                      : `TPM rate limit pacing: waiting ${msg.rateLimitStatus.wait_sec}s for token bucket refill...`}
                                  </span>
                                </div>
                              )}
                              <div className="flex items-center gap-1.5 py-1 text-[#A5B49A] font-mono text-xs">
                                <span className="w-1.5 h-1.5 rounded-full bg-ember-copper animate-bounce" />
                                <span
                                  className="w-1.5 h-1.5 rounded-full bg-ember-copper animate-bounce"
                                  style={{ animationDelay: '0.15s' }}
                                />
                                <span
                                  className="w-1.5 h-1.5 rounded-full bg-ember-copper animate-bounce"
                                  style={{ animationDelay: '0.3s' }}
                                />
                                <span className="ml-2 text-[#929B89]">
                                  {msg.rateLimitStatus && msg.rateLimitStatus.status !== 'ready'
                                    ? 'Awaiting Groq token bucket capacity...'
                                    : 'Orchestrating via LangGraph...'}
                                </span>
                              </div>
                            </div>
                          )
                        ) : editingMessageId === msg.id ? (
                          /* Inline User Message Editor */
                          <div className="space-y-2 select-text">
                            <textarea
                              value={editingMessageContent}
                              onChange={(e) => setEditingMessageContent(e.target.value)}
                              className="w-full bg-[#10130F] border border-ember-copper text-[#F1EEE6] text-xs sm:text-sm p-2 rounded focus:outline-none resize-none font-sans"
                              rows={3}
                              autoFocus
                            />
                            <div className="flex items-center justify-end gap-2">
                              <button
                                onClick={() => setEditingMessageId(null)}
                                className="px-2.5 py-1 text-xs font-mono text-[#929B89] hover:text-[#F1EEE6] rounded bg-[#20271F] border border-[#394336] cursor-pointer"
                              >
                                Cancel
                              </button>
                              <button
                                onClick={() => handleSaveEditedMessage(msg.id, editingMessageContent)}
                                className="px-2.5 py-1 text-xs font-mono font-semibold bg-[#E59A64] text-[#10130F] hover:bg-[#F0AB78] rounded shadow-sm cursor-pointer"
                              >
                                Save & Regenerate
                              </button>
                            </div>
                          </div>
                        ) : (
                          <div>
                            <p className="whitespace-pre-wrap select-text">{msg.content}</p>
                            <div className="mt-2 pt-1 border-t border-[#394336]/40 flex items-center justify-end">
                              <button
                                onClick={() => {
                                  setEditingMessageId(msg.id);
                                  setEditingMessageContent(msg.content);
                                }}
                                className="flex items-center gap-1 text-[11px] font-mono text-[#929B89] hover:text-ember-copper transition-colors cursor-pointer"
                                title="Edit prompt & regenerate"
                              >
                                <Pencil size={11} />
                                <span>Edit</span>
                              </button>
                            </div>
                          </div>
                        )}

                        {/* Performance & Response Lifecycle Telemetry */}
                        {isAssistant && (msg.latencyMs || msg.ttftMs || msg.ragStatus || msg.backendTimings) && (
                          <div className="mt-3 pt-2.5 border-t border-[#394336]/60 flex flex-col gap-1.5">
                            {/* Tier 1: Frontend User Experience (Actual Client Wall-Clock Time) */}
                            <div className="flex items-center gap-2 flex-wrap">
                              <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded bg-[#10130F] border border-[#394336] text-[10px] font-mono shadow-xs">
                                <span className="text-[#929B89] flex items-center gap-1">
                                  <Zap size={11} className="text-ember-copper" />
                                  <span>Client E2E:</span>
                                </span>
                                <span className="text-[#E59A64] font-semibold">
                                  Total: {msg.latencyMs ? `${(msg.latencyMs / 1000).toFixed(2)}s` : 'streaming...'}
                                </span>
                                <span className="text-[#394336]">•</span>
                                <span className="text-[#929B89]">
                                  TTFT: <span className="text-[#A5B49A] font-medium">{msg.ttftMs ? `${msg.ttftMs}ms` : 'waiting...'}</span>
                                </span>
                                {msg.latencyMs && msg.backendTimings && (
                                  <>
                                    <span className="text-[#394336]">•</span>
                                    <span className="text-[#929B89]">
                                      Network/UI: <span className="text-[#F1EEE6] font-medium">
                                        {Math.max(0, msg.latencyMs - msg.backendTimings.backend_total_ms).toFixed(0)}ms
                                      </span>
                                    </span>
                                  </>
                                )}
                              </div>

                              {/* RAG Candidate Pill (if RAG is active) */}
                              {msg.ragStatus && (
                                <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded bg-[#10130F] border border-[#394336] text-[10px] font-mono shadow-xs">
                                  <span className="text-[#929B89]">Retrieval: <span className="text-[#A5B49A] font-semibold">{msg.ragStatus.retrieval}</span></span>
                                  <span className="text-[#394336]">•</span>
                                  <span className="text-[#929B89]">Reranker: <span className={msg.ragStatus.reranker === 'OFF' ? 'text-[#929B89]' : 'text-ember-copper font-semibold'}>{msg.ragStatus.reranker}</span></span>
                                  <span className="text-[#394336]">•</span>
                                  <span className="text-[#929B89]">Candidates: <span className="text-[#F1EEE6] font-medium">{msg.ragStatus.candidates_count} → {msg.ragStatus.reranked_count}</span></span>
                                </div>
                              )}
                            </div>

                            {/* Tier 2: Backend Pipeline Processing Breakdown */}
                            {msg.backendTimings && (
                              <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded bg-[#171C17] border border-[#394336]/80 text-[10px] font-mono text-[#929B89] shadow-xs flex-wrap">
                                <span className="text-[#B7BDAE] font-medium">Backend: <span className="text-[#F1EEE6] font-semibold">{(msg.backendTimings.backend_total_ms / 1000).toFixed(2)}s</span></span>
                                <span className="text-[#394336]">[</span>
                                <span>Retrieve: <span className="text-[#A5B49A]">{msg.backendTimings.retrieval_ms}ms</span></span>
                                <span className="text-[#394336]">•</span>
                                <span>Rerank: <span className="text-[#E59A64]">{msg.backendTimings.rerank_ms}ms</span></span>
                                {msg.backendTimings.queue_wait_ms !== undefined && msg.backendTimings.queue_wait_ms > 0 && (
                                  <>
                                    <span className="text-[#394336]">•</span>
                                    <span>Queue: <span className="text-[#E59A64]">{msg.backendTimings.queue_wait_ms >= 1000 ? `${(msg.backendTimings.queue_wait_ms / 1000).toFixed(1)}s` : `${msg.backendTimings.queue_wait_ms}ms`}</span></span>
                                  </>
                                )}
                                {msg.backendTimings.memory_ms !== undefined && msg.backendTimings.memory_ms > 0 && (
                                  <>
                                    <span className="text-[#394336]">•</span>
                                    <span>Mem: <span className="text-[#A5B49A]">{msg.backendTimings.memory_ms}ms</span></span>
                                  </>
                                )}
                                {msg.backendTimings.provider_ttft_ms !== undefined && (
                                  <>
                                    <span className="text-[#394336]">•</span>
                                    <span>LLM TTFT: <span className="text-[#A5B49A]">{msg.backendTimings.provider_ttft_ms}ms</span></span>
                                  </>
                                )}
                                <span className="text-[#394336]">•</span>
                                <span>LLM Gen: <span className="text-[#A5B49A]">{(msg.backendTimings.llm_generation_ms / 1000).toFixed(2)}s</span></span>
                                {msg.backendTimings.prompt_tokens !== undefined && (
                                  <>
                                    <span className="text-[#394336]">•</span>
                                    <span>Prompt: <span className="text-ember-copper font-medium">{msg.backendTimings.prompt_tokens}t</span></span>
                                  </>
                                )}
                                <span className="text-[#394336]">]</span>
                              </div>
                            )}

                            {/* Tier 3: Conversation Memory Status & Token Budgeting */}
                            {msg.memoryStatus && (
                              <div
                                data-testid="memory-status-badge"
                                className="inline-flex items-center gap-2 px-2.5 py-1 rounded bg-[#10130F] border border-[#394336] text-[10px] font-mono shadow-xs flex-wrap"
                              >
                                <span className="text-[#929B89] flex items-center gap-1">
                                  <span className={`w-1.5 h-1.5 rounded-full ${msg.memoryStatus.active ? 'bg-[#A5B49A]' : 'bg-[#929B89]'}`} />
                                  <span>Memory:</span>
                                  <span className={msg.memoryStatus.active ? 'text-[#A5B49A] font-semibold' : 'text-[#929B89]'}>
                                    {msg.memoryStatus.active ? 'Active' : 'Inactive'}
                                  </span>
                                </span>
                                <span className="text-[#394336]">•</span>
                                <span className="text-[#929B89]">
                                  Rolling summary:{' '}
                                  <span
                                    className={
                                      msg.memoryStatus.summary_status === 'Updated'
                                        ? 'text-[#E59A64] font-semibold'
                                        : msg.memoryStatus.summary_status === 'Available'
                                        ? 'text-[#A5B49A] font-medium'
                                        : 'text-[#929B89]'
                                    }
                                  >
                                    {msg.memoryStatus.summary_status}
                                  </span>
                                </span>
                                <span className="text-[#394336]">•</span>
                                <span className="text-[#929B89]">
                                  Recent context tokens:{' '}
                                  <span className="text-[#A5B49A] font-medium">{msg.memoryStatus.recent_tokens}</span>
                                </span>
                                <span className="text-[#394336]">•</span>
                                <span className="text-[#929B89]">
                                  Summary turns:{' '}
                                  <span className="text-[#F1EEE6] font-medium">{msg.memoryStatus.summary_turns}</span>
                                </span>
                                <span className="text-[#394336]">•</span>
                                <span className="text-[#929B89]">
                                  Historical recall:{' '}
                                  <span className={msg.memoryStatus.recall_used === 'Used' ? 'text-[#E59A64] font-semibold' : 'text-[#929B89]'}>
                                    {msg.memoryStatus.recall_used}
                                  </span>
                                </span>
                                <span className="text-[#394336]">•</span>
                                <span className="text-[#929B89]">
                                  Prompt tokens:{' '}
                                  <span className="text-ember-copper font-medium">{msg.memoryStatus.prompt_tokens}</span>
                                </span>
                              </div>
                            )}
                          </div>
                        )}

                        {/* Verified Grounded Sources Badges */}
                        {isAssistant && msg.citations && msg.citations.length > 0 && (
                          <div className={`pt-2.5 ${msg.ragStatus ? 'mt-1' : 'mt-3 border-t border-[#394336]/60'}`}>
                            <div className="flex items-center gap-2 mb-1.5">
                              <BookOpen size={13} className="text-ember-copper" />
                              <span className="font-mono text-[10px] text-[#A5B49A] font-semibold tracking-wider uppercase">
                                Verified Sources ({msg.citations.length})
                              </span>
                              <span className="text-[10px] font-mono text-[#929B89]">
                                • Click badge to inspect source passage
                              </span>
                            </div>
                            <div className="flex flex-wrap gap-1.5">
                              {msg.citations.map((cite, cIdx) => (
                                <button
                                  key={cIdx}
                                  type="button"
                                  onClick={() => {
                                    setPreviewCitations(msg.citations || []);
                                    setPreviewInitialIndex(cIdx);
                                    setPreviewDrawerOpen(true);
                                  }}
                                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-mono bg-[#10130F] border border-[#394336] text-[#E59A64] hover:border-ember-copper hover:bg-[#20271F] transition-all cursor-pointer shadow-xs active:scale-95 group"
                                  title={`Click to preview passage from ${cite.document_name}`}
                                >
                                  <FileText size={11} className="text-[#A5B49A] group-hover:text-ember-copper transition-colors" />
                                  <span>{cite.citation_label}</span>
                                  <span className="text-[10px] text-[#A5B49A] opacity-70 group-hover:opacity-100">↗</span>
                                </button>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Assistant Message Footer Metadata & Actions */}
                        {isAssistant && msg.content && (
                          <div className="mt-4 pt-2.5 border-t border-[#394336]/60 flex items-center justify-between text-[11px] font-mono text-[#A5B49A]">
                            <div className="flex items-center gap-3">
                              {msg.latencyMs && (
                                <span className="flex items-center gap-1">
                                  <Zap size={11} className="text-ember-copper" />
                                  <span>{msg.latencyMs}ms</span>
                                </span>
                              )}
                              <span>•</span>
                              <span className="text-[#929B89]">Postgres Checkpointed</span>
                            </div>

                            <div className="flex items-center gap-2">
                              {/* Retry / Regenerate Buttons */}
                              {msg.content.includes('*[Error:') || msg.content.includes('*[Generation stopped]*') ? (
                                <button
                                  onClick={() => handleRetry(idx)}
                                  className="flex items-center gap-1 px-2 py-0.5 rounded bg-[#20271F] border border-ember-copper text-ember-copper hover:bg-ember-copper/20 transition-colors cursor-pointer shadow-sm"
                                  title="Retry generation"
                                >
                                  <RotateCcw size={11} />
                                  <span className="font-mono text-[10px] font-semibold">Retry</span>
                                </button>
                              ) : (
                                <button
                                  onClick={() => handleRegenerate(idx)}
                                  className="flex items-center gap-1 text-[#A5B49A] hover:text-[#E59A64] transition-colors cursor-pointer"
                                  title="Regenerate response"
                                >
                                  <RotateCcw size={11} />
                                  <span className="font-mono text-[10px]">Regenerate</span>
                                </button>
                              )}

                              {/* Copy Response Button */}
                              <button
                                onClick={() => handleCopyResponse(msg.id, msg.content)}
                                className="flex items-center gap-1 text-[#A5B49A] hover:text-[#E59A64] transition-colors cursor-pointer"
                                title="Copy response"
                              >
                                {copiedMessageId === msg.id ? (
                                  <>
                                    <Check size={12} className="text-emerald-400" />
                                    <span className="text-emerald-400 font-mono text-[10px]">
                                      Copied
                                    </span>
                                  </>
                                ) : (
                                  <>
                                    <Copy size={12} />
                                    <span className="font-mono text-[10px]">Copy</span>
                                  </>
                                )}
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* =========================================================================
            BOTTOM FLOATING COMPOSER DOCK (ChatGPT-Style Inline Attachments)
        ========================================================================= */}
        <footer className="p-4 sm:p-6 z-20 flex-shrink-0 select-none">
          <div className="max-w-4xl mx-auto">
            {/* Dock Container with 3D Copper Block Shadow */}
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const file = e.dataTransfer.files?.[0];
                if (file) {
                  const fakeEvent = { target: { files: [file] } } as any;
                  handleUploadFile(fakeEvent);
                }
              }}
              className="rounded-xl border border-[#394336] bg-[#171C17]/95 backdrop-blur-md shadow-[4px_4px_0px_0px_#E59A64] flex flex-col transition-all overflow-hidden"
            >
              {/* ChatGPT-Style File Attachment Pills directly above input */}
              {(selectedDocIds.length > 0 || isUploadingDoc) && (
                <div className="flex flex-wrap items-center gap-2 px-3 pt-3 pb-1 border-b border-[#394336]/40">
                  {documents
                    .filter((d) => selectedDocIds.includes(d.id))
                    .map((doc) => (
                      <div
                        key={doc.id}
                        className="relative group flex items-center gap-2.5 px-3 py-1.5 rounded-lg bg-[#20271F] border border-[#394336] hover:border-ember-copper/70 transition-all text-xs font-mono shadow-sm"
                      >
                        <div className="w-6 h-6 rounded bg-[#10130F] border border-[#394336] flex items-center justify-center flex-shrink-0 text-ember-copper">
                          <FileText size={13} />
                        </div>
                        <div className="min-w-0 pr-4">
                          <div className="text-[#F1EEE6] font-medium text-xs truncate max-w-[180px] sm:max-w-[240px]">
                            {doc.filename}
                          </div>
                          <div className="flex items-center gap-1.5 text-[10px]">
                            {doc.status === 'processing' ? (
                              <span className="flex items-center gap-1 text-ember-copper font-medium">
                                <Loader2 size={10} className="animate-spin" />
                                Indexing...
                              </span>
                            ) : doc.status === 'completed' ? (
                              <span className="flex items-center gap-1 text-emerald-400">
                                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                                Ready ({doc.page_count}p)
                              </span>
                            ) : doc.status === 'partial' ? (
                              <span
                                className="flex items-center gap-1 text-amber-400 font-medium"
                                title="Partially indexed (some visual pages pending OCR credentials)"
                              >
                                <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                                Ready (Partial)
                              </span>
                            ) : (
                              <span className="text-red-400 font-mono" title={doc.error_message || 'Indexing failed'}>
                                {doc.error_message ? `Failed: ${doc.error_message.slice(0, 24)}...` : 'Error'}
                              </span>
                            )}
                            <span className="text-[#929B89]">
                              • {doc.file_size_bytes < 1024 * 1024 ? `${(doc.file_size_bytes / 1024).toFixed(0)}KB` : `${(doc.file_size_bytes / 1024 / 1024).toFixed(1)}MB`}
                            </span>
                          </div>
                        </div>
                        <button
                          onClick={() => toggleDocSelection(doc.id)}
                          className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-[#10130F] border border-[#394336] hover:border-red-400 text-[#929B89] hover:text-red-400 flex items-center justify-center transition-colors cursor-pointer shadow-sm"
                          title="Detach file from chat"
                        >
                          <X size={11} />
                        </button>
                      </div>
                    ))}

                  {isUploadingDoc && selectedDocIds.length === 0 && (
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#20271F] border border-ember-copper/60 text-xs font-mono text-[#E59A64] animate-pulse">
                      <Loader2 size={13} className="animate-spin text-ember-copper" />
                      <span>Uploading & Indexing...</span>
                    </div>
                  )}

                  {/* Reranker Mode Selector */}
                  <div className="ml-auto flex items-center gap-1.5 font-mono text-[10px]">
                    <span className="text-[#929B89]">Reranker:</span>
                    {(['onnx', 'baseline'] as const).map((r) => (
                      <button
                        key={r}
                        type="button"
                        onClick={() => setSelectedReranker(r)}
                        className={`px-2 py-0.5 rounded border transition-all cursor-pointer ${
                          selectedReranker === r
                            ? 'bg-ember-copper/20 border-ember-copper text-[#E59A64] font-semibold shadow-xs'
                            : 'bg-[#10130F] border-[#394336] text-[#929B89] hover:text-[#F1EEE6] hover:border-[#A5B49A]'
                        }`}
                        title={
                          r === 'onnx'
                            ? 'ONNX Cross-Encoder (Quantized INT8 - Render Safe)'
                            : 'Baseline (OFF - Base Dense + FTS + RRF)'
                        }
                      >
                        {r === 'baseline' ? 'OFF' : r.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Main Input Controls Row */}
              <div className="flex items-center gap-3 px-3.5 py-2.5">
                {/* Attachment Paperclip Button (Direct File Chooser like ChatGPT) */}
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className={`w-9 h-9 rounded-lg border flex items-center justify-center transition-colors flex-shrink-0 cursor-pointer ${
                    selectedDocIds.length > 0
                      ? 'bg-[#20271F] border-ember-copper text-[#E59A64]'
                      : 'bg-[#20271F] border-[#394336] text-[#A5B49A] hover:text-[#F1EEE6] hover:border-ember-copper'
                  }`}
                  title="Attach document (.pdf, .md, .txt, .py, .json)"
                >
                  <Paperclip size={16} />
                </button>

                {/* Textarea Input */}
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={handleTextareaInput}
                  onKeyDown={handleKeyDown}
                  placeholder={
                    selectedDocIds.length > 0
                      ? `Ask anything grounded in ${documents.find((d) => selectedDocIds.includes(d.id))?.filename || 'attached file'}...`
                      : 'Ask Groq anything...'
                  }
                  rows={1}
                  className="flex-1 bg-transparent resize-none text-sm text-[#F1EEE6] placeholder:text-[#929B89] focus:outline-none py-1 max-h-32 font-sans select-text"
                />

                {/* Shift + Enter Hint */}
                <div className="hidden sm:flex flex-col items-end justify-center font-mono text-[10px] text-[#A5B49A] leading-tight select-none flex-shrink-0 px-1">
                  <span>Shift + Enter</span>
                  <span className="text-[#929B89]">New line</span>
                </div>

                {/* Send or Stop Action Button */}
                {isGenerating ? (
                  <button
                    onClick={handleStop}
                    className="w-10 h-10 rounded-lg bg-[#20271F] border border-red-500/80 text-red-400 hover:bg-red-950/40 flex items-center justify-center transition-colors flex-shrink-0 cursor-pointer shadow-md"
                    title="Stop generating"
                  >
                    <Square size={14} fill="currentColor" />
                  </button>
                ) : (
                  <button
                    onClick={() => handleSend()}
                    disabled={!input.trim()}
                    className={`w-10 h-10 rounded-lg flex items-center justify-center transition-all flex-shrink-0 ${
                      input.trim()
                        ? 'bg-[#E59A64] hover:bg-[#F0AB78] text-[#10130F] cursor-pointer shadow-[1px_1px_0px_0px_#A5B49A]'
                        : 'bg-[#20271F] border border-[#394336] text-ember-muted cursor-not-allowed opacity-60'
                    }`}
                    title="Send prompt"
                  >
                    <Send size={16} className={input.trim() ? 'fill-current' : ''} />
                  </button>
                )}
              </div>
            </div>
          </div>
        </footer>
      </section>

      {/* Hidden File Input for Document Uploads */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.md,.txt,.py,.json"
        className="hidden"
        onChange={handleUploadFile}
      />

      {/* =========================================================================
          DOCUMENT INTELLIGENCE (RAG) DRAWER / MODAL
      ========================================================================= */}
      {isDocDrawerOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
          <div className="w-full max-w-2xl bg-[#171C17] border border-[#394336] rounded-xl shadow-[8px_8px_0px_0px_#E59A64] flex flex-col max-h-[85vh] overflow-hidden">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-4 sm:p-5 border-b border-[#394336] bg-[#10130F]">
              <div className="flex items-center gap-2.5">
                <BookOpen size={20} className="text-[#E59A64]" />
                <div>
                  <h3 className="font-sans font-bold text-base text-[#F1EEE6] tracking-wide">
                    Document Intelligence & RAG
                  </h3>
                  <p className="font-mono text-xs text-[#A5B49A]">
                    Vector Embeddings (768d nomic-embed-text) + PostgreSQL Full-Text Search
                  </p>
                </div>
              </div>
              <button
                onClick={() => setIsDocDrawerOpen(false)}
                className="w-8 h-8 rounded-lg bg-[#20271F] border border-[#394336] text-[#A5B49A] hover:text-[#F1EEE6] flex items-center justify-center cursor-pointer transition-colors"
              >
                <X size={16} />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-4 sm:p-5 space-y-4 overflow-y-auto flex-1 font-sans">
              {/* Upload Action Card */}
              <div className="p-4 rounded-lg bg-[#20271F] border border-[#394336] flex flex-col sm:flex-row items-center justify-between gap-3">
                <div>
                  <h4 className="font-medium text-sm text-[#F1EEE6]">Upload Knowledge Document</h4>
                  <p className="font-mono text-xs text-[#929B89]">
                    Supported formats: PDF, Markdown, TXT, Safe Python AST, JSON (Max 50MB)
                  </p>
                </div>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isUploadingDoc}
                  className="w-full sm:w-auto px-4 py-2 bg-[#E59A64] hover:bg-[#F0AB78] text-[#10130F] font-semibold text-xs rounded-lg flex items-center justify-center gap-2 shadow-[2px_2px_0px_0px_#A5B49A] transition-all cursor-pointer disabled:opacity-50"
                >
                  {isUploadingDoc ? (
                    <>
                      <Loader2 size={14} className="animate-spin" />
                      <span>Vectorizing...</span>
                    </>
                  ) : (
                    <>
                      <UploadCloud size={16} />
                      <span>Upload File</span>
                    </>
                  )}
                </button>
              </div>

              {/* Ingested Documents List */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs font-mono text-[#A5B49A] px-1">
                  <span>INGESTED DOCUMENTS ({documents.length})</span>
                  <span>SELECT TO GROUND CHAT</span>
                </div>

                {documents.length === 0 ? (
                  <div className="py-8 text-center border border-dashed border-[#394336] rounded-lg">
                    <FileText size={32} className="mx-auto text-[#929B89] mb-2 opacity-50" />
                    <p className="text-xs font-mono text-[#929B89]">
                      No documents uploaded yet. Upload a file to enable grounded RAG.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
                    {documents.map((doc) => {
                      const isSelected = selectedDocIds.includes(doc.id);
                      return (
                        <div
                          key={doc.id}
                          className={`p-3 rounded-lg border transition-all flex items-center justify-between gap-3 ${
                            isSelected
                              ? 'bg-[#20271F] border-ember-copper shadow-sm'
                              : 'bg-[#10130F] border-[#394336] hover:border-[#A5B49A]/50'
                          }`}
                        >
                          {/* Checkbox & File Info */}
                          <div
                            onClick={() => toggleDocSelection(doc.id)}
                            className="flex items-center gap-3 flex-1 min-w-0 cursor-pointer"
                          >
                            <div className="text-ember-copper flex-shrink-0">
                              {isSelected ? (
                                <CheckSquare size={18} className="text-[#E59A64]" />
                              ) : (
                                <Square size={18} className="text-[#929B89]" />
                              )}
                            </div>
                            <div className="min-w-0 flex-1">
                              <div className="flex items-center gap-2">
                                <span className="font-semibold text-sm text-[#F1EEE6] truncate">
                                  {doc.filename}
                                </span>
                                <span className="px-1.5 py-0.2 rounded text-[10px] font-mono uppercase bg-[#171C17] border border-[#394336] text-[#A5B49A]">
                                  {doc.file_type}
                                </span>
                              </div>
                              <div className="flex items-center gap-3 text-xs font-mono text-[#929B89] mt-0.5">
                                <span>{doc.page_count} pages</span>
                                <span>•</span>
                                <span>{doc.chunk_count} chunks</span>
                                <span>•</span>
                                <span>
                                  {doc.file_size_bytes < 1024 * 1024 ? `${(doc.file_size_bytes / 1024).toFixed(0)} KB` : `${(doc.file_size_bytes / 1024 / 1024).toFixed(2)} MB`}
                                </span>
                              </div>
                            </div>
                          </div>

                          {/* Status Badge & Actions */}
                          <div className="flex items-center gap-3 flex-shrink-0">
                            {doc.status === 'completed' ? (
                              <span className="text-[10px] font-mono text-emerald-400 bg-emerald-950/40 border border-emerald-800/60 px-2 py-0.5 rounded">
                                Ready
                              </span>
                            ) : doc.status === 'partial' ? (
                              <span
                                className="text-[10px] font-mono text-amber-400 bg-amber-950/40 border border-amber-800/60 px-2 py-0.5 rounded cursor-help"
                                title="Partially indexed (some visual pages pending OCR credentials)"
                              >
                                Ready (Partial)
                              </span>
                            ) : doc.status === 'processing' ? (
                              <span className="text-[10px] font-mono text-ember-copper bg-amber-950/40 border border-amber-800/60 px-2 py-0.5 rounded animate-pulse">
                                Indexing...
                              </span>
                            ) : (
                              <span
                                className="text-[10px] font-mono text-red-400 bg-red-950/40 border border-red-800/60 px-2 py-0.5 rounded cursor-help"
                                title={doc.error_message || 'Indexing failed'}
                              >
                                {doc.error_message ? `Error: ${doc.error_message.slice(0, 20)}...` : 'Error'}
                              </span>
                            )}

                            <button
                              onClick={() => handleDeleteDocument(doc.id)}
                              className="text-[#929B89] hover:text-red-400 p-1 transition-colors cursor-pointer"
                              title="Delete document and chunk vectors"
                            >
                              <Trash2 size={15} />
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>

            {/* Modal Footer */}
            <div className="p-4 border-t border-[#394336] bg-[#10130F] flex items-center justify-between font-mono text-xs">
              <span className="text-[#A5B49A]">
                {selectedDocIds.length} document{selectedDocIds.length !== 1 ? 's' : ''} selected for chat grounding
              </span>
              <button
                onClick={() => setIsDocDrawerOpen(false)}
                className="px-4 py-1.5 bg-[#E59A64] hover:bg-[#F0AB78] text-[#10130F] font-semibold rounded shadow-sm cursor-pointer"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Verified Evidence & Citation Preview Drawer */}
      <CitationDrawer
        isOpen={previewDrawerOpen}
        onClose={() => setPreviewDrawerOpen(false)}
        citations={previewCitations}
        initialIndex={previewInitialIndex}
      />
    </div>
  );
}
