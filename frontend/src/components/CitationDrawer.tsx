import { useEffect, useState } from 'react';
import {
  X,
  BookOpen,
  FileText,
  Globe,
  ExternalLink,
  Copy,
  Check,
  ChevronLeft,
  ChevronRight,
  Hash,
  ShieldCheck,
  Layers,
} from 'lucide-react';
import type { CitationItem } from '../App';

interface CitationDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  citations: CitationItem[];
  initialIndex?: number;
}

export function CitationDrawer({
  isOpen,
  onClose,
  citations,
  initialIndex = 0,
}: CitationDrawerProps) {
  const [currentIndex, setCurrentIndex] = useState(initialIndex);
  const [copied, setCopied] = useState(false);

  // Sync index when initialIndex changes or citations change
  useEffect(() => {
    if (initialIndex >= 0 && initialIndex < citations.length) {
      setCurrentIndex(initialIndex);
    } else {
      setCurrentIndex(0);
    }
  }, [initialIndex, citations]);

  // Handle ESC key to close
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;
      if (e.key === 'Escape') {
        onClose();
      } else if (e.key === 'ArrowLeft') {
        setCurrentIndex((prev) => (prev > 0 ? prev - 1 : prev));
      } else if (e.key === 'ArrowRight') {
        setCurrentIndex((prev) => (prev < citations.length - 1 ? prev + 1 : prev));
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, citations.length, onClose]);

  if (!isOpen || citations.length === 0) return null;

  const currentCitation = citations[currentIndex] || citations[0];

  const handleCopyExcerpt = async () => {
    if (!currentCitation?.excerpt) return;
    try {
      await navigator.clipboard.writeText(currentCitation.excerpt);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy excerpt', err);
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden flex justify-end animate-in fade-in duration-200">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/70 backdrop-blur-xs transition-opacity cursor-pointer"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Slide-over Drawer Panel */}
      <div className="relative w-full max-w-xl md:max-w-2xl bg-[#171C17] border-l border-[#394336] shadow-[-12px_0_32px_rgba(0,0,0,0.8)] h-full z-10 flex flex-col font-sans">
        {/* Top Header */}
        <div className="p-4 sm:p-5 border-b border-[#394336] bg-[#121511] flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded bg-[#20271F] border border-[#394336] flex items-center justify-center text-ember-copper shadow-xs">
              <BookOpen size={16} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-serif text-lg text-[#F1EEE6] font-normal tracking-wide">
                  Evidence & Grounding
                </h3>
                <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono bg-[#20271F] text-[#A5B49A] border border-[#394336]">
                  <ShieldCheck size={11} className="text-[#A5B49A]" /> Verified
                </span>
              </div>
              <p className="text-[11px] font-mono text-[#929B89]">
                Original extracted document passage used for synthesis
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="w-8 h-8 rounded flex items-center justify-center text-[#929B89] hover:text-[#F1EEE6] hover:bg-[#20271F] border border-transparent hover:border-[#394336] transition-colors cursor-pointer"
            title="Close drawer (Esc)"
          >
            <X size={18} />
          </button>
        </div>

        {/* Source Navigation Tabs (if multiple citations) */}
        {citations.length > 1 && (
          <div className="px-5 py-2.5 bg-[#171C17] border-b border-[#394336] flex items-center justify-between gap-2 overflow-x-auto">
            <div className="flex items-center gap-1.5 overflow-x-auto py-0.5">
              {citations.map((c, idx) => (
                <button
                  key={idx}
                  onClick={() => setCurrentIndex(idx)}
                  className={`px-2.5 py-1 rounded text-[11px] font-mono whitespace-nowrap border transition-all cursor-pointer ${
                    currentIndex === idx
                      ? 'bg-[#20271F] border-ember-copper text-[#E59A64] font-semibold shadow-xs'
                      : 'bg-[#10130F] border-[#394336] text-[#929B89] hover:text-[#F1EEE6]'
                  }`}
                >
                  {c.citation_label || `Source ${idx + 1}`}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-1 flex-shrink-0">
              <button
                disabled={currentIndex === 0}
                onClick={() => setCurrentIndex((prev) => prev - 1)}
                className="w-7 h-7 rounded flex items-center justify-center bg-[#20271F] border border-[#394336] text-[#F1EEE6] disabled:opacity-40 disabled:cursor-not-allowed hover:border-ember-copper transition-colors cursor-pointer"
                title="Previous Source"
              >
                <ChevronLeft size={14} />
              </button>
              <span className="font-mono text-[11px] text-[#A5B49A] px-1">
                {currentIndex + 1}/{citations.length}
              </span>
              <button
                disabled={currentIndex === citations.length - 1}
                onClick={() => setCurrentIndex((prev) => prev + 1)}
                className="w-7 h-7 rounded flex items-center justify-center bg-[#20271F] border border-[#394336] text-[#F1EEE6] disabled:opacity-40 disabled:cursor-not-allowed hover:border-ember-copper transition-colors cursor-pointer"
                title="Next Source"
              >
                <ChevronRight size={14} />
              </button>
            </div>
          </div>
        )}

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          {/* Metadata Card */}
          <div className="p-4 rounded-lg bg-[#20271F] border border-[#394336] space-y-2.5 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                {currentCitation.source_type === 'web' ? (
                  <Globe size={15} className="text-[#E59A64]" />
                ) : (
                  <FileText size={15} className="text-[#E59A64]" />
                )}
                <span className="font-mono text-xs font-semibold text-[#F1EEE6] truncate max-w-xs">
                  {currentCitation.document_name}
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                {currentCitation.source_type === 'web' && currentCitation.url ? (
                  <a
                    href={currentCitation.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono bg-[#10130F] border border-ember-copper/60 text-[#E59A64] hover:bg-ember-copper hover:text-[#10130F] transition-colors font-medium"
                  >
                    <span>Open Source</span>
                    <ExternalLink size={11} />
                  </a>
                ) : (
                  <>
                    <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-[#10130F] border border-ember-copper/60 text-[#E59A64] font-medium">
                      {currentCitation.book_page
                        ? `Page ${currentCitation.book_page}`
                        : `Page ${currentCitation.viewer_page}`}
                    </span>
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#10130F] border border-[#394336] text-[#929B89]">
                      PDF p. {currentCitation.viewer_page}
                    </span>
                  </>
                )}
              </div>
            </div>

            {currentCitation.heading && (
              <div className="flex items-center gap-1.5 text-xs text-[#A5B49A] font-mono pt-1 border-t border-[#394336]/60">
                <Hash size={12} className="text-ember-copper flex-shrink-0" />
                <span className="truncate">{currentCitation.heading}</span>
              </div>
            )}

            <div className="flex items-center gap-2 pt-1 text-[10px] font-mono text-[#929B89]">
              <Layers size={11} className="text-[#A5B49A]" />
              <span>
                {currentCitation.source_type === 'web'
                  ? 'Pipeline: LLM Intent Router + Tavily Web Search (Sanitized Grounding)'
                  : 'Pipeline: nomic-embed-text (768d) + PostgreSQL FTS (RRF k=60)'}
              </span>
            </div>
          </div>

          {/* Excerpt Container */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-semibold text-[#A5B49A] uppercase tracking-wider">
                Extracted Document Excerpt
              </span>
              <span className="font-mono text-[10px] text-[#929B89]">
                {currentCitation.excerpt?.length || 0} characters
              </span>
            </div>

            <div className="relative rounded-lg bg-[#10130F] border border-[#394336] p-4 sm:p-5 text-sm text-[#F1EEE6] leading-relaxed font-sans whitespace-pre-wrap select-text selection:bg-ember-copper/30 max-h-[50vh] overflow-y-auto">
              {currentCitation.excerpt ? (
                currentCitation.excerpt
              ) : (
                <div className="text-xs text-[#929B89] italic font-mono py-6 text-center">
                  No textual excerpt available for this chunk.
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="p-4 sm:p-5 border-t border-[#394336] bg-[#121511] flex items-center justify-between gap-3">
          <div className="text-[11px] font-mono text-[#929B89]">
            Source label:{' '}
            <span className="text-[#E59A64] font-semibold">
              {currentCitation.citation_label}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopyExcerpt}
              disabled={!currentCitation.excerpt}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-mono font-medium bg-[#20271F] hover:bg-[#2a3329] border border-[#394336] hover:border-ember-copper text-[#F1EEE6] transition-colors cursor-pointer disabled:opacity-40"
              title="Copy excerpt to clipboard"
            >
              {copied ? (
                <>
                  <Check size={13} className="text-[#A5B49A]" />
                  <span>Copied!</span>
                </>
              ) : (
                <>
                  <Copy size={13} className="text-ember-copper" />
                  <span>Copy Excerpt</span>
                </>
              )}
            </button>

            <button
              onClick={onClose}
              className="px-4 py-1.5 rounded text-xs font-mono font-semibold bg-[#E59A64] hover:bg-[#F0AB78] text-[#10130F] transition-colors cursor-pointer shadow-sm"
            >
              Done
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
