import { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { BookOpen } from 'lucide-react';
import { CodeBlock } from './CodeBlock';

interface MarkdownRendererProps {
  content: string;
  onCitationClick?: (label: string) => void;
}

export function MarkdownRenderer({ content, onCitationClick }: MarkdownRendererProps) {
  // Pre-process citations like [Page 1] or [Page 15 (PDF p. 21)] into clickable markdown links
  const processedContent = useMemo(() => {
    if (!content) return '';
    // Normalize unicode non-breaking and thin spaces
    const normalized = content.replace(/[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]/g, ' ');
    // Match [Page 1], [Page 1 (PDF p. 2)], [p. 15], [Source 1], etc.
    return normalized.replace(/\[((?:[^\]\n]*\b)?(?:Page|p\.|Source)\s*[0-9]+[^\]\n]*)\]/gi, (_match, label) => {
      const cleanLabel = label.trim();
      return `[${cleanLabel}](#cite-${encodeURIComponent(cleanLabel)})`;
    });
  }, [content]);

  return (
    <div className="prose prose-invert max-w-none text-[#F1EEE6] select-text font-sans">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            const codeString = String(children);
            const isMultiline = codeString.includes('\n');

            // If it has language tag or is multiline, treat as a code block with syntax highlighting
            if (match || isMultiline) {
              return (
                <CodeBlock
                  code={codeString.replace(/\n$/, '')}
                  language={match ? match[1] : 'text'}
                />
              );
            }

            // Inline code badge
            return (
              <code
                className="bg-[#20271F] text-[#A5B49A] font-mono text-[12px] px-1.5 py-0.5 rounded border border-[#394336] select-text"
                {...props}
              >
                {children}
              </code>
            );
          },
          p({ children }) {
            return <p className="my-2.5 leading-relaxed text-[#F1EEE6] select-text">{children}</p>;
          },
          h1({ children }) {
            return (
              <h1 className="font-serif text-2xl font-normal text-[#F1EEE6] my-4 pb-1.5 border-b border-[#394336] select-text">
                {children}
              </h1>
            );
          },
          h2({ children }) {
            return (
              <h2 className="font-serif text-xl font-normal text-[#F1EEE6] my-3.5 pb-1 border-b border-[#394336]/60 select-text">
                {children}
              </h2>
            );
          },
          h3({ children }) {
            return (
              <h3 className="font-sans text-base font-semibold text-[#E59A64] my-2.5 select-text">
                {children}
              </h3>
            );
          },
          ul({ children }) {
            return (
              <ul className="my-2.5 ml-5 list-disc space-y-1.5 text-[#F1EEE6] select-text marker:text-ember-copper">
                {children}
              </ul>
            );
          },
          ol({ children }) {
            return (
              <ol className="my-2.5 ml-5 list-decimal space-y-1.5 text-[#F1EEE6] select-text marker:text-ember-copper font-mono text-sm">
                {children}
              </ol>
            );
          },
          li({ children }) {
            return <li className="leading-relaxed text-[#F1EEE6] select-text">{children}</li>;
          },
          blockquote({ children }) {
            return (
              <blockquote className="border-l-2 border-[#A5B49A] pl-4 my-3 italic text-[#B7BDAE] bg-[#171C17]/60 py-1.5 rounded-r select-text">
                {children}
              </blockquote>
            );
          },
          a({ href, children }) {
            if (href && href.startsWith('#cite-')) {
              const label = decodeURIComponent(href.replace(/^#cite-/, ''));
              return (
                <button
                  type="button"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    onCitationClick?.(label);
                  }}
                  className="inline-flex items-center gap-1 mx-1 px-1.5 py-0.5 rounded font-mono text-[11px] bg-[#20271F] border border-ember-copper/70 text-[#E59A64] hover:bg-ember-copper hover:text-[#10130F] transition-all cursor-pointer font-semibold shadow-xs align-baseline"
                  title={`Click to preview verified source for ${label}`}
                >
                  <BookOpen size={10} className="inline flex-shrink-0" />
                  <span>{children}</span>
                </button>
              );
            }

            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#E59A64] hover:text-[#F0AB78] underline underline-offset-2 transition-colors font-medium select-text"
              >
                {children}
              </a>
            );
          },
          table({ children }) {
            return (
              <div className="overflow-x-auto my-4 rounded-lg border border-[#394336] bg-[#131713] select-text">
                <table className="min-w-full text-left text-xs sm:text-sm border-collapse select-text">
                  {children}
                </table>
              </div>
            );
          },
          thead({ children }) {
            return <thead className="bg-[#20271F] border-b border-[#394336]">{children}</thead>;
          },
          th({ children }) {
            return (
              <th className="px-3.5 py-2.5 text-[#E59A64] font-mono font-semibold text-xs uppercase tracking-wider border-r border-[#394336]/60 last:border-r-0">
                {children}
              </th>
            );
          },
          td({ children }) {
            return (
              <td className="px-3.5 py-2.5 border-b border-[#394336]/40 border-r border-[#394336]/40 last:border-r-0 text-[#F1EEE6] text-xs sm:text-sm leading-relaxed">
                {children}
              </td>
            );
          },
          strong({ children }) {
            return <strong className="font-semibold text-[#E59A64] select-text">{children}</strong>;
          },
          hr() {
            return <hr className="my-5 border-[#394336]" />;
          },
        }}
      >
        {processedContent}
      </ReactMarkdown>
    </div>
  );
}
