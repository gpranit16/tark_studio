import { useState, useMemo } from 'react';
import { Copy, Check } from 'lucide-react';
import Prism from 'prismjs';

// Import essential language components for Prism
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-typescript';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-bash';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-sql';
import 'prismjs/components/prism-markdown';
import 'prismjs/components/prism-yaml';
import 'prismjs/components/prism-css';

interface CodeBlockProps {
  code: string;
  language?: string;
}

export function CodeBlock({ code, language = 'text' }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const cleanLang = (language || 'text').toLowerCase().replace('language-', '');

  const highlightedHtml = useMemo(() => {
    const grammar = Prism.languages[cleanLang] || Prism.languages.text || null;
    if (grammar) {
      try {
        return Prism.highlight(code, grammar, cleanLang);
      } catch {
        // Fallback to text escaping
      }
    }
    return code
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }, [code, cleanLang]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
      const textarea = document.createElement('textarea');
      textarea.value = code;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="my-4 rounded-lg border border-[#394336] bg-[#121612] overflow-hidden shadow-md select-text">
      {/* Code Block Header */}
      <div className="flex items-center justify-between px-3.5 py-1.5 bg-[#171C17] border-b border-[#394336] text-xs font-mono select-none">
        <span className="text-[#A5B49A] font-semibold uppercase tracking-wider text-[11px]">
          {cleanLang}
        </span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 py-1 px-2.5 rounded bg-[#20271F] hover:bg-[#273126] border border-[#394336] hover:border-[#A5B49A] text-[#B7BDAE] hover:text-[#F1EEE6] transition-colors cursor-pointer text-[11px]"
          title="Copy code to clipboard"
        >
          {copied ? (
            <>
              <Check size={12} className="text-[#A5B49A]" />
              <span className="text-[#A5B49A]">Copied</span>
            </>
          ) : (
            <>
              <Copy size={12} />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Code Block Body with Syntax Highlighting */}
      <pre className="p-4 overflow-x-auto font-mono text-xs sm:text-[13px] leading-relaxed text-[#F1EEE6] bg-[#10130F]/90 select-text">
        <code
          className={`language-${cleanLang} select-text`}
          dangerouslySetInnerHTML={{ __html: highlightedHtml }}
        />
      </pre>
    </div>
  );
}
