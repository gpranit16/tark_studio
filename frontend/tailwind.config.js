/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ember: {
          main: '#10130F',
          sidebar: '#171C17',
          elevated: '#20271F',
          border: '#394336',
          primary: '#F1EEE6',
          secondary: '#B7BDAE',
          muted: '#929B89',
          sage: '#A5B49A',
          copper: '#E59A64',
          hover: '#F0AB78',
        },
      },
      fontFamily: {
        serif: ['"Instrument Serif"', 'Georgia', 'serif'],
        sans: ['"Instrument Sans"', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'Menlo', 'monospace'],
      },
      borderRadius: {
        DEFAULT: '2px',
        sm: '1px',
        md: '3px',
        lg: '4px',
      },
    },
  },
  plugins: [],
};
