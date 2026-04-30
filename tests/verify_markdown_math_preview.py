from __future__ import annotations

from pyside_app.markdown_preview import build_markdown_preview_html, katex_assets_dir


def main() -> None:
    print("[verify][markdown-math] start", flush=True)
    assets = katex_assets_dir()
    print(f"[verify][markdown-math] assets_dir={assets}", flush=True)
    print(f"[verify][markdown-math] css_exists={(assets / 'katex.min.css').exists()}", flush=True)
    print(f"[verify][markdown-math] js_exists={(assets / 'katex.min.js').exists()}", flush=True)
    print(f"[verify][markdown-math] auto_render_exists={(assets / 'auto-render.min.js').exists()}", flush=True)
    html = build_markdown_preview_html("Inline $x^2$ and block $$\\\\int_0^1 x^2 dx$$")
    print(f"[verify][markdown-math] html_len={len(html)}", flush=True)
    print(f"[verify][markdown-math] has_render_hook={'renderMathInElement' in html}", flush=True)
    print(f"[verify][markdown-math] has_inline_math={'$x^2$' in html}", flush=True)
    print(f"[verify][markdown-math] has_katex_css={'katex.min.css' in html}", flush=True)
    print(f"[verify][markdown-math] has_auto_render={'auto-render.min.js' in html}", flush=True)
    print("[verify][markdown-math] PASS", flush=True)


if __name__ == "__main__":
    main()
