"""
Generate self-contained static HTML EDA report.
Usage:  python generate_eda_html.py
Output: eda_report/index.html
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_raw
from src.eda import generate_all_figures, CHART_META
from src.config import EDA_REPORT_DIR
import pandas as pd


def _build_nav(items: list[dict]) -> str:
    links = []
    for item in items:
        links.append(
            f'<a href="#{item["id"]}" onclick="scrollTo(\'{item["id"]}\')">'
            f'{item["label"]}</a>'
        )
    return "\n".join(links)


def _build_kpi_cards(df: pd.DataFrame) -> str:
    survived_pct = df["Survived"].mean() * 100
    missing_age_pct = df["Age"].isna().mean() * 100
    return f"""
    <div class="stats-grid">
      <div class="stat-card">
        <div class="value">{len(df)}</div>
        <div class="label">Training Samples</div>
      </div>
      <div class="stat-card">
        <div class="value">{survived_pct:.1f}%</div>
        <div class="label">Overall Survival Rate</div>
      </div>
      <div class="stat-card">
        <div class="value">{df.shape[1]}</div>
        <div class="label">Original Features</div>
      </div>
      <div class="stat-card">
        <div class="value">{missing_age_pct:.1f}%</div>
        <div class="label">Age Missing Rate</div>
      </div>
      <div class="stat-card">
        <div class="value">{int(df["Survived"].sum())}</div>
        <div class="label">Survived</div>
      </div>
      <div class="stat-card">
        <div class="value">{int((df["Survived"] == 0).sum())}</div>
        <div class="label">Died</div>
      </div>
    </div>"""


def build_html(df: pd.DataFrame) -> str:
    print("Generating EDA figures...")
    items = generate_all_figures(df)

    # First figure gets the full Plotly.js bundle; rest are minimal divs
    chart_sections = []
    for i, item in enumerate(items):
        include_js = (i == 0)
        div_html = item["fig"].to_html(
            full_html=False,
            include_plotlyjs=include_js,
            config={"responsive": True, "displayModeBar": True,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        )
        section = f"""
        <div class="chart-section" id="{item['id']}">
          <h2>{item['label']}</h2>
          <div class="chart-container">{div_html}</div>
          {'<div class="insight"><b>Key Insight:</b> ' + item["insight"] + '</div>' if item["insight"] else ''}
        </div>"""
        chart_sections.append(section)

    nav_html = _build_nav(items)
    kpi_html = _build_kpi_cards(df)
    charts_html = "\n".join(chart_sections)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Titanic EDA Report</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --accent: #38bdf8;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --survived: #38bdf8;
      --died: #f87171;
      --nav-width: 230px;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
    }}
    /* ── Sidebar Navigation ── */
    nav {{
      position: fixed;
      left: 0; top: 0;
      width: var(--nav-width);
      height: 100vh;
      background: var(--card);
      border-right: 1px solid var(--border);
      overflow-y: auto;
      padding: 1.5rem 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
      z-index: 100;
    }}
    .nav-title {{
      font-size: 1rem;
      font-weight: 700;
      color: var(--accent);
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 0.75rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}
    nav a {{
      display: block;
      padding: 0.4rem 0.75rem;
      color: var(--muted);
      text-decoration: none;
      border-radius: 6px;
      font-size: 0.83rem;
      transition: all 0.15s;
      cursor: pointer;
    }}
    nav a:hover, nav a.active {{
      background: rgba(56,189,248,0.12);
      color: var(--accent);
    }}
    /* ── Main Content ── */
    main {{
      margin-left: var(--nav-width);
      padding: 2rem 2.5rem;
      max-width: calc(100vw - var(--nav-width));
      width: 100%;
    }}
    .page-header {{
      margin-bottom: 2rem;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid var(--border);
    }}
    .page-header h1 {{
      font-size: 1.75rem;
      font-weight: 700;
      color: var(--accent);
      margin-bottom: 0.5rem;
    }}
    .page-header p {{
      color: var(--muted);
      font-size: 0.9rem;
    }}
    /* ── KPI Cards ── */
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 1rem;
      margin-bottom: 2.5rem;
    }}
    .stat-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
      text-align: center;
      transition: transform 0.2s;
    }}
    .stat-card:hover {{ transform: translateY(-2px); }}
    .stat-card .value {{
      font-size: 1.75rem;
      font-weight: 800;
      color: var(--accent);
      line-height: 1;
    }}
    .stat-card .label {{
      color: var(--muted);
      font-size: 0.78rem;
      margin-top: 0.4rem;
    }}
    /* ── Chart Sections ── */
    .chart-section {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.75rem;
      margin-bottom: 2rem;
      scroll-margin-top: 20px;
    }}
    .chart-section h2 {{
      font-size: 1.1rem;
      font-weight: 600;
      color: var(--accent);
      margin-bottom: 1.25rem;
      padding-bottom: 0.75rem;
      border-bottom: 1px solid var(--border);
    }}
    .chart-container {{
      width: 100%;
    }}
    .insight {{
      background: rgba(56,189,248,0.08);
      border-left: 3px solid var(--accent);
      padding: 0.85rem 1.1rem;
      margin-top: 1.1rem;
      border-radius: 0 8px 8px 0;
      font-size: 0.88rem;
      color: var(--text);
      line-height: 1.6;
    }}
    .insight b {{ color: var(--accent); }}
    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg); }}
    ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
    /* ── Responsive ── */
    @media (max-width: 768px) {{
      nav {{ display: none; }}
      main {{ margin-left: 0; padding: 1rem; max-width: 100vw; }}
    }}
  </style>
</head>
<body>
  <nav>
    <div class="nav-title">
      <span>🚢</span> Titanic EDA
    </div>
    {nav_html}
  </nav>

  <main>
    <div class="page-header">
      <h1>🚢 Titanic Dataset — EDA Report</h1>
      <p>Kaggle Titanic Competition · Training set: 891 passengers · Test set: 418 passengers</p>
    </div>

    {kpi_html}

    {charts_html}

    <div style="text-align:center; color:var(--muted); font-size:0.8rem; padding: 2rem 0;">
      Generated with Python · Plotly · scikit-learn
    </div>
  </main>

  <script>
    function scrollTo(id) {{
      const el = document.getElementById(id);
      if (el) el.scrollIntoView({{behavior: 'smooth', block: 'start'}});
    }}

    // Highlight active nav item on scroll
    const sections = document.querySelectorAll('.chart-section[id]');
    const navLinks = document.querySelectorAll('nav a');
    const observer = new IntersectionObserver((entries) => {{
      entries.forEach(e => {{
        if (e.isIntersecting) {{
          navLinks.forEach(a => a.classList.remove('active'));
          const active = document.querySelector(`nav a[href="#${{e.target.id}}"]`);
          if (active) active.classList.add('active');
        }}
      }});
    }}, {{threshold: 0.3}});
    sections.forEach(s => observer.observe(s));
  </script>
</body>
</html>"""

    return html


def main():
    train, _ = load_raw()
    EDA_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building HTML EDA report...")
    html = build_html(train)
    output_path = EDA_REPORT_DIR / "index.html"
    output_path.write_text(html, encoding="utf-8")
    size_kb = output_path.stat().st_size / 1024
    print(f"\nEDA report saved → {output_path}")
    print(f"File size: {size_kb:.0f} KB")
    print(f"\nOpen in browser: open {output_path}")


if __name__ == "__main__":
    main()
