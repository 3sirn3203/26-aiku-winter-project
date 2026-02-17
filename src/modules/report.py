from __future__ import annotations

import datetime as dt
import html
import json
import os
from typing import Any, Dict, List, Optional


def make_report(
    report_cfg: Dict[str, Any],
    diagnose_results: List[Dict[str, Any]],
    iteration_summaries: List[Dict[str, Any]],
    run_dir: str,
    best_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report_dir = os.path.join(run_dir, str(report_cfg.get("output_dir", ""))).strip()
    if report_dir == run_dir + "/":
        report_dir = run_dir
    if not report_dir:
        report_dir = run_dir
    os.makedirs(report_dir, exist_ok=True)

    report_filename = str(report_cfg.get("report_filename", "report.html"))
    report_json_filename = str(report_cfg.get("report_json_filename", "report.json"))
    report_path = os.path.join(report_dir, report_filename)
    report_json_path = os.path.join(report_dir, report_json_filename)

    by_iter_diagnose = _index_by_iteration(diagnose_results)
    merged_rows = _merge_iteration_rows(iteration_summaries, by_iter_diagnose)

    total_iterations = len(merged_rows)
    success_count = sum(1 for row in merged_rows if bool(row.get("success", False)))
    failure_count = total_iterations - success_count

    payload = {
        "generated_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "run_dir": run_dir,
        "total_iterations": total_iterations,
        "success_count": success_count,
        "failure_count": failure_count,
        "best_summary": best_summary,
        "iterations": merged_rows,
    }
    with open(report_json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    html_text = _render_html(payload)
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(html_text)

    return {
        "report_path": report_path,
        "report_json_path": report_json_path,
        "total_iterations": total_iterations,
        "success_count": success_count,
        "failure_count": failure_count,
    }


def _index_by_iteration(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    indexed: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        try:
            key = int(row.get("iteration"))
        except Exception:
            continue
        indexed[key] = row
    return indexed


def _merge_iteration_rows(
    iteration_summaries: List[Dict[str, Any]],
    by_iter_diagnose: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for row in iteration_summaries:
        item = dict(row)
        try:
            iteration = int(item.get("iteration"))
        except Exception:
            iteration = None

        diagnose = by_iter_diagnose.get(iteration, {})
        root_cause = diagnose.get("root_cause", {}) if isinstance(diagnose, dict) else {}
        feedback = diagnose.get("feedback_for_next_iteration", {}) if isinstance(diagnose, dict) else {}
        score_summary = diagnose.get("score_summary", {}) if isinstance(diagnose, dict) else {}
        comparison = diagnose.get("comparison_to_best_before_iteration", {}) if isinstance(diagnose, dict) else {}

        item["diagnose"] = {
            "status": diagnose.get("status"),
            "root_cause_category": root_cause.get("category"),
            "root_cause_message": root_cause.get("message"),
            "score_summary": score_summary,
            "comparison_to_best_before_iteration": comparison,
            "feedback_for_next_iteration": feedback,
        }
        merged.append(item)

    merged.sort(key=lambda x: int(x.get("iteration", 0)))
    return merged


def _render_html(payload: Dict[str, Any]) -> str:
    run_dir = str(payload.get("run_dir", ""))
    generated_at = str(payload.get("generated_at_utc", ""))
    total_iterations = int(payload.get("total_iterations", 0))
    success_count = int(payload.get("success_count", 0))
    failure_count = int(payload.get("failure_count", 0))
    best_summary = payload.get("best_summary")
    iterations = payload.get("iterations", [])

    best_html = _render_best_summary(best_summary)
    rows_html = "".join(_render_iteration_row(row) for row in iterations)
    feedback_sections = "".join(_render_feedback_section(row) for row in iterations)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Run Report</title>
  <style>
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #f7f5ef 0%, #ffffff 100%);
      color: #1f2937;
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
      letter-spacing: 0.2px;
    }}
    .muted {{
      color: #4b5563;
      font-size: 14px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin: 14px 0 18px;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 12px 14px;
    }}
    .k {{
      color: #6b7280;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }}
    .v {{
      margin-top: 6px;
      font-size: 20px;
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid #f3f4f6;
      padding: 10px 8px;
      text-align: left;
      font-size: 13px;
      vertical-align: top;
    }}
    th {{
      background: #f9fafb;
      font-weight: 700;
    }}
    .ok {{
      color: #065f46;
      font-weight: 700;
    }}
    .fail {{
      color: #991b1b;
      font-weight: 700;
    }}
    .section {{
      margin-top: 24px;
    }}
    .feedback-box {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 12px 14px;
      margin-bottom: 10px;
    }}
    .pill {{
      display: inline-block;
      font-size: 11px;
      border-radius: 999px;
      padding: 2px 8px;
      border: 1px solid #d1d5db;
      margin-right: 6px;
      margin-bottom: 4px;
      color: #374151;
    }}
    ul {{
      margin: 6px 0 0 18px;
      padding: 0;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Feature Engineering Run Report</h1>
    <div class="muted">Run Dir: {html.escape(run_dir)}</div>
    <div class="muted">Generated At (UTC): {html.escape(generated_at)}</div>

    <div class="cards">
      <div class="card"><div class="k">Iterations</div><div class="v">{total_iterations}</div></div>
      <div class="card"><div class="k">Successful</div><div class="v">{success_count}</div></div>
      <div class="card"><div class="k">Failed</div><div class="v">{failure_count}</div></div>
    </div>

    {best_html}

    <div class="section">
      <h2>Iteration Summary</h2>
      <table>
        <thead>
          <tr>
            <th>Iter</th>
            <th>Status</th>
            <th>Metric</th>
            <th>Mean CV</th>
            <th>Std CV</th>
            <th>Objective</th>
            <th>Exec Attempts</th>
            <th>Fallbacks</th>
            <th>Root Cause</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>

    <div class="section">
      <h2>Diagnose Feedback</h2>
      {feedback_sections}
    </div>
  </div>
</body>
</html>
"""


def _render_best_summary(best_summary: Optional[Dict[str, Any]]) -> str:
    if not isinstance(best_summary, dict):
        return """
<div class="section">
  <h2>Best Iteration</h2>
  <div class="feedback-box">No successful iteration available.</div>
</div>
"""

    iteration = best_summary.get("iteration")
    metric = best_summary.get("metric")
    mean_cv = _fmt_float(best_summary.get("mean_cv"))
    std_cv = _fmt_float(best_summary.get("std_cv"))
    objective = _fmt_float(best_summary.get("objective_mean"))
    return f"""
<div class="section">
  <h2>Best Iteration</h2>
  <div class="feedback-box">
    <span class="pill">iteration={html.escape(str(iteration))}</span>
    <span class="pill">metric={html.escape(str(metric))}</span>
    <span class="pill">mean_cv={mean_cv}</span>
    <span class="pill">std_cv={std_cv}</span>
    <span class="pill">objective_mean={objective}</span>
  </div>
</div>
"""


def _render_iteration_row(row: Dict[str, Any]) -> str:
    status_ok = bool(row.get("success", False))
    status_class = "ok" if status_ok else "fail"
    status_text = "SUCCESS" if status_ok else "FAILED"

    diagnose = row.get("diagnose", {}) if isinstance(row.get("diagnose"), dict) else {}
    root_cause = diagnose.get("root_cause_category") or "-"
    reason = row.get("reason") or "-"

    return (
        "<tr>"
        f"<td>{html.escape(str(row.get('iteration', '-')))}</td>"
        f"<td class=\"{status_class}\">{status_text}</td>"
        f"<td>{html.escape(str(row.get('metric', '-')))}</td>"
        f"<td>{_fmt_float(row.get('mean_cv'))}</td>"
        f"<td>{_fmt_float(row.get('std_cv'))}</td>"
        f"<td>{_fmt_float(row.get('objective_mean'))}</td>"
        f"<td>{html.escape(str(row.get('execute_attempts', '-')))}</td>"
        f"<td>{html.escape(str(row.get('implement_fallback_count', '-')))}</td>"
        f"<td>{html.escape(str(root_cause))}</td>"
        f"<td>{html.escape(str(reason))}</td>"
        "</tr>"
    )


def _render_feedback_section(row: Dict[str, Any]) -> str:
    diagnose = row.get("diagnose", {}) if isinstance(row.get("diagnose"), dict) else {}
    feedback = diagnose.get("feedback_for_next_iteration", {}) if isinstance(diagnose.get("feedback_for_next_iteration"), dict) else {}

    profile_focus = _render_list(feedback.get("profile_focus", []))
    hypothesis_focus = _render_list(feedback.get("hypothesis_focus", []))
    constraints = _render_list(feedback.get("implement_constraints", []))
    actions = _render_list(feedback.get("priority_actions", []))

    root_msg = diagnose.get("root_cause_message") or "-"
    return f"""
<div class="feedback-box">
  <div><strong>Iteration {html.escape(str(row.get("iteration", "-")))}</strong></div>
  <div class="muted">Root cause: {html.escape(str(root_msg))}</div>
  <div><strong>Profile focus</strong>{profile_focus}</div>
  <div><strong>Hypothesis focus</strong>{hypothesis_focus}</div>
  <div><strong>Implement constraints</strong>{constraints}</div>
  <div><strong>Priority actions</strong>{actions}</div>
</div>
"""


def _render_list(values: Any) -> str:
    if not isinstance(values, list) or not values:
        return "<ul><li>-</li></ul>"
    items = "".join(f"<li>{html.escape(str(v))}</li>" for v in values)
    return f"<ul>{items}</ul>"


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except Exception:
        return "-"
