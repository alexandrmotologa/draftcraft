"""FastAPI application entrypoint and dashboard rendering for DraftCraft."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from draftcraft import __version__
from draftcraft.web.routes import router

app = FastAPI(
    title="DraftCraft Speculative Decoding Studio",
    description="Turnkey speculative decoding matrix evaluator and EAGLE draft head trainer.",
    version=__version__,
)

app.include_router(router)

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DraftCraft - Speculative Decoding & EAGLE Studio</title>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111827;
      --border: #1f293d;
      --text: #f3f4f6;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --accent: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      line-height: 1.5;
      padding: 24px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }
    .logo-group { display: flex; align-items: center; gap: 12px; }
    .badge {
      display: inline-block;
      font-size: 11px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 9999px;
      background: rgba(56, 189, 248, 0.15);
      color: var(--primary);
      border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .nav-tabs {
      display: flex;
      gap: 8px;
      margin-bottom: 24px;
    }
    .tab-btn {
      background: var(--card-bg);
      color: var(--text-muted);
      border: 1px solid var(--border);
      padding: 10px 18px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 500;
      font-size: 14px;
      transition: all 0.2s;
    }
    .tab-btn:hover { color: var(--text); border-color: var(--primary); }
    .tab-btn.active {
      background: rgba(56, 189, 248, 0.1);
      color: var(--primary);
      border-color: var(--primary);
    }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    .grid { display: grid; grid-template-columns: 360px 1fr; gap: 24px; }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
    }
    .card h3 {
      font-size: 16px;
      font-weight: 600;
      margin-bottom: 16px;
      color: var(--primary);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .form-group { margin-bottom: 16px; }
    .form-group label {
      display: block;
      font-size: 12px;
      font-weight: 500;
      color: var(--text-muted);
      margin-bottom: 6px;
    }
    .form-control {
      width: 100%;
      background: #0b1120;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 13px;
    }
    .form-control:focus { outline: none; border-color: var(--primary); }
    .slider-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      margin-top: 4px;
      color: var(--primary);
    }
    .btn {
      background: var(--primary);
      color: #0b1120;
      border: none;
      padding: 10px 18px;
      border-radius: 6px;
      font-weight: 600;
      cursor: pointer;
      width: 100%;
      font-size: 14px;
      transition: opacity 0.2s;
    }
    .btn:hover { opacity: 0.9; }
    .kpi-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }
    .kpi-card {
      background: #0b1120;
      border: 1px solid var(--border);
      padding: 14px 18px;
      border-radius: 8px;
    }
    .kpi-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }
    .kpi-val { font-size: 24px; font-weight: 700; color: var(--text); margin-top: 4px; }
    .kpi-sub { font-size: 11px; color: var(--accent); margin-top: 2px; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }
    th, td {
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
    }
    th {
      background: #0b1120;
      color: var(--text-muted);
      font-weight: 600;
      font-size: 12px;
    }
    .optimal-tag {
      background: rgba(16, 185, 129, 0.15);
      color: var(--accent);
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 700;
    }
    .chart-container {
      background: #0b1120;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      margin-top: 16px;
    }
  </style>
</head>
<body>
  <header>
    <div class="logo-group">
      <h1 style="font-size: 22px; font-weight: 700; letter-spacing: -0.5px;">DraftCraft</h1>
      <span class="badge">EAGLE & Speculative Engine</span>
    </div>
    <div style="font-size: 12px; color: var(--text-muted);">
      Inference Acceleration Suite | Local LLM Turnkey Evaluator
    </div>
  </header>

  <div class="nav-tabs">
    <button class="tab-btn active" onclick="switchTab('tab-matrix')">Matrix Evaluator</button>
    <button class="tab-btn" onclick="switchTab('tab-tokenizer')">Tokenizer Verifier</button>
    <button class="tab-btn" onclick="switchTab('tab-profiler')">Empirical Profiler</button>
    <button class="tab-btn" onclick="switchTab('tab-trainer')">EAGLE Head Trainer</button>
  </div>

  <!-- TAB 1: MATRIX EVALUATOR -->
  <div id="tab-matrix" class="tab-content active">
    <div class="grid">
      <div class="card">
        <h3>Configuration Parameters</h3>
        <div class="form-group">
          <label>Target LLM Latency per Token (ms)</label>
          <input type="number" id="mat-tl" class="form-control" value="28.0" step="0.5" oninput="runMatrixEval()">
        </div>
        <div class="form-group">
          <label>Draft Latency per Token (ms)</label>
          <input type="number" id="mat-dl" class="form-control" value="3.5" step="0.1" oninput="runMatrixEval()">
        </div>
        <div class="form-group">
          <label>Base Acceptance Rate (alpha)</label>
          <input type="range" id="mat-alpha" min="0.4" max="0.95" step="0.01" value="0.75" class="form-control" oninput="updateAlphaSlider(); runMatrixEval();">
          <div class="slider-row">
            <span>Acceptance:</span>
            <span id="mat-alpha-val">75%</span>
          </div>
        </div>
        <div class="form-group">
          <label>Domain Evaluation Category</label>
          <select id="mat-domain" class="form-control" onchange="runMatrixEval()">
            <option value="general">General Natural Language</option>
            <option value="code">Code & Syntax Predictable</option>
            <option value="reasoning">Math & Reasoning</option>
            <option value="factual_qa">Factual QA</option>
            <option value="creative">Creative Prose</option>
          </select>
        </div>
        <button class="btn" onclick="runMatrixEval()">Re-evaluate Matrix</button>
      </div>

      <div>
        <div class="kpi-row">
          <div class="kpi-card">
            <div class="kpi-label">Peak Speedup</div>
            <div class="kpi-val" id="kpi-speedup" style="color: var(--accent);">3.22x</div>
            <div class="kpi-sub">Pareto Optimal</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Recommended Depth (K*)</div>
            <div class="kpi-val" id="kpi-depth">5</div>
            <div class="kpi-sub">Lookahead Steps</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Optimal Width (W*)</div>
            <div class="kpi-val" id="kpi-width">4</div>
            <div class="kpi-sub">Tree Speculation</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Exp. Tokens / Step</div>
            <div class="kpi-val" id="kpi-tokens">5.29</div>
            <div class="kpi-sub">Accepted + Fallback</div>
          </div>
        </div>

        <div class="card">
          <h3>Speculative Decoding Matrix Points</h3>
          <div style="overflow-x: auto; max-height: 400px;">
            <table id="matrix-table">
              <thead>
                <tr>
                  <th>Depth (K)</th>
                  <th>Speculation Strategy</th>
                  <th>Latency Ratio</th>
                  <th>Exp. Tokens</th>
                  <th>Speedup</th>
                  <th>Pareto Status</th>
                </tr>
              </thead>
              <tbody id="matrix-tbody">
                <!-- Dynamically populated -->
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 2: TOKENIZER VERIFIER -->
  <div id="tab-tokenizer" class="tab-content">
    <div class="grid">
      <div class="card">
        <h3>Tokenizer Compatibility Check</h3>
        <div class="form-group">
          <label>Target LLM Vocab Size</label>
          <input type="number" id="tok-target-size" class="form-control" value="128256">
        </div>
        <div class="form-group">
          <label>Draft Model / Head Vocab Size</label>
          <input type="number" id="tok-draft-size" class="form-control" value="128256">
        </div>
        <div class="form-group">
          <label>Target Hidden State Dimension</label>
          <input type="number" id="tok-target-dim" class="form-control" value="4096">
        </div>
        <div class="form-group">
          <label>Draft Head Hidden Dimension</label>
          <input type="number" id="tok-head-dim" class="form-control" value="1024">
        </div>
        <button class="btn" onclick="runTokenizerVerify()">Verify Alignment</button>
      </div>

      <div class="card">
        <h3>Verification Report</h3>
        <div id="tok-report-content" style="font-size: 13px; color: var(--text-muted);">
          Click "Verify Alignment" to inspect vocabulary compatibility.
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 3: PROFILER -->
  <div id="tab-profiler" class="tab-content">
    <div class="grid">
      <div class="card">
        <h3>Empirical Profiling Sweep</h3>
        <div class="form-group">
          <label>Target Latency (ms)</label>
          <input type="number" id="prof-tl" class="form-control" value="28.0">
        </div>
        <div class="form-group">
          <label>Draft Latency (ms)</label>
          <input type="number" id="prof-dl" class="form-control" value="3.5">
        </div>
        <div class="form-group">
          <label>Lookahead Depth K</label>
          <input type="number" id="prof-k" class="form-control" value="4" min="1" max="8">
        </div>
        <div class="form-group">
          <label>Acceptance Rate (alpha)</label>
          <input type="number" id="prof-alpha" class="form-control" value="0.75" step="0.05" min="0.1" max="0.99">
        </div>
        <button class="btn" onclick="runProfiler()">Run Empirical Profiler</button>
      </div>

      <div class="card">
        <h3>Empirical Benchmark Profile</h3>
        <div id="prof-results" style="font-size: 13px; color: var(--text-muted);">
          Ready to profile. Click above to execute verification cycle sweep.
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 4: TRAINER -->
  <div id="tab-trainer" class="tab-content">
    <div class="grid">
      <div class="card">
        <h3>Train Local EAGLE Head</h3>
        <div class="form-group">
          <label>Epochs</label>
          <input type="number" id="tr-epochs" class="form-control" value="2" min="1" max="10">
        </div>
        <div class="form-group">
          <label>Batch Size</label>
          <input type="number" id="tr-batch" class="form-control" value="8" min="2" max="64">
        </div>
        <div class="form-group">
          <label>Learning Rate</label>
          <input type="number" id="tr-lr" class="form-control" value="0.001" step="0.0001">
        </div>
        <div class="form-group">
          <label>Draft Head Hidden Dim</label>
          <input type="number" id="tr-hidden" class="form-control" value="128">
        </div>
        <div class="form-group">
          <label>Target LLM Dim</label>
          <input type="number" id="tr-target-dim" class="form-control" value="256">
        </div>
        <button class="btn" id="tr-start-btn" onclick="startTraining()">Start Training Run</button>
      </div>

      <div class="card">
        <h3>Training Telemetry & Live Loss</h3>
        <div id="tr-status-box" style="font-size: 13px; color: var(--text-muted);">
          Draft head trainer ready. Click "Start Training Run" to train in background.
        </div>
      </div>
    </div>
  </div>

  <script>
    function switchTab(tabId) {
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      event.target.classList.add('active');
    }

    function updateAlphaSlider() {
      const val = Math.round(document.getElementById('mat-alpha').value * 100);
      document.getElementById('mat-alpha-val').innerText = val + '%';
    }

    async function runMatrixEval() {
      const payload = {
        domain: document.getElementById('mat-domain').value,
        alpha: parseFloat(document.getElementById('mat-alpha').value),
        draft_latency_ms: parseFloat(document.getElementById('mat-dl').value) || 3.5,
        target_latency_ms: parseFloat(document.getElementById('mat-tl').value) || 28.0,
        min_lookahead: 1,
        max_lookahead: 6
      };

      try {
        const res = await fetch('/api/matrix-eval', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();

        document.getElementById('kpi-speedup').innerText = data.peak_speedup + 'x';
        document.getElementById('kpi-depth').innerText = data.recommended_depth;
        document.getElementById('kpi-width').innerText = data.recommended_tree_width;
        document.getElementById('kpi-tokens').innerText = data.optimal_point.expected_tokens;

        const tbody = document.getElementById('matrix-tbody');
        tbody.innerHTML = '';
        data.points.forEach(pt => {
          const tr = document.createElement('tr');
          const isOptimal = pt.is_pareto_optimal;
          const strategyLabel = pt.tree_width === 1 ? 'Greedy Chain' : `Tree (W=${pt.tree_width})`;
          tr.innerHTML = `
            <td><strong>${pt.lookahead_depth}</strong></td>
            <td>${strategyLabel}</td>
            <td>${pt.latency_ratio}</td>
            <td>${pt.expected_tokens}</td>
            <td style="color: ${pt.speedup >= 2.0 ? 'var(--accent)' : 'var(--text)'}; font-weight: 700;">${pt.speedup}x</td>
            <td>${isOptimal ? '<span class="optimal-tag">PARETO OPTIMAL</span>' : '-'}</td>
          `;
          tbody.appendChild(tr);
        });
      } catch (err) {
        console.error('Matrix eval error:', err);
      }
    }

    async function runTokenizerVerify() {
      const payload = {
        target_vocab_size: parseInt(document.getElementById('tok-target-size').value),
        draft_vocab_size: parseInt(document.getElementById('tok-draft-size').value),
        target_hidden_dim: parseInt(document.getElementById('tok-target-dim').value),
        head_hidden_dim: parseInt(document.getElementById('tok-head-dim').value),
        is_eagle: true
      };

      try {
        const res = await fetch('/api/verify-tokenizer', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();

        const box = document.getElementById('tok-report-content');
        const color = data.status === 'fully_aligned' ? 'var(--accent)' : 'var(--warning)';
        box.innerHTML = `
          <div style="background: #0b1120; border: 1px solid var(--border); border-radius: 8px; padding: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
              <span style="font-weight: 600; color: var(--text);">Alignment Status:</span>
              <span style="color: ${color}; font-weight: 700; text-transform: uppercase;">${data.status}</span>
            </div>
            <p>• Vocabulary Overlap Ratio: <strong>${(data.overlap_ratio * 100).toFixed(2)}%</strong></p>
            <p>• Shared Tokens: <strong>${data.shared_tokens_count.toLocaleString()}</strong></p>
            <p>• Divergent Count: <strong>${data.divergent_sample_count}</strong></p>
            <p>• Special Tokens Matched: <strong>${data.special_tokens_matched}</strong></p>
            <div style="margin-top: 14px; padding: 10px; background: rgba(56, 189, 248, 0.1); border-radius: 6px; color: var(--primary);">
              ${data.status === 'fully_aligned' ? 'Fully aligned: Speculative decoding will run with zero vocabulary remapping overhead.' : 'Warning: Mismatch detected. Remapping may incur latency overhead.'}
            </div>
          </div>
        `;
      } catch (err) {
        console.error(err);
      }
    }

    async function runProfiler() {
      const payload = {
        alpha: parseFloat(document.getElementById('prof-alpha').value),
        lookahead_depth: parseInt(document.getElementById('prof-k').value),
        draft_latency_ms: parseFloat(document.getElementById('prof-dl').value),
        target_latency_ms: parseFloat(document.getElementById('prof-tl').value),
        num_steps: 40
      };

      try {
        const res = await fetch('/api/profile', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();

        const box = document.getElementById('prof-results');
        box.innerHTML = `
          <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 16px;">
            <div class="kpi-card">
              <div class="kpi-label">Wallclock Speedup</div>
              <div class="kpi-val" style="color: var(--accent);">${data.wallclock_speedup.toFixed(2)}x</div>
            </div>
            <div class="kpi-card">
              <div class="kpi-label">Empirical Alpha</div>
              <div class="kpi-val">${(data.alpha_acceptance_rate * 100).toFixed(1)}%</div>
            </div>
            <div class="kpi-card">
              <div class="kpi-label">Speculative TPS</div>
              <div class="kpi-val">${data.tokens_per_second_speculative.toFixed(1)}</div>
            </div>
          </div>
          <div style="background: #0b1120; border: 1px solid var(--border); border-radius: 8px; padding: 14px;">
            <strong>Positional Acceptance Curve:</strong>
            <div style="display: flex; gap: 8px; margin-top: 8px;">
              ${data.acceptance_by_position.map((p, i) => `<div style="background: #1e293b; padding: 6px 12px; border-radius: 4px; font-size: 12px;">Pos ${i+1}: <strong>${(p*100).toFixed(1)}%</strong></div>`).join('')}
            </div>
          </div>
        `;
      } catch (err) {
        console.error(err);
      }
    }

    let activeJobId = null;
    let pollInterval = null;

    async function startTraining() {
      const payload = {
        epochs: parseInt(document.getElementById('tr-epochs').value),
        batch_size: parseInt(document.getElementById('tr-batch').value),
        lr: parseFloat(document.getElementById('tr-lr').value),
        hidden_dim: parseInt(document.getElementById('tr-hidden').value),
        target_dim: parseInt(document.getElementById('tr-target-dim').value),
        vocab_size: 500,
        num_samples: 32
      };

      const btn = document.getElementById('tr-start-btn');
      btn.disabled = true;
      btn.innerText = 'Training in Progress...';

      try {
        const res = await fetch('/api/train/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        activeJobId = data.job_id;

        pollInterval = setInterval(pollTrainingStatus, 1000);
      } catch (err) {
        console.error(err);
        btn.disabled = false;
        btn.innerText = 'Start Training Run';
      }
    }

    async function pollTrainingStatus() {
      if (!activeJobId) return;
      try {
        const res = await fetch(`/api/train/status/${activeJobId}`);
        const data = await res.json();
        const box = document.getElementById('tr-status-box');

        if (data.status === 'running') {
          box.innerHTML = `
            <div style="background: #0b1120; border: 1px solid var(--border); border-radius: 8px; padding: 16px;">
              <p style="color: var(--primary); font-weight: 600;">Status: Training Running (Job: ${data.job_id})</p>
              <p style="margin-top: 8px;">Epoch: <strong>${data.current_epoch}</strong> | Step: <strong>${data.current_step}</strong></p>
              <p>Current Distillation Loss: <strong>${data.current_loss.toFixed(4)}</strong></p>
              <p>Next-Token Top-1 Accuracy: <strong>${(data.current_acc * 100).toFixed(1)}%</strong></p>
            </div>
          `;
        } else if (data.status === 'completed') {
          clearInterval(pollInterval);
          document.getElementById('tr-start-btn').disabled = false;
          document.getElementById('tr-start-btn').innerText = 'Start Training Run';
          box.innerHTML = `
            <div style="background: #0b1120; border: 1px solid var(--accent); border-radius: 8px; padding: 16px;">
              <p style="color: var(--accent); font-weight: 700;">Training Completed Successfully</p>
              <p style="margin-top: 8px;">Total Steps: <strong>${data.result.total_steps}</strong></p>
              <p>Final Loss: <strong>${data.result.final_loss.toFixed(4)}</strong></p>
              <p>Saved Checkpoint: <code>${data.result.final_checkpoint}</code></p>
            </div>
          `;
        }
      } catch (err) {
        console.error(err);
      }
    }

    // Initialize matrix evaluation on load
    window.addEventListener('DOMContentLoaded', () => {
      runMatrixEval();
      runTokenizerVerify();
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(content=_DASHBOARD_HTML)
