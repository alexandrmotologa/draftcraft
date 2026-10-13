"""FastAPI application entrypoint and interactive dashboard rendering for DraftCraft."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

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
      --card-header: #0f172a;
      --border: #1f293d;
      --text: #f3f4f6;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --accent: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --purple: #a855f7;
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
      margin-bottom: 20px;
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
    .header-actions { display: flex; gap: 8px; }
    .btn-sm {
      background: #1e293b;
      color: var(--text);
      border: 1px solid var(--border);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      transition: all 0.2s;
    }
    .btn-sm:hover { background: #334155; border-color: var(--primary); }
    .nav-tabs {
      display: flex;
      gap: 8px;
      margin-bottom: 24px;
      flex-wrap: wrap;
    }
    .tab-btn {
      background: var(--card-bg);
      color: var(--text-muted);
      border: 1px solid var(--border);
      padding: 9px 16px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 500;
      font-size: 13px;
      transition: all 0.2s;
    }
    .tab-btn:hover { color: var(--text); border-color: var(--primary); }
    .tab-btn.active {
      background: rgba(56, 189, 248, 0.1);
      color: var(--primary);
      border-color: var(--primary);
      font-weight: 600;
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
      font-size: 15px;
      font-weight: 600;
      margin-bottom: 14px;
      color: var(--primary);
    }
    .form-group { margin-bottom: 14px; }
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
    .btn {
      background: var(--primary);
      color: #0b1120;
      border: none;
      padding: 10px 16px;
      border-radius: 6px;
      font-weight: 600;
      cursor: pointer;
      width: 100%;
      font-size: 13px;
      transition: opacity 0.2s;
    }
    .btn:hover { opacity: 0.9; }
    .kpi-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 20px;
    }
    .kpi-card {
      background: #0b1120;
      border: 1px solid var(--border);
      padding: 14px 16px;
      border-radius: 8px;
    }
    .kpi-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }
    .kpi-val { font-size: 22px; font-weight: 700; color: var(--text); margin-top: 4px; }
    .kpi-sub { font-size: 11px; color: var(--accent); margin-top: 2px; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }
    th, td {
      padding: 8px 12px;
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
    .canvas-container {
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
      <h1 style="font-size: 20px; font-weight: 700; letter-spacing: -0.5px;">DraftCraft</h1>
      <span class="badge">EAGLE Speculative Suite</span>
    </div>
    <div class="header-actions">
      <button class="btn-sm" onclick="exportData('markdown')">Export Markdown</button>
      <button class="btn-sm" onclick="exportData('vllm')">Export vLLM JSON</button>
      <button class="btn-sm" onclick="exportData('csv')">Export CSV</button>
    </div>
  </header>

  <div class="nav-tabs">
    <button class="tab-btn active" onclick="switchTab('tab-matrix')">Speculative Matrix</button>
    <button class="tab-btn" onclick="switchTab('tab-vram')">VRAM & KV Budget</button>
    <button class="tab-btn" onclick="switchTab('tab-temp')">Temperature Profiler</button>
    <button class="tab-btn" onclick="switchTab('tab-tokenizer')">Tokenizer Verifier</button>
    <button class="tab-btn" onclick="switchTab('tab-profiler')">Empirical Profiler</button>
    <button class="tab-btn" onclick="switchTab('tab-trainer')">EAGLE Head Trainer</button>
  </div>

  <!-- TAB 1: SPECULATIVE MATRIX EVALUATOR -->
  <div id="tab-matrix" class="tab-content active">
    <div class="grid">
      <div class="card">
        <h3>Matrix Configuration</h3>
        <div class="form-group">
          <label>Hardware Preset (Auto-Latency)</label>
          <select id="mat-preset" class="form-control" onchange="applyPresetToMatrix()">
            <option value="custom">-- Custom Latency --</option>
            <option value="rtx-4090">NVIDIA RTX 4090 (24GB)</option>
            <option value="rtx-4080">NVIDIA RTX 4080 (16GB)</option>
            <option value="rtx-4070-ti">NVIDIA RTX 4070 Ti (12GB)</option>
            <option value="rtx-4060-8gb" selected>NVIDIA RTX 4060 (8GB)</option>
            <option value="rtx-3060-12gb">NVIDIA RTX 3060 (12GB)</option>
            <option value="apple-m3-max">Apple M3 Max (36GB)</option>
          </select>
        </div>
        <div class="form-group">
          <label>Target LLM Latency (ms)</label>
          <input type="number" id="mat-tl" class="form-control" value="38.0" step="0.5" oninput="runMatrixEval()">
        </div>
        <div class="form-group">
          <label>Draft Latency (ms)</label>
          <input type="number" id="mat-dl" class="form-control" value="5.0" step="0.1" oninput="runMatrixEval()">
        </div>
        <div class="form-group">
          <label>Base Acceptance Rate (alpha)</label>
          <input type="range" id="mat-alpha" min="0.4" max="0.95" step="0.01" value="0.75" class="form-control" oninput="document.getElementById('mat-alpha-val').innerText = Math.round(this.value * 100) + '%'; runMatrixEval();">
          <div style="display: flex; justify-content: space-between; font-size: 12px; color: var(--primary); margin-top: 4px;">
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
            <div class="kpi-val" id="kpi-speedup" style="color: var(--accent);">--</div>
            <div class="kpi-sub">Pareto Optimal</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Recommended Depth (K*)</div>
            <div class="kpi-val" id="kpi-depth">--</div>
            <div class="kpi-sub">Lookahead Tokens</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Optimal Width (W*)</div>
            <div class="kpi-val" id="kpi-width">--</div>
            <div class="kpi-sub">Speculation Strategy</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Exp. Tokens / Step</div>
            <div class="kpi-val" id="kpi-tokens">--</div>
            <div class="kpi-sub">Accepted Length</div>
          </div>
        </div>

        <div class="card">
          <h3>Interactive Speedup Trajectory (Chain vs Tree)</h3>
          <div class="canvas-container">
            <canvas id="matrixCanvas" width="700" height="220" style="width: 100%; height: 220px; display: block;"></canvas>
          </div>

          <h3 style="margin-top: 20px;">Speculative Decoding Parameter Matrix</h3>
          <div style="overflow-x: auto; max-height: 280px;">
            <table>
              <thead>
                <tr>
                  <th>Depth (K)</th>
                  <th>Strategy</th>
                  <th>Latency Ratio</th>
                  <th>Exp. Tokens</th>
                  <th>Speedup</th>
                  <th>Pareto Status</th>
                </tr>
              </thead>
              <tbody id="matrix-tbody"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 2: VRAM & KV CACHE BUDGET -->
  <div id="tab-vram" class="tab-content">
    <div class="grid">
      <div class="card">
        <h3>VRAM Simulation Inputs</h3>
        <div class="form-group">
          <label>Hardware GPU Preset</label>
          <select id="vram-preset" class="form-control" onchange="applyVRAMPreset()">
            <option value="rtx-4090">NVIDIA RTX 4090 (24GB)</option>
            <option value="rtx-4080">NVIDIA RTX 4080 (16GB)</option>
            <option value="rtx-4070-ti">NVIDIA RTX 4070 Ti (12GB)</option>
            <option value="rtx-4060-8gb" selected>NVIDIA RTX 4060 (8GB)</option>
            <option value="rtx-3060-12gb">NVIDIA RTX 3060 (12GB)</option>
            <option value="apple-m3-max">Apple M3 Max (36GB)</option>
          </select>
        </div>
        <div class="form-group">
          <label>Target Parameters (Billions)</label>
          <input type="number" id="vram-params" class="form-control" value="8.0" step="0.5" oninput="runVRAMCalc()">
        </div>
        <div class="form-group">
          <label>Target Precision / Quantization</label>
          <select id="vram-prec" class="form-control" onchange="runVRAMCalc()">
            <option value="q4_k_m" selected>GGUF Q4_K_M (4.5 bits)</option>
            <option value="int4">INT4 (4.0 bits)</option>
            <option value="int8">INT8 (8.0 bits)</option>
            <option value="fp16">FP16 / BF16 (16.0 bits)</option>
          </select>
        </div>
        <div class="form-group">
          <label>Context Length (Tokens)</label>
          <input type="number" id="vram-ctx" class="form-control" value="4096" step="512" oninput="runVRAMCalc()">
        </div>
        <div class="form-group">
          <label>Lookahead Depth K</label>
          <input type="number" id="vram-k" class="form-control" value="4" min="1" max="16" oninput="runVRAMCalc()">
        </div>
        <button class="btn" onclick="runVRAMCalc()">Compute Memory Footprint</button>
      </div>

      <div class="card">
        <h3>Memory Allocation & Budget Analysis</h3>
        <div id="vram-report-box" style="margin-bottom: 16px;"></div>
        <div class="canvas-container">
          <canvas id="vramCanvas" width="700" height="200" style="width: 100%; height: 200px; display: block;"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 3: TEMPERATURE SENSITIVITY -->
  <div id="tab-temp" class="tab-content">
    <div class="grid">
      <div class="card">
        <h3>Temperature Sweep Controls</h3>
        <div class="form-group">
          <label>Greedy Baseline Acceptance (T = 0)</label>
          <input type="number" id="temp-alpha" class="form-control" value="0.85" step="0.05" min="0.5" max="0.99" oninput="runTempProfile()">
        </div>
        <div class="form-group">
          <label>Lookahead Depth K</label>
          <input type="number" id="temp-k" class="form-control" value="4" min="1" max="8" oninput="runTempProfile()">
        </div>
        <button class="btn" onclick="runTempProfile()">Profile Sensitivity</button>
      </div>

      <div class="card">
        <h3>Acceptance Degradation & Break-Even Threshold</h3>
        <div id="temp-summary-box" style="margin-bottom: 16px;"></div>
        <div class="canvas-container">
          <canvas id="tempCanvas" width="700" height="200" style="width: 100%; height: 200px; display: block;"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 4: TOKENIZER VERIFIER -->
  <div id="tab-tokenizer" class="tab-content">
    <div class="grid">
      <div class="card">
        <h3>Tokenizer Compatibility Check</h3>
        <div class="form-group">
          <label>Target Vocab Size</label>
          <input type="number" id="tok-target-size" class="form-control" value="128256">
        </div>
        <div class="form-group">
          <label>Draft Vocab Size</label>
          <input type="number" id="tok-draft-size" class="form-control" value="128256">
        </div>
        <div class="form-group">
          <label>Target Hidden Dimension</label>
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

  <!-- TAB 5: EMPIRICAL PROFILER -->
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
        <div id="prof-results" style="font-size: 13px; color: var(--text-muted); margin-bottom: 16px;"></div>
        <div class="canvas-container">
          <canvas id="profCanvas" width="700" height="180" style="width: 100%; height: 180px; display: block;"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 6: TRAINER -->
  <div id="tab-trainer" class="tab-content">
    <div class="grid">
      <div class="card">
        <h3>Train Local EAGLE Head</h3>
        <div class="form-group">
          <label>Epochs</label>
          <input type="number" id="tr-epochs" class="form-control" value="3" min="1" max="10">
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
        <div id="tr-status-box" style="font-size: 13px; color: var(--text-muted); margin-bottom: 16px;">
          Draft head trainer ready. Click "Start Training Run" to train in background.
        </div>
        <div class="canvas-container">
          <canvas id="trainCanvas" width="700" height="200" style="width: 100%; height: 200px; display: block;"></canvas>
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

      if (tabId === 'tab-vram') runVRAMCalc();
      if (tabId === 'tab-temp') runTempProfile();
    }

    function applyPresetToMatrix() {
      const p = document.getElementById('mat-preset').value;
      const presets = {
        'rtx-4090': { tl: 18.5, dl: 2.4 },
        'rtx-4080': { tl: 23.0, dl: 2.9 },
        'rtx-4070-ti': { tl: 27.5, dl: 3.4 },
        'rtx-4060-8gb': { tl: 38.0, dl: 5.0 },
        'rtx-3060-12gb': { tl: 42.0, dl: 5.5 },
        'apple-m3-max': { tl: 28.0, dl: 3.6 }
      };
      if (presets[p]) {
        document.getElementById('mat-tl').value = presets[p].tl;
        document.getElementById('mat-dl').value = presets[p].dl;
        runMatrixEval();
      }
    }

    async function runMatrixEval() {
      const payload = {
        domain: document.getElementById('mat-domain').value,
        alpha: parseFloat(document.getElementById('mat-alpha').value),
        draft_latency_ms: parseFloat(document.getElementById('mat-dl').value) || 5.0,
        target_latency_ms: parseFloat(document.getElementById('mat-tl').value) || 38.0,
        min_lookahead: 1,
        max_lookahead: 8
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
        document.getElementById('kpi-width').innerText = 'W=' + data.recommended_tree_width;
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

        drawMatrixChart(data.points);
      } catch (err) {
        console.error('Matrix eval error:', err);
      }
    }

    function drawMatrixChart(points) {
      const canvas = document.getElementById('matrixCanvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const chainPoints = points.filter(p => p.tree_width === 1);
      const treePoints = points.filter(p => p.tree_width === 2);
      const tree4Points = points.filter(p => p.tree_width === 4);

      const maxK = 8;
      const maxSpeedup = Math.max(...points.map(p => p.speedup), 3.5);

      function getX(k) { return 50 + ((k - 1) / (maxK - 1)) * (w - 80); }
      function getY(sp) { return (h - 35) - (sp / maxSpeedup) * (h - 60); }

      // Grid
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      for (let s = 1.0; s <= maxSpeedup; s += 0.5) {
        ctx.beginPath();
        ctx.moveTo(40, getY(s));
        ctx.lineTo(w - 20, getY(s));
        ctx.stroke();
        ctx.fillStyle = '#64748b';
        ctx.font = '10px sans-serif';
        ctx.fillText(s.toFixed(1) + 'x', 10, getY(s) + 3);
      }

      function drawLine(pts, color, label) {
        if (!pts.length) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        pts.forEach((p, idx) => {
          const x = getX(p.lookahead_depth), y = getY(p.speedup);
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();

        pts.forEach(p => {
          const x = getX(p.lookahead_depth), y = getY(p.speedup);
          ctx.fillStyle = p.is_pareto_optimal ? '#10b981' : color;
          ctx.beginPath();
          ctx.arc(x, y, p.is_pareto_optimal ? 5 : 3.5, 0, Math.PI * 2);
          ctx.fill();
        });
      }

      drawLine(chainPoints, '#38bdf8', 'Chain (W=1)');
      drawLine(treePoints, '#a855f7', 'Tree (W=2)');
      drawLine(tree4Points, '#10b981', 'Tree (W=4)');

      // Legend
      ctx.fillStyle = '#38bdf8'; ctx.fillText('■ Chain (W=1)', w - 240, 20);
      ctx.fillStyle = '#a855f7'; ctx.fillText('■ Tree (W=2)', w - 160, 20);
      ctx.fillStyle = '#10b981'; ctx.fillText('■ Tree (W=4)', w - 80, 20);
    }

    function applyVRAMPreset() {
      runVRAMCalc();
    }

    async function runVRAMCalc() {
      const p = document.getElementById('vram-preset').value;
      const payload = {
        target_param_count_billions: parseFloat(document.getElementById('vram-params').value) || 8.0,
        target_precision: document.getElementById('vram-prec').value,
        context_len: parseInt(document.getElementById('vram-ctx').value) || 4096,
        lookahead_k: parseInt(document.getElementById('vram-k').value) || 4,
        preset: p
      };

      try {
        const res = await fetch('/api/vram-calc', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();

        const box = document.getElementById('vram-report-box');
        const color = data.fits_in_gpu ? 'var(--accent)' : 'var(--danger)';
        box.innerHTML = `
          <div style="background: #0b1120; border: 1px solid var(--border); border-radius: 8px; padding: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
              <span style="font-weight: 600;">GPU Budget Status:</span>
              <span style="color: ${color}; font-weight: 700;">${data.fits_in_gpu ? 'FITS IN GPU' : 'VRAM EXCEEDED'}</span>
            </div>
            <p>• Total VRAM: <strong>${data.total_vram_gb} GB</strong> / ${data.gpu_vram_capacity_gb} GB</p>
            <p>• Headroom: <strong style="color: ${color}">${data.headroom_gb} GB</strong></p>
            <p>• Max Viable Context: <strong>${data.max_context_length_tokens.toLocaleString()} tokens</strong></p>
            <p style="margin-top: 6px; font-size: 12px; color: var(--text-muted);">${data.recommendation}</p>
          </div>
        `;

        drawVRAMChart(data);
      } catch (err) {
        console.error(err);
      }
    }

    function drawVRAMChart(data) {
      const canvas = document.getElementById('vramCanvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const totalCap = data.gpu_vram_capacity_gb;
      const barY = 60, barH = 50, barW = w - 60;

      const tw = (data.target_weights_gb / totalCap) * barW;
      const dw = (data.draft_head_weights_gb / totalCap) * barW;
      const kvw = (data.total_kv_cache_gb / totalCap) * barW;
      const actw = (data.activations_and_cuda_overhead_gb / totalCap) * barW;

      let curX = 30;
      // Target
      ctx.fillStyle = '#38bdf8'; ctx.fillRect(curX, barY, tw, barH); curX += tw;
      // Draft Head
      ctx.fillStyle = '#a855f7'; ctx.fillRect(curX, barY, dw, barH); curX += dw;
      // KV Cache
      ctx.fillStyle = '#10b981'; ctx.fillRect(curX, barY, kvw, barH); curX += kvw;
      // Activations
      ctx.fillStyle = '#f59e0b'; ctx.fillRect(curX, barY, actw, barH); curX += actw;

      // GPU Limit line
      ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(30 + barW, 40);
      ctx.lineTo(30 + barW, 130);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#ef4444'; ctx.font = '11px sans-serif';
      ctx.fillText('GPU Cap (' + totalCap + 'GB)', 30 + barW - 70, 35);

      // Legend
      ctx.font = '11px sans-serif';
      ctx.fillStyle = '#38bdf8'; ctx.fillText('■ Target Weights (' + data.target_weights_gb + 'G)', 30, 150);
      ctx.fillStyle = '#a855f7'; ctx.fillText('■ Draft Head (' + data.draft_head_weights_gb + 'G)', 210, 150);
      ctx.fillStyle = '#10b981'; ctx.fillText('■ KV Cache (' + data.total_kv_cache_gb + 'G)', 360, 150);
      ctx.fillStyle = '#f59e0b'; ctx.fillText('■ CUDA Overh. (' + data.activations_and_cuda_overhead_gb + 'G)', 510, 150);
    }

    async function runTempProfile() {
      const payload = {
        alpha: parseFloat(document.getElementById('temp-alpha').value) || 0.85,
        lookahead_depth: parseInt(document.getElementById('temp-k').value) || 4,
        draft_latency_ms: 3.5,
        target_latency_ms: 28.0
      };

      try {
        const res = await fetch('/api/temperature-profile', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();

        const box = document.getElementById('temp-summary-box');
        box.innerHTML = `
          <div style="background: #0b1120; border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-size: 13px;">
            <p>• Peak Speedup: <strong style="color: var(--accent);">${data.peak_speedup}x</strong> at T=${data.peak_temperature}</p>
            <p>• Guidance: ${data.recommendation}</p>
          </div>
        `;

        drawTempChart(data.points);
      } catch (err) {
        console.error(err);
      }
    }

    function drawTempChart(points) {
      const canvas = document.getElementById('tempCanvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const maxSp = 3.0;
      function getX(t) { return 40 + t * (w - 70); }
      function getY(sp) { return (h - 30) - (sp / maxSp) * (h - 50); }

      // Baseline 1.0x line
      ctx.strokeStyle = '#ef4444';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(40, getY(1.0));
      ctx.lineTo(w - 20, getY(1.0));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#ef4444';
      ctx.font = '10px sans-serif';
      ctx.fillText('1.0x Baseline', 10, getY(1.0) + 3);

      // Curve
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      points.forEach((p, idx) => {
        const x = getX(p.temperature), y = getY(p.speedup);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      points.forEach(p => {
        const x = getX(p.temperature), y = getY(p.speedup);
        ctx.fillStyle = '#10b981';
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#f3f4f6';
        ctx.font = '10px sans-serif';
        ctx.fillText(p.speedup + 'x', x - 10, y - 8);
      });
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
          <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 12px;">
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
        `;

        drawProfilerChart(data.acceptance_by_position);
      } catch (err) {
        console.error(err);
      }
    }

    function drawProfilerChart(posAlphas) {
      const canvas = document.getElementById('profCanvas');
      if (!canvas || !posAlphas) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const barW = Math.min(60, (w - 60) / posAlphas.length - 20);
      posAlphas.forEach((alpha, i) => {
        const x = 50 + i * (barW + 25);
        const barH = alpha * (h - 50);
        const y = h - 30 - barH;

        ctx.fillStyle = alpha >= 0.75 ? '#10b981' : alpha >= 0.5 ? '#f59e0b' : '#ef4444';
        ctx.fillRect(x, y, barW, barH);

        ctx.fillStyle = '#f3f4f6';
        ctx.font = '11px sans-serif';
        ctx.fillText((alpha * 100).toFixed(1) + '%', x + 5, y - 6);
        ctx.fillStyle = '#94a3b8';
        ctx.fillText('Pos ' + (i + 1), x + 10, h - 12);
      });
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
            <div style="background: #0b1120; border: 1px solid var(--border); border-radius: 8px; padding: 14px;">
              <p style="color: var(--primary); font-weight: 600;">Status: Training Running (Job: ${data.job_id})</p>
              <p style="margin-top: 6px;">Epoch: <strong>${data.current_epoch}</strong> | Step: <strong>${data.current_step}</strong></p>
              <p>Distillation Loss: <strong>${data.current_loss.toFixed(4)}</strong> | Next-Token Acc: <strong>${(data.current_acc * 100).toFixed(1)}%</strong></p>
            </div>
          `;
          drawTrainChart(data.history);
        } else if (data.status === 'completed') {
          clearInterval(pollInterval);
          document.getElementById('tr-start-btn').disabled = false;
          document.getElementById('tr-start-btn').innerText = 'Start Training Run';
          box.innerHTML = `
            <div style="background: #0b1120; border: 1px solid var(--accent); border-radius: 8px; padding: 14px;">
              <p style="color: var(--accent); font-weight: 700;">Training Completed Successfully</p>
              <p style="margin-top: 6px;">Total Steps: <strong>${data.result.total_steps}</strong> | Final Loss: <strong>${data.result.final_loss.toFixed(4)}</strong></p>
              <p>Saved Checkpoint: <code>${data.result.final_checkpoint}</code></p>
            </div>
          `;
          drawTrainChart(data.history);
        }
      } catch (err) {
        console.error(err);
      }
    }

    function drawTrainChart(history) {
      const canvas = document.getElementById('trainCanvas');
      if (!canvas || !history || !history.length) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const maxLoss = Math.max(...history.map(h => h.total_loss), 1.0);
      function getX(i) { return 40 + (i / Math.max(history.length - 1, 1)) * (w - 70); }
      function getY(l) { return (h - 25) - (l / maxLoss) * (h - 45); }

      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      history.forEach((pt, i) => {
        const x = getX(i), y = getY(pt.total_loss);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      ctx.fillStyle = '#ef4444';
      ctx.font = '11px sans-serif';
      ctx.fillText('■ Loss Trajectory', w - 120, 20);
    }

    async function exportData(format) {
      try {
        const res = await fetch('/api/export-report', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ format: format })
        });
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = format === 'markdown' ? 'draftcraft_report.md' : format === 'csv' ? 'matrix.csv' : 'vllm_config.json';
        document.body.appendChild(a);
        a.click();
        a.remove();
      } catch (err) {
        console.error(err);
      }
    }

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
