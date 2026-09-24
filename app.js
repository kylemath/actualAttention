const TAB_MAP = {
  comparison: "tabs/comparison.html",
  architectures: "tabs/architectures.html",
  "bio-mechanisms": "tabs/bio-mechanisms.html",
  statistics: "tabs/statistics.html",
  overview: "tabs/overview.html",
  module: "tabs/module.html",
  traditional: "tabs/traditional.html",
  physio: "tabs/physio.html",
  variants: "tabs/variants.html",
  direction1: "tabs/direction1.html",
  direction2: "tabs/direction2.html",
  results: "tabs/results.html",
};

const STATE = {
  inputs: [],
  experimentLogKey: "aalab-experiments",
  lastRun: {},
};

const DEFAULT_INPUTS = [
  [1, 0],
  [0.5, 1],
  [-0.5, 0.5],
];

const TARGET = [0, 0, 1];

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

const seededRandom = (seed) => {
  let t = seed + 0x6d2b79f5;
  return () => {
    t += 0x6d2b79f5;
    let result = Math.imul(t ^ (t >>> 15), 1 | t);
    result ^= result + Math.imul(result ^ (result >>> 7), 61 | result);
    return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
  };
};

const softmax = (arr, temperature = 1) => {
  const invTemp = 1 / Math.max(temperature, 1e-6);
  const maxVal = Math.max(...arr);
  const exps = arr.map((v) => Math.exp((v - maxVal) * invTemp));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
};

const matVecMul = (mat, vec) =>
  mat.map((row) => row.reduce((sum, value, idx) => sum + value * vec[idx], 0));

const outer = (a, b) =>
  a.map((ai) => b.map((bi) => ai * bi));

const addMatrices = (a, b) =>
  a.map((row, r) => row.map((val, c) => val + b[r][c]));

const scaleMatrix = (mat, scale) =>
  mat.map((row) => row.map((val) => val * scale));

const matrixToString = (mat, precision = 3) =>
  mat
    .map((row) => row.map((v) => v.toFixed(precision).padStart(7)).join(" "))
    .join("\n");

const randomMatrix = (rows, cols, seed = 1) => {
  const rand = seededRandom(seed);
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (rand() - 0.5) * 0.8)
  );
};

const computeScores = (inputs, Wq, Wk) => {
  const d = inputs[0].length;
  const scale = 1 / Math.sqrt(d);
  const Q = inputs.map((input) => matVecMul(Wq, input));
  const K = inputs.map((input) => matVecMul(Wk, input));
  const scores = Q.map((q) =>
    K.map((k) => q.reduce((sum, qv, idx) => sum + qv * k[idx], 0) * scale)
  );
  return { Q, K, scores };
};

const applyMechanisms = (scores, opts) => {
  const { biasStrength, inhibition, sharpening } = opts;
  let updated = scores.map((row) => [...row]);

  if (biasStrength > 0) {
    updated = updated.map((row, idx) =>
      row.map((val, col) => val + biasStrength * (idx === 0 ? col * 0.15 : 0))
    );
  }

  if (inhibition > 0) {
    updated = updated.map((row) => {
      const mean = row.reduce((a, b) => a + b, 0) / row.length;
      return row.map((val) => val - inhibition * mean);
    });
  }

  if (sharpening && sharpening !== 1) {
    return updated.map((row) => row.map((val) => val * sharpening));
  }

  return updated;
};

const computeAttention = (inputs, Wq, Wk, opts) => {
  const { scores } = computeScores(inputs, Wq, Wk);
  const adjusted = applyMechanisms(scores, opts);
  const temperature = opts.temperature || 1;
  const weights = adjusted.map((row) => softmax(row, temperature));
  return { scores: adjusted, weights };
};

const VARIANT_COLORS = {
  "Traditional": "#2563eb",
  "Physio": "#16a34a",
  "Variant": "#ef4444",
  "Sparse": "#f59e0b",
  "Predictive": "#8b5cf6",
  "default": "#64748b"
};

const drawLineChart = (canvas, dataSets, options = {}) => {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = width;
  canvas.height = height;
  ctx.clearRect(0, 0, width, height);

  const padding = { top: 20, right: 20, bottom: 30, left: 40 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Draw grid
  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + (chartHeight / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }

  // Find global max for consistent scaling
  const allData = dataSets.flatMap(set => set.data || []);
  const maxVal = Math.max(...allData, 1);

  // Draw Y-axis labels
  ctx.fillStyle = "#64748b";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const val = maxVal * (1 - i / 4);
    const y = padding.top + (chartHeight / 4) * i;
    ctx.fillText(val.toFixed(3), padding.left - 5, y + 3);
  }

  // Draw lines
  dataSets.forEach((set, index) => {
    const data = set.data;
    if (!data || data.length < 2) return;
    const color = set.color || VARIANT_COLORS[set.label] || VARIANT_COLORS.default;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    data.forEach((val, idx) => {
      const x = padding.left + (idx / (data.length - 1)) * chartWidth;
      const y = padding.top + chartHeight - (val / maxVal) * chartHeight;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  // Draw X-axis label
  ctx.fillStyle = "#64748b";
  ctx.font = "11px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Training Steps", width / 2, height - 5);
  
  // Draw Y-axis label
  ctx.save();
  ctx.translate(12, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Loss", 0, 0);
  ctx.restore();
};

const drawImageCanvas = (canvas, matrix) => {
  if (!canvas || !matrix || !matrix.length) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = width;
  canvas.height = height;
  const rows = matrix.length;
  const cols = matrix[0].length;
  const cellW = width / cols;
  const cellH = height / rows;
  const maxVal = Math.max(...matrix.flat(), 1);
  matrix.forEach((row, r) => {
    row.forEach((val, c) => {
      const intensity = Math.round((val / maxVal) * 255);
      ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
      ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
    });
  });
};

const drawNetwork = (svg, weights, options = {}) => {
  if (!svg || !weights || !weights.length) return;
  const rows = weights.length;
  const cols = weights[0].length;
  const width = svg.clientWidth || 320;
  const height = svg.clientHeight || 220;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";

  const NS = "http://www.w3.org/2000/svg";
  const leftX = width * 0.25;
  const rightX = width * 0.75;
  const topPad = 30;
  const bottomPad = 30;
  const leftSpacing = (height - topPad - bottomPad) / (rows - 1 || 1);
  const rightSpacing = (height - topPad - bottomPad) / (cols - 1 || 1);
  const leftNodes = Array.from({ length: rows }, (_, i) => ({
    x: leftX,
    y: topPad + i * leftSpacing,
  }));
  const rightNodes = Array.from({ length: cols }, (_, i) => ({
    x: rightX,
    y: topPad + i * rightSpacing,
  }));

  // Add column headers
  const leftHeader = document.createElementNS(NS, "text");
  leftHeader.setAttribute("x", leftX);
  leftHeader.setAttribute("y", 15);
  leftHeader.setAttribute("font-size", "11");
  leftHeader.setAttribute("font-weight", "bold");
  leftHeader.setAttribute("fill", "#2563eb");
  leftHeader.setAttribute("text-anchor", "middle");
  leftHeader.textContent = options.leftLabel || "Queries (Q)";
  svg.appendChild(leftHeader);

  const rightHeader = document.createElementNS(NS, "text");
  rightHeader.setAttribute("x", rightX);
  rightHeader.setAttribute("y", 15);
  rightHeader.setAttribute("font-size", "11");
  rightHeader.setAttribute("font-weight", "bold");
  rightHeader.setAttribute("fill", "#16a34a");
  rightHeader.setAttribute("text-anchor", "middle");
  rightHeader.textContent = options.rightLabel || "Keys (K)";
  svg.appendChild(rightHeader);

  // Draw edges with weights
  const maxVal = Math.max(...weights.flat(), 1);
  const showWeights = options.showWeights !== false && rows <= 4 && cols <= 4;
  
  weights.forEach((row, r) => {
    row.forEach((val, c) => {
      const alpha = clamp(val / maxVal, 0.05, 1);
      const stroke = options.stroke || "#2563eb";
      const line = document.createElementNS(NS, "line");
      line.setAttribute("x1", leftNodes[r].x);
      line.setAttribute("y1", leftNodes[r].y);
      line.setAttribute("x2", rightNodes[c].x);
      line.setAttribute("y2", rightNodes[c].y);
      line.setAttribute("stroke", stroke);
      line.setAttribute("stroke-width", (0.5 + alpha * 3.5).toFixed(2));
      line.setAttribute("stroke-opacity", alpha.toFixed(2));
      
      // Add tooltip
      const title = document.createElementNS(NS, "title");
      title.textContent = `Q${r} → K${c}: ${val.toFixed(3)}`;
      line.appendChild(title);
      svg.appendChild(line);

      // Optionally show weight labels on edges
      if (showWeights && val > 0.15) {
        const midX = (leftNodes[r].x + rightNodes[c].x) / 2;
        const midY = (leftNodes[r].y + rightNodes[c].y) / 2;
        const weightLabel = document.createElementNS(NS, "text");
        weightLabel.setAttribute("x", midX);
        weightLabel.setAttribute("y", midY - 3);
        weightLabel.setAttribute("font-size", "8");
        weightLabel.setAttribute("fill", "#dc2626");
        weightLabel.setAttribute("text-anchor", "middle");
        weightLabel.setAttribute("font-weight", "bold");
        weightLabel.textContent = val.toFixed(2);
        svg.appendChild(weightLabel);
      }
    });
  });

  const makeNode = (node, label, color) => {
    const group = document.createElementNS(NS, "g");
    const circle = document.createElementNS(NS, "circle");
    circle.setAttribute("cx", node.x);
    circle.setAttribute("cy", node.y);
    circle.setAttribute("r", 10);
    circle.setAttribute("fill", color || "#0f172a");
    circle.setAttribute("stroke", "white");
    circle.setAttribute("stroke-width", "2");
    svg.appendChild(circle);
    if (label) {
      const text = document.createElementNS(NS, "text");
      text.setAttribute("x", node.x + 15);
      text.setAttribute("y", node.y + 4);
      text.setAttribute("font-size", "11");
      text.setAttribute("font-weight", "600");
      text.setAttribute("fill", "#374151");
      text.textContent = label;
      group.appendChild(text);
      svg.appendChild(group);
    }
  };

  leftNodes.forEach((node, idx) => makeNode(node, `Q${idx}`, "#2563eb"));
  rightNodes.forEach((node, idx) => makeNode(node, `K${idx}`, "#16a34a"));
  
  // Add legend at bottom
  const legendY = height - 8;
  const legend = document.createElementNS(NS, "text");
  legend.setAttribute("x", width / 2);
  legend.setAttribute("y", legendY);
  legend.setAttribute("font-size", "9");
  legend.setAttribute("fill", "#64748b");
  legend.setAttribute("text-anchor", "middle");
  legend.textContent = "Line thickness & opacity = attention weight";
  svg.appendChild(legend);
};

const getExperimentLog = () => {
  try {
    return JSON.parse(localStorage.getItem(STATE.experimentLogKey) || "[]");
  } catch (error) {
    return [];
  }
};

const computeVarianceExplained = (matrix) => {
  try {
    const eigenData = approximateEigenvalues(matrix, Math.min(3, matrix.length));
    const eigenvalues = eigenData.map(e => Math.abs(e.value));
    const total = eigenvalues.reduce((a, b) => a + b, 0) + 1e-8;
    const top1 = (eigenvalues[0] || 0) / total;
    const top2 = (eigenvalues.slice(0, 2).reduce((a, b) => a + b, 0)) / total;
    return { top1: top1 * 100, top2: top2 * 100, eigenvalues };
  } catch (e) {
    return { top1: 0, top2: 0, eigenvalues: [] };
  }
};

const saveExperiment = (entry) => {
  const varianceInfo = computeVarianceExplained(entry.attention);
  const log = getExperimentLog();
  log.push({ ...entry, varianceExplained: varianceInfo, id: Date.now() });
  localStorage.setItem(STATE.experimentLogKey, JSON.stringify(log));
  showToast(`✅ Logged ${entry.label || entry.variant} (loss: ${entry.loss.toFixed(4)}, var: ${varianceInfo.top1.toFixed(1)}%)`);
};

const showToast = (message, duration = 3000) => {
  // Remove any existing toast
  const existing = document.getElementById("toast-notification");
  if (existing) existing.remove();
  
  const toast = document.createElement("div");
  toast.id = "toast-notification";
  toast.textContent = message;
  toast.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: #10b981;
    color: white;
    padding: 12px 20px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    z-index: 10000;
    font-size: 14px;
    font-weight: 600;
    animation: slideIn 0.3s ease-out;
  `;
  document.body.appendChild(toast);
  
  setTimeout(() => {
    toast.style.animation = "slideOut 0.3s ease-in";
    setTimeout(() => toast.remove(), 300);
  }, duration);
};

// Add animation keyframes to document
if (!document.getElementById("toast-styles")) {
  const style = document.createElement("style");
  style.id = "toast-styles";
  style.textContent = `
    @keyframes slideIn {
      from { transform: translateX(400px); opacity: 0; }
      to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
      from { transform: translateX(0); opacity: 1; }
      to { transform: translateX(400px); opacity: 0; }
    }
  `;
  document.head.appendChild(style);
}

const resetExperiments = () => {
  localStorage.removeItem(STATE.experimentLogKey);
};

const simulateTraining = (opts) => {
  const steps = opts.steps || 80;
  const lr = opts.lr || 0.1;
  const inputs = opts.inputs || DEFAULT_INPUTS;
  const d = inputs[0].length;
  const scale = 1 / Math.sqrt(d);
  const temperature = opts.temperature || 1;
  const biasStrength = opts.biasStrength || 0;
  const inhibition = opts.inhibition || 0;
  const sharpening = opts.sharpening || 1;
  const noiseCorr = opts.noiseCorr || 0;
  const rand = seededRandom(opts.seed || 3);

  let Wq = randomMatrix(d, d, 1);
  let Wk = randomMatrix(d, d, 2);
  const losses = [];

  for (let step = 0; step < steps; step += 1) {
    const noise = noiseCorr
      ? inputs.map((row) =>
          row.map((val) => val + noiseCorr * (rand() - 0.5))
        )
      : inputs;

    const { Q, K, scores } = computeScores(noise, Wq, Wk);
    let rowScores = scores[0];
    let adjustedRow = applyMechanisms([rowScores], {
      biasStrength,
      inhibition,
      sharpening,
    })[0];

    const attention = softmax(adjustedRow, temperature);
    const loss =
      0.5 * attention.reduce((sum, val, idx) => sum + (val - TARGET[idx]) ** 2, 0);
    losses.push(loss);

    const dLdA = attention.map((val, idx) => val - TARGET[idx]);
    const dot = dLdA.reduce((sum, val, idx) => sum + val * attention[idx], 0);
    let dLdS = attention.map((val, idx) => val * (dLdA[idx] - dot));

    if (temperature !== 1) {
      dLdS = dLdS.map((val) => val / temperature);
    }

    if (inhibition > 0) {
      const mean = dLdS.reduce((a, b) => a + b, 0) / dLdS.length;
      dLdS = dLdS.map((val) => val - inhibition * mean);
    }

    if (sharpening !== 1) {
      dLdS = dLdS.map((val) => val * sharpening);
    }

    const q0 = Q[0];
    const dLdq0 = dLdS.reduce(
      (acc, grad, idx) =>
        acc.map((val, dim) => val + (grad * K[idx][dim]) / scale),
      Array.from({ length: d }, () => 0)
    );

    const dLdWq = outer(inputs[0], dLdq0);

    const dLdK = dLdS.map((grad) =>
      q0.map((val) => (grad * val) / scale)
    );

    const dLdWk = inputs.reduce((acc, inputRow, idx) => {
      const contribution = outer(inputRow, dLdK[idx]);
      return addMatrices(acc, contribution);
    }, Array.from({ length: d }, () => Array.from({ length: d }, () => 0)));

    Wq = addMatrices(Wq, scaleMatrix(dLdWq, -lr));
    Wk = addMatrices(Wk, scaleMatrix(dLdWk, -lr));
  }

  const attentionState = computeAttention(inputs, Wq, Wk, {
    temperature,
    biasStrength,
    inhibition,
    sharpening,
  });

  return { losses, attention: attentionState.weights, Wq, Wk };
};

const updateMatrix = (id, matrix) => {
  const el = document.getElementById(id);
  if (el) el.textContent = matrixToString(matrix);
};

const drawArchitectureDiagram = (svg) => {
  if (!svg) return;
  
  const NS = "http://www.w3.org/2000/svg";
  svg.innerHTML = "";
  svg.setAttribute("viewBox", "0 0 600 440");
  
  // Helper to create elements
  const el = (tag, attrs = {}, text = "") => {
    const elem = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([k, v]) => elem.setAttribute(k, v));
    if (text) elem.textContent = text;
    return elem;
  };
  
  // Layer positions
  const layers = [
    { y: 40, label: "Input (X)", color: "#2563eb", desc: "N×d tokens" },
    { y: 100, label: "Q, K, V", color: "#7c3aed", desc: "W_Q·X, W_K·X, W_V·X" },
    { y: 160, label: "Scores", color: "#dc2626", desc: "Q·K^T / √d_k" },
    { y: 220, label: "Softmax", color: "#ea580c", desc: "exp(·/τ) / Σ" },
    { y: 280, label: "Attention", color: "#16a34a", desc: "Normalized weights" },
    { y: 340, label: "Output (Z)", color: "#0891b2", desc: "Attention × V" },
  ];
  
  const boxX = 150;
  const boxW = 140;
  const boxH = 45;
  
  // Draw flow arrows and boxes
  layers.forEach((layer, i) => {
    // Arrow from previous layer
    if (i > 0) {
      const arrowY = layers[i - 1].y + boxH / 2;
      const arrowEndY = layer.y - boxH / 2;
      svg.appendChild(el("line", {
        x1: boxX + boxW / 2,
        y1: arrowY + 5,
        x2: boxX + boxW / 2,
        y2: arrowEndY - 5,
        stroke: "#94a3b8",
        "stroke-width": 2,
        "marker-end": "url(#arrowhead)"
      }));
    }
    
    // Box
    svg.appendChild(el("rect", {
      x: boxX,
      y: layer.y - boxH / 2,
      width: boxW,
      height: boxH,
      fill: layer.color,
      "fill-opacity": 0.15,
      stroke: layer.color,
      "stroke-width": 2,
      rx: 6
    }));
    
    // Label
    svg.appendChild(el("text", {
      x: boxX + boxW / 2,
      y: layer.y - 8,
      "text-anchor": "middle",
      "font-weight": "bold",
      "font-size": 14,
      fill: layer.color
    }, layer.label));
    
    // Description
    svg.appendChild(el("text", {
      x: boxX + boxW / 2,
      y: layer.y + 10,
      "text-anchor": "middle",
      "font-size": 11,
      fill: "#64748b"
    }, layer.desc));
  });
  
  // Arrowhead marker
  const defs = el("defs");
  const marker = el("marker", {
    id: "arrowhead",
    markerWidth: 10,
    markerHeight: 10,
    refX: 5,
    refY: 3,
    orient: "auto"
  });
  marker.appendChild(el("polygon", {
    points: "0 0, 10 3, 0 6",
    fill: "#94a3b8"
  }));
  defs.appendChild(marker);
  svg.insertBefore(defs, svg.firstChild);
  
  // Side annotations
  const annotations = [
    { y: 130, text: "Linear projections create 3 views of input" },
    { y: 190, text: "Similarity matrix (each query vs all keys)" },
    { y: 250, text: "Temperature τ controls peakiness" },
    { y: 310, text: "Each row sums to 1.0" },
    { y: 370, text: "Weighted sum of values per token" },
  ];
  
  annotations.forEach(ann => {
    svg.appendChild(el("text", {
      x: boxX + boxW + 20,
      y: ann.y,
      "font-size": 11,
      fill: "#475569"
    }, ann.text));
  });
  
  // Title
  svg.appendChild(el("text", {
    x: 300,
    y: 20,
    "text-anchor": "middle",
    "font-size": 13,
    "font-weight": "bold",
    fill: "#0f172a"
  }, "Forward Pass: Token Inputs → Attention Weights → Mixed Outputs"));
  
  // Data flow example on the right
  const exX = 430;
  const exY = 60;
  svg.appendChild(el("text", {
    x: exX,
    y: exY,
    "font-size": 11,
    "font-weight": "bold",
    fill: "#334155"
  }, "Example: 3 tokens"));
  
  const exampleSteps = [
    "X = [[x₁], [x₂], [x₃]]  (3×d)",
    "Q·K^T → 3×3 matrix",
    "Softmax → weights[i,j]",
    "Z[i] = Σⱼ weights[i,j]·V[j]"
  ];
  
  exampleSteps.forEach((step, i) => {
    svg.appendChild(el("text", {
      x: exX,
      y: exY + 25 + i * 22,
      "font-size": 10,
      "font-family": "monospace",
      fill: "#64748b"
    }, step));
  });
};

const setupOverview = () => {
  const tempSlider = document.getElementById("overview-temp");
  const seedSlider = document.getElementById("overview-seed");
  const tempValue = document.getElementById("overview-temp-value");
  const seedValue = document.getElementById("overview-seed-value");
  const reroll = document.getElementById("overview-reroll");
  const chart = document.getElementById("overview-chart");

  // Draw architecture diagram
  drawArchitectureDiagram(document.getElementById("architecture-diagram"));

  const render = () => {
    const temperature = parseFloat(tempSlider.value || 1);
    const seed = parseInt(seedSlider.value || 10, 10);
    tempValue.textContent = temperature.toFixed(1);
    seedValue.textContent = seed;

    const rand = seededRandom(seed);
    const inputs = DEFAULT_INPUTS.map((row) =>
      row.map((val) => val + (rand() - 0.5) * 0.6)
    );
    STATE.inputs = inputs;
    const Wq = randomMatrix(2, 2, 7);
    const Wk = randomMatrix(2, 2, 9);
    const { weights } = computeAttention(inputs, Wq, Wk, { temperature });
    updateMatrix("overview-matrix", weights);
    drawNetwork(document.getElementById("overview-network"), weights);

    const sim = simulateTraining({
      steps: 40,
      lr: 0.12,
      inputs,
      temperature,
    });
    drawLineChart(chart, [{ data: sim.losses }]);
  };

  tempSlider.value = "1.0";
  seedSlider.value = "10";
  render();

  [tempSlider, seedSlider].forEach((input) =>
    input.addEventListener("input", render)
  );
  reroll.addEventListener("click", render);
};

const setupModule = () => {
  const inputsEl = document.getElementById("module-inputs");
  const qkvEl = document.getElementById("module-qkv");
  const attentionEl = document.getElementById("module-attention");
  const networkEl = document.getElementById("module-network");
  const stepButton = document.getElementById("module-step");
  const resetButton = document.getElementById("module-reset");
  const labelEl = document.getElementById("module-step-label");

  const inputs = DEFAULT_INPUTS;
  const Wq = randomMatrix(2, 2, 4);
  const Wk = randomMatrix(2, 2, 5);
  const Wv = randomMatrix(2, 2, 6);
  const { Q, K, scores } = computeScores(inputs, Wq, Wk);
  const V = inputs.map((input) => matVecMul(Wv, input));
  const weights = scores.map((row) => softmax(row));

  const stageLabels = [
    "Stage 1: Input tokens (X)",
    "Stage 2: Linear projections (Q, K, V)",
    "Stage 3: Scores & attention weights"
  ];

  const stages = [
    () => {
      inputsEl.textContent = matrixToString(inputs);
      qkvEl.textContent = "➡️ Click 'Step' to compute Q, K, V";
      attentionEl.textContent = "⏳ Waiting...";
      if (labelEl) labelEl.textContent = stageLabels[0];
    },
    () => {
      qkvEl.textContent = `Q (queries):\n${matrixToString(Q)}\n\nK (keys):\n${matrixToString(
        K
      )}\n\nV (values):\n${matrixToString(V)}`;
      attentionEl.textContent = "➡️ Click 'Step' to compute attention";
      if (labelEl) labelEl.textContent = stageLabels[1];
    },
    () => {
      attentionEl.textContent = `Scores (Q·K^T/√d):\n${matrixToString(
        scores
      )}\n\nWeights (softmax):\n${matrixToString(weights)}`;
      drawNetwork(networkEl, weights);
      if (labelEl) labelEl.textContent = stageLabels[2];
    },
  ];

  let stage = 0;
  const runStage = () => {
    stages[stage]();
    stage = (stage + 1) % stages.length;
  };

  runStage();
  stepButton.addEventListener("click", runStage);
  resetButton.addEventListener("click", () => {
    stage = 0;
    runStage();
  });
};

const setupTraditional = () => {
  const stepsSlider = document.getElementById("traditional-steps");
  const lrSlider = document.getElementById("traditional-lr");
  const stepsValue = document.getElementById("traditional-steps-value");
  const lrValue = document.getElementById("traditional-lr-value");
  const matrix = document.getElementById("traditional-matrix");
  const chart = document.getElementById("traditional-chart");
  const image = document.getElementById("traditional-image");
  const network = document.getElementById("traditional-network");
  const runButton = document.getElementById("traditional-run");
  const logButton = document.getElementById("traditional-log");

  const run = () => {
    const steps = parseInt(stepsSlider.value || 80, 10);
    const lr = parseFloat(lrSlider.value || 0.12);
    stepsValue.textContent = steps;
    lrValue.textContent = lr.toFixed(2);

    const results = simulateTraining({ steps, lr, seed: 11 });
    matrix.textContent = matrixToString(results.attention);
    drawLineChart(chart, [{ data: results.losses }]);
    drawImageCanvas(image, results.attention);
    drawNetwork(network, results.attention);
    STATE.lastRun.traditional = {
      label: "Traditional",
      steps,
      loss: results.losses[results.losses.length - 1],
      attention: results.attention,
      losses: results.losses,
    };
  };

  stepsSlider.value = "80";
  lrSlider.value = "0.12";
  run();

  [stepsSlider, lrSlider].forEach((input) =>
    input.addEventListener("input", run)
  );
  runButton.addEventListener("click", run);
  logButton.addEventListener("click", () => {
    if (STATE.lastRun.traditional) {
      saveExperiment({ ...STATE.lastRun.traditional, type: "traditional" });
    }
  });
};

const setupPhysio = () => {
  const bias = document.getElementById("physio-bias");
  const inhibition = document.getElementById("physio-inhibition");
  const sharpen = document.getElementById("physio-sharpen");
  const noise = document.getElementById("physio-noise");
  const biasValue = document.getElementById("physio-bias-value");
  const inhibitionValue = document.getElementById("physio-inhibition-value");
  const sharpenValue = document.getElementById("physio-sharpen-value");
  const noiseValue = document.getElementById("physio-noise-value");
  const matrix = document.getElementById("physio-matrix");
  const chart = document.getElementById("physio-chart");
  const image = document.getElementById("physio-image");
  const network = document.getElementById("physio-network");
  const runButton = document.getElementById("physio-run");
  const logButton = document.getElementById("physio-log");

  const run = () => {
    const biasStrength = parseFloat(bias.value || 0.25);
    const inhibitionStrength = parseFloat(inhibition.value || 0.3);
    const sharpening = parseFloat(sharpen.value || 1.2);
    const noiseCorr = parseFloat(noise.value || 0.15);
    biasValue.textContent = biasStrength.toFixed(2);
    inhibitionValue.textContent = inhibitionStrength.toFixed(2);
    sharpenValue.textContent = sharpening.toFixed(1);
    noiseValue.textContent = noiseCorr.toFixed(2);

    const results = simulateTraining({
      steps: 80,
      lr: 0.12,
      seed: 13,
      biasStrength,
      inhibition: inhibitionStrength,
      sharpening,
      noiseCorr,
    });
    matrix.textContent = matrixToString(results.attention);
    drawLineChart(chart, [{ data: results.losses }]);
    drawImageCanvas(image, results.attention);
    drawNetwork(network, results.attention);
    STATE.lastRun.physio = {
      label: "Physio",
      steps: 80,
      loss: results.losses[results.losses.length - 1],
      attention: results.attention,
      losses: results.losses,
      params: { biasStrength, inhibition: inhibitionStrength, sharpening, noiseCorr },
    };
  };

  bias.value = "0.25";
  inhibition.value = "0.3";
  sharpen.value = "1.2";
  noise.value = "0.15";
  run();

  [bias, inhibition, sharpen, noise].forEach((input) =>
    input.addEventListener("input", run)
  );
  runButton.addEventListener("click", run);
  logButton.addEventListener("click", () => {
    if (STATE.lastRun.physio) {
      saveExperiment({ ...STATE.lastRun.physio, type: "physio" });
    }
  });
};

const setupVariants = () => {
  const temp = document.getElementById("variant-temp");
  const sparsity = document.getElementById("variant-sparsity");
  const competition = document.getElementById("variant-competition");
  const tempValue = document.getElementById("variant-temp-value");
  const sparsityValue = document.getElementById("variant-sparsity-value");
  const competitionValue = document.getElementById("variant-competition-value");
  const matrix = document.getElementById("variant-matrix");
  const chart = document.getElementById("variant-chart");
  const network = document.getElementById("variant-network");
  const runButton = document.getElementById("variant-run");
  const logButton = document.getElementById("variant-log");

  const run = () => {
    const temperature = parseFloat(temp.value || 1);
    const k = parseInt(sparsity.value || 2, 10);
    const comp = parseFloat(competition.value || 0.4);
    tempValue.textContent = temperature.toFixed(1);
    sparsityValue.textContent = k;
    competitionValue.textContent = comp.toFixed(1);

    const results = simulateTraining({
      steps: 60,
      lr: 0.1,
      seed: 17,
      temperature,
      biasStrength: comp,
    });

    let weights = results.attention.map((row) => {
      const threshold = [...row]
        .sort((a, b) => b - a)
        .slice(k - 1, k)[0];
      return row.map((val) => (val >= threshold ? val : 0));
    });
    const rowSums = weights.map((row) => row.reduce((a, b) => a + b, 0));
    weights = weights.map((row, idx) =>
      row.map((val) => (rowSums[idx] ? val / rowSums[idx] : 0))
    );

    matrix.textContent = matrixToString(weights);
    drawLineChart(chart, [{ data: results.losses }]);
    drawNetwork(network, weights);
    STATE.lastRun.variant = {
      label: "Variant",
      steps: 60,
      loss: results.losses[results.losses.length - 1],
      attention: weights,
      losses: results.losses,
      params: { temperature, k, competition: comp },
    };
  };

  temp.value = "1.0";
  sparsity.value = "2";
  competition.value = "0.4";
  run();

  [temp, sparsity, competition].forEach((input) =>
    input.addEventListener("input", run)
  );
  runButton.addEventListener("click", run);
  logButton.addEventListener("click", () => {
    if (STATE.lastRun.variant) {
      saveExperiment({ ...STATE.lastRun.variant, type: "variant" });
    }
  });
};

const setupDirection1 = () => {
  const kInput = document.getElementById("dir1-k");
  const tempInput = document.getElementById("dir1-temperature");
  const kValue = document.getElementById("dir1-k-value");
  const tempValue = document.getElementById("dir1-temperature-value");
  const matrix = document.getElementById("dir1-matrix");
  const network = document.getElementById("dir1-network");
  const runButton = document.getElementById("dir1-run");
  const logButton = document.getElementById("dir1-log");

  const run = () => {
    const k = parseInt(kInput.value || 1, 10);
    const temperature = parseFloat(tempInput.value || 1);
    kValue.textContent = k;
    tempValue.textContent = temperature.toFixed(1);

    const results = simulateTraining({
      steps: 50,
      lr: 0.1,
      seed: 21,
      temperature,
    });

    let weights = results.attention.map((row) => {
      const threshold = [...row]
        .sort((a, b) => b - a)
        .slice(k - 1, k)[0];
      return row.map((val) => (val >= threshold ? val : 0));
    });
    const rowSums = weights.map((row) => row.reduce((a, b) => a + b, 0));
    weights = weights.map((row, idx) =>
      row.map((val) => (rowSums[idx] ? val / rowSums[idx] : 0))
    );

    matrix.textContent = matrixToString(weights);
    drawNetwork(network, weights);
    STATE.lastRun.direction1 = {
      label: "Sparse",
      steps: 50,
      loss: results.losses[results.losses.length - 1],
      attention: weights,
      losses: results.losses,
      params: { k, temperature },
    };
  };

  kInput.value = "1";
  tempInput.value = "1.0";
  run();

  [kInput, tempInput].forEach((input) =>
    input.addEventListener("input", run)
  );
  runButton.addEventListener("click", run);
  logButton.addEventListener("click", () => {
    if (STATE.lastRun.direction1) {
      saveExperiment({ ...STATE.lastRun.direction1, type: "direction1" });
    }
  });
};

const setupDirection2 = () => {
  const control = document.getElementById("dir2-control");
  const interrupt = document.getElementById("dir2-interrupt");
  const controlValue = document.getElementById("dir2-control-value");
  const interruptValue = document.getElementById("dir2-interrupt-value");
  const matrix = document.getElementById("dir2-matrix");
  const network = document.getElementById("dir2-network");
  const runButton = document.getElementById("dir2-run");
  const logButton = document.getElementById("dir2-log");

  const run = () => {
    const controlStrength = parseFloat(control.value || 0.4);
    const interruptStrength = parseFloat(interrupt.value || 0.2);
    controlValue.textContent = controlStrength.toFixed(2);
    interruptValue.textContent = interruptStrength.toFixed(2);

    const results = simulateTraining({
      steps: 60,
      lr: 0.1,
      seed: 29,
      biasStrength: controlStrength,
      inhibition: interruptStrength * 0.4,
    });

    matrix.textContent = matrixToString(results.attention);
    drawNetwork(network, results.attention);
    STATE.lastRun.direction2 = {
      label: "Predictive",
      steps: 60,
      loss: results.losses[results.losses.length - 1],
      attention: results.attention,
      losses: results.losses,
      params: { controlStrength, interruptStrength },
    };
  };

  control.value = "0.4";
  interrupt.value = "0.2";
  run();

  [control, interrupt].forEach((input) =>
    input.addEventListener("input", run)
  );
  runButton.addEventListener("click", run);
  logButton.addEventListener("click", () => {
    if (STATE.lastRun.direction2) {
      saveExperiment({ ...STATE.lastRun.direction2, type: "direction2" });
    }
  });
};

const setupResults = () => {
  const table = document.getElementById("results-table");
  const matrixContainer = document.getElementById("results-matrices");
  const imageContainer = document.getElementById("results-images");
  const chart = document.getElementById("results-chart");
  const legendContainer = document.getElementById("results-legend");
  const refreshButton = document.getElementById("results-refresh");

  const render = () => {
    const log = getExperimentLog();
    if (!log.length) {
      table.textContent = "No experiments logged yet.";
      matrixContainer.innerHTML = "<p class='muted'>No matrices logged.</p>";
      imageContainer.innerHTML = "<p class='muted'>No images logged.</p>";
      if (legendContainer) legendContainer.innerHTML = "";
      drawLineChart(chart, []);
      return;
    }

    table.textContent = log
      .slice(-8)
      .map(
        (entry) => {
          const varInfo = entry.varianceExplained;
          const varStr = varInfo ? ` | var1: ${varInfo.top1.toFixed(1)}%` : "";
          return `${entry.label.padEnd(12)} | steps: ${String(entry.steps).padEnd(
            3
          )} | loss: ${entry.loss.toFixed(4)}${varStr}`;
        }
      )
      .join("\n");

    matrixContainer.innerHTML = log
      .slice(-4)
      .map(
        (entry) =>
          `<pre class="matrix">${entry.label}\n${matrixToString(
            entry.attention
          )}</pre>`
      )
      .join("");

    imageContainer.innerHTML = log
      .slice(-3)
      .map(
        (entry, idx) =>
          `<div class="card"><h3>${entry.label}</h3><canvas class="chart" id="results-image-${idx}"></canvas></div>`
      )
      .join("");

    log.slice(-3).forEach((entry, idx) => {
      const canvas = document.getElementById(`results-image-${idx}`);
      drawImageCanvas(canvas, entry.attention);
    });

    const recentRuns = log.slice(-5);
    const dataSets = recentRuns.map((entry) => ({
      data: entry.losses || [],
      label: entry.label || entry.variant || "Unknown",
    }));
    drawLineChart(chart, dataSets);

    // Build legend
    if (legendContainer) {
      legendContainer.innerHTML = recentRuns
        .map((entry) => {
          const label = entry.label || entry.variant || "Unknown";
          const color = VARIANT_COLORS[label] || VARIANT_COLORS.default;
          return `
            <div style="display: flex; align-items: center; gap: 0.5rem;">
              <div style="width: 20px; height: 3px; background: ${color}; border-radius: 2px;"></div>
              <span style="font-size: 0.9rem; font-weight: 500; color: #334155;">${label}</span>
              <span style="font-size: 0.85rem; color: #64748b;">(loss: ${entry.loss.toFixed(4)})</span>
            </div>
          `;
        })
        .join("");
    }
  };

  render();
  refreshButton.addEventListener("click", render);
};

// Simple eigenvalue approximation (power iteration for largest eigenvalue)
const powerIteration = (matrix, iterations = 20) => {
  const n = matrix.length;
  let vec = Array(n).fill(1 / Math.sqrt(n));
  
  for (let iter = 0; iter < iterations; iter++) {
    // Multiply matrix by vector
    const newVec = matrix.map(row => 
      row.reduce((sum, val, i) => sum + val * vec[i], 0)
    );
    // Normalize
    const norm = Math.sqrt(newVec.reduce((sum, v) => sum + v * v, 0));
    vec = newVec.map(v => v / norm);
  }
  
  // Compute eigenvalue
  const Av = matrix.map(row => 
    row.reduce((sum, val, i) => sum + val * vec[i], 0)
  );
  const eigenvalue = vec.reduce((sum, v, i) => sum + v * Av[i], 0);
  
  return { value: eigenvalue, vector: vec };
};

// Approximate top-k eigenvalues (simple, not perfect but good for demo)
const approximateEigenvalues = (matrix, k = 3) => {
  const results = [];
  let remaining = matrix.map(row => [...row]);
  
  for (let i = 0; i < Math.min(k, matrix.length); i++) {
    const { value, vector } = powerIteration(remaining, 20);
    results.push({ value: Math.abs(value), vector });
    
    // Deflate: subtract outer product of eigenvector
    remaining = remaining.map((row, r) =>
      row.map((val, c) => val - value * vector[r] * vector[c])
    );
  }
  
  return results;
};

// Correlation matrix
const correlationMatrix = (matrix) => {
  const n = matrix.length;
  const means = matrix.map(row => row.reduce((a,b) => a+b, 0) / row.length);
  const stds = matrix.map((row, i) => {
    const variance = row.reduce((sum, val) => sum + (val - means[i]) ** 2, 0) / row.length;
    return Math.sqrt(variance + 1e-8);
  });
  
  const corr = [];
  for (let i = 0; i < n; i++) {
    corr[i] = [];
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < matrix[i].length; k++) {
        sum += ((matrix[i][k] - means[i]) / stds[i]) * ((matrix[j][k] - means[j]) / stds[j]);
      }
      corr[i][j] = sum / matrix[i].length;
    }
  }
  return corr;
};

// Low-rank approximation using top-k components
const lowRankApprox = (matrix, eigenData, k) => {
  const n = matrix.length;
  const approx = Array.from({ length: n }, () => Array(n).fill(0));
  
  for (let i = 0; i < Math.min(k, eigenData.length); i++) {
    const { value, vector } = eigenData[i];
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        approx[r][c] += value * vector[r] * vector[c];
      }
    }
  }
  
  return approx;
};

const setupComparison = () => {
  const gradientChart = document.getElementById("comparison-gradient-chart");
  
  if (gradientChart) {
    // Simulate gradient norms across layers for different architectures
    const layers = [1, 2, 3, 4, 5, 6, 7, 8];
    
    // Transformer with residuals: stable gradient norms
    const transformer = layers.map(() => 0.8 + Math.random() * 0.3);
    
    // RNN: vanishing gradients (exponential decay)
    const rnn = layers.map((l) => 1.0 * Math.pow(0.6, l - 1) + Math.random() * 0.1);
    
    // Without residuals: degrading gradients
    const noResidual = layers.map((l) => 1.0 * Math.pow(0.75, l - 1) + Math.random() * 0.15);
    
    const ctx = gradientChart.getContext("2d");
    const width = gradientChart.clientWidth;
    const height = gradientChart.clientHeight;
    gradientChart.width = width;
    gradientChart.height = height;
    ctx.clearRect(0, 0, width, height);
    
    const padding = { top: 20, right: 20, bottom: 40, left: 50 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    
    // Draw grid
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (chartHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }
    
    // Y-axis labels
    ctx.fillStyle = "#64748b";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "right";
    for (let i = 0; i <= 5; i++) {
      const val = 1.2 - (i * 0.24);
      const y = padding.top + (chartHeight / 5) * i;
      ctx.fillText(val.toFixed(1), padding.left - 5, y + 3);
    }
    
    // Draw lines
    const datasets = [
      { data: transformer, color: "#2563eb", label: "Transformer (residual)" },
      { data: noResidual, color: "#f59e0b", label: "No residual" },
      { data: rnn, color: "#ef4444", label: "RNN" }
    ];
    
    datasets.forEach(({ data, color, label }) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      data.forEach((val, idx) => {
        const x = padding.left + (idx / (data.length - 1)) * chartWidth;
        const y = padding.top + chartHeight - (val / 1.2) * chartHeight;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
    
    // X-axis labels
    ctx.fillStyle = "#64748b";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    layers.forEach((layer, idx) => {
      const x = padding.left + (idx / (layers.length - 1)) * chartWidth;
      ctx.fillText(`L${layer}`, x, height - 20);
    });
    
    // Axis titles
    ctx.font = "11px sans-serif";
    ctx.fillText("Layer Depth", width / 2, height - 5);
    
    ctx.save();
    ctx.translate(12, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Gradient Norm", 0, 0);
    ctx.restore();
    
    // Legend
    const legendY = padding.top + 10;
    datasets.forEach((ds, i) => {
      const x = padding.left + i * 120;
      ctx.fillStyle = ds.color;
      ctx.fillRect(x, legendY, 15, 3);
      ctx.fillStyle = "#0f172a";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(ds.label, x + 20, legendY + 3);
    });
  }
};

const setupStatistics = () => {
  const covEl = document.getElementById("stats-cov");
  const eigenEl = document.getElementById("stats-eigen");
  const eigenChart = document.getElementById("stats-eigen-chart");
  const svdChart = document.getElementById("stats-svd-chart");
  const reconEl = document.getElementById("stats-recon");
  const corrEl = document.getElementById("stats-corr");
  const attnEl = document.getElementById("stats-attn");
  const rankSlider = document.getElementById("stats-rank-slider");
  const rankValue = document.getElementById("stats-rank-value");
  const varianceEl = document.getElementById("stats-variance-explained");

  // Generate sample data
  const inputs = DEFAULT_INPUTS;
  const Wq = randomMatrix(2, 2, 11);
  const Wk = randomMatrix(2, 2, 13);
  const { Q, K, scores } = computeScores(inputs, Wq, Wk);
  const weights = scores.map(row => softmax(row, 1.0));
  
  // Q·K^T (covariance-like matrix)
  const QKT = Q.map(q => K.map(k => q.reduce((sum, qv, i) => sum + qv * k[i], 0)));
  
  if (covEl) covEl.textContent = matrixToString(QKT);
  if (attnEl) attnEl.textContent = matrixToString(weights);
  
  // Correlation
  const corr = correlationMatrix(Q);
  if (corrEl) corrEl.textContent = matrixToString(corr);
  
  // Eigenvalue decomposition
  const eigenData = approximateEigenvalues(weights, 3);
  const eigenvalues = eigenData.map(e => e.value);
  const eigenvectors = eigenData.map(e => e.vector);
  
  if (eigenEl) {
    eigenEl.textContent = eigenvectors
      .map((vec, i) => `λ${i+1} = ${eigenvalues[i].toFixed(3)}\nv${i+1} = [${vec.map(v => v.toFixed(3)).join(", ")}]`)
      .join("\n\n");
  }
  
  // Eigenvalue chart (bar chart)
  if (eigenChart) {
    const ctx = eigenChart.getContext("2d");
    const width = eigenChart.clientWidth;
    const height = eigenChart.clientHeight;
    eigenChart.width = width;
    eigenChart.height = height;
    ctx.clearRect(0, 0, width, height);
    
    const maxEigen = Math.max(...eigenvalues, 1);
    const barWidth = width / (eigenvalues.length * 2);
    const padding = 30;
    
    eigenvalues.forEach((val, i) => {
      const barHeight = (val / maxEigen) * (height - padding * 2);
      const x = padding + i * barWidth * 2;
      const y = height - padding - barHeight;
      
      ctx.fillStyle = "#2563eb";
      ctx.fillRect(x, y, barWidth, barHeight);
      
      ctx.fillStyle = "#0f172a";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(`λ${i+1}`, x + barWidth / 2, height - 10);
      ctx.fillText(val.toFixed(2), x + barWidth / 2, y - 5);
    });
  }
  
  // SVD / Rank approximation
  const updateRank = () => {
    const k = parseInt(rankSlider.value, 10);
    if (rankValue) rankValue.textContent = k;
    
    const approx = lowRankApprox(weights, eigenData, k);
    if (reconEl) reconEl.textContent = matrixToString(approx);
    
    // Variance explained
    const totalVariance = eigenvalues.reduce((a, b) => a + b, 0);
    const explainedVariance = eigenvalues.slice(0, k).reduce((a, b) => a + b, 0);
    const pct = ((explainedVariance / totalVariance) * 100).toFixed(1);
    if (varianceEl) varianceEl.textContent = `Variance explained: ${pct}%`;
    
    // SVD chart (same as eigenvalue chart for simplicity)
    if (svdChart) {
      const ctx = svdChart.getContext("2d");
      const width = svdChart.clientWidth;
      const height = svdChart.clientHeight;
      svdChart.width = width;
      svdChart.height = height;
      ctx.clearRect(0, 0, width, height);
      
      const maxSV = Math.max(...eigenvalues, 1);
      const barWidth = width / (eigenvalues.length * 2);
      const padding = 25;
      
      eigenvalues.forEach((val, i) => {
        const barHeight = (val / maxSV) * (height - padding * 2);
        const x = padding + i * barWidth * 2;
        const y = height - padding - barHeight;
        
        const color = i < k ? "#16a34a" : "#cbd5e1";
        ctx.fillStyle = color;
        ctx.fillRect(x, y, barWidth, barHeight);
        
        ctx.fillStyle = "#0f172a";
        ctx.font = "9px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(`σ${i+1}`, x + barWidth / 2, height - 8);
      });
      
      // Legend
      ctx.fillStyle = "#16a34a";
      ctx.fillRect(10, 10, 12, 8);
      ctx.fillStyle = "#0f172a";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("Used", 25, 17);
      
      ctx.fillStyle = "#cbd5e1";
      ctx.fillRect(70, 10, 12, 8);
      ctx.fillStyle = "#0f172a";
      ctx.fillText("Dropped", 85, 17);
    }
  };
  
  if (rankSlider) {
    rankSlider.value = "3";
    updateRank();
    rankSlider.addEventListener("input", updateRank);
  }
};

const initTab = (tab) => {
  switch (tab) {
    case "comparison":
      setupComparison();
      break;
    case "statistics":
      setupStatistics();
      break;
    case "overview":
      setupOverview();
      break;
    case "module":
      setupModule();
      break;
    case "traditional":
      setupTraditional();
      break;
    case "physio":
      setupPhysio();
      break;
    case "variants":
      setupVariants();
      break;
    case "direction1":
      setupDirection1();
      break;
    case "direction2":
      setupDirection2();
      break;
    case "results":
      setupResults();
      break;
    default:
      break;
  }
};

const setActiveTab = (tab) => {
  document.querySelectorAll("#tab-nav button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });
};

const loadTab = async (tab) => {
  const container = document.getElementById("tab-content");
  container.innerHTML = "<div class='loading'>Loading...</div>";
  try {
    const response = await fetch(TAB_MAP[tab]);
    if (!response.ok) {
      console.error(`Failed to load tab: ${tab}, status: ${response.status}`);
      throw new Error(`Failed to load: ${response.status}`);
    }
    const html = await response.text();
    container.innerHTML = html;
    initTab(tab);
    setActiveTab(tab);
    window.location.hash = tab;
  } catch (error) {
    console.error(`Error loading tab ${tab}:`, error);
    container.innerHTML =
      `<div class='loading'>Failed to load tab: ${tab}. Error: ${error.message}<br>Try opening index.html via a local server (e.g., python3 -m http.server)</div>`;
  }
};

const setupTabs = () => {
  const nav = document.getElementById("tab-nav");
  nav.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-tab]");
    if (!button) return;
    loadTab(button.dataset.tab);
  });

  const startTab = window.location.hash.replace("#", "") || "comparison";
  loadTab(TAB_MAP[startTab] ? startTab : "comparison");
};

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  const resetButton = document.getElementById("reset-state");
  resetButton.addEventListener("click", () => {
    resetExperiments();
    alert("Experiment log cleared.");
  });
});
