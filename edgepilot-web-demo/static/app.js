const steps = [
  ["环境检查", "检测 GPU、CUDA、TensorRT 与项目布局"],
  ["需求解析", "从提示词提取模型路径、硬件和约束"],
  ["Dense/Baseline", "建立原始精度与延迟基线"],
  ["FP16 TRT", "导出 TensorRT FP16 engine"],
  ["INT8 PTQ", "使用校准集快速量化并评估精度"],
  ["INT8 QAT", "插入 Q/DQ 节点后微调恢复精度"],
  ["Prune + QAT", "DepGraph 结构化剪枝后联合量化"],
  ["交付输出", "生成 report、commands 和推荐策略"]
];

let latest = { report: "", commands: "" };

const $ = (id) => document.getElementById(id);

function icon(name) {
  return `<svg><use href="#icon-${name}"></use></svg>`;
}

function initTimeline(state = "idle") {
  const timeline = $("timeline");
  timeline.innerHTML = steps.map(([title, text], idx) => {
    const cls = state === "done" ? "done" : state === "running" && idx < 2 ? "running" : "";
    return `<div class="step ${cls}"><div class="step-title"><span class="dot"></span>${title}</div><p>${text}</p></div>`;
  }).join("");
}

async function loadTemplate() {
  const res = await fetch("/api/template");
  const data = await res.json();
  $("prompt").value = data.prompt;
}

function collectForm() {
  const strategies = [...document.querySelectorAll(".strategy:checked")].map((el) => el.value);
  return {
    modelPath: $("modelPath").value.trim(),
    dataYaml: $("dataYaml").value.trim(),
    targetHardware: $("targetHardware").value.trim(),
    demoHardware: $("demoHardware").value.trim(),
    metric: $("metric").value.trim(),
    accuracyDropMax: $("accuracyDropMax").value,
    speedupMin: $("speedupMin").value,
    latencyMax: $("latencyMax").value,
    strategies
  };
}

function fmt(value, digits = 2) {
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(digits) : "-";
}

function renderResults(data) {
  const req = data.request;
  const evalData = data.evaluation;
  const recommended = evalData.recommended || {};
  const gpu = data.env?.tools?.nvidia_smi?.stdout || "not detected";

  $("modelName").textContent = req.model.includes("yolov8") || req.project === "yolov8" ? "YOLOv8s-pose" : req.project;
  $("targetText").textContent = `${req.target.hardware} / ≤${req.target.accuracy_drop_max_pct}% / ≥${req.target.speedup_min}x`;
  $("recommendText").textContent = recommended.name || "暂无推荐";
  $("gpuText").textContent = gpu;

  $("resultRows").innerHTML = (evalData.evaluated || []).map((item) => {
    const status = item.accepted
      ? `<span class="badge ${item.name === recommended.name ? "best" : "pass"}">${item.name === recommended.name ? "推荐" : "通过"}</span>`
      : `<span class="badge fail">未通过</span>`;
    return `<tr>
      <td><strong>${item.name}</strong></td>
      <td>${item.strategy}</td>
      <td>${fmt(item.latency_ms, 3)} ms</td>
      <td>${fmt(item.speedup, 2)}x</td>
      <td>${fmt(item.accuracy, 3)}</td>
      <td>${fmt(item.accuracy_drop_pct, 3)}%</td>
      <td>${status}</td>
    </tr>`;
  }).join("");

  const artifacts = [
    ["report.md", data.paths.report],
    ["commands.sh", data.paths.commands],
    ["plan.json", data.paths.plan],
    ["env.json", data.paths.env],
    ["evaluation.json", data.paths.evaluation]
  ];
  $("artifactList").innerHTML = artifacts.map(([name, path]) => `
    <div class="artifact">${icon("file")}<div><strong>${name}</strong><span>${path}</span></div></div>
  `).join("");

  latest.report = data.report;
  latest.commands = data.commands;
  $("preview").textContent = latest.report;
}

async function runAgent() {
  const btn = $("runBtn");
  btn.disabled = true;
  btn.innerHTML = `${icon("play")} Agent 运行中`;
  initTimeline("running");
  $("serverStatus").innerHTML = "<span></span> 正在生成方案";

  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: $("prompt").value,
        form: collectForm(),
        execute: $("executeReal").checked
      })
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      throw new Error(data.stderr || data.error || "Agent run failed");
    }
    renderResults(data);
    initTimeline("done");
    $("serverStatus").innerHTML = "<span></span> 已生成交付结果";
  } catch (err) {
    $("preview").textContent = String(err);
    $("serverStatus").innerHTML = "<span></span> 运行失败";
  } finally {
    btn.disabled = false;
    btn.innerHTML = `${icon("play")} 运行 Agent Demo`;
  }
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    tab.classList.add("active");
    $("preview").textContent = latest[tab.dataset.tab] || "";
  });
});

$("runBtn").addEventListener("click", runAgent);

initTimeline();
loadTemplate();
