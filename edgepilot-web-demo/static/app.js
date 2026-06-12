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
let pollTimer = null;

const $ = (id) => document.getElementById(id);

function icon(name) {
  return `<svg><use href="#icon-${name}"></use></svg>`;
}

function initTimeline(state = "idle", candidateStatus = {}) {
  const runningNames = Object.entries(candidateStatus).filter(([, status]) => status === "running").map(([name]) => name);
  const timeline = $("timeline");
  timeline.innerHTML = steps.map(([title, text], idx) => {
    let cls = "";
    if (state === "done") cls = "done";
    if (state === "running" && idx < 2) cls = "done";
    if (state === "running" && idx === 2) cls = "running";
    if (state === "failed" && idx === 2) cls = "failed";
    const note = runningNames.length && idx >= 3 && idx <= 6 ? `当前候选: ${runningNames.join(", ")}` : text;
    return `<div class="step ${cls}"><div class="step-title"><span class="dot"></span>${title}</div><p>${note}</p></div>`;
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

function statusBadge(status) {
  const label = {
    pending: "待执行",
    running: "执行中",
    done: "已执行",
    failed: "失败"
  }[status] || status;
  const cls = status === "done" ? "pass" : status === "running" ? "best" : status === "failed" ? "fail" : "";
  return `<span class="badge ${cls}">${label}</span>`;
}

function renderCandidateStatus(job) {
  const items = Object.entries(job.candidate_status || {});
  if (!items.length) return;
  $("resultRows").innerHTML = items.map(([name, status]) => `
    <tr>
      <td><strong>${name}</strong></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>${statusBadge(status)}</td>
    </tr>
  `).join("");
}

function renderResults(data, mode) {
  const req = data.result ? data.result.plan.request : data.request;
  const evalData = data.result ? data.result.evaluation : data.evaluation;
  const recommended = evalData?.recommended || {};
  const gpu = data.result?.env?.tools?.nvidia_smi?.stdout || data.env?.tools?.nvidia_smi?.stdout || "not detected";

  $("modelName").textContent = req?.project === "yolov8" ? "YOLOv8s-pose" : (req?.project || "-");
  $("targetText").textContent = `${req?.target?.hardware || "NVIDIA T4"} / ≤${req?.target?.accuracy_drop_max_pct || 1}% / ≥${req?.target?.speedup_min || 2}x`;
  $("recommendText").textContent = recommended.name || (mode === "real_search" ? "等待真实指标" : "暂无推荐");
  $("gpuText").textContent = gpu;

  const rows = evalData?.evaluated || [];
  if (rows.length) {
    $("resultRows").innerHTML = rows.map((item) => {
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
  }

  const paths = data.result?.paths;
  if (paths) {
    const artifacts = [
      ["report.md", paths.report],
      ["commands.sh", paths.commands],
      ["plan.json", paths.plan],
      ["env.json", paths.env],
      ["evaluation.json", paths.evaluation]
    ];
    $("artifactList").innerHTML = artifacts.map(([name, path]) => `
      <div class="artifact">${icon("file")}<div><strong>${name}</strong><span>${path}</span></div></div>
    `).join("");
    latest.report = data.result.report;
    latest.commands = data.result.commands;
    $("preview").textContent = latest.report;
  }
}

function updateJob(job) {
  $("serverStatus").innerHTML = `<span></span> ${job.stage || job.status}`;
  $("liveLog").textContent = (job.log || []).join("");
  $("liveLog").scrollTop = $("liveLog").scrollHeight;
  initTimeline(job.status === "done" ? "done" : job.status === "failed" ? "failed" : "running", job.candidate_status);
  renderCandidateStatus(job);
  if (job.result) renderResults(job, job.mode);
}

async function pollJob(jobId) {
  const res = await fetch(`/api/job/${jobId}`);
  const data = await res.json();
  if (!data.ok) throw new Error(data.error || "job not found");
  updateJob(data.job);
  if (data.job.status === "done" || data.job.status === "failed") {
    clearInterval(pollTimer);
    pollTimer = null;
    $("runBtn").disabled = false;
    $("runBtn").innerHTML = `${icon("play")} 运行 Agent Demo`;
  }
}

async function runAgent() {
  if (pollTimer) clearInterval(pollTimer);
  const btn = $("runBtn");
  const realSearch = $("realSearch").checked;
  btn.disabled = true;
  btn.innerHTML = `${icon("play")} Agent 运行中`;
  $("runMode").textContent = realSearch ? "真实候选搜索" : "快速演示模式";
  $("liveLog").textContent = "正在提交任务...\n";
  initTimeline("running");
  $("serverStatus").innerHTML = "<span></span> 正在启动任务";

  try {
    const res = await fetch("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: $("prompt").value,
        form: collectForm(),
        realSearch
      })
    });
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || "Agent start failed");
    pollTimer = setInterval(() => pollJob(data.job_id).catch((err) => {
      clearInterval(pollTimer);
      pollTimer = null;
      $("liveLog").textContent += `\n[frontend error] ${err}`;
      btn.disabled = false;
    }), 1000);
    await pollJob(data.job_id);
  } catch (err) {
    $("preview").textContent = String(err);
    $("liveLog").textContent += `\n${err}`;
    $("serverStatus").innerHTML = "<span></span> 运行失败";
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

$("realSearch").addEventListener("change", () => {
  $("runMode").textContent = $("realSearch").checked ? "真实候选搜索" : "快速演示模式";
});
$("runBtn").addEventListener("click", runAgent);

initTimeline();
loadTemplate();
