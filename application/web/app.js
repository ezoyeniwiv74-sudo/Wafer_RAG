const FIELD_LABELS = {
  platform: "平台", defect_type: "缺陷类型", machine: "机器 / 腔体", station: "站点",
  product_id: "产品", lot_id: "LotID", map_location: "Map 位置", dft_location: "DFT 位置",
};
const DETAIL_FIELDS = [
  ["dn_no", "DN 编号"], ["class_type", "类型"], ["lot_id", "LotID"],
  ["product_id", "产品"], ["platform", "平台"], ["station", "站点"],
  ["machine", "机器 / 腔体"], ["scan_time", "扫描时间"], ["defect_type", "缺陷类型"],
  ["map_location", "Map 位置"], ["dft_location", "DFT 位置"],
  ["total_wafers", "Wafer 总数"], ["scanned_wafers", "已扫描 Wafer"], ["bad_wafers", "坏 Wafer"],
];
const PIE_COLORS = ["#2f6fed", "#19a974", "#f59e0b", "#8b5cf6", "#ef5da8", "#06b6d4", "#f06a4b", "#64748b"];

const state = {
  route: "text", textMode: "semantic", filterOptions: {}, selectedFilters: {},
  textQueryByMode: {semantic: "", dn: ""},
  fieldConditions: [{id: 1, field: "platform", value: ""}], nextConditionId: 2,
  pptFiles: [], pptHistory: [], pptDictionary: {}, imageFiles: [], imagePreviewUrls: [], lastImageSearch: null,
  imageSearchData: null, imageResultOptions: {}, imageResultFilters: {}, imageManualRegions: {}, imageModalities: {},
  imageParsingPpts: 0,
  roiEditorIndex: null, roiEditorDraft: null, roiEditorRegions: [], roiEditorMode: "rectangle", roiEditorPoints: [], roiFreehandActive: false,
  comparisonSelection: new Set(),
  textAnalysisSelection: new Set(), imageAnalysisSelection: new Set(),
  reportReturnRoute: "text", reportData: null,
  reportDraft: {query:"", results:[], images:[]}, reportImagePreviewUrls: [],
  lastTextSearchData: null, lastTextSearchContext: "",
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const clean = value => value === null || value === undefined || ["", "UNKNOWN", "UNKNOWN_DN", "nan"].includes(String(value).trim()) ? "—" : String(value);
const escapeHtml = value => String(value ?? "").replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
const percent = value => `${Math.max(0, Math.min(1, Number(value) || 0)) * 100}`.replace(/(\.\d).*/, "$1") + "%";

function toast(message) {
  const node = $("#toast"); node.textContent = message; node.classList.add("show");
  clearTimeout(toast.timer); toast.timer = setTimeout(() => node.classList.remove("show"), 2600);
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({detail: `请求失败 (${response.status})`}));
  if (!response.ok) throw new Error(body.detail || "请求失败");
  return body;
}

function route() {
  const target = (location.hash.match(/^#\/(ppt|text|image|report)/) || [])[1] || "text";
  state.route = target;
  $$(".page").forEach(page => page.classList.toggle("active", page.id === `page-${target}`));
  $$(".nav-list a").forEach(link => link.classList.toggle("active", link.dataset.route === target));
  const names = {ppt:["PPT 解析","导入标准缺陷报告"], text:["文字检索","查询历史异常记录"], image:["图片检索","查找视觉相近记录"], report:["分析报告","选定记录的可视化根因分析"]};
  $("#pageTitle").textContent = names[target][0]; $("#pageSubtitle").textContent = names[target][1];
  renderAnalysisFloat();
  renderComparisonFloat();
}

function renderAnalysisFloat() {
  const node = $("#analysisFloat");
  if (!node) return;
  const count = state.route === "text" ? state.textAnalysisSelection.size : state.route === "image" ? state.imageAnalysisSelection.size : 0;
  node.hidden = !count;
  $("#analysisFloatCount").textContent = `已选 ${count} 条`;
  $("#analysisFloatRun").textContent = `立即分析 ${count} 条`;
  if(count)requestAnimationFrame(()=>clampFloatingNode(node));
}

function renderComparisonFloat() {
  const node = $("#comparisonFloat");
  if (!node) return;
  const count = state.route === "image" ? state.comparisonSelection.size : 0;
  node.hidden = !count;
  $("#comparisonFloatCount").textContent = `已选 ${count} 条`;
  $("#comparisonFloatOpen").textContent = count < 2 ? "再选 1 条" : `对比 ${count} 条`;
  $("#comparisonFloatOpen").disabled = count < 2;
  if(count)requestAnimationFrame(()=>clampFloatingNode(node));
}

function clampFloatingNode(node) {
  if (!node || node.hidden || !node.style.left) return;
  const rect = node.getBoundingClientRect();
  const left = Math.max(8, Math.min(window.innerWidth - rect.width - 8, Number.parseFloat(node.style.left) || rect.left));
  const top = Math.max(8, Math.min(window.innerHeight - rect.height - 8, Number.parseFloat(node.style.top) || rect.top));
  node.style.left = `${left}px`; node.style.top = `${top}px`;
}

function enableFloatingDrag(node, storageKey) {
  const handle = node?.querySelector("[data-float-drag]");
  if (!node || !handle) return;
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
    if (saved && Number.isFinite(saved.left) && Number.isFinite(saved.top)) {
      node.style.left = `${saved.left}px`; node.style.top = `${saved.top}px`; node.style.right = "auto"; node.style.bottom = "auto";
    }
  } catch (_) {}
  let drag = null;
  handle.addEventListener("pointerdown",event=>{
    if(event.button!==0)return;
    const rect=node.getBoundingClientRect();drag={pointerId:event.pointerId,dx:event.clientX-rect.left,dy:event.clientY-rect.top};
    node.style.left=`${rect.left}px`;node.style.top=`${rect.top}px`;node.style.right="auto";node.style.bottom="auto";
    node.classList.add("dragging");handle.setPointerCapture(event.pointerId);event.preventDefault();
  });
  handle.addEventListener("pointermove",event=>{
    if(!drag||event.pointerId!==drag.pointerId)return;
    const left=Math.max(8,Math.min(window.innerWidth-node.offsetWidth-8,event.clientX-drag.dx));
    const top=Math.max(8,Math.min(window.innerHeight-node.offsetHeight-8,event.clientY-drag.dy));
    node.style.left=`${left}px`;node.style.top=`${top}px`;
  });
  const finish=event=>{
    if(!drag||event.pointerId!==drag.pointerId)return;
    drag=null;node.classList.remove("dragging");
    try{localStorage.setItem(storageKey,JSON.stringify({left:Number.parseFloat(node.style.left),top:Number.parseFloat(node.style.top)}));}catch(_){}
  };
  handle.addEventListener("pointerup",finish);handle.addEventListener("pointercancel",finish);
}

function selectedCount() { return Object.values(state.selectedFilters).reduce((sum, values) => sum + values.size, 0); }
function selectedAsObject() {
  return Object.fromEntries(Object.entries(state.selectedFilters).map(([key, set]) => [key, [...set]]).filter(([, values]) => values.length));
}
function appendFilters(params) {
  for (const [key, values] of Object.entries(selectedAsObject())) values.forEach(value => params.append(key, value));
}
function appendFilterObject(params, filters) {
  for (const [key, values] of Object.entries(filters)) values.forEach(value => params.append(key, value));
}
function mergeFilterObjects(...objects) {
  const merged = {};
  for (const filters of objects) for (const [key, values] of Object.entries(filters || {})) {
    merged[key] ||= new Set();
    values.forEach(value => merged[key].add(value));
  }
  return Object.fromEntries(Object.entries(merged).map(([key, values]) => [key, [...values]]));
}

function renderFilters() {
  const groups = Object.entries(FIELD_LABELS).filter(([key]) => (state.filterOptions[key] || []).length);
  $$('[data-filter-host]').forEach(host => {
    host.innerHTML = groups.map(([key, label]) => {
      const options = state.filterOptions[key] || [];
      const visible = options.slice(0, 12);
      return `<section class="filter-group" data-filter-group="${key}">
        <h3>${label}<span>${options.length} 项</span></h3>
        ${options.length > 8 ? `<div class="filter-search"><span class="icon-mask icon-search" aria-hidden="true"></span><input data-filter-search="${key}" placeholder="搜索${label}" autocomplete="off"></div>` : ""}
        <div class="filter-options" data-filter-options="${key}">${visible.map(value => `<label class="filter-option"><input type="checkbox" data-field="${key}" value="${escapeHtml(value)}" ${state.selectedFilters[key]?.has(value) ? "checked" : ""}><span title="${escapeHtml(value)}">${escapeHtml(value)}</span></label>`).join("")}</div>
        ${options.length > 12 ? `<button class="filter-more" data-expand="${key}">显示全部</button>` : ""}
      </section>`;
    }).join("");
  });
  renderActiveFilters();
}

function renderActiveFilters() {
  const chips = [];
  for (const [key, values] of Object.entries(selectedAsObject())) for (const value of values) chips.push(`<span class="filter-chip">${FIELD_LABELS[key]}：${escapeHtml(value)}</span>`);
  $("#textActiveFilters").innerHTML = chips.join("");
}

function renderFieldBuilder() {
  const host = $("#fieldConditions");
  if (!host) return;
  host.innerHTML = state.fieldConditions.map((condition, index) => {
    const usedElsewhere = new Set(state.fieldConditions.filter(item => item.id !== condition.id).map(item => item.field));
    const fieldOptions = Object.entries(FIELD_LABELS)
      .filter(([key]) => !usedElsewhere.has(key) || key === condition.field)
      .map(([key, label]) => `<option value="${key}" ${key === condition.field ? "selected" : ""}>${label}</option>`).join("");
    return `<div class="field-condition-row" data-condition-id="${condition.id}">
      <span class="condition-index">${String(index + 1).padStart(2, "0")}</span>
      <label class="condition-control condition-field"><span>字段</span><select data-condition-field>${fieldOptions}</select></label>
      <label class="condition-control condition-value"><span>${escapeHtml(FIELD_LABELS[condition.field])}</span><div class="condition-combobox"><input data-condition-value value="${escapeHtml(condition.value)}" placeholder="输入关键词或展开选择" autocomplete="off" role="combobox" aria-expanded="false"><button type="button" data-toggle-condition aria-label="展开${escapeHtml(FIELD_LABELS[condition.field])}候选值"><span class="icon-mask icon-chevron" aria-hidden="true"></span></button><div class="condition-menu" data-condition-menu hidden></div></div></label>
      <button class="condition-remove" type="button" data-remove-condition="${condition.id}" aria-label="删除第 ${index + 1} 个条件" ${state.fieldConditions.length === 1 ? "disabled" : ""}>×</button>
    </div>`;
  }).join("");
  const count = state.fieldConditions.length;
  $("#fieldConditionCount").textContent = `已使用 ${count} / 3 个条件`;
  $("#addFieldCondition").disabled = count >= 3;
}

function updateConditionMenu(row, query = "", open = true) {
  const condition = state.fieldConditions.find(item => item.id === Number(row.dataset.conditionId));
  const input = row.querySelector("[data-condition-value]"); const menu = row.querySelector("[data-condition-menu]");
  const keyword = query.trim().toLocaleLowerCase();
  const options = (state.filterOptions[condition.field] || []).filter(value => !keyword || String(value).toLocaleLowerCase().includes(keyword)).slice(0, 50);
  menu.innerHTML = options.length ? options.map(value => `<button type="button" data-condition-option="${escapeHtml(value)}">${escapeHtml(value)}</button>`).join("") : '<span class="condition-empty">没有匹配的候选值</span>';
  menu.hidden = !open; input.setAttribute("aria-expanded", String(open));
}

function closeConditionMenus(exceptRow = null) {
  $$(".field-condition-row").forEach(row => {
    if (row === exceptRow) return;
    const menu = row.querySelector("[data-condition-menu]"); const input = row.querySelector("[data-condition-value]");
    if (menu) menu.hidden = true; if (input) input.setAttribute("aria-expanded", "false");
  });
}

function syncConditionToSidebar(condition, value) {
  if (condition.syncedValue && condition.syncedValue !== value) {
    const stillUsed = state.fieldConditions.some(item => item.id !== condition.id && item.field === condition.field && item.syncedValue === condition.syncedValue);
    if (!stillUsed) state.selectedFilters[condition.field]?.delete(condition.syncedValue);
  }
  condition.syncedValue = value;
  if (value) { state.selectedFilters[condition.field] ||= new Set(); state.selectedFilters[condition.field].add(value); }
  renderFilters();
}

function fieldBuilderFilters(validate = false) {
  const filters = {};
  for (const condition of state.fieldConditions) {
    const rawValue = condition.value.trim();
    if (!rawValue) continue;
    const options = state.filterOptions[condition.field] || [];
    const exactValue = options.find(value => String(value).toLocaleLowerCase() === rawValue.toLocaleLowerCase());
    if (validate && options.length && !exactValue) throw new Error(`${FIELD_LABELS[condition.field]}请从数据库候选值中选择`);
    const value = exactValue || rawValue;
    filters[condition.field] ||= [];
    if (!filters[condition.field].includes(value)) filters[condition.field].push(value);
  }
  return filters;
}

document.addEventListener("change", event => {
  const input = event.target.closest("input[data-field]");
  if (!input) return;
  const key = input.dataset.field; state.selectedFilters[key] ||= new Set();
  input.checked ? state.selectedFilters[key].add(input.value) : state.selectedFilters[key].delete(input.value);
  $$(`input[data-field="${key}"]`).filter(node => node.value === input.value).forEach(node => node.checked = input.checked);
  if (!input.checked) {
    let changed = false;
    state.fieldConditions.forEach(condition => {if(condition.field===key&&condition.syncedValue===input.value){condition.value="";condition.syncedValue="";changed=true;}});
    if (changed) renderFieldBuilder();
  }
  renderActiveFilters();
});

document.addEventListener("input", event => {
  const search = event.target.closest("[data-filter-search]");
  if (!search) return;
  const key = search.dataset.filterSearch; const keyword = search.value.trim().toLocaleLowerCase();
  const group = search.closest(".filter-group"); const options = (state.filterOptions[key] || []).filter(value => !keyword || String(value).toLocaleLowerCase().includes(keyword)).slice(0, 80);
  group.querySelector(".filter-options").innerHTML = options.length ? options.map(value => `<label class="filter-option"><input type="checkbox" data-field="${key}" value="${escapeHtml(value)}" ${state.selectedFilters[key]?.has(value) ? "checked" : ""}><span title="${escapeHtml(value)}">${escapeHtml(value)}</span></label>`).join("") : '<div class="filter-no-result">没有匹配项</div>';
  const more = group.querySelector(".filter-more"); if (more) more.hidden = Boolean(keyword);
});

document.addEventListener("click", event => {
  if (!event.target.closest(".condition-combobox")) closeConditionMenus();
  const expand = event.target.closest("[data-expand]");
  if (expand) {
    const key = expand.dataset.expand; const host = expand.closest(".filter-group");
    host.querySelector(".filter-options").innerHTML = state.filterOptions[key].map(value => `<label class="filter-option"><input type="checkbox" data-field="${key}" value="${escapeHtml(value)}" ${state.selectedFilters[key]?.has(value) ? "checked" : ""}><span title="${escapeHtml(value)}">${escapeHtml(value)}</span></label>`).join("");
    expand.remove(); return;
  }
  if (event.target.closest("[data-clear-filters]")) {
    state.selectedFilters = {};state.fieldConditions.forEach(condition=>{condition.value="";condition.syncedValue="";});renderFilters();renderFieldBuilder();
  }
  const zoom = event.target.closest("[data-zoom]");
  if (zoom) { event.preventDefault(); event.stopPropagation(); openLightbox(zoom.dataset.zoom); return; }
  const compareToggle=event.target.closest("[data-compare-dn]");
  if(compareToggle){event.preventDefault();event.stopPropagation();toggleComparison(compareToggle.dataset.compareDn);return;}
  if(event.target.closest(".match-breakdown"))return;
  const detail = event.target.closest("[data-detail]");
  if (detail) { openDetail(detail.dataset.detail, detail.dataset.detailContext === "image"); return; }
});

function recordCard(item) {
  const f = item.fields || {}; const rid = item.record_id || f.dn_no;
  const selected = state.textAnalysisSelection.has(String(rid));
  const tags = [
    [f.defect_type, "warm"], [f.platform, "accent"], [f.station, ""], [f.machine, ""],
    [f.lot_id && `LotID: ${f.lot_id}`, ""], [f.map_location && `Map: ${f.map_location}`, ""],
  ].filter(([value]) => clean(value) !== "—");
  return `<article class="record-card ${selected ? "selected-for-analysis" : ""}">
    <div class="record-card-head"><label class="analysis-check"><input type="checkbox" data-text-analysis-id="${escapeHtml(rid)}" ${selected ? "checked" : ""}><span>加入分析</span></label><h3>${escapeHtml(rid)}</h3>${item.score !== undefined && state.textMode === "semantic" ? `<span class="score">${percent(item.score)}</span>` : ""}</div>
    <div class="tag-row">${tags.map(([value, kind]) => `<span class="tag ${kind}">${escapeHtml(value)}</span>`).join("")}</div>
    <p class="record-text">${escapeHtml(f.text_content || "暂无完整描述")}</p>
    <div class="record-actions"><button class="link-button" data-detail="${escapeHtml(rid)}">查看详情</button></div>
  </article>`;
}

function textResultIds() {
  return (state.lastTextSearchData?.results || []).map(item => String(item.record_id || item.fields?.dn_no || "")).filter(Boolean);
}

function renderTextBatchState() {
  const ids = textResultIds(), selected = ids.filter(id => state.textAnalysisSelection.has(id));
  const toolbar = $("#textBatchToolbar"), selectAll = $("#selectAllTextResults"), button = $("#generateReport");
  toolbar.hidden = !ids.length;
  $("#textAnalysisCount").textContent = `已选 ${selected.length} 条（最多 20 条）`;
  selectAll.checked = Boolean(ids.length) && selected.length === ids.length;
  selectAll.indeterminate = selected.length > 0 && selected.length < ids.length;
  button.disabled = selected.length === 0;
  button.textContent = selected.length ? `分析已选 ${selected.length} 条` : "分析已选记录";
  renderAnalysisFloat();
}

function setTextMode(mode) {
  const input = $("#textQuery");
  if (input && state.textMode !== "filter") state.textQueryByMode[state.textMode] = input.value;
  state.textMode = mode;
  $$("#textTabs button").forEach(button => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("active", active); button.setAttribute("aria-selected", String(active));
  });
  $("#page-text .work-layout").classList.toggle("dn-mode", mode === "dn");
  $("#textSimplePanel").hidden = mode === "filter";
  $("#fieldBuilderPanel").hidden = mode !== "filter";
  const settings = {
    semantic: {label: "异常描述或关键词", prefix: "语义", placeholder: "例如：晶圆边缘出现颗粒残留，良率下降", hint: "可使用自然语言描述异常现象；左侧条件可进一步缩小结果范围。", submit: "搜索记录"},
    dn: {label: "DN 编号", prefix: "DN", placeholder: "输入完整 DN 编号，例如 DN-20241020-02", hint: "按唯一编号精确匹配一条记录，不使用左侧筛选条件。", submit: "精确查找"},
    filter: {submit: "应用筛选"},
  };
  const setting = settings[mode];
  if (mode !== "filter") {
    $("#textQueryLabel").textContent = setting.label; $("#textInputPrefix").textContent = setting.prefix;
    input.placeholder = setting.placeholder; input.value = state.textQueryByMode[mode] || "";
    $("#textModeHint").textContent = setting.hint;
  } else renderFieldBuilder();
  $("#textSubmitLabel").textContent = setting.submit;
}

async function runTextSearch(event) {
  event?.preventDefault();
  const query = $("#textQuery").value.trim();
  if (state.textMode !== "filter" && !query) return toast(state.textMode === "dn" ? "请输入完整 DN 编号" : "请输入异常描述或关键词");
  let filters = selectedAsObject();
  if (state.textMode === "filter") {
    try { filters = mergeFilterObjects(filters, fieldBuilderFilters(true)); }
    catch (error) { return toast(error.message); }
    if (!Object.keys(filters).length) return toast("请填写至少一个字段条件");
  }
  $("#textResults").innerHTML = '<div class="loading">正在检索记录</div>'; $("#textMeta").textContent = "";
  state.lastTextSearchData = null; state.textAnalysisSelection.clear(); renderAnalysisFloat(); $("#textBatchToolbar").hidden = true; $("#reportPanel").hidden = true; $("#reportOutput").innerHTML = '<div class="empty small">选择记录后点击分析，报告不会改动原始检索数据。</div>';
  const started = performance.now();
  try {
    let data;
    if (state.textMode === "dn") data = await fetchJson(`/api/dn-search?dn_no=${encodeURIComponent(query)}`);
    else if (state.textMode === "filter") { const params = new URLSearchParams({limit:"500"}); appendFilterObject(params, filters); data = await fetchJson(`/api/filter-search?${params}`); }
    else data = await fetchJson("/api/search", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({query, limit:100, filters})});
    $("#textMeta").textContent = `检索结果：${data.count ?? data.results.length} 条 · 用时 ${((performance.now()-started)/1000).toFixed(2)} 秒`;
    $("#textResults").innerHTML = data.results?.length ? data.results.map(recordCard).join("") : '<div class="empty large"><span>⌕</span><strong>没有找到匹配记录</strong><p>可以调整关键词或减少筛选条件。</p></div>';
    if (data.results?.length) {
      state.lastTextSearchData = data;
      state.lastTextSearchContext = query || Object.entries(filters).map(([key, values]) => `${FIELD_LABELS[key] || key}=${values.join("/")}`).join("，");
      $("#reportPanel").hidden = false;
      renderTextBatchState();
    }
  } catch (error) { $("#textResults").innerHTML = `<div class="empty large"><strong>${escapeHtml(error.message)}</strong></div>`; }
}

function stringList(values) {
  return (Array.isArray(values) ? values : []).filter(Boolean).map(value => `<li>${escapeHtml(value)}</li>`).join("");
}

function reportPieChart(chart) {
  const items = (chart.items || []).filter(item => Number(item.count) > 0);
  const total = items.reduce((sum,item) => sum + Number(item.count), 0);
  let cursor = 0;
  const segments = items.map((item,index) => {
    const start = cursor, end = cursor + Number(item.count) / Math.max(total,1) * 100;
    cursor = end; return `${PIE_COLORS[index % PIE_COLORS.length]} ${start.toFixed(3)}% ${end.toFixed(3)}%`;
  });
  const legend = items.map((item,index) => {
    const ratio = Number(item.count) / Math.max(total,1) * 100;
    return `<li><i style="background:${PIE_COLORS[index % PIE_COLORS.length]}"></i><span title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</span><b>${escapeHtml(item.count)}</b><small>${ratio.toFixed(1)}%</small></li>`;
  }).join("");
  return `<article class="report-chart report-pie-card"><h3>${escapeHtml(chart.title)}</h3><div class="report-pie-layout"><div class="report-pie" style="background:conic-gradient(${segments.join(",") || "#e8edf5 0 100%"})"><span><b>${total}</b><small>记录</small></span></div><ul class="report-pie-legend">${legend}</ul></div></article>`;
}

function renderReportImagePreviews() {
  state.reportImagePreviewUrls.forEach(url=>URL.revokeObjectURL(url));
  state.reportImagePreviewUrls = state.reportDraft.images.map(file=>URL.createObjectURL(file));
  const host = $("#reportImagePreviews"); if(!host)return;
  host.innerHTML = state.reportDraft.images.length ? state.reportDraft.images.map((file,index)=>`<article><img src="${escapeHtml(state.reportImagePreviewUrls[index])}" alt="${escapeHtml(file.name)}"><span title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</span><button type="button" data-remove-report-image="${index}" aria-label="移除${escapeHtml(file.name)}">×</button></article>`).join("") : '<div class="empty small">未添加图片</div>';
}

function setReportImages(files, append=true) {
  const valid=[...files].filter(file=>file.type.startsWith("image/")&&file.size<=20*1024*1024);
  const base=append?state.reportDraft.images:[],before=base.length,known=new Set(base.map(file=>`${file.name}|${file.size}|${file.lastModified}`));
  for(const file of valid){const key=`${file.name}|${file.size}|${file.lastModified}`;if(base.length<8&&!known.has(key)){base.push(file);known.add(key);}}
  state.reportDraft.images=base.slice(0,8);renderReportImagePreviews();
  if(valid.length!==files.length||state.reportDraft.images.length-before<valid.length)toast("仅接收不超过 20MB 的图片，单次最多 8 张");
}

function openReportPage(source, sourceCount, draft) {
  state.reportReturnRoute = source;
  state.reportData = null;
  state.reportDraft = {query:draft?.query||"", results:draft?.results||[], images:[...(draft?.images||[])].slice(0,8)};
  $("#reportPageSource").textContent = `${source === "image" ? "图片" : "文字"}检索 · ${sourceCount} 条记录`;
  try { sessionStorage.setItem("waferReportMeta", JSON.stringify({returnRoute:source, sourceCount})); } catch (_) {}
  $("#reportUserContext").value="";$("#reportContextCount").textContent="0";
  $("#reportEvidenceHint").textContent=`已选 ${sourceCount} 条基础证据；可补充现场文字和图片后综合分析。`;
  renderReportImagePreviews();
  $("#standaloneReportOutput").innerHTML = '<div class="empty large"><span class="empty-symbol">AI</span><strong>输入材料已准备</strong><p>确认或补充现场信息后，点击“开始综合分析”。</p></div>';
  location.hash = "#/report";
}

async function runMultimodalReport(){
  const context=$("#reportUserContext").value.trim();
  if(!state.reportDraft.results.length&&!context&&!state.reportDraft.images.length)return toast("请提供记录、文字或图片中的至少一项");
  const button=$("#runMultimodalReport");button.disabled=true;button.textContent="正在分析…";
  $("#standaloneReportOutput").innerHTML='<div class="report-page-loading"><div class="loading">正在检索 Qdrant 并生成综合报告，首次运行可能需要 1–3 分钟</div><p>正在融合已选记录、现场文字、图片相似结果和工业知识。</p></div>';
  try{
    const images=await Promise.all(state.reportDraft.images.map(async file=>({name:file.name,data_base64:await fileAsBase64(file)})));
    const data=await fetchJson("/api/report/multimodal",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({query:state.reportDraft.query,user_context:context,results:state.reportDraft.results,images})});
    renderReport(data);button.textContent="重新综合分析";
  }catch(error){$("#standaloneReportOutput").innerHTML=`<div class="empty large"><strong>报告生成失败</strong><p>${escapeHtml(error.message)}</p></div>`;button.textContent="重新分析";}
  finally{button.disabled=false;}
}

function renderReport(data, outputSelector = "#standaloneReportOutput") {
  const report = data.report || {};
  const observations = (report.observations || []).map(item => `<article><p>${escapeHtml(item.finding || "")}</p><small>证据：${escapeHtml((item.source_ids || []).join("、") || "未标注")}</small></article>`).join("");
  const hypotheses = (report.root_cause_hypotheses || []).map(item => `<article class="hypothesis-card"><header><b>#${escapeHtml(item.rank || "-")} ${escapeHtml(item.hypothesis || "待验证假设")}</b><span class="confidence ${item.confidence === "高" ? "high" : item.confidence === "中" ? "medium" : "low"}">${escapeHtml(item.confidence || "低")}置信度</span></header><p>${escapeHtml(item.evidence || "证据不足")}</p><small>证据记录：${escapeHtml((item.source_ids || []).join("、") || "未标注")}</small><h4>验证动作</h4><ol>${stringList(item.validation_actions)}</ol></article>`).join("");
  const charts = (data.visualizations || []).map(reportPieChart).join("");
  const references = (data.methodology_references || []).map(item => `<li><a href="${escapeHtml(item.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(item.title)}</a></li>`).join("");
  const breakdown=data.evidence_breakdown||data.input_metadata||{};
  const inputSummary=Object.keys(breakdown).length?`<section class="report-input-summary"><h3>本次综合输入</h3><div><span>勾选记录 <b>${escapeHtml(breakdown.selected_count||0)}</b></span><span>文字召回 <b>${escapeHtml(breakdown.text_retrieved_count||0)}</b></span><span>图片召回 <b>${escapeHtml(breakdown.image_retrieved_count||0)}</b></span><span>合并证据 <b>${escapeHtml(breakdown.merged_evidence_count||data.source_count||0)}</b></span></div></section>`:"";
  const knowledgeStatus=data.knowledge_status||{};
  $(outputSelector).innerHTML = `<div class="report-document">
    <div class="report-title"><div><span>模型 ${escapeHtml(data.model)} · ${escapeHtml(data.source_count)} 条证据记录</span><h2>${escapeHtml(report.title || "晶圆异常分析报告")}</h2></div><span class="status-pill">本地生成</span></div>
    <section class="report-summary"><h3>执行摘要</h3><p>${escapeHtml(report.executive_summary || "暂无摘要")}</p></section>
    ${inputSummary}
    <section><h3>证据观察</h3><div class="observation-grid">${observations || '<div class="empty small">模型未列出观察项</div>'}</div></section>
    <section><h3>根因假设（需验证）</h3><div class="hypothesis-grid">${hypotheses || '<div class="empty small">现有证据不足以提出根因假设</div>'}</div></section>
    <section><h3>检索结果可视化</h3><div class="report-chart-grid">${charts || '<div class="empty small">可统计字段不足</div>'}</div></section>
    <div class="report-action-grid"><section><h3>立即行动</h3><ul>${stringList(report.immediate_actions)}</ul></section><section><h3>验证计划</h3><ol>${stringList(report.verification_plan)}</ol></section></div>
    <section class="report-limitations"><h3>限制与边界</h3><ul>${stringList(report.limitations)}</ul><p>本报告用于辅助排查，不替代工程签核、量测复验和设备处置流程。</p></section>
    <section class="report-references"><h3>Qdrant 检索的行业分析方法</h3>${references?`<ul>${references}</ul>`:`<p>本次未检索到行业方法知识。${escapeHtml(knowledgeStatus.detail||"")}</p>`}</section>
  </div>`;
  state.reportData = data;
  try { sessionStorage.setItem("waferLatestReport", JSON.stringify(data)); } catch (_) {}
}

function restoreLatestReport() {
  try {
    const saved = sessionStorage.getItem("waferLatestReport");
    if (!saved) return;
    const data = JSON.parse(saved), meta = JSON.parse(sessionStorage.getItem("waferReportMeta") || "{}");
    state.reportReturnRoute = meta.returnRoute || "text";
    $("#reportPageSource").textContent = `${state.reportReturnRoute === "image" ? "图片" : "文字"}检索 · ${meta.sourceCount || data.source_count || 0} 条记录`;
    renderReport(data);
  } catch (_) {
    sessionStorage.removeItem("waferLatestReport");
  }
}

async function generateTextReport() {
  if (!state.lastTextSearchData?.results?.length) return toast("请先完成一次检索");
  const selected = state.lastTextSearchData.results.filter(item => state.textAnalysisSelection.has(String(item.record_id || item.fields?.dn_no || ""))).slice(0,20);
  if (!selected.length) return toast("请先勾选需要分析的记录");
  openReportPage("text", selected.length,{query:state.lastTextSearchContext,results:selected,images:[]});
  renderTextBatchState();
}

function imageAnalysisRecords() {
  return (state.imageSearchData?.results || []).filter(item => state.imageAnalysisSelection.has(String(item.dn_no)));
}

function generateImageReport() {
  const selected = imageAnalysisRecords().slice(0,20).map(item => ({
    record_id: item.dn_no,
    score: Number(item.average_similarity ?? item.score ?? 0),
    fields: item.fields || {},
  }));
  if (!selected.length) return toast("请先选择需要分析的相似记录");
  const names = (state.lastImageSearch?.images || []).map(item => item.name).filter(Boolean).join("、");
  const context = `基于图片检索结果分析晶圆异常${names ? `；输入材料：${names}` : ""}`;
  openReportPage("image", selected.length,{query:context,results:selected,images:state.imageFiles});
}

function fieldTable(fields) {
  return `<table class="field-table"><tbody>${DETAIL_FIELDS.filter(([key]) => clean(fields[key]) !== "—").map(([key,label]) => `<tr><th>${label}</th><td>${escapeHtml(clean(fields[key]))}</td></tr>`).join("")}</tbody></table>`;
}

async function openDetail(dn, compareWithQuery = false) {
  $("#detailTitle").textContent = dn; $("#detailBody").innerHTML = '<div class="loading">正在读取详情</div>';
  $("#drawerBackdrop").classList.add("open"); $("#detailDrawer").classList.add("open"); $("#detailDrawer").setAttribute("aria-hidden","false");
  try {
    const data = await fetchJson(`/api/yedn/${encodeURIComponent(dn)}`); const f = data.fields || {};
    const query = compareWithQuery ? state.lastImageSearch : null;
    const queryImages = query?.images || [];
    const matchedResult=(state.imageSearchData?.results||[]).find(item=>item.dn_no===dn);
    const matchedBlocks=matchedResult?.block_evidence||[];
    const sharedQueryBlocks=state.imageSearchData?.query_blocks||[];
    const queryImage = queryImages.map((image,index) => {const boxes=matchedBlocks.filter(block=>block.query_index===index).map(block=>{const shared=sharedQueryBlocks.find(item=>item.query_index===block.query_index&&String(item.block_id)===String(block.query_block_id));return {bbox_norm:block.query_bbox_norm,block_id:block.query_block_id,zoom_url:shared?.preview_url||""};});return `<div class="detail-image query-detail-image" data-zoom="${escapeHtml(image.url)}"><div class="detail-image-frame"><img src="${escapeHtml(image.url)}" alt="检索输入图 ${index+1}"><span class="image-role-badge">输入 ${index+1}</span>${matchBoxesHtml(boxes,"query")}</div><span>${escapeHtml(image.name || `查询图片 ${index+1}`)}</span></div>`}).join("");
    const historyImages = data.images.map((image,index) => {const boxes=matchedBlocks.filter(block=>block.history_file_name===image.file_name).map(block=>({bbox_norm:block.history_bbox_norm,block_id:block.query_block_id,zoom_url:block.history_patch_url}));return `<div class="detail-image" data-zoom="${escapeHtml(image.url)}"><div class="detail-image-frame"><img loading="${index<2?"eager":"lazy"}" decoding="async" src="${escapeHtml(image.url)}" alt="${escapeHtml(image.image_type)}"><span class="image-role-badge history">历史图片</span>${matchBoxesHtml(boxes,"history")}</div><span>${escapeHtml(image.image_type)}</span></div>`}).join("");
    $("#detailBody").innerHTML = `<div class="detail-grid">
      <section class="detail-section"><h3>异常记录</h3><div class="description-box">${escapeHtml(f.text_content || "暂无完整描述")}</div>${fieldTable(f)}</section>
      <section class="detail-section"><div class="detail-image-heading"><h3>${query ? "检索输入与记录图片" : "记录图片"}</h3><span>${query ? `${queryImages.length} 张输入图 · ${data.images.length} 张历史图片` : `${data.images.length} 张`}</span></div><div class="detail-images ${query ? "has-query-image" : ""}">${queryImage}${historyImages}</div></section>
    </div>`;
  } catch (error) { $("#detailBody").innerHTML = `<div class="empty large"><strong>${escapeHtml(error.message)}</strong></div>`; }
}
function closeDetail(){ $("#drawerBackdrop").classList.remove("open"); $("#detailDrawer").classList.remove("open"); $("#detailDrawer").setAttribute("aria-hidden","true"); }
function openLightbox(src){ $("#lightboxImage").src=src; $("#lightbox").classList.add("open"); }

function toggleComparison(dn){
  if(state.comparisonSelection.has(dn))state.comparisonSelection.delete(dn);
  else if(state.comparisonSelection.size>=4)return toast("最多同时对比 4 条记录");
  else state.comparisonSelection.add(dn);
  if(state.imageSearchData)renderImageOutput(state.imageSearchData,false);
}

function closeComparison(){
  $("#comparisonBackdrop").classList.remove("open");$("#comparisonModal").classList.remove("open");$("#comparisonModal").setAttribute("aria-hidden","true");
}

async function openComparison(){
  const dns=[...state.comparisonSelection];if(dns.length<2)return toast("请至少选择 2 条记录");
  $("#comparisonTitle").textContent=`记录对比（${dns.length}）`;$("#comparisonBody").innerHTML='<div class="loading">正在读取记录</div>';
  $("#comparisonBackdrop").classList.add("open");$("#comparisonModal").classList.add("open");$("#comparisonModal").setAttribute("aria-hidden","false");
  try{
    const records=await Promise.all(dns.map(dn=>fetchJson(`/api/yedn/${encodeURIComponent(dn)}`)));
    const selectedResults=records.map(record=>(state.imageSearchData?.results||[]).find(item=>item.dn_no===record.dn_no)||{});
    const queryImages=state.lastImageSearch?.images||[],queryBlocks=state.imageSearchData?.query_blocks||[];
    const fieldRows=DETAIL_FIELDS.map(([key,label])=>`<tr><th>${label}</th><td>—</td>${records.map(record=>`<td>${escapeHtml(clean(record.fields?.[key]))}</td>`).join("")}</tr>`).join("");
    const scoreRows=`<tr class="comparison-score-row"><th>综合相似度</th><td>检索基准</td>${selectedResults.map(result=>`<td><strong>${percent(result.average_similarity)}</strong></td>`).join("")}</tr>${queryBlocks.map(block=>`<tr class="comparison-score-row"><th>区域 ${block.block_id} 相似度</th><td>区域 ${block.block_id}</td>${selectedResults.map(result=>{const match=(result.block_matches||result.block_evidence||[]).find(item=>String(item.query_block_id)===String(block.block_id));return `<td>${match?percent(match.similarity):"—"}</td>`}).join("")}</tr>`).join("")}`;
    const heads=`<th><strong>输入样本</strong><span>${queryImages.length} 张图片 · ${queryBlocks.length} 个区域</span></th>`+records.map(record=>`<th><strong>${escapeHtml(record.dn_no)}</strong><span>${escapeHtml(clean(record.fields?.defect_type))}</span></th>`).join("");
    const queryOriginals=queryImages.map((image,index)=>`<button type="button" data-zoom="${escapeHtml(image.url||"")}"><img src="${escapeHtml(image.url||"")}" alt="输入图片 ${index+1}"><span>${escapeHtml(image.name||`输入图片 ${index+1}`)}</span></button>`).join("");
    const queryRegions=queryBlocks.map(block=>`<button type="button" class="comparison-region-tile" data-zoom="${escapeHtml(block.preview_url||"")}"><img src="${escapeHtml(block.preview_url||"")}" alt="输入区域 ${block.block_id}"><span>输入区域 ${block.block_id}</span></button>`).join("");
    const queryColumn=`<section class="comparison-record-images comparison-input-sample"><h3>输入样本</h3><div>${queryOriginals}${queryRegions}</div></section>`;
    const imageColumns=records.map(record=>{
      const result=(state.imageSearchData?.results||[]).find(item=>item.dn_no===record.dn_no)||{};
      const matches=result.block_matches||result.block_evidence||[];
      const originals=(record.images||[]).map(image=>`<button type="button" data-zoom="${escapeHtml(image.url)}"><img src="${escapeHtml(image.url)}" alt="${escapeHtml(image.image_type)}"><span>${escapeHtml(image.image_type)}</span></button>`).join("");
      const regions=matches.map(match=>`<button type="button" class="comparison-region-tile" data-zoom="${escapeHtml(match.history_patch_url||"")}"><img src="${escapeHtml(match.history_patch_url||"")}" alt="${escapeHtml(record.dn_no)} 匹配区域 ${match.query_block_id}"><span>区域 ${match.query_block_id} · ${percent(match.similarity)}</span></button>`).join("");
      return `<section class="comparison-record-images"><h3>${escapeHtml(record.dn_no)}</h3><div>${originals}${regions||""}</div></section>`;
    }).join("");
    $("#comparisonBody").innerHTML=`<div class="comparison-table-wrap"><table class="comparison-table"><thead><tr><th>字段</th>${heads}</tr></thead><tbody>${scoreRows}${fieldRows}</tbody></table></div><div class="comparison-image-columns">${queryColumn}${imageColumns}</div>`;
  }catch(error){$("#comparisonBody").innerHTML=`<div class="empty large"><strong>${escapeHtml(error.message)}</strong></div>`;}
}

function bindDrop(zone, input, onFile, multiple = false) {
  ["dragenter","dragover"].forEach(name => zone.addEventListener(name, event => {event.preventDefault(); zone.classList.add("dragging")}));
  ["dragleave","drop"].forEach(name => zone.addEventListener(name, event => {event.preventDefault(); zone.classList.remove("dragging")}));
  zone.addEventListener("drop", event => {const files=[...event.dataTransfer.files];if(files.length)onFile(multiple?files:files[0]);});
  input.addEventListener("change", () => {const files=[...input.files];if(files.length)onFile(multiple?files:files[0]);});
}

function renderSelectedPptFiles() {
  const files = state.pptFiles;
  $("#pptImport").disabled = !files.length;
  $("#pptImport").textContent = files.length ? `批量导入（${files.length}）` : "批量导入";
  $("#pptDrop strong").textContent = files.length ? `已选择 ${files.length} 个 PPTX` : "选择或拖入一个或多个 PPTX";
  $("#pptDrop small").textContent = files.length ? `合计 ${(files.reduce((sum,file)=>sum+file.size,0)/1024/1024).toFixed(2)} MB，可继续添加` : "支持批量选择；单个文件不超过 100MB";
  $("#pptSelectedFiles").innerHTML = files.map((file,index)=>`<div class="selected-file"><span class="icon-mask icon-presentation" aria-hidden="true"></span><div><strong title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</strong><small>${(file.size/1024/1024).toFixed(2)} MB</small></div><button type="button" data-remove-ppt="${index}" aria-label="移除${escapeHtml(file.name)}">×</button></div>`).join("");
}

function selectPpts(incomingFiles) {
  const files = incomingFiles.filter(file => file.name.toLowerCase().endsWith(".pptx") && file.size <= 100 * 1024 * 1024);
  const invalidCount = incomingFiles.length - files.length;
  const known = new Set(state.pptFiles.map(file => `${file.name}|${file.size}|${file.lastModified}`));
  files.forEach(file => {const key=`${file.name}|${file.size}|${file.lastModified}`;if(!known.has(key)){state.pptFiles.push(file);known.add(key);}});
  renderSelectedPptFiles();
  if (invalidCount) toast(`${invalidCount} 个文件不是 PPTX 或超过 100MB，已跳过`);
}

function renderPptBatch(results, total, currentName = "") {
  const success = results.filter(item=>item.ok); const failed = results.filter(item=>!item.ok);
  const slides = success.reduce((sum,item)=>sum+(item.data.slide_count||0),0);
  const records = success.reduce((sum,item)=>sum+(item.data.record_count||0),0);
  const images = success.reduce((sum,item)=>sum+(item.data.image_count||0),0);
  $("#pptResult").innerHTML = `<div class="import-summary"><div><h2>${results.length < total ? "正在批量导入" : "批量导入完成"}</h2><span>${results.length} / ${total}${currentName ? ` · ${escapeHtml(currentName)}` : ""}</span></div><span class="status-pill">成功 ${success.length}${failed.length ? ` · 失败 ${failed.length}` : ""}</span></div>
    <div class="metric-row"><div class="metric"><strong>${slides}</strong><span>幻灯片</span></div><div class="metric"><strong>${records}</strong><span>记录</span></div><div class="metric"><strong>${images}</strong><span>图片</span></div></div>
    <h2 class="section-title">文件处理结果</h2><div class="batch-result-list">${results.map(item=>`<div class="batch-result ${item.ok?"success":"failed"}"><span class="icon-mask ${item.ok?"icon-file":"icon-x"}" aria-hidden="true"></span><div><strong>${escapeHtml(item.file.name)}</strong><small>${item.ok?`${item.data.record_count} 条记录 · ${item.data.image_count} 张图片${item.data.already_imported?" · 已存在":""}`:escapeHtml(item.error)}</small></div><b>${item.ok?"已入库":"失败"}</b></div>`).join("")}${results.length < total ? '<div class="loading batch-loading">正在处理下一个文件</div>' : ""}</div>`;
}

async function importPpt() {
  if (!state.pptFiles.length) return;
  const queue=[...state.pptFiles];const results=[];$("#pptImport").disabled=true;renderPptBatch(results,queue.length,queue[0].name);
  for(const file of queue){
    try{const data=await fetchJson("/api/ppt/import",{method:"POST",headers:{"X-File-Name":encodeURIComponent(file.name)},body:file});results.push({ok:true,file,data});}
    catch(error){results.push({ok:false,file,error:error.message});}
    renderPptBatch(results,queue.length,queue[results.length]?.name||"");
  }
  state.pptFiles=[];$("#pptFile").value="";renderSelectedPptFiles();await loadPptHistory();await loadFilters();
  toast(results.some(item=>!item.ok)?`导入完成：成功 ${results.filter(item=>item.ok).length}，失败 ${results.filter(item=>!item.ok).length}`:`${results.length} 个 PPT 已全部导入`);
}
function renderPptResult(data){
  $("#pptResult").innerHTML=`<div class="import-summary"><div><h2>${escapeHtml(data.file_name)}</h2><span>${escapeHtml(data.imported_at || "")}</span></div><span class="status-pill">已入库</span></div>
  <div class="metric-row"><div class="metric"><strong>${data.slide_count}</strong><span>幻灯片</span></div><div class="metric"><strong>${data.record_count}</strong><span>记录</span></div><div class="metric"><strong>${data.image_count}</strong><span>图片</span></div></div>
  <h2 class="section-title">导入内容</h2><div class="slide-grid">${(data.records||[]).map(record=>{const image=(record.images||[])[0];const url=image?`/api/yedn/image/${encodeURIComponent(record.dn_no)}/${encodeURIComponent(image.file_name)}`:"";return `<article class="slide-card interactive" data-ppt-record-dn="${escapeHtml(record.dn_no)}" tabindex="0">${url?`<img src="${url}" alt="${escapeHtml(record.dn_no)}">`:""}<div><strong>${escapeHtml(record.dn_no)}</strong><span>第 ${record.slide} 页 · ${(record.images||[]).length} 张图片</span></div></article>`}).join("")}</div>`;
}
async function loadPptHistory(){
  try{
    const keyword=$("#pptLibrarySearch")?.value.trim()||"";
    const data=await fetchJson(`/api/ppt/library?query=${encodeURIComponent(keyword)}`);
    state.pptHistory=data.files||[];state.pptDictionary=data.dictionaries||{};
    $("#pptLibraryCount").textContent=`${state.pptHistory.length} / ${data.total||0}`;
    $("#pptHistory").innerHTML=state.pptHistory.length?state.pptHistory.map((item,index)=>`<div class="file-card" data-history="${index}"><strong>${escapeHtml(item.file_name)}</strong><span>${item.record_count} 条记录 · ${item.image_count} 张图片</span></div>`).join(""):'<div class="empty small">没有匹配的已导入文件</div>';
    renderPptDictionary();
  }catch(_){ }
}

function renderPptDictionary(){
  const labels={product_id:"产品",platform:"平台",machine:"设备",defect_type:"缺陷类型"};
  const groups=Object.entries(labels).map(([key,label])=>{
    const values=state.pptDictionary[key]||[];
    return `<section class="dictionary-group"><header><strong>${label}</strong><span>${values.length}</span></header><div>${values.slice(0,12).map(item=>`<button type="button" data-ppt-library-query="${escapeHtml(item.value)}"><b>${escapeHtml(item.value)}</b><span>${item.count}</span></button>`).join("")||'<small>暂无数据</small>'}</div></section>`;
  }).join("");
  const host=$("#pptLibraryOverview");if(host)host.innerHTML=`<div class="library-overview"><div class="section-heading"><h2>资料字典</h2></div><div class="dictionary-grid">${groups}</div></div>`;
}

function clearImageFiles(){
  state.imagePreviewUrls.forEach(url=>URL.revokeObjectURL(url));
  state.imageFiles=[];state.imagePreviewUrls=[];state.imageManualRegions={};state.imageModalities={};state.lastImageSearch=null;state.imageSearchData=null;state.imageResultFilters={};state.comparisonSelection.clear();state.imageAnalysisSelection.clear();renderAnalysisFloat();renderComparisonFloat();
  $("#imageFile").value="";$("#imageSearch").disabled=true;$("#imageOutput").innerHTML="";renderSelectedSearchFiles();
}

function fileKey(file){return `${file.name}|${file.size}|${file.lastModified}`;}

function renderSelectedSearchFiles(){
  const files=state.imageFiles;$("#imageSearch").disabled=!files.length||state.imageParsingPpts>0;
  const status=state.imageParsingPpts?"正在读取图片":(files.length?`已选择 ${files.length} 张图片`:"选择或拖入文件");
  $("#imagePreview").innerHTML=`<span class="upload-icon image-icon" aria-hidden="true"><i></i></span><strong>${status}</strong>`;
  $("#imageSelectedFiles").innerHTML=files.length?files.map((file,index)=>{
    const preview=state.imagePreviewUrls[index];
    const region=state.imageManualRegions[fileKey(file)];
    const regionCount=normalizedRegionList(region).length;
    const modality=state.imageModalities[fileKey(file)]||"unknown";
    const canDraw=preview&&modality==="binmap";
    return `<div class="query-file-item">${preview?`<img src="${escapeHtml(preview)}" alt="">`:`<span class="query-file-type">FILE</span>`}<div><strong title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</strong></div><div class="query-file-actions">${canDraw?`<button class="roi-button ${regionCount?"active":""}" type="button" data-draw-search-file="${index}">${regionCount?`${regionCount} 个区域`:"画框"}</button>`:""}<button type="button" data-remove-search-file="${index}" aria-label="移除${escapeHtml(file.name)}">×</button></div></div>`;
  }).join(""):'<div class="empty small">尚未选择文件</div>';
}

function isRectangleRegion(region){return Array.isArray(region)&&region.length===4&&region.every(value=>Number.isFinite(Number(value)));}

function normalizedRegionList(saved){
  if(!saved)return [];
  if(isRectangleRegion(saved)||saved?.type==="polygon")return [JSON.parse(JSON.stringify(saved))];
  return Array.isArray(saved)?saved.filter(region=>isRectangleRegion(region)||region?.type==="polygon").map(region=>JSON.parse(JSON.stringify(region))):[];
}

function regionBounds(region){
  if(isRectangleRegion(region))return region;
  const points=region?.points||[];if(!points.length)return [0,0,0,0];
  const xs=points.map(point=>point[0]),ys=points.map(point=>point[1]);
  const left=Math.min(...xs),top=Math.min(...ys);return [left,top,Math.max(...xs)-left,Math.max(...ys)-top];
}

function roiOverlayHtml(region,index,draft=false){
  const label=draft?state.roiEditorRegions.length+1:index+1;
  const remove=draft?"":`<button type="button" class="roi-remove-region" data-remove-roi-region="${index}" aria-label="删除区域 ${label}">×</button>`;
  if(region?.type==="polygon"){
    const points=region.points||[];if(!points.length)return "";
    const [left,top]=regionBounds(region);const pointString=points.map(point=>`${point[0]*1000},${point[1]*1000}`).join(" ");
    return `<svg class="roi-region polygon ${draft?"draft":""}" viewBox="0 0 1000 1000" preserveAspectRatio="none"><polyline points="${pointString}"></polyline></svg><span class="roi-region-number" style="left:${left*100}%;top:${top*100}%">${label}</span>${remove?`<span class="roi-region-remove-wrap" style="left:${left*100}%;top:${top*100}%">${remove}</span>`:""}`;
  }
  if(!isRectangleRegion(region))return "";const [x,y,w,h]=region;
  return `<div class="roi-region rectangle ${draft?"draft":""}" style="left:${x*100}%;top:${y*100}%;width:${w*100}%;height:${h*100}%"><b>${label}</b>${remove}</div>`;
}

function updateRoiSelection(){
  const layer=$("#roiOverlayLayer");
  layer.innerHTML=state.roiEditorRegions.map((region,index)=>roiOverlayHtml(region,index,false)).join("")+(state.roiEditorDraft?roiOverlayHtml(state.roiEditorDraft,state.roiEditorRegions.length,true):"");
  const count=state.roiEditorRegions.length+(state.roiEditorDraft?1:0);
  $("#roiEditorStatus").textContent=state.roiEditorMode==="polygon"
    ? `${count?`当前 ${count} 个区域；`:""}按住鼠标一笔画出闭合区域`
    : `${count?`当前 ${count} 个区域；`:""}拖动可继续添加矩形区域`;
}

function setRoiMode(mode){
  state.roiEditorMode=mode;state.roiEditorDraft=null;state.roiEditorPoints=[];state.roiFreehandActive=false;
  $("#roiModeRectangle").classList.toggle("active",mode==="rectangle");
  $("#roiModePolygon").classList.toggle("active",mode==="polygon");updateRoiSelection();
}

function openRoiEditor(index){
  const file=state.imageFiles[index],preview=state.imagePreviewUrls[index];if(!file||!preview)return;
  const saved=state.imageManualRegions[fileKey(file)];state.roiEditorIndex=index;
  state.roiEditorRegions=normalizedRegionList(saved);state.roiEditorDraft=null;
  state.roiEditorMode="rectangle";state.roiEditorPoints=[];
  $("#roiModeRectangle").classList.toggle("active",state.roiEditorMode==="rectangle");
  $("#roiModePolygon").classList.toggle("active",state.roiEditorMode==="polygon");
  $("#roiImage").src=preview;updateRoiSelection();$("#roiEditor").classList.add("open");$("#roiEditor").setAttribute("aria-hidden","false");
}

function closeRoiEditor(){
  $("#roiEditor").classList.remove("open");$("#roiEditor").setAttribute("aria-hidden","true");state.roiEditorIndex=null;state.roiEditorDraft=null;state.roiEditorRegions=[];state.roiEditorPoints=[];state.roiFreehandActive=false;
}

function commitRoiDraft(){
  const region=state.roiEditorDraft;if(!region)return true;
  if(region?.type==="polygon"&&(region.points||[]).length<3){toast("不规则选区至少需要三个点");return false;}
  const bounds=regionBounds(region);if(bounds[2]<.02||bounds[3]<.02){toast("选区太小，请重新选择");return false;}
  if(state.roiEditorRegions.length>=3){toast("一张图片最多选择 3 个区域");return false;}
  state.roiEditorRegions.push(JSON.parse(JSON.stringify(region)));state.roiEditorDraft=null;state.roiEditorPoints=[];updateRoiSelection();return true;
}

function applyRoiEditor(){
  const file=state.imageFiles[state.roiEditorIndex];if(!file)return closeRoiEditor();
  if(state.roiEditorDraft&&!commitRoiDraft())return;
  const key=fileKey(file);if(state.roiEditorRegions.length)state.imageManualRegions[key]=JSON.parse(JSON.stringify(state.roiEditorRegions));else delete state.imageManualRegions[key];
  state.lastImageSearch=null;state.imageSearchData=null;state.imageResultFilters={};state.imageAnalysisSelection.clear();state.comparisonSelection.clear();renderAnalysisFloat();renderComparisonFloat();$("#imageOutput").innerHTML="";closeRoiEditor();renderSelectedSearchFiles();
}

function addSearchImage(file,modality="unknown"){
  if(state.imageFiles.length>=30)return false;
  const key=fileKey(file);if(state.imageFiles.some(item=>fileKey(item)===key))return false;
  state.imageFiles.push(file);state.imagePreviewUrls.push(URL.createObjectURL(file));state.imageModalities[key]=modality;
  return true;
}

function extractedImageFile(item,pptFile,index){
  const binary=atob(item.data_base64||"");const bytes=new Uint8Array(binary.length);
  for(let offset=0;offset<binary.length;offset++)bytes[offset]=binary.charCodeAt(offset);
  return new File([bytes],item.name||`${pptFile.name} · 图片${index+1}.png`,{type:item.content_type||"image/png",lastModified:pptFile.lastModified+index+1});
}

async function expandSearchPpt(file){
  state.imageParsingPpts++;renderSelectedSearchFiles();
  try{
    const data=await fetchJson("/api/image/ppt-extract",{method:"POST",headers:{"X-File-Name":encodeURIComponent(file.name)},body:file});
    let added=0;(data.images||[]).forEach((item,index)=>{const image=extractedImageFile(item,file,index);if(addSearchImage(image,item.modality||"unknown"))added++;});
    if(!added)throw new Error("PPT中没有可添加的图片");
  }finally{state.imageParsingPpts--;renderSelectedSearchFiles();}
}

async function detectSearchImageModality(file){
  const data=await fetchJson("/api/image/detect-modality",{method:"POST",headers:{"X-File-Name":encodeURIComponent(file.name)},body:file});
  return data.modality==="binmap"?"binmap":"sem";
}

async function selectSearchFiles(incoming){
  const files=Array.isArray(incoming)?incoming:[incoming];
  let skipped=0;
  for(const file of files){
    if(file.name.toLowerCase().endsWith(".pptx")){
      try{await expandSearchPpt(file);}catch(error){skipped++;toast(error.message);}
      continue;
    }
    const imageLike=file.type.startsWith("image/")||/\.(jpe?g|png|bmp|webp|tiff?)$/i.test(file.name);
    if(!imageLike){skipped++;continue;}
    state.imageParsingPpts++;renderSelectedSearchFiles();
    try{
      const modality=await detectSearchImageModality(file);
      if(!addSearchImage(file,modality))skipped++;
    }catch(error){skipped++;toast(`${file.name}：${error.message}`);}
    finally{state.imageParsingPpts--;renderSelectedSearchFiles();}
  }
  state.lastImageSearch=null;state.imageSearchData=null;state.imageResultFilters={};state.imageAnalysisSelection.clear();state.comparisonSelection.clear();renderAnalysisFloat();renderComparisonFloat();$("#imageOutput").innerHTML="";renderSelectedSearchFiles();
  if(skipped)toast(`已忽略 ${skipped} 个重复文件或超出 30 个上限的文件`);
}

function fileAsBase64(file){
  return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(",",2)[1]||"");reader.onerror=()=>reject(new Error(`无法读取 ${file.name}`));reader.readAsDataURL(file);});
}

async function runImageSearch(){
  if(!state.imageFiles.length)return;
  $("#imageOutput").innerHTML='<div class="panel image-results-panel loading">正在解析输入并综合计算相似度</div>';$("#imageSearch").disabled=true;
  try{
    const files=await Promise.all(state.imageFiles.map(async file=>({name:file.name,content_type:file.type||"application/octet-stream",data_base64:await fileAsBase64(file),manual_region:state.imageManualRegions[fileKey(file)]||null})));
    const data=await fetchJson("/api/image/search",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({files,limit:50})});
    state.lastImageSearch={images:data.query_images||[]};renderImageOutput(data,true);
  }catch(error){$("#imageOutput").innerHTML=`<div class="panel image-results-panel empty large"><strong>${escapeHtml(error.message)}</strong></div>`;}
  finally{$("#imageSearch").disabled=!state.imageFiles.length;}
}

function imageResultOptions(results){
  const options={};for(const key of Object.keys(FIELD_LABELS))options[key]=[];
  for(const item of results)for(const key of Object.keys(FIELD_LABELS)){const value=item.fields?.[key];if(clean(value)==="—")continue;const text=String(value);if(!options[key].includes(text))options[key].push(text);}
  for(const values of Object.values(options))values.sort((a,b)=>a.localeCompare(b,"zh-CN"));
  return options;
}

function filteredImageResults(){
  const results=state.imageSearchData?.results||[];
  return results.filter(item=>Object.entries(state.imageResultFilters).every(([key,values])=>!values.size||values.has(String(item.fields?.[key]??""))));
}

function resultStatistics(results){
  return Object.entries(FIELD_LABELS).map(([field,label])=>{
    const counts=new Map();for(const item of results){const value=item.fields?.[field];if(clean(value)==="—")continue;counts.set(String(value),(counts.get(String(value))||0)+1);}
    const dominant=[...counts.entries()].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0],"zh-CN"))[0];
    return dominant?{field,label,dominant_value:dominant[0]}:null;
  }).filter(Boolean);
}

function postFilterHtml(){
  const groups=Object.entries(FIELD_LABELS).filter(([key])=>(state.imageResultOptions[key]||[]).length);
  const current=filteredImageResults().length,total=state.imageSearchData?.results?.length||0;
  const active=[];for(const [key,values] of Object.entries(state.imageResultFilters))for(const value of values)active.push(`<button type="button" class="result-filter-chip" data-remove-image-filter="${key}" data-filter-value="${escapeHtml(value)}"><span>${FIELD_LABELS[key]} · ${escapeHtml(value)}</span><b>×</b></button>`);
  return `<section class="panel image-results-panel post-search-filter"><div class="result-filter-toolbar"><div><h2>筛选结果</h2><span>${current} / ${total}</span></div><div class="facet-menu-row">${groups.map(([key,label])=>{const selected=state.imageResultFilters[key]?.size||0;const values=state.imageResultOptions[key];return `<details class="post-filter-group"><summary><span>${label}</span>${selected?`<b>${selected}</b>`:""}</summary><div class="facet-popover">${values.length>8?`<label class="facet-search"><span class="icon-mask icon-search"></span><input type="search" data-image-filter-search placeholder="搜索${label}"></label>`:""}<div class="facet-options">${values.map(value=>`<label class="filter-option"><input type="checkbox" data-image-result-field="${key}" value="${escapeHtml(value)}" ${state.imageResultFilters[key]?.has(value)?"checked":""}><span title="${escapeHtml(value)}">${escapeHtml(value)}</span></label>`).join("")}</div></div></details>`}).join("")}</div><button class="text-button" type="button" id="clearImageResultFilters" ${active.length?"":"hidden"}>清空</button></div>${active.length?`<div class="result-filter-chips">${active.join("")}</div>`:""}</section>`;
}

function matchBoxesHtml(boxes,type="query"){
  return (boxes||[]).map((box,index)=>{
    const value=box.bbox_norm||box;
    if(!Array.isArray(value)||value.length!==4)return "";
    const [x,y,w,h]=value.map(number=>Math.max(0,Math.min(1,Number(number)||0)));
    const label=box.block_id||box.query_block_id||index+1;
    const zoom=box.zoom_url||box.preview_url||box.history_patch_url||"";
    const tag=zoom?"button":"span";const attrs=zoom?` type="button" data-zoom="${escapeHtml(zoom)}" aria-label="查看区域 ${label}" title="查看区域 ${label}"`:"";
    return `<${tag} class="match-box ${type}"${attrs} style="left:${x*100}%;top:${y*100}%;width:${w*100}%;height:${h*100}%"><b>${label}</b></${tag}>`;
  }).join("");
}

function resultBlockBreakdown(item,queryBlocks){
  const matches=(item.block_matches||item.block_evidence||[]).slice().sort((a,b)=>Number(a.query_block_id)-Number(b.query_block_id));
  if(!matches.length)return "";
  const chips=matches.map(match=>`<span>区域 ${match.query_block_id} <b>${percent(match.similarity)}</b></span>`).join("");
  const rows=matches.map(match=>{
    const query=queryBlocks.find(block=>String(block.block_id)===String(match.query_block_id));
    return `<div class="block-match-row"><button type="button" data-zoom="${escapeHtml(query?.preview_url||"")}"><img src="${escapeHtml(query?.preview_url||"")}" alt="检索区域 ${match.query_block_id}"><span>输入 ${match.query_block_id}</span></button><div class="block-match-score"><b>${percent(match.similarity)}</b><span>↔</span></div><button type="button" data-zoom="${escapeHtml(match.history_patch_url||"")}"><img src="${escapeHtml(match.history_patch_url||"")}" alt="历史区域 ${match.query_block_id}"><span>匹配 ${match.query_block_id}</span></button></div>`;
  }).join("");
  return `<details class="match-breakdown"><summary><span>区域相似度</span><div>${chips}</div></summary><div class="block-match-list">${rows}</div></details>`;
}

function renderImageOutputLegacy(data,initialize=false){
  if(initialize){
    state.imageSearchData=data;state.imageResultOptions=imageResultOptions(data.results||[]);state.imageResultFilters={};state.comparisonSelection.clear();state.imageAnalysisSelection.clear();
  }
  const results=filteredImageResults();const rows=resultStatistics(results);const queryImages=state.lastImageSearch?.images||[];const total=state.imageSearchData?.results?.length||0;
  const queryBlocks=state.imageSearchData?.query_blocks||[];
  $("#imageOutput").innerHTML=`${postFilterHtml()}<section class="panel image-results-panel"><h2 class="section-title first">当前结果统计</h2>${rows.length?`<table class="statistics-table"><thead><tr><th>统计字段</th><th>历史记录常见值</th></tr></thead><tbody>${rows.map(row=>`<tr><td>${escapeHtml(row.label)}</td><td>${escapeHtml(row.dominant_value)}</td></tr>`).join("")}</tbody></table>`:'<div class="empty small">当前筛选条件下暂无可统计记录</div>'}</section>
  <section class="panel image-results-panel"><div class="import-summary"><div><h2>检索图片</h2></div><span>${queryImages.length} 张</span></div><div class="visual-grid query-input-grid">${queryImages.map((image,index)=>{const boxes=queryBlocks.filter(block=>block.query_index===index);return `<article class="visual-card query-reference-card" data-zoom="${escapeHtml(image.url)}"><div class="visual-card-image"><img src="${escapeHtml(image.url)}" alt="输入图片 ${index+1}"><span class="query-badge">输入 ${index+1}</span>${matchBoxesHtml(boxes,"query")}</div><div class="visual-card-body"><strong>${escapeHtml(image.name||`输入图片 ${index+1}`)}</strong></div></article>`}).join("")}</div></section>
  <section class="panel image-results-panel"><div class="import-summary"><div><h2>相似记录</h2></div><span>${results.length} 条</span></div>${results.length?`<div class="visual-grid comparison-grid match-result-grid">${results.map(item=>{const image=item.best_match||{};const f=item.fields||{};const evidences=item.block_evidence||[];const boxes=evidences.length?evidences.map(evidence=>({bbox_norm:evidence.history_bbox_norm,block_id:evidence.query_block_id,zoom_url:evidence.history_patch_url})):(image.bbox_norm?[image]:[]);const selected=state.comparisonSelection.has(item.dn_no);return `<article class="visual-card match-result-card ${selected?"selected-for-compare":""}" data-detail="${escapeHtml(item.dn_no)}" data-detail-context="image"><button class="compare-select ${selected?"active":""}" type="button" data-compare-dn="${escapeHtml(item.dn_no)}" aria-pressed="${selected}">${selected?"已选择":"对比"}</button><div class="visual-card-image"><img loading="lazy" decoding="async" src="${escapeHtml(image.url||"")}" alt="${escapeHtml(item.dn_no)}">${matchBoxesHtml(boxes,"history")}<span class="similarity-badge">综合 ${percent(item.average_similarity)}</span></div><div class="visual-card-body"><strong>${escapeHtml(item.dn_no)}</strong><span>${escapeHtml(clean(f.defect_type))} · ${escapeHtml(clean(f.platform))}</span>${resultBlockBreakdown(item,queryBlocks)}</div></article>`}).join("")}</div>`:'<div class="empty large"><strong>当前筛选条件下没有记录</strong></div>'}</section>${state.comparisonSelection.size?`<div class="comparison-tray"><span>已选择 <b>${state.comparisonSelection.size}</b> 条</span><div><button type="button" class="button ghost compact" id="clearComparisonSelection">清空</button><button type="button" class="button primary compact" id="openComparison" ${state.comparisonSelection.size<2?"disabled":""}>开始对比</button></div></div>`:""}`;
}

function resultMetric(item,metric){
  if(metric==="overall")return Number(item.average_similarity)||0;
  const match=(item.block_matches||item.block_evidence||[]).find(entry=>String(entry.query_block_id)===String(metric));
  return Number(match?.similarity)||0;
}

function resultRankingTile(item,metric){
  const image=item.best_match||{},fields=item.fields||{};
  const matches=item.block_matches||item.block_evidence||[];
  const metricMatch=metric==="overall"?null:matches.find(entry=>String(entry.query_block_id)===String(metric));
  const boxes=metricMatch?[{bbox_norm:metricMatch.history_bbox_norm,block_id:metricMatch.query_block_id,zoom_url:metricMatch.history_patch_url}]:(item.block_evidence||[]).map(entry=>({bbox_norm:entry.history_bbox_norm,block_id:entry.query_block_id,zoom_url:entry.history_patch_url}));
  const selected=state.comparisonSelection.has(item.dn_no),analysisSelected=state.imageAnalysisSelection.has(String(item.dn_no)),score=resultMetric(item,metric);
  return `<article class="visual-card ranking-result-card ${selected?"selected-for-compare":""} ${analysisSelected?"selected-for-analysis":""}">
    <div class="visual-card-image" data-zoom="${escapeHtml(image.url||"")}"><img loading="lazy" decoding="async" src="${escapeHtml(image.url||"")}" alt="${escapeHtml(item.dn_no)}">${matchBoxesHtml(boxes,"history")}<span class="similarity-badge">${percent(score)}</span></div>
    <div class="visual-card-body"><strong>${escapeHtml(item.dn_no)}</strong><span>${escapeHtml(clean(fields.defect_type))} · ${escapeHtml(clean(fields.platform))}</span><div class="result-card-actions"><button class="analysis-select-inline ${analysisSelected?"active":""}" type="button" data-image-analysis-id="${escapeHtml(item.dn_no)}" aria-pressed="${analysisSelected}">${analysisSelected?"已加入分析":"加入分析"}</button><button class="compare-select-inline ${selected?"active":""}" type="button" data-compare-dn="${escapeHtml(item.dn_no)}" aria-pressed="${selected}">${selected?"已选对比":"对比"}</button><button class="record-detail-link" type="button" data-detail="${escapeHtml(item.dn_no)}" data-detail-context="image">详情</button></div></div>
  </article>`;
}

function rankingGroup(title,metric,results,open=false){
  const ranked=[...results].sort((a,b)=>resultMetric(b,metric)-resultMetric(a,metric));
  return `<details class="ranking-group" ${open?"open":""}><summary><span>${escapeHtml(title)}</span><b>${ranked.length} 条</b></summary><div class="ranking-result-grid">${ranked.map(item=>resultRankingTile(item,metric)).join("")}</div></details>`;
}

function renderImageOutput(data,initialize=false){
  if(initialize){state.imageSearchData=data;state.imageResultOptions=imageResultOptions(data.results||[]);state.imageResultFilters={};state.comparisonSelection.clear();state.imageAnalysisSelection.clear();}
  const results=filteredImageResults(),rows=resultStatistics(results),queryImages=state.lastImageSearch?.images||[],queryBlocks=state.imageSearchData?.query_blocks||[];
  const queryRegionTiles=queryBlocks.map(block=>`<button type="button" class="query-region-tile" data-zoom="${escapeHtml(block.preview_url||"")}"><img src="${escapeHtml(block.preview_url||"")}" alt="区域 ${block.block_id}"><span>区域 ${block.block_id}</span></button>`).join("");
  $("#imageOutput").innerHTML=`${postFilterHtml()}
  <section class="panel image-results-panel"><h2 class="section-title first">当前结果统计</h2>${rows.length?`<table class="statistics-table"><thead><tr><th>统计字段</th><th>历史记录常见值</th></tr></thead><tbody>${rows.map(row=>`<tr><td>${escapeHtml(row.label)}</td><td>${escapeHtml(row.dominant_value)}</td></tr>`).join("")}</tbody></table>`:'<div class="empty small">当前筛选条件下暂无可统计记录</div>'}</section>
  <section class="panel image-results-panel"><div class="import-summary"><div><h2>检索图片</h2></div><span>${queryImages.length} 张</span></div><div class="visual-grid query-input-grid">${queryImages.map((image,index)=>{const boxes=queryBlocks.filter(block=>block.query_index===index);return `<article class="visual-card query-reference-card"><div class="visual-card-image" data-zoom="${escapeHtml(image.url)}"><img src="${escapeHtml(image.url)}" alt="输入图片 ${index+1}"><span class="query-badge">输入 ${index+1}</span>${matchBoxesHtml(boxes,"query")}</div><div class="visual-card-body"><strong>${escapeHtml(image.name||`输入图片 ${index+1}`)}</strong></div></article>`}).join("")}</div>${queryRegionTiles?`<div class="query-region-strip">${queryRegionTiles}</div>`:""}</section>
  <section class="panel image-results-panel"><div class="import-summary"><div><h2>相似记录</h2></div><span>${results.length} 条</span></div>${results.length?`<div class="ranking-groups">${rankingGroup("综合相似","overall",results,true)}${queryBlocks.map(block=>rankingGroup(`区域 ${block.block_id} 相似`,block.block_id,results)).join("")}</div>`:'<div class="empty large"><strong>当前筛选条件下没有记录</strong></div>'}</section>
  ${results.length?`<section class="panel image-results-panel image-analysis-panel"><div class="report-panel-head"><div><span class="eyebrow">BATCH ANALYSIS</span><h2>批量异常分析报告</h2><p>从相似记录中选择多个 DN，汇总设备、站点、缺陷和位置分布，并生成根因假设。</p></div><button class="button primary" type="button" id="generateImageReport" ${state.imageAnalysisSelection.size?"":"disabled"}>分析已选 ${state.imageAnalysisSelection.size} 条</button></div><div class="batch-analysis-toolbar"><label><input type="checkbox" id="selectAllImageResults" ${results.length&&results.every(item=>state.imageAnalysisSelection.has(String(item.dn_no)))?"checked":""}><span>选择当前筛选结果</span></label><div><span>已选 ${state.imageAnalysisSelection.size} 条（最多 20 条）</span><button class="text-button" type="button" id="clearImageAnalysis">清空选择</button></div></div><div id="imageReportOutput"><div class="empty small">选择相似记录后生成报告；原始检索数据不会被修改。</div></div></section>`:""}
  ${state.comparisonSelection.size?`<div class="comparison-tray"><span>已选择 <b>${state.comparisonSelection.size}</b> 条</span><div><button type="button" class="button ghost compact" id="clearComparisonSelection">清空</button><button type="button" class="button primary compact" id="openComparison" ${state.comparisonSelection.size<2?"disabled":""}>开始对比</button></div></div>`:""}`;
  renderAnalysisFloat();
  renderComparisonFloat();
}

async function loadFilters(){
  try{state.filterOptions=await fetchJson("/api/filters");renderFilters();renderFieldBuilder();}catch(error){toast(`筛选项加载失败：${error.message}`);}
}

function resetTextSearch() {
  state.textQueryByMode = {semantic: "", dn: ""}; $("#textQuery").value = "";
  state.selectedFilters = {}; state.fieldConditions = [{id: state.nextConditionId++, field: "platform", value: ""}];
  renderFilters(); renderFieldBuilder();
  $("#textResults").innerHTML = '<div class="empty large"><span class="empty-symbol icon-mask icon-search" aria-hidden="true"></span><strong>输入内容开始检索</strong><p>综合搜索、DN 编号和字段组合分别适用于不同查询任务。</p></div>';
  $("#textMeta").textContent = "";
  state.lastTextSearchData = null; state.lastTextSearchContext = ""; state.textAnalysisSelection.clear(); renderAnalysisFloat(); $("#textBatchToolbar").hidden = true; $("#reportPanel").hidden = true;
}

function init(){
  restoreLatestReport();window.addEventListener("hashchange",route);route();loadFilters();loadPptHistory();
  enableFloatingDrag($("#analysisFloat"),"wafer-analysis-float-position");
  enableFloatingDrag($("#comparisonFloat"),"wafer-comparison-float-position");
  window.addEventListener("resize",()=>{$$(".analysis-float").forEach(clampFloatingNode);});
  $("#backToResults").addEventListener("click",()=>{location.hash=`#/${state.reportReturnRoute || "text"}`;});
  $("#page-report").addEventListener("click",event=>{if(event.target.closest("[data-back-report]"))location.hash=`#/${state.reportReturnRoute || "text"}`;});
  $("#runMultimodalReport").addEventListener("click",runMultimodalReport);
  $("#reportUserContext").addEventListener("input",event=>{$("#reportContextCount").textContent=event.target.value.length;});
  $("#reportUserImages").addEventListener("change",event=>{setReportImages([...event.target.files]);event.target.value="";});
  $("#clearReportImages").addEventListener("click",()=>{state.reportDraft.images=[];renderReportImagePreviews();});
  $("#reportImagePreviews").addEventListener("click",event=>{const button=event.target.closest("[data-remove-report-image]");if(!button)return;state.reportDraft.images.splice(Number(button.dataset.removeReportImage),1);renderReportImagePreviews();});
  $("#analysisFloatRun").addEventListener("click",()=>{if(state.route==="text")generateTextReport();else if(state.route==="image")generateImageReport();});
  $("#analysisFloatClear").addEventListener("click",()=>{
    if(state.route==="text"){
      state.textAnalysisSelection.clear();$("#textResults").innerHTML=(state.lastTextSearchData?.results||[]).map(recordCard).join("");renderTextBatchState();
    }else if(state.route==="image"){
      state.imageAnalysisSelection.clear();if(state.imageSearchData)renderImageOutput(state.imageSearchData,false);else renderAnalysisFloat();
    }
  });
  $("#comparisonFloatOpen").addEventListener("click",openComparison);
  $("#comparisonFloatClear").addEventListener("click",()=>{state.comparisonSelection.clear();if(state.imageSearchData)renderImageOutput(state.imageSearchData,false);else renderComparisonFloat();});
  $$("#textTabs button").forEach(button=>button.addEventListener("click",()=>setTextMode(button.dataset.mode)));
  $("#textSearchForm").addEventListener("submit",runTextSearch);$("#textReset").addEventListener("click",resetTextSearch);
  $("#generateReport").addEventListener("click",generateTextReport);
  $("#textResults").addEventListener("change",event=>{
    const input=event.target.closest("[data-text-analysis-id]");if(!input)return;
    const id=input.dataset.textAnalysisId;
    if(input.checked&&state.textAnalysisSelection.size>=20){input.checked=false;return toast("单次最多选择 20 条记录");}
    input.checked?state.textAnalysisSelection.add(id):state.textAnalysisSelection.delete(id);
    input.closest(".record-card")?.classList.toggle("selected-for-analysis",input.checked);renderTextBatchState();
  });
  $("#selectAllTextResults").addEventListener("change",event=>{
    state.textAnalysisSelection.clear();
    if(event.target.checked)textResultIds().slice(0,20).forEach(id=>state.textAnalysisSelection.add(id));
    $("#textResults").innerHTML=(state.lastTextSearchData?.results||[]).map(recordCard).join("");renderTextBatchState();
    if(event.target.checked&&textResultIds().length>20)toast("已选择前 20 条，单次分析上限为 20 条");
  });
  $("#clearTextAnalysis").addEventListener("click",()=>{state.textAnalysisSelection.clear();$("#textResults").innerHTML=(state.lastTextSearchData?.results||[]).map(recordCard).join("");renderTextBatchState();});
  $("#addFieldCondition").addEventListener("click",()=>{
    if(state.fieldConditions.length>=3)return;
    const used=new Set(state.fieldConditions.map(item=>item.field));
    const field=Object.keys(FIELD_LABELS).find(key=>!used.has(key))||"platform";
    state.fieldConditions.push({id:state.nextConditionId++,field,value:""});renderFieldBuilder();
  });
  $("#fieldConditions").addEventListener("click",event=>{
    const row=event.target.closest("[data-condition-id]");if(!row)return;const condition=state.fieldConditions.find(item=>item.id===Number(row.dataset.conditionId));
    const option=event.target.closest("[data-condition-option]");if(option){event.preventDefault();condition.value=option.dataset.conditionOption;row.querySelector("[data-condition-value]").value=condition.value;syncConditionToSidebar(condition,condition.value);closeConditionMenus();return;}
    const toggle=event.target.closest("[data-toggle-condition]");if(toggle){event.preventDefault();const menu=row.querySelector("[data-condition-menu]");const willOpen=menu.hidden;closeConditionMenus(row);updateConditionMenu(row,row.querySelector("[data-condition-value]").value,willOpen);return;}
    const button=event.target.closest("[data-remove-condition]");if(button){syncConditionToSidebar(condition,"");state.fieldConditions=state.fieldConditions.filter(item=>item.id!==condition.id);renderFieldBuilder();}
  });
  $("#fieldConditions").addEventListener("change",event=>{const row=event.target.closest("[data-condition-id]");if(!row)return;const condition=state.fieldConditions.find(item=>item.id===Number(row.dataset.conditionId));if(event.target.matches("[data-condition-field]")){syncConditionToSidebar(condition,"");condition.field=event.target.value;condition.value="";renderFieldBuilder();return;}if(event.target.matches("[data-condition-value]")){const exact=(state.filterOptions[condition.field]||[]).find(value=>String(value).toLocaleLowerCase()===condition.value.trim().toLocaleLowerCase());if(exact){condition.value=exact;event.target.value=exact;syncConditionToSidebar(condition,exact);}}});
  $("#fieldConditions").addEventListener("input",event=>{if(!event.target.matches("[data-condition-value]"))return;const row=event.target.closest("[data-condition-id]");const condition=state.fieldConditions.find(item=>item.id===Number(row.dataset.conditionId));condition.value=event.target.value;if(condition.syncedValue&&condition.syncedValue!==condition.value)syncConditionToSidebar(condition,"");closeConditionMenus(row);updateConditionMenu(row,condition.value,true);});
  $("#fieldConditions").addEventListener("focusin",event=>{if(!event.target.matches("[data-condition-value]"))return;const row=event.target.closest("[data-condition-id]");closeConditionMenus(row);updateConditionMenu(row,event.target.value,true);});
  $("#detailClose").addEventListener("click",closeDetail);$("#drawerBackdrop").addEventListener("click",closeDetail);
  $("#comparisonClose").addEventListener("click",closeComparison);$("#comparisonBackdrop").addEventListener("click",closeComparison);
  $("#lightboxClose").addEventListener("click",()=>$("#lightbox").classList.remove("open"));$("#lightbox").addEventListener("click",event=>event.target===$("#lightbox")&&$("#lightbox").classList.remove("open"));
  bindDrop($("#pptDrop"),$("#pptFile"),selectPpts,true);$("#pptImport").addEventListener("click",importPpt);
  let pptSearchTimer;$("#pptLibrarySearch").addEventListener("input",()=>{clearTimeout(pptSearchTimer);pptSearchTimer=setTimeout(loadPptHistory,180);});
  $("#pptSelectedFiles").addEventListener("click",event=>{const button=event.target.closest("[data-remove-ppt]");if(!button)return;state.pptFiles.splice(Number(button.dataset.removePpt),1);renderSelectedPptFiles();});
  $("#page-ppt").addEventListener("click",event=>{const record=event.target.closest("[data-ppt-record-dn]");if(record){openDetail(record.dataset.pptRecordDn);return;}const query=event.target.closest("[data-ppt-library-query]");if(query){$("#pptLibrarySearch").value=query.dataset.pptLibraryQuery;loadPptHistory();return;}const card=event.target.closest("[data-history]");if(card)renderPptResult(state.pptHistory[Number(card.dataset.history)]);});
  bindDrop($("#imageDrop"),$("#imageFile"),selectSearchFiles,true);$("#imageSearch").addEventListener("click",runImageSearch);
  $("#clearImageFiles").addEventListener("click",clearImageFiles);
  $("#imageSelectedFiles").addEventListener("click",event=>{
    const draw=event.target.closest("[data-draw-search-file]");if(draw){openRoiEditor(Number(draw.dataset.drawSearchFile));return;}
    const button=event.target.closest("[data-remove-search-file]");if(!button)return;
    const index=Number(button.dataset.removeSearchFile),file=state.imageFiles[index];const url=state.imagePreviewUrls[index];if(url)URL.revokeObjectURL(url);if(file){delete state.imageManualRegions[fileKey(file)];delete state.imageModalities[fileKey(file)];}
    state.imageFiles.splice(index,1);state.imagePreviewUrls.splice(index,1);state.lastImageSearch=null;state.imageSearchData=null;state.imageResultFilters={};state.imageAnalysisSelection.clear();state.comparisonSelection.clear();renderAnalysisFloat();renderComparisonFloat();$("#imageOutput").innerHTML="";renderSelectedSearchFiles();
  });
  $("#roiEditorClose").addEventListener("click",closeRoiEditor);$("#roiEditorCancel").addEventListener("click",closeRoiEditor);$("#roiEditorApply").addEventListener("click",applyRoiEditor);
  $("#roiModeRectangle").addEventListener("click",()=>setRoiMode("rectangle"));
  $("#roiModePolygon").addEventListener("click",()=>setRoiMode("polygon"));
  $("#roiEditorReset").addEventListener("click",()=>{state.roiEditorRegions=[];state.roiEditorDraft=null;state.roiEditorPoints=[];updateRoiSelection();});
  $("#roiOverlayLayer").addEventListener("click",event=>{const remove=event.target.closest("[data-remove-roi-region]");if(!remove)return;event.stopPropagation();state.roiEditorRegions.splice(Number(remove.dataset.removeRoiRegion),1);updateRoiSelection();});
  $("#roiEditor").addEventListener("click",event=>{if(event.target===$("#roiEditor"))closeRoiEditor();});
  let roiStart=null;
  const roiPoint=event=>{const rect=$("#roiStage").getBoundingClientRect();return [Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)),Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height))];};
  const sampleFreehandPoints=(points,maxPoints=80)=>{
    if(points.length<=maxPoints)return points;
    const step=(points.length-1)/(maxPoints-1);
    return Array.from({length:maxPoints},(_,index)=>points[Math.round(index*step)]);
  };
  $("#roiStage").addEventListener("pointerdown",event=>{
    if(event.target.closest("[data-remove-roi-region]"))return;
    if(event.button!==0)return;
    if(state.roiEditorRegions.length>=3){toast("每张图片最多保留 3 个区域");return;}
    const point=roiPoint(event);
    $("#roiStage").setPointerCapture(event.pointerId);
    if(state.roiEditorMode==="polygon"){
      state.roiFreehandActive=true;state.roiEditorPoints=[point];state.roiEditorDraft={type:"polygon",points:[point]};updateRoiSelection();return;
    }
    roiStart=point;
    state.roiEditorDraft=[roiStart[0],roiStart[1],0,0];updateRoiSelection();
  });
  $("#roiStage").addEventListener("pointermove",event=>{
    if(state.roiFreehandActive){
      const point=roiPoint(event),last=state.roiEditorPoints[state.roiEditorPoints.length-1];
      if(!last||Math.hypot(point[0]-last[0],point[1]-last[1])>=.005){state.roiEditorPoints.push(point);state.roiEditorDraft={type:"polygon",points:[...state.roiEditorPoints]};updateRoiSelection();}
      return;
    }
    if(!roiStart)return;const [x,y]=roiPoint(event);
    state.roiEditorDraft=[Math.min(roiStart[0],x),Math.min(roiStart[1],y),Math.abs(x-roiStart[0]),Math.abs(y-roiStart[1])];updateRoiSelection();
  });
  const finishRoi=()=>{
    if(state.roiFreehandActive){
      state.roiFreehandActive=false;
      const points=sampleFreehandPoints(state.roiEditorPoints);
      if(points.length<3){state.roiEditorDraft=null;state.roiEditorPoints=[];updateRoiSelection();toast("请按住鼠标画出一个闭合区域");return;}
      state.roiEditorDraft={type:"polygon",points};commitRoiDraft();return;
    }
    if(!roiStart)return;roiStart=null;if(state.roiEditorDraft&&(state.roiEditorDraft[2]<.02||state.roiEditorDraft[3]<.02)){state.roiEditorDraft=null;updateRoiSelection();toast("选区太小，请重新选择");return;}commitRoiDraft();
  };
  $("#roiStage").addEventListener("pointerup",finishRoi);$("#roiStage").addEventListener("pointercancel",finishRoi);
  $("#imageOutput").addEventListener("change",event=>{
    const analysisAll=event.target.closest("#selectAllImageResults");
    if(analysisAll){state.imageAnalysisSelection.clear();if(analysisAll.checked)filteredImageResults().slice(0,20).forEach(item=>state.imageAnalysisSelection.add(String(item.dn_no)));renderImageOutput(state.imageSearchData,false);if(analysisAll.checked&&filteredImageResults().length>20)toast("已选择前 20 条，单次分析上限为 20 条");return;}
    const input=event.target.closest("[data-image-result-field]");if(!input)return;
    const key=input.dataset.imageResultField;state.imageResultFilters[key]||=new Set();input.checked?state.imageResultFilters[key].add(input.value):state.imageResultFilters[key].delete(input.value);renderImageOutput(state.imageSearchData,false);
  });
  $("#imageOutput").addEventListener("input",event=>{
    const input=event.target.closest("[data-image-filter-search]");if(!input)return;
    const keyword=input.value.trim().toLocaleLowerCase();
    input.closest(".facet-popover").querySelectorAll(".filter-option").forEach(option=>{option.hidden=keyword&&!option.textContent.toLocaleLowerCase().includes(keyword);});
  });
  $("#imageOutput").addEventListener("click",event=>{
    const analysis=event.target.closest("[data-image-analysis-id]");
    if(analysis){const id=analysis.dataset.imageAnalysisId;if(!state.imageAnalysisSelection.has(id)&&state.imageAnalysisSelection.size>=20)return toast("单次最多选择 20 条记录");state.imageAnalysisSelection.has(id)?state.imageAnalysisSelection.delete(id):state.imageAnalysisSelection.add(id);renderImageOutput(state.imageSearchData,false);return;}
    if(event.target.closest("#clearImageAnalysis")){state.imageAnalysisSelection.clear();renderImageOutput(state.imageSearchData,false);return;}
    if(event.target.closest("#generateImageReport")){generateImageReport();return;}
    if(event.target.closest("#openComparison")){openComparison();return;}
    if(event.target.closest("#clearComparisonSelection")){state.comparisonSelection.clear();renderImageOutput(state.imageSearchData,false);return;}
    const remove=event.target.closest("[data-remove-image-filter]");
    if(remove){state.imageResultFilters[remove.dataset.removeImageFilter]?.delete(remove.dataset.filterValue);renderImageOutput(state.imageSearchData,false);return;}
    if(event.target.closest("#clearImageResultFilters")){state.imageResultFilters={};renderImageOutput(state.imageSearchData,false);}
  });
}
document.addEventListener("DOMContentLoaded",init);
