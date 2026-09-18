document.addEventListener("DOMContentLoaded", () => {
  // Core controls
  const dateInput = document.getElementById("dateInput");
  const predictBtn = document.getElementById("predictBtn");
  const btnSpinner = document.getElementById("btnSpinner");
  const messageContainer = document.getElementById("messageContainer");
  const resultsSection = document.getElementById("resultsSection");
  const statusText = document.getElementById("statusText");
  const versionTag = document.getElementById("versionTag");
  const niftyBadge = document.getElementById("niftyBadge");
  const niftyReturnValue = document.getElementById("niftyReturnValue");
  const dateResolutionNote = document.getElementById("dateResolutionNote");

  // KPI elements
  const kpiTotalStocks = document.getElementById("kpiTotalStocks");
  const kpiPredictedBeatPct = document.getElementById("kpiPredictedBeatPct");
  const kpiBeatAccuracy = document.getElementById("kpiBeatAccuracy");
  const kpiBenchmarkAccuracy = document.getElementById("kpiBenchmarkAccuracy");

  // Tab controls
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");
  const shortlistBadgeCount = document.getElementById("shortlistBadgeCount");

  // Table bodies & counters
  const signalsTableBody = document.getElementById("signalsTableBody");
  const outcomesTableBody = document.getElementById("outcomesTableBody");
  const shortlistTableBody = document.getElementById("shortlistTableBody");
  const signalsCount = document.getElementById("signalsCount");
  const outcomesCount = document.getElementById("outcomesCount");

  // Shortlist banner elements
  const shortlistBannerCount = document.getElementById("shortlistBannerCount");
  const shortlistBannerAlpha = document.getElementById("shortlistBannerAlpha");

  // Slide-out Drawer Elements
  const drawerBackdrop = document.getElementById("drawerBackdrop");
  const stockExplainDrawer = document.getElementById("stockExplainDrawer");
  const closeDrawerBtn = document.getElementById("closeDrawerBtn");
  const drawerStockTitle = document.getElementById("drawerStockTitle");
  const drawerSignalBadge = document.getElementById("drawerSignalBadge");
  const drawerStockClose = document.getElementById("drawerStockClose");
  const drawerStockProb = document.getElementById("drawerStockProb");
  const drawerProbVal = document.getElementById("drawerProbVal");
  const drawerAlphaVal = document.getElementById("drawerAlphaVal");
  const drawerPillarsContainer = document.getElementById("drawerPillarsContainer");
  const drawerDriversList = document.getElementById("drawerDriversList");
  const drawerRisksList = document.getElementById("drawerRisksList");

  // Open & populate drawer
  function openDrawer(item) {
    if (!item) return;
    drawerStockTitle.textContent = item.Stock;
    drawerStockClose.textContent = item.formatted_close || `₹${item.CLOSE?.toFixed(2) || '0.00'}`;
    drawerStockProb.textContent = `${item.formatted_prob} Beat Nifty`;
    drawerProbVal.textContent = item.formatted_prob;

    const alphaStr = item.formatted_predicted_alpha || item.formatted_actual_alpha || "N/A";
    drawerAlphaVal.textContent = alphaStr;
    drawerAlphaVal.className = alphaStr.startsWith("+") ? "snapshot-val highlight-green" : alphaStr.startsWith("-") ? "snapshot-val highlight-red" : "snapshot-val";

    const isYes = item.Beat_Nifty_Signal === "Yes" || (item.prob_beat_nifty100 >= 0.5);
    drawerSignalBadge.textContent = isYes ? "BEAT NIFTY: YES" : "BEAT NIFTY: NO";
    drawerSignalBadge.className = isYes ? "signal-pill signal-yes" : "signal-pill signal-no";

    const attr = item.attribution;
    if (attr) {
      // Pillars
      drawerPillarsContainer.innerHTML = (attr.pillars || []).map(p => {
        const isPos = p.net_impact > 0;
        const isNeg = p.net_impact < 0;
        const scoreClass = isPos ? "pillar-pos" : isNeg ? "pillar-neg" : "pillar-neutral";
        return `
          <div class="pillar-chip">
            <span class="pillar-name" title="${p.pillar}">${p.pillar}</span>
            <span class="pillar-score ${scoreClass}">${p.formatted_net_impact}</span>
          </div>
        `;
      }).join("");

      // Top Positive Drivers
      if (attr.drivers && attr.drivers.length > 0) {
        const maxImpact = Math.max(...attr.drivers.map(d => Math.abs(d.impact)), 1.0);
        drawerDriversList.innerHTML = attr.drivers.map(d => {
          const barPct = Math.min(Math.max((Math.abs(d.impact) / maxImpact) * 100, 15), 100);
          return `
            <div class="factor-item">
              <div class="factor-top-row">
                <div>
                  <div class="factor-name">${d.name}</div>
                  <div class="factor-pillar-tag">${d.pillar}</div>
                </div>
                <div class="factor-impact pos">${d.formatted_impact}</div>
              </div>
              <div class="factor-bar-wrapper">
                <div class="factor-bar pos" style="width: ${barPct}%;"></div>
              </div>
            </div>
          `;
        }).join("");
      } else {
        drawerDriversList.innerHTML = `<div style="color:var(--text-muted);font-size:0.8rem;padding:0.5rem 0;">No strong positive drivers detected for this asset.</div>`;
      }

      // Top Negative Risks
      if (attr.risks && attr.risks.length > 0) {
        const maxRisk = Math.max(...attr.risks.map(r => Math.abs(r.impact)), 1.0);
        drawerRisksList.innerHTML = attr.risks.map(r => {
          const barPct = Math.min(Math.max((Math.abs(r.impact) / maxRisk) * 100, 15), 100);
          return `
            <div class="factor-item">
              <div class="factor-top-row">
                <div>
                  <div class="factor-name">${r.name}</div>
                  <div class="factor-pillar-tag">${r.pillar}</div>
                </div>
                <div class="factor-impact neg">${r.formatted_impact}</div>
              </div>
              <div class="factor-bar-wrapper">
                <div class="factor-bar neg" style="width: ${barPct}%;"></div>
              </div>
            </div>
          `;
        }).join("");
      } else {
        drawerRisksList.innerHTML = `<div style="color:var(--text-muted);font-size:0.8rem;padding:0.5rem 0;">No significant headwinds identified.</div>`;
      }
    } else {
      drawerPillarsContainer.innerHTML = `<div style="color:var(--text-muted);font-size:0.8rem;grid-column: span 2;">Attribution unavailable for this model mode.</div>`;
      drawerDriversList.innerHTML = `<div style="color:var(--text-muted);font-size:0.8rem;">TreeSHAP attribution requires loaded XGBoost model.</div>`;
      drawerRisksList.innerHTML = `<div style="color:var(--text-muted);font-size:0.8rem;">TreeSHAP attribution requires loaded XGBoost model.</div>`;
    }

    drawerBackdrop.classList.remove("hidden");
    stockExplainDrawer.classList.remove("hidden");
    document.body.style.overflow = "hidden";
  }

  function closeDrawer() {
    drawerBackdrop.classList.add("hidden");
    stockExplainDrawer.classList.add("hidden");
    document.body.style.overflow = "";
  }

  if (closeDrawerBtn) closeDrawerBtn.addEventListener("click", closeDrawer);
  if (drawerBackdrop) drawerBackdrop.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !stockExplainDrawer.classList.contains("hidden")) {
      closeDrawer();
    }
  });

  // Message alert helper
  function showMessage(type, text) {
    messageContainer.className = `alert alert-${type}`;
    messageContainer.innerHTML = `
      <span class="alert-icon">${type === "warning" ? "⚠️" : type === "danger" ? "❌" : "ℹ️"}</span>
      <span class="alert-text">${text}</span>
    `;
    messageContainer.classList.remove("hidden");
  }

  function hideMessage() {
    messageContainer.classList.add("hidden");
  }

  // 1. Tab switching logic
  function setupTabs() {
    tabButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetId = btn.getAttribute("data-tab");

        tabButtons.forEach((b) => {
          b.classList.remove("active");
          b.setAttribute("aria-selected", "false");
        });
        tabContents.forEach((c) => c.classList.remove("active"));

        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");

        const targetPanel = document.getElementById(targetId);
        if (targetPanel) {
          targetPanel.classList.add("active");
        }
      });
    });
  }

  // 2. Initialize metadata from backend
  async function loadMetadata() {
    try {
      const res = await fetch("/api/meta");
      if (!res.ok) throw new Error("Failed to load metadata");
      const data = await res.json();

      if (data.min_date) {
        dateInput.min = data.min_date;
        const maxValidDate = data.max_prediction_date || "2024-09-30";
        dateInput.max = maxValidDate;
        dateInput.value = data.default_date || maxValidDate;
      }

      if (data.active_model_mode && versionTag) {
        if (data.active_model_mode.startsWith("v2")) {
          versionTag.textContent = "Version 2 (Dual Model)";
          statusText.textContent = `Model + Data Loaded (${data.unique_stocks} Stocks, 2024 Test Set)`;
        } else {
          versionTag.textContent = "Version 1";
          statusText.textContent = `Model + Data Loaded (${data.unique_stocks} Stocks)`;
        }
      }
    } catch (err) {
      console.error("Metadata load error:", err);
      showMessage("danger", "Could not load dataset metadata. Please check the backend server.");
    }
  }

  // 3. Predict handler
  async function handlePredict() {
    const selectedDate = dateInput.value;
    if (!selectedDate) {
      showMessage("warning", "Please select a trading date first.");
      return;
    }

    hideMessage();
    dateResolutionNote.classList.add("hidden");
    resultsSection.classList.add("hidden");
    predictBtn.disabled = true;
    btnSpinner.classList.remove("hidden");

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ date: selectedDate })
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server responded with status ${res.status}`);
      }

      const data = await res.json();

      // Show date resolution note if needed
      if (!data.is_exact_date) {
        dateResolutionNote.textContent = `Note: Selected date was not an active trading day. Using closest active trading date: ${data.resolved_date}`;
        dateResolutionNote.classList.remove("hidden");
      }

      // Update Nifty Expected Return Badge
      if (data.nifty_expected_return !== undefined) {
        niftyReturnValue.textContent = data.formatted_nifty_expected || `${data.nifty_expected_return.toFixed(2)}%`;
        niftyBadge.classList.remove("hidden");
      }

      // Render KPIs
      renderKPIs(data.performance_summary);

      // Render Tables
      renderSignals(data.signals || []);
      renderOutcomes(data.outcomes || []);
      renderShortlist(data.shortlist || [], data.num_shortlisted || 0, data.formatted_avg_actual_alpha || "+0.00%");

      resultsSection.classList.remove("hidden");
    } catch (err) {
      console.error("Prediction error:", err);
      showMessage("danger", `❌ Prediction failed: ${err.message}`);
    } finally {
      predictBtn.disabled = false;
      btnSpinner.classList.add("hidden");
    }
  }

  // 4. Render KPIs
  function renderKPIs(summary) {
    if (!summary) return;
    kpiTotalStocks.textContent = summary.total_stocks || "0";
    kpiPredictedBeatPct.textContent = `${summary.predicted_beat_pct}%`;
    kpiBeatAccuracy.textContent = `${summary.beat_accuracy_pct}%`;
    kpiBenchmarkAccuracy.textContent = `${summary.benchmark_adjusted_accuracy_pct}%`;
  }

  // 5. Render Signals Table
  function renderSignals(signals) {
    signalsTableBody.innerHTML = "";
    signalsCount.textContent = `${signals.length} stocks`;

    if (signals.length === 0) {
      signalsTableBody.innerHTML = `<tr><td colspan="8" style="text-align:center;color:var(--text-muted);">No signals available.</td></tr>`;
      return;
    }

    signals.forEach((item) => {
      const tr = document.createElement("tr");
      tr.className = "clickable-row";
      const rankBadgeClass = item.rank <= 3 ? "rank-badge top-3" : "rank-badge";
      const isYes = item.Beat_Nifty_Signal === "Yes";
      const signalPillClass = isYes ? "signal-pill signal-yes" : "signal-pill signal-no";

      const alphaVal = item.predicted_alpha_vs_nifty;
      const alphaClass = alphaVal > 0 ? "alpha-badge alpha-pos" : alphaVal < 0 ? "alpha-badge alpha-neg" : "alpha-badge";

      tr.innerHTML = `
        <td class="col-rank"><span class="${rankBadgeClass}">${item.rank}</span></td>
        <td class="col-stock"><strong>${item.Stock}</strong></td>
        <td class="col-num">${item.formatted_close}</td>
        <td class="col-center"><span class="${signalPillClass}">${item.Beat_Nifty_Signal}</span></td>
        <td class="col-num" style="color:#34d399;font-weight:600;">${item.formatted_prob}</td>
        <td class="col-num">${item.formatted_pred_return}</td>
        <td class="col-num ${alphaClass}">${item.formatted_predicted_alpha}</td>
        <td class="col-center">
          <button class="btn-explain" type="button" title="View Model Drivers for ${item.Stock}">⚡ Explain</button>
        </td>
      `;

      tr.addEventListener("click", () => openDrawer(item));
      signalsTableBody.appendChild(tr);
    });
  }

  // 6. Render Outcomes Table
  function renderOutcomes(outcomes) {
    outcomesTableBody.innerHTML = "";
    outcomesCount.textContent = `${outcomes.length} stocks`;

    if (outcomes.length === 0) {
      outcomesTableBody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:var(--text-muted);">No outcomes available.</td></tr>`;
      return;
    }

    outcomes.forEach((item, idx) => {
      const tr = document.createElement("tr");
      const devVal = item.return_deviation;
      const devClass = devVal > 0 ? "alpha-pos" : devVal < 0 ? "alpha-neg" : "";

      tr.innerHTML = `
        <td class="col-rank">${idx + 1}</td>
        <td class="col-stock">${item.Stock}</td>
        <td class="col-num">${item.formatted_pred_return}</td>
        <td class="col-num">${item.formatted_actual_return}</td>
        <td class="col-num ${devClass}">${item.formatted_deviation}</td>
      `;
      outcomesTableBody.appendChild(tr);
    });
  }

  // 7. Render Shortlist Table
  function renderShortlist(shortlist, count, avgAlpha) {
    shortlistTableBody.innerHTML = "";
    shortlistBadgeCount.textContent = count;
    shortlistBannerCount.textContent = `${count} stocks`;
    shortlistBannerAlpha.textContent = `Average Actual Alpha: ${avgAlpha}`;

    if (shortlist.length === 0) {
      shortlistTableBody.innerHTML = `<tr><td colspan="9" style="text-align:center;color:var(--text-muted);padding:2rem;">No stocks met the consensus winning criteria on this date.</td></tr>`;
      return;
    }

    shortlist.forEach((item) => {
      const tr = document.createElement("tr");
      tr.className = "clickable-row";
      const rankBadgeClass = item.rank <= 3 ? "rank-badge top-3" : "rank-badge";

      tr.innerHTML = `
        <td class="col-rank"><span class="${rankBadgeClass}">${item.rank}</span></td>
        <td class="col-stock"><strong>${item.Stock}</strong></td>
        <td class="col-num">${item.formatted_close}</td>
        <td class="col-num" style="color:#34d399;font-weight:600;">${item.formatted_prob}</td>
        <td class="col-num">${item.formatted_pred_return}</td>
        <td class="col-num">${item.formatted_actual_return}</td>
        <td class="col-num">${item.formatted_nifty_return}</td>
        <td class="col-num alpha-pos" style="font-weight:700;">${item.formatted_actual_alpha}</td>
        <td class="col-center">
          <button class="btn-explain" type="button" title="View Model Drivers for ${item.Stock}">⚡ Explain</button>
        </td>
      `;

      tr.addEventListener("click", () => openDrawer(item));
      shortlistTableBody.appendChild(tr);
    });
  }

  // Initialize
  setupTabs();
  loadMetadata();
  predictBtn.addEventListener("click", handlePredict);
});
