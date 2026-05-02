(function () {
  var SOURCE_LABELS = {
    arxiv: "arXiv",
    biorxiv: "bioRxiv",
    medrxiv: "medRxiv",
  };

  function contextFromPath() {
    var path = window.location.pathname.replace(/\/+$/, "");
    if (/\/daily-papers\/latest$/.test(path)) {
      return { type: "latest", date: "latest", source: "" };
    }

    var match = path.match(/\/daily-papers\/(\d{4}-\d{2}-\d{2})(?:-(arxiv|biorxiv|medrxiv))?$/);
    if (!match) {
      return null;
    }

    return {
      type: "daily",
      date: match[1],
      source: match[2] || "",
    };
  }

  function apiUrl(context) {
    if (context.type === "latest") {
      return new URL("../../api/latest.json", window.location.href).href;
    }
    return new URL("../../api/daily/" + context.date + ".json", window.location.href).href;
  }

  function fileDate(payload, context) {
    return payload.date || (context.type === "latest" ? "latest" : context.date);
  }

  function filteredPapers(payload, context) {
    var papers = payload.papers || [];
    if (!context.source) {
      return papers;
    }
    return papers.filter(function (paper) {
      return paper.source === context.source;
    });
  }

  function valueOrEmpty(value) {
    return value == null ? "" : String(value);
  }

  function csvCell(value) {
    return '"' + valueOrEmpty(value).replace(/"/g, '""') + '"';
  }

  function toCsv(payload, context) {
    var rows = [["date", "source", "published_date", "rank", "title", "author", "pdf_url", "paper_url", "summary"]];
    filteredPapers(payload, context).forEach(function (paper) {
      rows.push([
        paper.date || payload.date || "",
        paper.source_label || SOURCE_LABELS[paper.source] || paper.source || "",
        paper.published_date || "",
        paper.rank || "",
        paper.translated_title || paper.title || "",
        paper.author || "",
        paper.pdf_url || "",
        paper.paper_url || "",
        paper.translated_summary || paper.summary || "",
      ]);
    });
    return rows
      .map(function (row) {
        return row.map(csvCell).join(",");
      })
      .join("\n") + "\n";
  }

  function toMarkdown(payload, context) {
    var date = fileDate(payload, context);
    var scope = context.source ? " " + (SOURCE_LABELS[context.source] || context.source) : "";
    var lines = ["# " + date + scope + " 论文", ""];
    filteredPapers(payload, context).forEach(function (paper, index) {
      var title = paper.translated_title || paper.title || "Untitled";
      lines.push((index + 1) + ". [" + title + "](" + (paper.pdf_url || paper.paper_url || "#") + ")");
      if (paper.author) {
        lines.push("   - 作者: " + paper.author);
      }
      if (paper.published_date) {
        lines.push("   - 发布日期: " + paper.published_date);
      }
      if (paper.source_label || paper.source) {
        lines.push("   - 来源: " + (paper.source_label || paper.source));
      }
      if (paper.title && paper.title !== title) {
        lines.push("   - 原题: " + paper.title);
      }
      if (paper.paper_url) {
        lines.push("   - 页面: " + paper.paper_url);
      }
      var summary = paper.translated_summary || paper.summary || "";
      if (summary) {
        lines.push("   - 摘要: " + summary);
      }
      lines.push("");
    });
    return lines.join("\n");
  }

  function downloadFile(filename, content, type) {
    var blob = new Blob([content], { type: type + ";charset=utf-8" });
    var link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(function () {
      URL.revokeObjectURL(link.href);
    }, 1000);
  }

  function setStatus(toolbar, message, isError) {
    var status = toolbar.querySelector(".paper-export__status");
    if (!status) {
      return;
    }
    status.textContent = message;
    status.classList.toggle("is-error", Boolean(isError));
  }

  function fetchPayload(context) {
    return fetch(apiUrl(context), { cache: "no-store" }).then(function (response) {
      if (!response.ok) {
        throw new Error("HTTP " + response.status);
      }
      return response.json();
    });
  }

  function addButton(toolbar, label, handler) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "paper-export__button";
    button.textContent = label;
    button.addEventListener("click", handler);
    toolbar.insertBefore(button, toolbar.querySelector(".paper-export__status"));
  }

  function insertToolbar() {
    var context = contextFromPath();
    if (!context || document.querySelector(".paper-export")) {
      return;
    }

    var heading = document.querySelector(".md-content h1");
    if (!heading || !heading.parentNode) {
      return;
    }

    var toolbar = document.createElement("section");
    toolbar.className = "paper-export";
    toolbar.setAttribute("aria-label", "论文导出");

    var status = document.createElement("span");
    status.className = "paper-export__status";
    status.textContent = "导出当前页面对应的论文数据";
    toolbar.appendChild(status);

    addButton(toolbar, "导出 Markdown", function () {
      setStatus(toolbar, "正在生成 Markdown...", false);
      fetchPayload(context)
        .then(function (payload) {
          var suffix = context.source ? "-" + context.source : "";
          downloadFile(fileDate(payload, context) + suffix + "-papers.md", toMarkdown(payload, context), "text/markdown");
          setStatus(toolbar, "Markdown 已生成", false);
        })
        .catch(function (error) {
          setStatus(toolbar, "导出失败: " + error.message, true);
        });
    });

    addButton(toolbar, "导出 CSV", function () {
      setStatus(toolbar, "正在生成 CSV...", false);
      fetchPayload(context)
        .then(function (payload) {
          var suffix = context.source ? "-" + context.source : "";
          downloadFile(fileDate(payload, context) + suffix + "-papers.csv", toCsv(payload, context), "text/csv");
          setStatus(toolbar, "CSV 已生成", false);
        })
        .catch(function (error) {
          setStatus(toolbar, "导出失败: " + error.message, true);
        });
    });

    addButton(toolbar, "导出 JSON", function () {
      setStatus(toolbar, "正在生成 JSON...", false);
      fetchPayload(context)
        .then(function (payload) {
          var suffix = context.source ? "-" + context.source : "";
          var data = context.source
            ? Object.assign({}, payload, {
                count: filteredPapers(payload, context).length,
                papers: filteredPapers(payload, context),
              })
            : payload;
          downloadFile(
            fileDate(payload, context) + suffix + "-papers.json",
            JSON.stringify(data, null, 2) + "\n",
            "application/json"
          );
          setStatus(toolbar, "JSON 已生成", false);
        })
        .catch(function (error) {
          setStatus(toolbar, "导出失败: " + error.message, true);
        });
    });

    heading.insertAdjacentElement("afterend", toolbar);
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(insertToolbar);
  } else {
    document.addEventListener("DOMContentLoaded", insertToolbar);
  }
})();
