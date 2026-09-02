(function () {
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }

    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.top = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();

    try {
      document.execCommand("copy");
      return Promise.resolve();
    } catch (error) {
      return Promise.reject(error);
    } finally {
      textarea.remove();
    }
  }

  function codeTextFrom(block) {
    const code =
      block.querySelector("td.code code") ||
      block.querySelector("pre > code") ||
      block.querySelector("code");

    return code ? code.textContent : "";
  }

  function setupCopyButtons() {
    document.querySelectorAll(".md-typeset div.highlight > .code-copy-button").forEach((button) => {
      if (button.dataset.copyReady === "true") {
        return;
      }

      button.dataset.copyReady = "true";

      button.addEventListener("click", async () => {
        const block = button.closest("div.highlight");

        if (!block) {
          return;
        }

        try {
          await copyText(codeTextFrom(block));
          button.textContent = "Copied";
          window.setTimeout(() => {
            button.textContent = "Copy";
          }, 1400);
        } catch (error) {
          button.textContent = "Error";
          window.setTimeout(() => {
            button.textContent = "Copy";
          }, 1400);
        }
      });
    });
  }

  const sourceBaseUrl =
    "https://github.com/kyjung2357/specular-differentiation/blob/main/";

  const sourcePathMap = [
    { page: "/api/calculation/", source: "specular/calculation.py" },
    { page: "/api/backend/", source: "specular/backends/_registry.py" },
  ];

  function sourceUrlForCurrentPage() {
    const path = window.location.pathname.replace(/\/+$/, "/");
    const match = sourcePathMap.find((item) => path.includes(item.page));
    return match ? sourceBaseUrl + match.source : "";
  }

  function setupFullSourceLinks() {
    const sourceUrl = sourceUrlForCurrentPage();

    if (!sourceUrl) {
      return;
    }

    document.querySelectorAll(".md-typeset div.doc-signature.highlight").forEach((block) => {
      if (block.dataset.fullSourceReady === "true") {
        return;
      }

      block.dataset.fullSourceReady = "true";

      const link = document.createElement("a");
      link.className = "full-source-button";
      link.href = sourceUrl;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = "Full source";
      block.appendChild(link);
    });
  }

  function setupCodeActions() {
    setupCopyButtons();
    setupFullSourceLinks();
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(setupCodeActions);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setupCodeActions);
  } else {
    setupCodeActions();
  }
})();
