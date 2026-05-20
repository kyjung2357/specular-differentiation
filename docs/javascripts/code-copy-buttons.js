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

  if (typeof document$ !== "undefined") {
    document$.subscribe(setupCopyButtons);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setupCopyButtons);
  } else {
    setupCopyButtons();
  }
})();
