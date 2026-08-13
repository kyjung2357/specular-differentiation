(function () {
  var title = document.querySelector(".md-header__title");
  var logo = document.querySelector(".md-header .md-logo");

  if (!title || !logo) {
    return;
  }

  title.setAttribute("role", "link");
  title.setAttribute("tabindex", "0");
  title.setAttribute("aria-label", "Go to homepage");

  function goHome() {
    window.location.href = logo.href;
  }

  title.addEventListener("click", goHome);
  title.addEventListener("keydown", function (event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      goHome();
    }
  });
})();
