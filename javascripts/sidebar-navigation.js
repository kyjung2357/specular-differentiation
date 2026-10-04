// Close the mobile drawer even when the selected heading is already in the URL.
document.addEventListener("click", function (event) {
  if (event.defaultPrevented || event.button !== 0 || event.ctrlKey ||
      event.metaKey || event.shiftKey || event.altKey) return;

  const link = event.target.closest('.specular-sidebar__toc a[href^="#"]');
  const drawer = document.getElementById("__drawer");
  if (link && drawer && drawer.checked) drawer.click();
});
