// Preserve bookmarks into the former combined API pages.
(function () {
  const modules = {
    calculation: {
      members: ["scaled_mean", "derivative", "gradient", "jacobian"],
      sections: ["examples", "scaled-angular-mean", "vector-derivative"],
    },
    optimization: {
      members: ["specular_gradient", "minimize", "make_direction", "make_step_size", "OptimizationResult", "LineSearchError"],
      sections: ["speg", "compose-rules", "step-size-rules", "classical-and-specular-search-derivatives", "results-and-execution", "backend-selection"],
    },
    ode: {
      members: ["ellipse_scheme", "euler_scheme_1", "euler_scheme_2", "euler_scheme_5", "ODEResult"],
      sections: ["unscaled-specular-euler-methods", "type-1", "type-2", "type-5", "prescribed-scale", "nonlinear-solve-controls", "automatic-scale-selection-modes", "result"],
    },
    backends: {
      members: ["get_backend", "available_backends", "set_backend", "use_backend", "BackendName"],
      sections: ["persistent-selection", "temporary-selection", "backend-behavior"],
    },
  };

  function followLegacyAnchor() {
    const match = window.location.pathname.match(/^(.*)\/api\/(calculation|optimization|ode|backends|backend)\/$/);
    if (!match || !window.location.hash) return;
    const module = match[2] === "backend" ? "backends" : match[2];
    const fragment = window.location.hash.slice(1);
    const member = modules[module].members.find(name =>
      fragment === name || fragment.endsWith("." + name)
    );
    if (member) {
      window.location.replace(`${match[1]}/api/${module}/${member}/`);
    } else if (modules[module].sections.includes(fragment)) {
      window.location.replace(`${match[1]}/user-guide/${module}/#${fragment}`);
    }
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(followLegacyAnchor);
  } else {
    followLegacyAnchor();
  }
})();
