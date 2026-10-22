"use strict";

// One interface for the local Python API and the static GitHub Pages worker.
(() => {
  const isStatic = document.documentElement.dataset.mode === "static";
  let worker,
    serial = 0;
  const pending = new Map();
  window.resultsRequest = async (endpoint, filters = {}, signal) => {
    if (!isStatic) {
      const response = await fetch(
        `api/${endpoint}?${new URLSearchParams(filters)}`,
        { signal },
      );
      if (!response.ok)
        throw new Error(
          `Could not read ${endpoint} (HTTP ${response.status}).`,
        );
      return response.json();
    }
    if (!worker) {
      worker = new Worker(new URL("static-worker.js", document.baseURI));
      worker.onmessage = ({ data }) => {
        const request = pending.get(data.id);
        if (!request) return;
        pending.delete(data.id);
        request.cleanup();
        data.error
          ? request.reject(new Error(data.error))
          : request.resolve(data.result);
      };
      worker.onerror = () => {
        for (const request of pending.values()) {
          request.cleanup();
          request.reject(
            new Error("The results worker could not load. Refresh the page."),
          );
        }
        pending.clear();
        worker.terminate();
        worker = null;
      };
    }
    return new Promise((resolve, reject) => {
      const id = ++serial;
      const abort = () => {
        pending.delete(id);
        reject(new DOMException("Aborted", "AbortError"));
      };
      if (signal?.aborted) {
        abort();
        return;
      }
      signal?.addEventListener("abort", abort, { once: true });
      pending.set(id, {
        resolve,
        reject,
        cleanup: () => signal?.removeEventListener("abort", abort),
      });
      worker.postMessage({ id, endpoint, filters });
    });
  };
})();
