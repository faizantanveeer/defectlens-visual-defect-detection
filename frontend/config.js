// Runtime settings for the test UI. Edit here instead of in app.js.
window.APP_CONFIG = {
  title: "Visual Defect Detection",   // product name shown in the header and tab
  apiUrl: "http://localhost:8000",    // FastAPI backend
  maxBatch: 16,                       // must not exceed the API's VDD_MAX_BATCH_SIZE
};
