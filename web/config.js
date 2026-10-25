/**
 * ChamAI Configuration
 * Model CDN endpoints, default detection thresholds, and storage keys.
 */

export const STORAGE_KEY_CDN = "chamai_cdn_settings";

export const DEFAULT_CONFIG = {
  // CDN Provider: 'cloudflare_r2', 'huggingface', or 'custom'
  provider: "huggingface",

  // Base URLs for CDN providers
  r2BaseUrl: "",
  hfBaseUrl: "https://huggingface.co/buckets/Khanh434/cham-models",
  customBaseUrl: "",

  // Individual model filenames on CDN
  models: {
    l1_dewarp: "uvdocnet.onnx",
    l2_qbox: "yolo-qbox.onnx",
    l3_cbox: "yolo-cbox.onnx",
    l4_acnet: "acnet.onnx"
  },

  // Pipeline execution settings
  enableDewarp: true, // L1 UVDocNet (can be disabled on low-end devices)
  qboxConf: 0.35,
  qboxIou: 0.45,
  cboxConf: 0.25,
  cboxIou: 0.45,
  acnetConf: 0.40,

  // Runtime engine
  ortCdn: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/"
};

/**
 * Loads configured CDN settings from localStorage or defaults
 */
export function loadConfig() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY_CDN);
    if (saved) {
      return { ...DEFAULT_CONFIG, ...JSON.parse(saved) };
    }
  } catch (e) {
    console.warn("Could not read stored CDN settings:", e);
  }
  return { ...DEFAULT_CONFIG };
}

/**
 * Saves CDN settings to localStorage
 */
export function saveConfig(cfg) {
  try {
    localStorage.setItem(STORAGE_KEY_CDN, JSON.stringify(cfg));
  } catch (e) {
    console.warn("Could not save CDN settings:", e);
  }
}

/**
 * Resolves full URL for a specific model key ('l1_dewarp', 'l2_qbox', 'l3_cbox', 'l4_acnet')
 */
export function getModelUrl(config, modelKey) {
  let base = "";
  if (config.provider === "cloudflare_r2") {
    base = config.r2BaseUrl.trim();
  } else if (config.provider === "custom") {
    base = config.customBaseUrl.trim();
  } else {
    base = config.hfBaseUrl.trim();
  }

  // Remove trailing slash
  base = base.replace(/\/+$/, "");
  const filename = config.models[modelKey] || "";
  return `${base}/${filename}`;
}
