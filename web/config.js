/**
 * ChamAI Configuration
 * Default model source from Hugging Face Bucket Khanh434/cham-models
 */

export const CONFIG = {
  // Hugging Face Storage Bucket Base URL
  bucketBaseUrl: "https://huggingface.co/buckets/Khanh434/cham-models/resolve",

  // Model filenames
  models: {
    l1_dewarp: "uvdocnet.onnx",
    l2_qbox: "yolo-qbox.onnx",
    l3_cbox: "yolo-cbox.onnx",
    l4_acnet: "acnet.onnx"
  },

  // Pipeline detection parameters
  enableDewarp: true,
  qboxConf: 0.4,
  qboxIou: 0.15,
  cboxConf: 0.4,
  cboxIou: 0.15,
  acnetConf: 0.30,

  // Runtime engine
  ortCdn: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/"
};

/**
 * Resolves full direct download URL for a model key
 */
export function getModelUrl(modelKey) {
  const filename = CONFIG.models[modelKey] || "";
  return `${CONFIG.bucketBaseUrl}/${filename}`;
}
