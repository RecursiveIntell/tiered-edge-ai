//! # Tiered Bridge — Adaptive Quantization for Edge AI
//!
//! Exports the same neural network model at multiple quantization levels,
//! creating a confidence bridge between hardware tiers.
//!
//! ## Concept
//!
//! A single model is trained at full precision (float32). The bridge then
//! exports it at two levels:
//!
//! - **Tier 0 (ESP32-S3)**: Aggressive quantization (int4/int8) via fib-quant's
//!   radial-angular vector quantization. Fits in 8MB PSRAM. Runs on MicroFlow.
//!
//! - **Tier 1 (UNO Q 4GB)**: Light quantization (int8/float16) or full precision.
//!   Runs on TensorFlow Lite / PyTorch with GPU acceleration.
//!
//! The bridge also calibrates the confidence threshold: at what point does the
//! tier-0 model's uncertainty justify waking the tier-1 model?
//!
//! ## Quantization Mapping
//!
//! Traditional int8 quantization: scale + zero-point, per-channel or per-tensor.
//!
//! fib-quant approach: radial-angular vector quantization in a Fibonacci
//! lattice. Instead of uniform quantization, the codebook is structured
//! so that nearby values in the original space are nearby in code space.
//! This gives better reconstruction at low bit-widths (4-bit) than
//! uniform quantization.
//!
//! turbo-quant approach: PolarQuant (polar coordinate quantization after
//! seeded rotation) + residual sketch. Data-oblivious — no training needed.
//! The ESP32-S3 can decode without a trained codebook, using only
//! (dim, bits, seed) parameters.

use serde::{Deserialize, Serialize};

/// Quantization tier identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tier {
    /// ESP32-S3: aggressive quantization (int4/int8)
    Tier0,
    /// Arduino UNO Q: light quantization (int8/float16)
    Tier1,
}

/// Quantization scheme for a model layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantScheme {
    /// Standard int8 uniform quantization (TFLite compatible)
    Int8Uniform {
        scale: f32,
        zero_point: i32,
    },
    /// int4 uniform quantization — half the size of int8
    Int4Uniform {
        scale: f32,
        zero_point: i32,
    },
    /// fib-quant radial-angular quantization
    /// Uses Fibonacci lattice codebook for better low-bit reconstruction
    FibQuant {
        dimensions: usize,
        codebook_bits: u8,
        seed: u64,
    },
    /// turbo-quant PolarQuant
    /// Polar coordinate quantization after seeded rotation
    PolarQuant {
        dimensions: usize,
        bits: u8,
        seed: u64,
    },
    /// No quantization (float32 or float16)
    Float {
        half_precision: bool,
    },
}

/// Model export configuration for a specific tier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierConfig {
    pub tier: Tier,
    pub scheme: QuantScheme,
    /// Maximum model size in bytes
    pub max_size_bytes: usize,
    /// Expected inference time (milliseconds)
    pub expected_latency_ms: f32,
}

/// Default configurations for each tier
impl TierConfig {
    /// ESP32-S3 sentinel: 8MB PSRAM, needs to fit in MicroFlow's static model
    /// Uses int4 or aggressive int8
    pub fn tier0_default() -> Self {
        Self {
            tier: Tier::Tier0,
            scheme: QuantScheme::Int4Uniform {
                scale: 0.01,
                zero_point: 8, // centered for int4 range [-8, 7]
            },
            max_size_bytes: 4 * 1024 * 1024, // 4MB max (leave room for firmware)
            expected_latency_ms: 50.0,
        }
    }

    /// UNO Q heavy: 4GB RAM, GPU acceleration
    /// Uses int8 or float16 — much higher precision
    pub fn tier1_default() -> Self {
        Self {
            tier: Tier::Tier1,
            scheme: QuantScheme::Float {
                half_precision: true, // float16 — 2x smaller, GPU-friendly
            },
            max_size_bytes: 500 * 1024 * 1024, // 500MB (plenty of room)
            expected_latency_ms: 10.0, // GPU-accelerated, faster despite higher precision
        }
    }
}

/// Confidence calibration between tiers
///
/// Maps the tier-0 model's confidence score to a boolean decision:
/// should we wake tier-1 for a second opinion?
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceCalibration {
    /// Wake tier-1 if confidence drops below this
    pub threshold: f32,
    /// Minimum time between wake signals (rate limiting, ms)
    pub min_wake_interval_ms: u64,
    /// Maximum consecutive wake signals before escalating
    pub max_consecutive_wakes: u8,
}

impl Default for ConfidenceCalibration {
    fn default() -> Self {
        Self {
            threshold: 0.75,
            min_wake_interval_ms: 5000,
            max_consecutive_wakes: 3,
        }
    }
}

/// Result of a tiered inference cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub tier: Tier,
    pub prediction: Vec<f32>,
    pub confidence: f32,
    pub inference_time_ms: f32,
    /// If this was a tier-0 result that triggered a wake
    pub woke_tier1: bool,
}

/// Feature vector transmitted between tiers
///
/// When tier-0 wakes tier-1, it sends the raw feature vector so tier-1
/// can run inference on the same input at higher precision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureTransfer {
    /// Magic bytes "TEAI"
    pub magic: [u8; 4],
    /// Tier-0's prediction (for cross-tier comparison)
    pub sentinel_prediction: Vec<f32>,
    /// Tier-0's confidence
    pub sentinel_confidence: f32,
    /// Raw input features that triggered the wake
    pub features: Vec<f32>,
    /// Timestamp (ms since boot)
    pub timestamp_ms: u64,
}

impl FeatureTransfer {
    pub const MAGIC: [u8; 4] = *b"TEAI";

    pub fn new(
        prediction: Vec<f32>,
        confidence: f32,
        features: Vec<f32>,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            magic: Self::MAGIC,
            sentinel_prediction: prediction,
            sentinel_confidence: confidence,
            features,
            timestamp_ms,
        }
    }

    /// Serialize to bytes for WiFi transmission
    pub fn to_bytes(&self) -> Vec<u8> {
        // Simple binary format: magic + prediction_len + prediction + conf + features_len + features + timestamp
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.magic);
        bytes.extend_from_slice(&(self.sentinel_prediction.len() as u32).to_le_bytes());
        for &v in &self.sentinel_prediction {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&self.sentinel_confidence.to_le_bytes());
        bytes.extend_from_slice(&(self.features.len() as u32).to_le_bytes());
        for &v in &self.features {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&self.timestamp_ms.to_le_bytes());
        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 4 + 4 {
            return None;
        }
        let magic: [u8; 4] = data[0..4].try_into().ok()?;
        if magic != Self::MAGIC {
            return None;
        }

        let mut pos = 4;
        let pred_len = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;

        let mut prediction = Vec::with_capacity(pred_len);
        for _ in 0..pred_len {
            prediction.push(f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?));
            pos += 4;
        }

        let confidence = f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
        pos += 4;

        let feat_len = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;

        let mut features = Vec::with_capacity(feat_len);
        for _ in 0..feat_len {
            features.push(f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?));
            pos += 4;
        }

        let timestamp_ms = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);

        Some(Self {
            magic,
            sentinel_prediction: prediction,
            sentinel_confidence: confidence,
            features,
            timestamp_ms,
        })
    }
}

/// Estimate model size after quantization
pub fn estimate_model_size(
    num_params: usize,
    scheme: &QuantScheme,
) -> usize {
    match scheme {
        QuantScheme::Int8Uniform { .. } => num_params, // 1 byte per param
        QuantScheme::Int4Uniform { .. } => (num_params + 1) / 2, // 0.5 bytes per param
        QuantScheme::FibQuant { codebook_bits, .. } => {
            // Each param is encoded as a codebook index
            (num_params * *codebook_bits as usize) / 8
        }
        QuantScheme::PolarQuant { bits, .. } => {
            // Each param is encoded as a polar quantized value
            (num_params * *bits as usize) / 8
        }
        QuantScheme::Float { half_precision } => {
            if *half_precision {
                num_params * 2 // float16: 2 bytes per param
            } else {
                num_params * 4 // float32: 4 bytes per param
            }
        }
    }
}

/// Compute the compression ratio between two tiers
pub fn compression_ratio(tier0: &TierConfig, tier1: &TierConfig) -> f32 {
    // Assuming same number of params, ratio is tier1_size / tier0_size
    let dummy_params = 1_000_000; // 1M params for estimation
    let size0 = estimate_model_size(dummy_params, &tier0.scheme);
    let size1 = estimate_model_size(dummy_params, &tier1.scheme);
    size1 as f32 / size0 as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_transfer_roundtrip() {
        let original = FeatureTransfer::new(
            vec![0.5, -0.3],
            0.82,
            vec![1.0, 2.0, 3.0],
            12345,
        );
        let bytes = original.to_bytes();
        let decoded = FeatureTransfer::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.sentinel_prediction, original.sentinel_prediction);
        assert_eq!(decoded.sentinel_confidence, original.sentinel_confidence);
        assert_eq!(decoded.features, original.features);
        assert_eq!(decoded.timestamp_ms, original.timestamp_ms);
    }

    #[test]
    fn test_int4_halves_size() {
        let int8_size = estimate_model_size(1000, &QuantScheme::Int8Uniform { scale: 0.01, zero_point: 0 });
        let int4_size = estimate_model_size(1000, &QuantScheme::Int4Uniform { scale: 0.01, zero_point: 8 });
        assert_eq!(int4_size, 500);
        assert_eq!(int8_size, 1000);
    }

    #[test]
    fn test_compression_ratio() {
        let tier0 = TierConfig::tier0_default();
        let tier1 = TierConfig::tier1_default();
        let ratio = compression_ratio(&tier0, &tier1);
        // float16 (2 bytes) vs int4 (0.5 bytes) = 4x compression
        assert!((ratio - 4.0).abs() < 0.01, "expected 4x, got {}", ratio);
    }

    #[test]
    fn test_bad_magic_rejected() {
        let bad_data = [0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0];
        assert!(FeatureTransfer::from_bytes(&bad_data).is_none());
    }
}