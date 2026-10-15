const mongoose = require('mongoose');

const observationSchema = new mongoose.Schema({
  sol: { type: Number, required: true },
  ltst: { type: Number, required: true },
  ls: { type: Number, required: true },
  t_air: { type: Number, required: true },
  pressure: { type: Number, required: true },
  is_anomaly: { type: Boolean, default: false },
  anomaly_type: { type: String },
  analysis_result: { type: Object }
}, { timestamps: true });

module.exports = mongoose.model('Observation', observationSchema);
