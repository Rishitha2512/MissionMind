const mongoose = require('mongoose');

const instrumentSchema = new mongoose.Schema({
  name: { type: String, required: true },
  acronym: { type: String },
  mission_id: { type: mongoose.Schema.Types.ObjectId, ref: 'Mission' },
  description: { type: String },
  type: { type: String }
}, { timestamps: true });

module.exports = mongoose.model('Instrument', instrumentSchema);
