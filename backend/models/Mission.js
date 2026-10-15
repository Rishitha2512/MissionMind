const mongoose = require('mongoose');

const missionSchema = new mongoose.Schema({
  name: { type: String, required: true },
  description: { type: String },
  launch_date: { type: Date },
  landing_date: { type: Date },
  status: { type: String }
}, { timestamps: true });

module.exports = mongoose.model('Mission', missionSchema);
