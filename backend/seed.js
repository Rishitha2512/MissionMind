const mongoose = require('mongoose');
const Mission = require('./models/Mission');
const Instrument = require('./models/Instrument');
const Observation = require('./models/Observation');
const fs = require('fs');
const path = require('path');
const readline = require('readline');
require('dotenv').config({ path: path.resolve(__dirname, '../.env') });

const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017/missionmind';

async function seed(mongoUri) {
  try {
    await mongoose.connect(mongoUri || MONGODB_URI);
    console.log('Connected to DB for seeding');

    // Clear DB
    await Mission.deleteMany({});
    await Instrument.deleteMany({});
    await Observation.deleteMany({});

    // 1. Seed Mission & Instrument
    const mission = new Mission({
      name: 'Mars 2020',
      description: 'Perseverance rover mission',
      launch_date: new Date('2020-07-30'),
      landing_date: new Date('2021-02-18'),
      status: 'active'
    });
    await mission.save();

    const instrument = new Instrument({
      name: 'Mars Environmental Dynamics Analyzer',
      acronym: 'MEDA',
      mission_id: mission._id,
      description: 'Sensors that provide measurements of temperature, wind speed and direction, pressure, relative humidity and dust size and shape.',
      type: 'weather'
    });
    await instrument.save();
    console.log('Seeded Mission & Instrument');

    // 2. Seed Observations from CSV if it exists
    const csvPath = path.join(__dirname, '../data/processed/meda_clean.csv');
    if (fs.existsSync(csvPath)) {
      console.log('Found meda_clean.csv, seeding observations... (this might take a minute)');
      const fileStream = fs.createReadStream(csvPath);
      const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

      let isHeader = true;
      let batch = [];
      
      for await (const line of rl) {
        if (isHeader) { isHeader = false; continue; }
        
        // CSV format: sol,ltst,ls,t_air,pressure,is_anomaly,anomaly_type
        const [sol, ltst, ls, t_air, pressure, is_anomaly, anomaly_type] = line.split(',');
        
        if (sol && ltst) {
          batch.push({
            sol: Number(sol),
            ltst: Number(ltst),
            ls: Number(ls),
            t_air: Number(t_air),
            pressure: Number(pressure),
            is_anomaly: is_anomaly === 'True',
            anomaly_type: anomaly_type === 'none' ? null : anomaly_type
          });
        }

        if (batch.length === 500) {
          await Observation.insertMany(batch);
          batch = [];
        }
      }

      if (batch.length > 0) {
        await Observation.insertMany(batch);
      }
      console.log('Seeded observations');
    } else {
      console.log('No CSV found to seed observations. Generate python data first.');
    }

    console.log('Seeding complete!');
  } catch (error) {
    console.error('Seeding error:', error);
  }
}

module.exports = seed;

if (require.main === module) {
  seed().then(() => process.exit(0));
}
