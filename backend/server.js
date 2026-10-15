const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');
require('dotenv').config({ path: path.resolve(__dirname, '../.env') });

const app = express();
const PORT = Number(process.env.BACKEND_PORT || process.env.PORT || 3001);

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use('/api/auth', require('./routes/auth'));
app.use('/api/observations', require('./routes/observations'));

const frontendPath = path.resolve(__dirname, '../frontend/dist');
app.use(express.static(frontendPath));
app.get('/login', (req, res) => {
  res.sendFile(path.join(frontendPath, 'index.html'));
});

const { MongoMemoryServer } = require('mongodb-memory-server');
const seed = require('./seed');

// Database connection
async function startServer() {
  try {
    let mongoUri = process.env.MONGODB_URI;
    let isMemory = false;
    if (!mongoUri) {
      console.log('No MONGODB_URI found. Starting in-memory MongoDB server...');
      const mongoServer = await MongoMemoryServer.create();
      mongoUri = mongoServer.getUri();
      isMemory = true;
    }
    
    await mongoose.connect(mongoUri);
    console.log('Connected to MongoDB at', mongoUri);
    
    if (isMemory) {
       console.log('Seeding in-memory database automatically...');
       await seed(mongoUri);
    }
    
    app.listen(PORT, () => {
      console.log(`Backend server running on port ${PORT}`);
    });
  } catch (err) {
    console.error('MongoDB connection error:', err);
  }
}
startServer();
