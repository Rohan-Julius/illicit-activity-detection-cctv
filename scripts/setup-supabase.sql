-- Create cameras table
CREATE TABLE IF NOT EXISTS cameras (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  location VARCHAR(255) NOT NULL,
  latitude FLOAT NOT NULL,
  longitude FLOAT NOT NULL,
  status VARCHAR(20) DEFAULT 'live',
  video_url TEXT,
  stream_type VARCHAR DEFAULT 'websocket',
  last_seen_at TIMESTAMPTZ DEFAULT now(),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create incidents table
CREATE TABLE IF NOT EXISTS incidents (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  camera_id UUID NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
  camera_name VARCHAR(255),
  class VARCHAR(50) NOT NULL CHECK (class IN ('Fighting', 'Robbery', 'Vandalism')),
  confidence FLOAT DEFAULT 0.95,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  clip_url TEXT,
  twilio_status VARCHAR DEFAULT 'pending',
  acknowledged BOOLEAN DEFAULT false,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert dummy cameras
INSERT INTO cameras (name, location, latitude, longitude, status, video_url) VALUES
  ('Main Street Intersection', 'Downtown - 5th & Main', 40.7128, -74.0060, 'live', 'https://images.unsplash.com/photo-1611339555312-e607c25352ba?w=500&h=300&fit=crop'),
  ('Park Entrance', 'Central Park North', 40.7829, -73.9654, 'live', 'https://images.unsplash.com/photo-1517694712202-14dd9538aa97?w=500&h=300&fit=crop'),
  ('Shopping Mall', 'East Side Mall', 40.7505, -73.9680, 'live', 'https://images.unsplash.com/photo-1542401886-65d27afda266?w=500&h=300&fit=crop'),
  ('Bridge Toll', 'Brooklyn Bridge', 40.7061, -73.9969, 'offline', 'https://images.unsplash.com/photo-1557618666-a2a1c8f33a7d?w=500&h=300&fit=crop'),
  ('Transit Station', 'Grand Central', 40.7527, -73.9772, 'live', 'https://images.unsplash.com/photo-1552664730-d307ca884978?w=500&h=300&fit=crop'),
  ('Highway Exit 42', 'West Side Highway', 40.7614, -74.0037, 'live', 'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=500&h=300&fit=crop');

-- Insert dummy incidents
INSERT INTO incidents (camera_id, camera_name, class, confidence, timestamp) 
SELECT id, 'Main Street Intersection', 'Fighting', 0.92, CURRENT_TIMESTAMP - INTERVAL '45 minutes' FROM cameras WHERE name = 'Main Street Intersection'
UNION ALL
SELECT id, 'Park Entrance', 'Robbery', 0.87, CURRENT_TIMESTAMP - INTERVAL '2 hours' FROM cameras WHERE name = 'Park Entrance'
UNION ALL
SELECT id, 'Shopping Mall', 'Vandalism', 0.94, CURRENT_TIMESTAMP - INTERVAL '30 minutes' FROM cameras WHERE name = 'Shopping Mall'
UNION ALL
SELECT id, 'Main Street Intersection', 'Fighting', 0.89, CURRENT_TIMESTAMP - INTERVAL '1 hour' FROM cameras WHERE name = 'Main Street Intersection'
UNION ALL
SELECT id, 'Transit Station', 'Robbery', 0.91, CURRENT_TIMESTAMP - INTERVAL '15 minutes' FROM cameras WHERE name = 'Transit Station'
UNION ALL
SELECT id, 'Shopping Mall', 'Fighting', 0.85, CURRENT_TIMESTAMP - INTERVAL '3 hours' FROM cameras WHERE name = 'Shopping Mall'
UNION ALL
SELECT id, 'Highway Exit 42', 'Vandalism', 0.88, CURRENT_TIMESTAMP - INTERVAL '20 minutes' FROM cameras WHERE name = 'Highway Exit 42'
UNION ALL
SELECT id, 'Park Entrance', 'Fighting', 0.93, CURRENT_TIMESTAMP - INTERVAL '90 minutes' FROM cameras WHERE name = 'Park Entrance';

-- Create index for performance
CREATE INDEX IF NOT EXISTS idx_incidents_timestamp ON incidents(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_incidents_camera_id ON incidents(camera_id);

-- Enable RLS if needed (optional)
ALTER TABLE cameras ENABLE ROW LEVEL SECURITY;
ALTER TABLE incidents ENABLE ROW LEVEL SECURITY;

-- Create policies for authenticated users
CREATE POLICY "Allow authenticated users to view cameras" 
ON cameras FOR SELECT 
USING (auth.role() = 'authenticated');

CREATE POLICY "Allow authenticated users to view incidents" 
ON incidents FOR SELECT 
USING (auth.role() = 'authenticated');
