-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create cameras table
CREATE TABLE IF NOT EXISTS cameras (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL,
  location TEXT NOT NULL,
  latitude FLOAT NOT NULL,
  longitude FLOAT NOT NULL,
  status TEXT DEFAULT 'online',
  video_url TEXT,
  last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create incidents table
CREATE TABLE IF NOT EXISTS incidents (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  camera_id UUID NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
  class TEXT NOT NULL,
  confidence FLOAT DEFAULT 0.0,
  detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create incidents_count view for quick statistics
CREATE OR REPLACE VIEW incidents_count AS
SELECT 
  class,
  COUNT(*) as count
FROM incidents
WHERE detected_at > NOW() - INTERVAL '24 hours'
GROUP BY class;

-- Enable Row Level Security (RLS)
ALTER TABLE cameras ENABLE ROW LEVEL SECURITY;
ALTER TABLE incidents ENABLE ROW LEVEL SECURITY;

-- Create RLS policies (allow authenticated users to read all data)
CREATE POLICY "Enable read access for all users" ON cameras
  FOR SELECT USING (TRUE);

CREATE POLICY "Enable read access for all users" ON incidents
  FOR SELECT USING (TRUE);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_incidents_camera_id ON incidents(camera_id);
CREATE INDEX IF NOT EXISTS idx_incidents_detected_at ON incidents(detected_at);
CREATE INDEX IF NOT EXISTS idx_incidents_class ON incidents(class);

-- Insert demo cameras
INSERT INTO cameras (name, location, latitude, longitude, status, video_url)
VALUES
  ('Main Entrance', 'Building A, Ground Floor', 40.7128, -74.0060, 'online', 'https://sample-videos.com/video.mp4'),
  ('Parking Lot 1', 'Parking Lot Level 2', 40.7135, -74.0065, 'online', 'https://sample-videos.com/video.mp4'),
  ('Server Room', 'Building C, Floor 3', 40.7120, -74.0055, 'online', 'https://sample-videos.com/video.mp4')
ON CONFLICT DO NOTHING;
