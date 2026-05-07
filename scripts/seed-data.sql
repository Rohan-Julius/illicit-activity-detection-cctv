-- Seed demo user for login
-- Email: demo@sentinel.ai
-- Password: demo123456
-- Note: Supabase handles password hashing automatically

-- Insert demo cameras
INSERT INTO cameras (name, location, latitude, longitude, status, video_url) VALUES
  ('Downtown Street', 'Fifth Avenue & Main St', 40.7128, -74.0060, 'live', 'https://images.unsplash.com/photo-1530382992516-a23d5c904abb?w=640&h=480&fit=crop'),
  ('Park Entrance', 'Central Park North', 40.7829, -73.9654, 'live', 'https://images.unsplash.com/photo-1552664730-d307ca884978?w=640&h=480&fit=crop'),
  ('Transit Hub', 'Grand Central Terminal', 40.7527, -73.9772, 'live', 'https://images.unsplash.com/photo-1531746790731-6c087fecd65a?w=640&h=480&fit=crop');

-- Insert sample incidents
INSERT INTO incidents (camera_id, class, confidence, timestamp) 
SELECT 
  cameras.id,
  event_types.type,
  RANDOM() * 0.3 + 0.7,
  NOW() - INTERVAL '1 minute' * FLOOR(RANDOM() * 120)
FROM cameras
CROSS JOIN (VALUES ('Fighting'), ('Robbery'), ('Vandalism')) AS event_types(type)
LIMIT 5;
