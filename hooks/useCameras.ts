import { useEffect, useState } from 'react';
import { createClient } from '@supabase/supabase-js';

export interface Camera {
  id: string;
  name: string;
  location: string;
  latitude: number;
  longitude: number;
  status: 'online' | 'offline' | 'live' | 'alert' | string;
  video_url: string | null;
  stream_type?: 'websocket' | 'video' | string | null;
  last_seen_at?: string | null;
}

export function useCameras() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchCameras = async () => {
      try {
        const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
        const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

        if (!supabaseUrl || !supabaseKey) {
          setError('Supabase is not configured');
          setLoading(false);
          return;
        }

        const supabase = createClient(supabaseUrl, supabaseKey);
        const { data, error: fetchError } = await supabase
          .from('cameras')
          .select('*')
          .order('created_at', { ascending: true });

        if (fetchError) throw fetchError;
        setCameras(data || []);

        // Subscribe to realtime changes
        const subscription = supabase
          .channel('cameras')
          .on(
            'postgres_changes',
            { event: '*', schema: 'public', table: 'cameras' },
            (payload) => {
              if (payload.eventType === 'DELETE') {
                setCameras((prev) => prev.filter((c) => c.id !== (payload.old as any).id));
              } else {
                const next = payload.new as unknown as Camera;
                setCameras((prev) => {
                  const index = prev.findIndex((c) => c.id === next.id);
                  if (index > -1) {
                    const updated = [...prev];
                    updated[index] = next;
                    return updated;
                  }
                  return [...prev, next];
                });
              }
            }
          )
          .subscribe();

        return () => {
          subscription.unsubscribe();
        };
      } catch (err) {
        console.error('Error fetching cameras:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch cameras');
      } finally {
        setLoading(false);
      }
    };

    fetchCameras();
  }, []);

  return { cameras, loading, error };
}
