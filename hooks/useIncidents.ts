import { useEffect, useState } from 'react';
import { createClient } from '@supabase/supabase-js';

export type IncidentClass = 'Fighting' | 'Robbery' | 'Vandalism';

export interface Incident {
  id: string;
  camera_id: string;
  class: IncidentClass;
  confidence: number;
  timestamp: string;
  camera_name?: string;
  clip_url?: string | null;
  twilio_status?: 'pending' | 'sent' | 'failed' | string | null;
  acknowledged?: boolean | null;
}

export function useIncidents() {
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [incidentCounts, setIncidentCounts] = useState({
    fighting: 0,
    robbery: 0,
    vandalism: 0,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let subscription: { unsubscribe: () => void } | null = null;

    const normalizeClass = (value: unknown): string => String(value || '').trim().toLowerCase();

    const computeCounts = (rows: Array<{ class?: string }>) => {
      const next = { fighting: 0, robbery: 0, vandalism: 0 };
      for (const row of rows) {
        const key = normalizeClass(row.class);
        if (key === 'fighting') next.fighting += 1;
        else if (key === 'robbery') next.robbery += 1;
        else if (key === 'vandalism') next.vandalism += 1;
      }
      setIncidentCounts(next);
    };

    const fetchIncidents = async () => {
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
          .from('incidents')
          .select('*')
          .order('timestamp', { ascending: false })
          .limit(100);

        if (fetchError) throw fetchError;
        setIncidents(data || []);

        // Counters should come from all incident rows in Supabase, not only the recent list.
        const { data: allClasses, error: countErr } = await supabase
          .from('incidents')
          .select('class');
        if (countErr) throw countErr;
        computeCounts((allClasses as Array<{ class?: string }>) || []);

        // Subscribe to realtime new incidents
        subscription = supabase
          .channel('incidents-realtime')
          .on(
            'postgres_changes',
            { event: 'INSERT', schema: 'public', table: 'incidents' },
            (payload) => {
              const inserted = payload.new as Incident;
              setIncidents((prev) => [inserted, ...prev]);
              const key = normalizeClass((inserted as any).class);
              if (key === 'fighting' || key === 'robbery' || key === 'vandalism') {
                setIncidentCounts((prev) => ({ ...prev, [key]: prev[key as keyof typeof prev] + 1 }));
              }
            }
          )
          .subscribe();
      } catch (err) {
        console.error('Error fetching incidents:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch incidents');
      } finally {
        setLoading(false);
      }
    };

    fetchIncidents();

    return () => {
      subscription?.unsubscribe();
    };
  }, []);

  return {
    incidents,
    loading,
    error,
    incidentCounts: {
      fighting: incidentCounts.fighting,
      robbery: incidentCounts.robbery,
      vandalism: incidentCounts.vandalism,
    },
  };
}
