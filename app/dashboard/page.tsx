'use client';

import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { createClient } from '@supabase/supabase-js';
import dynamic from 'next/dynamic';
import TopBar from '@/components/TopBar';
import IncidentLog from '@/components/IncidentLog';
import { useCameras } from '@/hooks/useCameras';
import { useIncidents } from '@/hooks/useIncidents';

const MapContainer = dynamic(() => import('@/components/MapContainer'), {
  ssr: false,
  loading: () => <div className="w-full h-screen bg-background flex items-center justify-center"><p className="text-foreground">Loading map...</p></div>,
});

export default function DashboardPage() {
  const router = useRouter();
  const { cameras, loading: camerasLoading } = useCameras();
  const { incidents, incidentCounts } = useIncidents();
  const [selectedCamera, setSelectedCamera] = useState<any>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [user, setUser] = useState<any>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const lastObjectUrlRef = useRef<string | null>(null);
  const [wsStatus, setWsStatus] = useState<'idle' | 'connecting' | 'live' | 'error'>('idle');

  useEffect(() => {
    if (!selectedCamera || !isModalOpen) return;

    const isOnline =
      selectedCamera.status === 'online' || selectedCamera.status === 'live' || selectedCamera.status === 'alert';
    if (!isOnline) {
      setWsStatus('idle');
      return;
    }

    // Default to WS if stream_type is missing.
    if (selectedCamera.stream_type && selectedCamera.stream_type !== 'websocket') return;

    // Best-effort: default to localhost backend. Override with NEXT_PUBLIC_BACKEND_WS_BASE.
    const base = process.env.NEXT_PUBLIC_BACKEND_WS_BASE || 'ws://localhost:8000';
    const ws = new WebSocket(`${base}/ws/camera/${selectedCamera.id}`);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;
    setWsStatus('connecting');

    ws.onmessage = (e) => {
      if (!imgRef.current) return;
      const blob = new Blob([e.data], { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      if (lastObjectUrlRef.current) {
        URL.revokeObjectURL(lastObjectUrlRef.current);
      }
      lastObjectUrlRef.current = url;
      imgRef.current.src = url;
      setWsStatus('live');
    };
    ws.onerror = () => setWsStatus('error');
    ws.onclose = () => setWsStatus((s) => (s === 'live' ? 'error' : s));

    return () => {
      try {
        ws.close();
      } catch {}
      wsRef.current = null;
      if (lastObjectUrlRef.current) {
        URL.revokeObjectURL(lastObjectUrlRef.current);
        lastObjectUrlRef.current = null;
      }
      setWsStatus('idle');
    };
  }, [isModalOpen, selectedCamera]);

  useEffect(() => {
    // Check if user is authenticated
    const checkAuth = async () => {
      try {
        const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
        const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

        if (!supabaseUrl || !supabaseKey) {
          console.error('Missing Supabase environment variables');
          router.push('/login');
          return;
        }

        const supabase = createClient(supabaseUrl, supabaseKey);
        const {
          data: { user: currentUser },
        } = await supabase.auth.getUser();

        if (!currentUser) {
          router.push('/login');
          return;
        }

        setUser(currentUser);
      } catch (error) {
        console.error('Auth check failed:', error);
        router.push('/login');
      }
    };

    checkAuth();
  }, [router]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsModalOpen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  if (!user || camerasLoading) {
    return (
      <div className="w-full h-screen bg-background flex items-center justify-center">
        <p className="text-foreground">Loading dashboard...</p>
      </div>
    );
  }

  return (
    <div className="w-full h-screen bg-background overflow-hidden">
      {/* Map Container */}
      <MapContainer
        cameras={cameras}
        incidents={incidents}
        useTestVideo={false}
        hideMarkers={!!(selectedCamera && isModalOpen)}
        onCameraSelect={(cam) => {
          setSelectedCamera(cam);
          setIsModalOpen(true);
        }}
      />

      {/* Top Bar */}
      <TopBar cameraCount={cameras.length} />

      {/* Incident Log */}
      <IncidentLog incidents={incidents} counts={incidentCounts} />

      {/* Full-screen camera modal (click state) */}
      {selectedCamera && isModalOpen && (() => {
        const isOnline = selectedCamera.status === 'online' || selectedCamera.status === 'live' || selectedCamera.status === 'alert';
        const cameraIncidents = incidents.filter((i: any) => i.camera_id === selectedCamera.id);
        const latestIncident = cameraIncidents[0] ?? null;
        const statusLabel = selectedCamera.status === 'alert' ? 'ALERT' : isOnline ? 'LIVE' : 'OFFLINE';

        return (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) setIsModalOpen(false);
          }}
        >
          <div
            className="w-[92vw] max-w-[1400px] h-[88vh] bg-[#0d0d10]/95 backdrop-blur-2xl border border-white/[0.08] rounded-2xl overflow-hidden flex flex-col"
            style={{ boxShadow: '0 25px 80px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,255,255,0.04) inset' }}
            onMouseDown={(e) => e.stopPropagation()}
          >
            {/* Top bar */}
            <div className="h-14 px-5 flex items-center justify-between border-b border-white/[0.06] shrink-0">
              <div className="flex items-center gap-3 min-w-0">
                <div className="w-8 h-8 rounded-lg bg-white/[0.06] flex items-center justify-center">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white/70"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>
                </div>
                <div className="min-w-0">
                  <div className="text-sm font-semibold text-white tracking-wide truncate">{selectedCamera.name}</div>
                  <div className="text-[11px] text-white/40 truncate">{selectedCamera.location}</div>
                </div>
                <span
                  className={[
                    'ml-2 px-2.5 py-1 rounded-md text-[10px] font-bold tracking-wider uppercase',
                    isOnline
                      ? selectedCamera.status === 'alert'
                        ? 'bg-red-500/20 text-red-300 border border-red-400/25'
                        : 'bg-emerald-500/15 text-emerald-300 border border-emerald-400/25'
                      : 'bg-white/[0.06] text-white/40 border border-white/[0.08]',
                  ].join(' ')}
                >
                  {statusLabel}
                </span>
              </div>
              <button
                className="w-8 h-8 rounded-lg border border-white/[0.08] hover:bg-white/[0.06] text-white/60 hover:text-white text-sm flex items-center justify-center transition-colors"
                onClick={() => setIsModalOpen(false)}
              >
                ✕
              </button>
            </div>

            {/* Video container */}
            <div className="flex-1 min-h-0 p-3">
              <div className="w-full h-full rounded-xl overflow-hidden relative bg-black/80 border border-white/[0.04]">
                {/* Faint scanline overlay */}
                <div
                  className="absolute inset-0 pointer-events-none z-10 opacity-[0.03]"
                  style={{
                    backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.15) 2px, rgba(255,255,255,0.15) 4px)',
                  }}
                />

                {/* REC / LIVE badge */}
                <div className="absolute top-3 left-3 z-20 flex items-center gap-2">
                  {isOnline && (
                    <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-red-600/80 backdrop-blur-sm">
                      <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                      <span className="text-[10px] font-bold text-white tracking-widest">REC</span>
                    </div>
                  )}
                </div>

                {/* Timestamp overlay */}
                <div className="absolute bottom-3 left-3 z-20">
                  <span className="text-[11px] font-mono text-white/50 bg-black/50 px-2 py-1 rounded">
                    {new Date().toLocaleString()}
                  </span>
                </div>

                {/* Camera ID overlay */}
                <div className="absolute bottom-3 right-3 z-20">
                  <span className="text-[10px] font-mono text-white/30 bg-black/40 px-2 py-1 rounded uppercase tracking-wider">
                    CAM {selectedCamera.id.slice(0, 8)}
                  </span>
                </div>

                {/* Actual video/stream */}
                {!selectedCamera.stream_type || selectedCamera.stream_type === 'websocket' ? (
                  <div className="relative w-full h-full">
                    <img ref={imgRef} className="w-full h-full object-contain" alt="" />
                    {wsStatus !== 'live' && (
                      <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
                        <div className="w-12 h-12 rounded-full bg-white/[0.06] flex items-center justify-center">
                          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-white/40"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>
                        </div>
                        <span className="text-white/40 text-sm font-medium">
                          {!isOnline
                            ? 'Camera offline'
                            : wsStatus === 'connecting'
                              ? 'Connecting to live feed…'
                              : wsStatus === 'error'
                                ? 'No frames received'
                                : 'Awaiting stream…'}
                        </span>
                      </div>
                    )}
                  </div>
                ) : (
                  <video
                    className="w-full h-full object-contain"
                    src={selectedCamera.video_url || '/api/test-video'}
                    autoPlay
                    muted
                    loop
                    playsInline
                    controls
                  />
                )}
              </div>
            </div>

            {/* Bottom info — all data from DB */}
            <div className="shrink-0 px-4 pb-4">
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2">
                {/* Camera Location (from cameras table) */}
                <div className="rounded-lg bg-white/[0.03] border border-white/[0.06] px-3 py-2.5">
                  <div className="text-[10px] uppercase tracking-wider text-white/30 mb-1">Location</div>
                  <div className="text-sm font-medium text-white/90 truncate">{selectedCamera.location || '—'}</div>
                </div>

                {/* Camera Status (from cameras table) */}
                <div className="rounded-lg bg-white/[0.03] border border-white/[0.06] px-3 py-2.5">
                  <div className="text-[10px] uppercase tracking-wider text-white/30 mb-1">Status</div>
                  <div className="flex items-center gap-1.5">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: isOnline ? '#34d399' : '#f87171' }}
                    />
                    <span className={`text-sm font-medium ${isOnline ? 'text-emerald-300' : 'text-red-300'}`}>
                      {statusLabel}
                    </span>
                  </div>
                </div>

                {/* Last Detection (from incidents table) */}
                <div className="rounded-lg bg-white/[0.03] border border-white/[0.06] px-3 py-2.5">
                  <div className="text-[10px] uppercase tracking-wider text-white/30 mb-1">Last Detection</div>
                  <div className="text-sm font-medium text-white/90">
                    {latestIncident?.timestamp
                      ? new Date(latestIncident.timestamp).toLocaleString(undefined, { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' })
                      : '—'}
                  </div>
                </div>

                {/* Incident Count (from incidents table) */}
                <div className="rounded-lg bg-white/[0.03] border border-white/[0.06] px-3 py-2.5">
                  <div className="text-[10px] uppercase tracking-wider text-white/30 mb-1">Incidents</div>
                  <div className="text-sm font-medium text-white/90">{cameraIncidents.length}</div>
                </div>

                {/* Latest Class + Confidence (from incidents table) */}
                <div className="rounded-lg bg-white/[0.03] border border-white/[0.06] px-3 py-2.5">
                  <div className="text-[10px] uppercase tracking-wider text-white/30 mb-1">Latest Detection</div>
                  <div className="text-sm font-medium text-white/90">
                    {latestIncident?.class
                      ? `${String(latestIncident.class).toUpperCase()} · ${Math.round((latestIncident.confidence ?? 0) * 100)}%`
                      : '—'}
                  </div>
                </div>
              </div>

              {/* Secondary info row */}
              <div className="mt-2 flex items-center gap-4 text-[10px] text-white/25">
                <span>Camera ID: {selectedCamera.id.slice(0, 12)}…</span>
                {selectedCamera.last_seen_at && (
                  <span>Last seen: {new Date(selectedCamera.last_seen_at).toLocaleString(undefined, { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' })}</span>
                )}
                <span>Stream: {selectedCamera.stream_type || 'websocket'}</span>
                <span className="ml-auto">Press Esc to close</span>
              </div>
            </div>
          </div>
        </div>
        );
      })()}
    </div>
  );
}
