'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { createRoot, type Root } from 'react-dom/client';

interface Camera {
  id: string;
  name: string;
  location: string;
  latitude: number;
  longitude: number;
  status: 'online' | 'offline' | 'live' | 'alert' | string;
  video_url: string | null;
  stream_type?: 'websocket' | 'video' | string | null;
}

interface Incident {
  id: string;
  camera_id: string;
  class: 'Fighting' | 'Robbery' | 'Vandalism' | string;
  confidence: number;
  timestamp: string;
}

interface MapContainerProps {
  cameras: Camera[];
  incidents?: Incident[];
  onCameraSelect: (camera: Camera) => void;
  useTestVideo?: boolean;
  hideMarkers?: boolean;
}

function formatTime(ts: string | undefined | null) {
  if (!ts) return '—';
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleString(undefined, { hour: '2-digit', minute: '2-digit', month: 'short', day: '2-digit' });
}


/** Map marker preview: WebSocket JPEG (same as backend pipeline) or HTTP <video> when stream_type is video. */
function MarkerStreamPreview(props: {
  camera: Camera;
  /** http(s) URL for <video> when stream_type is not websocket */
  httpVideoSrc: string;
  /** When true (fullscreen modal open), disconnect marker WS to avoid duplicate streams */
  pauseStream: boolean;
}) {
  const { camera, httpVideoSrc, pauseStream } = props;
  const imgRef = useRef<HTMLImageElement | null>(null);
  const lastUrlRef = useRef<string | null>(null);
  const useWs = !camera.stream_type || camera.stream_type === 'websocket';
  const isOnline = camera.status === 'online' || camera.status === 'live' || camera.status === 'alert';

  useEffect(() => {
    if (!useWs || pauseStream || !isOnline) return;

    const base = process.env.NEXT_PUBLIC_BACKEND_WS_BASE || 'ws://localhost:8000';
    const ws = new WebSocket(`${base}/ws/camera/${camera.id}`);
    ws.binaryType = 'arraybuffer';

    ws.onmessage = (e) => {
      if (!imgRef.current) return;
      const blob = new Blob([e.data], { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      if (lastUrlRef.current) URL.revokeObjectURL(lastUrlRef.current);
      lastUrlRef.current = url;
      imgRef.current.src = url;
    };

    return () => {
      try {
        ws.close();
      } catch {}
      if (lastUrlRef.current) {
        URL.revokeObjectURL(lastUrlRef.current);
        lastUrlRef.current = null;
      }
    };
  }, [camera.id, useWs, pauseStream, isOnline]);

  if (!isOnline) {
    return (
      <div className="w-full h-full bg-black/70 flex items-center justify-center">
        <div className="text-xs font-semibold text-white/70">CAMERA OFFLINE</div>
      </div>
    );
  }

  if (useWs) {
    return (
      <img
        ref={imgRef}
        className="w-full h-full object-cover bg-black"
        alt=""
        decoding="async"
      />
    );
  }

  if (httpVideoSrc) {
    return (
      <video
        className="w-full h-full object-cover"
        src={httpVideoSrc}
        autoPlay
        muted
        loop
        playsInline
        preload="metadata"
      />
    );
  }

  return (
    <div className="w-full h-full bg-black/60 flex items-center justify-center">
      <div className="text-xs text-white/70">Preview unavailable</div>
    </div>
  );
}

function CameraMarkerCard(props: {
  camera: Camera;
  lastDetectionTs?: string;
  lastIncident?: Incident | null;
  totalIncidents: number;
  alertLabel?: string | null;
  isAlerting: boolean;
  httpVideoSrc: string;
  pauseMarkerStream: boolean;
  zoom: number;
  onClick: () => void;
  onHoverChange: (hovered: boolean) => void;
}) {
  const {
    camera,
    lastDetectionTs,
    lastIncident,
    totalIncidents,
    alertLabel,
    isAlerting,
    httpVideoSrc,
    pauseMarkerStream,
    zoom,
    onClick,
    onHoverChange,
  } = props;

  const dotOnly = zoom <= 10.25;
  const shrink = zoom > 10.25 && zoom <= 11.25;
  const isOnline = camera.status === 'online' || camera.status === 'live' || camera.status === 'alert';
  const isCameraAlert = camera.status === 'alert';

  return (
    <div
      className="sentinel-marker group"
      role="button"
      tabIndex={0}
      onMouseEnter={() => onHoverChange(true)}
      onMouseLeave={() => onHoverChange(false)}
      onClick={onClick}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') onClick();
      }}
      // Keep the marker element effectively 0x0 so Mapbox anchoring stays stable
      // even when the card grows/shrinks with zoom/hover.
      style={{ pointerEvents: 'auto', position: 'relative', width: 0, height: 0 }}
    >
      {!dotOnly && (
        <>
          {/* Card (always visible) */}
          <div
            className={[
              'relative overflow-hidden rounded-xl',
              'transition-[width,height,transform,box-shadow,border-color] duration-[350ms]',
              '[transition-timing-function:cubic-bezier(0.34,1.56,0.64,1)]',
              'w-[200px] h-[130px] group-hover:w-[340px] group-hover:h-[310px]',
              'cursor-pointer',
              isAlerting ? 'sentinel-alert-card' : '',
            ].join(' ')}
            style={{
              position: 'absolute',
              left: 0,
              bottom: 0,
              transform: shrink ? 'scale(0.72)' : 'scale(1)',
              transformOrigin: 'left bottom',
              background: 'linear-gradient(145deg, rgba(20,20,24,0.92) 0%, rgba(12,12,16,0.95) 100%)',
              border: isAlerting
                ? '1px solid rgba(255,51,51,0.6)'
                : '1px solid rgba(255,255,255,0.08)',
              boxShadow: isAlerting
                ? '0 0 0 1px rgba(255,51,51,0.9), 0 0 26px rgba(255,51,51,0.55), 0 8px 32px rgba(0,0,0,0.5)'
                : '0 8px 32px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.03) inset',
            }}
          >
            {/* Video / live JPEG (WebSocket) */}
            <div className="relative w-full h-full group-hover:h-[160px] transition-[height] duration-[350ms] [transition-timing-function:cubic-bezier(0.34,1.56,0.64,1)]">
              <MarkerStreamPreview
                camera={camera}
                httpVideoSrc={httpVideoSrc}
                pauseStream={pauseMarkerStream}
              />

              {/* Vignette overlay for cinematic look */}
              <div
                className="absolute inset-0 pointer-events-none"
                style={{
                  background: 'radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.5) 100%)',
                }}
              />

              {/* Red flash overlay on alert */}
              <div
                className="absolute inset-0 pointer-events-none opacity-0"
                style={
                  isAlerting
                    ? {
                        animation: 'sentinelFlash 10s ease-out 1',
                        background: 'rgba(255,0,0,0.2)',
                      }
                    : undefined
                }
              />

              {/* Top-left: status indicator */}
              <div className="absolute left-2 top-2 flex items-center gap-1.5">
                <span
                  className="w-2 h-2 rounded-full"
                  style={{
                    backgroundColor: isOnline ? '#34d399' : '#ef4444',
                    boxShadow: isOnline
                      ? '0 0 8px rgba(52,211,153,0.6)'
                      : '0 0 8px rgba(239,68,68,0.6)',
                  }}
                />
                <span className="text-[10px] font-semibold tracking-wider uppercase"
                  style={{ color: isOnline ? '#34d399' : '#ef4444' }}
                >
                  {isOnline ? (isCameraAlert ? 'ALERT' : 'LIVE') : 'OFF'}
                </span>
              </div>

              {/* Bottom gradient bar with camera name */}
              <div
                className="absolute bottom-0 left-0 right-0 px-2.5 py-1.5"
                style={{ background: 'linear-gradient(transparent, rgba(0,0,0,0.75))' }}
              >
                <span className="text-[11px] font-medium text-white/90 truncate block">{camera.name}</span>
              </div>

              {/* Alert badge top-center */}
              {isAlerting && alertLabel && (
                <div
                  className="absolute top-2 left-1/2 -translate-x-1/2 px-2.5 py-1 rounded-md text-[10px] font-bold text-white tracking-wide"
                  style={{
                    background: 'linear-gradient(135deg, rgba(220,38,38,0.9) 0%, rgba(185,28,28,0.9) 100%)',
                    border: '1px solid rgba(252,165,165,0.3)',
                    boxShadow: '0 2px 12px rgba(220,38,38,0.4)',
                  }}
                >
                  ⚠ {alertLabel.toUpperCase()}
                </div>
              )}
            </div>

            {/* Metadata section (only visible on hover) */}
            <div className="px-2.5 py-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
              {/* Location row */}
              <div className="text-[11px] text-white/40 truncate mb-2">{camera.location}</div>

              {/* Stats grid — Last, Incidents, Status only */}
              <div className="grid grid-cols-3 gap-1.5">
                <div className="rounded-md px-2 py-1.5" style={{ background: 'rgba(255,255,255,0.04)' }}>
                  <div className="text-[9px] uppercase tracking-wider text-white/25 mb-0.5">Last</div>
                  <div className="text-[11px] font-medium text-white/80 truncate">{formatTime(lastDetectionTs)}</div>
                </div>
                <div className="rounded-md px-2 py-1.5" style={{ background: 'rgba(255,255,255,0.04)' }}>
                  <div className="text-[9px] uppercase tracking-wider text-white/25 mb-0.5">Incidents</div>
                  <div className="text-[11px] font-medium text-white/80">{totalIncidents}</div>
                </div>
                <div className="rounded-md px-2 py-1.5" style={{ background: 'rgba(255,255,255,0.04)' }}>
                  <div className="text-[9px] uppercase tracking-wider text-white/25 mb-0.5">Status</div>
                  <div className="text-[11px] font-medium" style={{ color: isOnline ? '#34d399' : '#f87171' }}>
                    {isOnline ? (isCameraAlert ? 'Alert' : 'Online') : 'Offline'}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Connector line */}
          <div className="sentinel-connector" style={{ position: 'absolute', left: 0, bottom: 0 }} />
        </>
      )}

      {/* Crimson glowing dot */}
      <div
        className={isOnline ? 'sentinel-dot marker-pulse' : 'sentinel-dot'}
        aria-hidden="true"
        style={{ position: 'absolute', left: 0, bottom: 0 }}
      />
    </div>
  );
}

export default function MapContainer({
  cameras,
  incidents = [],
  onCameraSelect,
  useTestVideo = true,
  hideMarkers = false,
}: MapContainerProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const markersRef = useRef<Map<string, mapboxgl.Marker>>(new Map());
  const rootsRef = useRef<Map<string, Root>>(new Map());
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);
  const [zoom, setZoom] = useState(12);

  const derived = useMemo(() => {
    const lastByCamera = new Map<string, Incident>();
    const totalCounts = new Map<string, number>();
    for (const inc of incidents) {
      const prev = lastByCamera.get(inc.camera_id);
      if (!prev || new Date(inc.timestamp).getTime() > new Date(prev.timestamp).getTime()) {
        lastByCamera.set(inc.camera_id, inc);
      }
      totalCounts.set(inc.camera_id, (totalCounts.get(inc.camera_id) ?? 0) + 1);
    }
    return { lastByCamera, totalCounts };
  }, [incidents]);

  useEffect(() => {
    if (!mapContainer.current) return;

    mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_TOKEN!;

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [77.5946, 12.9716], // Default to India (Bengaluru)
      zoom: 12,
      attributionControl: false,
    });

    // Remove default controls
    map.current.addControl(new mapboxgl.NavigationControl(), 'bottom-right');
    setZoom(map.current.getZoom());

    let raf = 0;
    const onZoom = () => {
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => setZoom(map.current?.getZoom() ?? 12));
    };
    map.current.on('zoom', onZoom);

    return () => {
      markersRef.current.forEach((marker) => marker.remove());
      const roots = Array.from(rootsRef.current.values());
      rootsRef.current.clear();
      markersRef.current.clear();
      // Defer unmount to avoid React strict-mode concurrent render warnings.
      setTimeout(() => roots.forEach((r) => r.unmount()), 0);
      map.current?.off('zoom', onZoom);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  useEffect(() => {
    if (!map.current) return;

    const nextIds = new Set(cameras.map((c) => c.id));

    // Remove markers for deleted cameras
    for (const [id, marker] of markersRef.current.entries()) {
      if (!nextIds.has(id)) {
        marker.remove();
        markersRef.current.delete(id);
        const root = rootsRef.current.get(id);
        rootsRef.current.delete(id);
        if (root) root.unmount();
      }
    }

    // Add/update markers for current cameras
    cameras.forEach((camera) => {
      let marker = markersRef.current.get(camera.id);
      let root = rootsRef.current.get(camera.id);
      let markerElement: HTMLDivElement | null = null;

      if (!marker || !root) {
        markerElement = document.createElement('div');
        markerElement.style.pointerEvents = 'auto';
        markerElement.style.willChange = 'transform';

        marker = new mapboxgl.Marker({ element: markerElement, anchor: 'bottom' })
          .setLngLat([camera.longitude, camera.latitude])
          .addTo(map.current!);

        root = createRoot(markerElement);
        markersRef.current.set(camera.id, marker);
        rootsRef.current.set(camera.id, root);
      } else {
        marker.setLngLat([camera.longitude, camera.latitude]);
        markerElement = marker.getElement() as HTMLDivElement;
      }

      const last = derived.lastByCamera.get(camera.id);
      const lastTs = last?.timestamp;
      const totalIncidents = derived.totalCounts.get(camera.id) ?? 0;
      const now = Date.now();
      const isAlerting = !!lastTs && now - new Date(lastTs).getTime() <= 10_000;
      const alertLabel = isAlerting ? (last as any)?.class ?? null : null;

      // NOTE: Local file paths in `video_url` cannot be played by the browser.
      // For the marker card, only show a preview when the URL is actually fetchable (http/https).
      const url = camera.video_url || '';
      const isHttp = /^https?:\/\//i.test(url);
      const httpVideoSrc = useTestVideo && cameras[0]?.id === camera.id ? '/api/test-video' : isHttp ? url : '';

      root.render(
        <CameraMarkerCard
          camera={camera}
          lastDetectionTs={lastTs}
          lastIncident={last ?? null}
          totalIncidents={totalIncidents}
          isAlerting={isAlerting}
          alertLabel={alertLabel}
          httpVideoSrc={httpVideoSrc}
          pauseMarkerStream={hideMarkers}
          zoom={zoom}
          onHoverChange={(hovered) => {
            markerElement!.style.zIndex = hovered ? '999' : selectedCameraId === camera.id ? '998' : '1';
          }}
          onClick={() => {
            setSelectedCameraId(camera.id);
            markerElement!.style.zIndex = '998';
            onCameraSelect(camera);
          }}
        />
      );
    });
  }, [cameras, derived.lastByCamera, derived.totalCounts, hideMarkers, onCameraSelect, selectedCameraId, useTestVideo, zoom]);

  useEffect(() => {
    // When fullscreen modal is open, hard-hide marker DOM so it can't overlay the modal
    // (markers may have high z-index while hovered/selected).
    markersRef.current.forEach((marker) => {
      const el = marker.getElement() as HTMLElement | null;
      if (!el) return;
      el.style.display = hideMarkers ? 'none' : '';
    });
  }, [hideMarkers]);

  return (
    <div
      ref={mapContainer}
      className={['w-full h-full bg-background', hideMarkers ? 'sentinel-hide-markers' : ''].join(' ')}
    />
  );
}
