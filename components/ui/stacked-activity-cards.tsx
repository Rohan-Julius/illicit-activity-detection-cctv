'use client';

import React, { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import { AlertTriangle, Skull, Hammer, ChevronDown, ChevronUp } from "lucide-react";
import type { Incident } from "@/hooks/useIncidents";

/* ── colour / icon mapping (mirrors IncidentLog.tsx) ────────────────── */
const eventTypeConfig: Record<
  string,
  {
    label: string;
    color: string;
    icon: React.ElementType;
    bgColor: string;
    borderColor: string;
    textColor: string;
  }
> = {
  fighting: {
    label: "Fighting",
    color: "#ff3333",
    icon: AlertTriangle,
    bgColor: "bg-red-500/10",
    borderColor: "border-red-500/50",
    textColor: "text-red-500",
  },
  robbery: {
    label: "Robbery",
    color: "#ff8c00",
    icon: Skull,
    bgColor: "bg-orange-500/10",
    borderColor: "border-orange-500/50",
    textColor: "text-orange-500",
  },
  vandalism: {
    label: "Vandalism",
    color: "#ffd700",
    icon: Hammer,
    bgColor: "bg-yellow-500/10",
    borderColor: "border-yellow-500/50",
    textColor: "text-yellow-500",
  },
};

/* ── injected stylesheet (replaces styled-jsx) ─────────────────────── */
const STACKED_CARDS_CSS = `
.stacked-cards-root {
  position: relative;
  width: 100%;
}

.stacked-cards-stack {
  position: relative;
  width: 100%;
  min-height: 70px;
  transition: min-height 0.35s ease;
}

.stacked-cards-stack.is-expanded {
  min-height: unset;
}

.stacked-card {
  position: relative;
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 14px;
  box-sizing: border-box;
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
  transition: all 0.35s cubic-bezier(0.68, -0.25, 0.27, 1.25);
  background: rgba(20, 20, 22, 0.75);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45);
}

.stacked-card.expanded {
  transform: none !important;
  position: relative;
}

.stacked-card-pos-0 {
  position: relative;
  z-index: 3;
  transform: scale(1);
}
.stacked-card-pos-1 {
  position: absolute;
  top: 8px;
  left: 6px;
  right: 6px;
  width: auto;
  z-index: 2;
  transform: scale(0.96);
  opacity: 0.7;
}
.stacked-card-pos-2 {
  position: absolute;
  top: 14px;
  left: 12px;
  right: 12px;
  width: auto;
  z-index: 1;
  transform: scale(0.92);
  opacity: 0.4;
}

.stacked-cards-btn {
  position: relative;
  padding: 6px 18px;
  background: rgba(20, 20, 22, 0.75);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border-radius: 14px;
  border: 1px solid var(--border);
  color: var(--foreground);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  display: inline-flex;
  align-items: center;
}

.stacked-cards-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(255, 51, 51, 0.2);
  border-color: rgba(255, 51, 51, 0.4);
}
`;

/* ── helper ──────────────────────────────────────────────────────────── */
function getTimeAgo(timestamp: string): string {
  const now = new Date();
  const time = new Date(timestamp);
  const diffMs = now.getTime() - time.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);

  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  return time.toLocaleDateString();
}

/* ── component ───────────────────────────────────────────────────────── */
export interface StackedActivityCardsProps {
  /** Latest incidents from useIncidents(). Shows up to 10 when expanded. */
  incidents: Incident[];
}

export const StackedActivityCards: React.FC<StackedActivityCardsProps> = ({
  incidents,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  /* Inject the stylesheet once on mount */
  useEffect(() => {
    const STYLE_ID = "stacked-activity-cards-styles";
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = STACKED_CARDS_CSS;
    document.head.appendChild(style);
    return () => {
      const el = document.getElementById(STYLE_ID);
      if (el) el.remove();
    };
  }, []);

  const allCards = incidents.slice(0, 10);

  if (allCards.length === 0) {
    return (
      <div className="py-4 text-center">
        <p className="text-sm text-muted-foreground">No incidents detected</p>
      </div>
    );
  }

  // When collapsed, only show first 3 (visually stacked)
  const visibleCards = isExpanded ? allCards : allCards.slice(0, 3);

  return (
    <div className="stacked-cards-root">
      <div className={cn("stacked-cards-stack", isExpanded && "is-expanded")}>
        {visibleCards.map((incident, index) => {
          const key = String(incident.class ?? "").trim().toLowerCase();
          const config = eventTypeConfig[key] ?? eventTypeConfig.fighting;
          const Icon = config.icon;

          return (
            <div
              key={incident.id}
              className={cn(
                "stacked-card",
                !isExpanded && `stacked-card-pos-${index}`,
                isExpanded && "expanded"
              )}
              style={
                !isExpanded
                  ? {
                      zIndex: visibleCards.length - index,
                    }
                  : undefined
              }
            >
              <div className="flex items-center gap-3 min-w-0 flex-1">
                {/* Icon badge */}
                <span
                  className={cn(
                    "flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center border",
                    config.bgColor,
                    config.borderColor
                  )}
                >
                  <Icon className={cn("w-4 h-4", config.textColor)} />
                </span>

                {/* Text */}
                <div className="min-w-0 flex-1">
                  <h3 className={cn("text-sm font-semibold leading-tight", config.textColor)}>
                    {config.label}
                  </h3>
                  <span className="text-xs text-muted-foreground truncate block">
                    {incident.camera_name || `Camera ${incident.camera_id?.slice(0, 8)}`}
                  </span>
                </div>
              </div>

              {/* Right side — date + confidence */}
              <div className="flex flex-col items-end flex-shrink-0 ml-2">
                <span className="text-xs text-muted-foreground whitespace-nowrap">
                  {getTimeAgo(incident.timestamp)}
                </span>
                <span className="text-[11px] text-muted-foreground/70 whitespace-nowrap">
                  {Math.round(incident.confidence * 100)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Toggle button — only if there are items */}
      {allCards.length > 1 && (
        <div className="flex items-center justify-center mt-1 mb-1">
          <button
            className="stacked-cards-btn"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? (
              <>
                Collapse <ChevronUp className="w-3.5 h-3.5 ml-1.5 inline-block" />
              </>
            ) : (
              <>
                Show All ({allCards.length}) <ChevronDown className="w-3.5 h-3.5 ml-1.5 inline-block" />
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
};

export default StackedActivityCards;

