'use client';

import type { Incident } from '@/hooks/useIncidents';
import { StackedActivityCards } from '@/components/ui/stacked-activity-cards';
import { AlertTriangle, Skull, Hammer } from 'lucide-react';

interface IncidentLogProps {
  incidents: Incident[];
  counts: {
    fighting: number;
    robbery: number;
    vandalism: number;
  };
}

const counterConfig = {
  fighting: {
    icon: AlertTriangle,
    bgColor: 'bg-red-500/10',
    borderColor: 'border-red-500/50',
    textColor: 'text-red-500',
  },
  robbery: {
    icon: Skull,
    bgColor: 'bg-orange-500/10',
    borderColor: 'border-orange-500/50',
    textColor: 'text-orange-500',
  },
  vandalism: {
    icon: Hammer,
    bgColor: 'bg-yellow-500/10',
    borderColor: 'border-yellow-500/50',
    textColor: 'text-yellow-500',
  },
};

export default function IncidentLog({ incidents, counts }: IncidentLogProps) {
  return (
    <div className="fixed bottom-6 right-6 z-20 w-96 flex flex-col pointer-events-auto">
      {/* Header with Counters */}
      <div className="backdrop-blur-md bg-card/30 border border-border/50 rounded-t-lg p-4 border-b-0">
        <h2 className="text-sm font-bold text-foreground mb-3">Recent Incidents</h2>
        <div className="grid grid-cols-3 gap-2">
          {Object.entries(counts).map(([type, count]) => {
            const config = counterConfig[type as keyof typeof counterConfig];
            const Icon = config.icon;
            return (
              <div key={type} className={`${config.bgColor} border ${config.borderColor} rounded-lg p-2 text-center`}>
                <Icon className={`w-4 h-4 ${config.textColor} mx-auto mb-1`} />
                <p className={`text-xs font-semibold ${config.textColor}`}>{count}</p>
                <p className="text-xs text-muted-foreground capitalize">{type}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Stacked Incident Cards */}
      <div className="backdrop-blur-md bg-card/30 border border-border/50 border-t-0 rounded-b-lg px-3 pb-3 pt-2">
        <StackedActivityCards incidents={incidents} />
      </div>
    </div>
  );
}

