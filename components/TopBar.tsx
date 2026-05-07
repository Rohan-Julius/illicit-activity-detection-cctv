'use client';

import { useRouter } from 'next/navigation';
import { createClient } from '@supabase/supabase-js';
import { LogOut, Camera } from 'lucide-react';

interface TopBarProps {
  cameraCount: number;
}

export default function TopBar({ cameraCount }: TopBarProps) {
  const router = useRouter();

  const handleLogout = async () => {
    try {
      const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
      const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

      if (!supabaseUrl || !supabaseKey) {
        router.push('/login');
        return;
      }

      const supabase = createClient(supabaseUrl, supabaseKey);
      await supabase.auth.signOut();
      router.push('/login');
    } catch (error) {
      console.error('Logout failed:', error);
      router.push('/login');
    }
  };

  return (
    <div className="fixed top-6 left-6 right-6 z-10 flex items-center justify-between pointer-events-none">
      {/* Left: Title */}
      <div className="flex items-center gap-3 pointer-events-auto">
        <div>
          <h1 className="text-lg font-bold text-foreground">CCTV Dashboard</h1>
          <p className="text-xs text-muted-foreground">Live Monitoring</p>
        </div>
      </div>

      {/* Right: Camera Count and Logout */}
      <div className="flex items-center gap-4">
        {/* Camera Count Badge */}
        <div className="backdrop-blur-md bg-card/30 border border-border/50 rounded-lg px-4 py-2 flex items-center gap-2 pointer-events-auto">
          <div className="flex items-center justify-center w-6 h-6 bg-secondary rounded-full">
            <Camera className="w-4 h-4 text-secondary-foreground" />
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Active Cameras</p>
            <p className="text-lg font-bold text-foreground">{cameraCount}</p>
          </div>
        </div>

        {/* Logout Button */}
        <button
          onClick={handleLogout}
          className="backdrop-blur-md bg-card/30 border border-border/50 hover:bg-destructive/10 hover:border-destructive/50 rounded-lg px-4 py-2 flex items-center gap-2 transition-all pointer-events-auto"
        >
          <LogOut className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">Logout</span>
        </button>
      </div>
    </div>
  );
}
