'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { createClient } from '@supabase/supabase-js';

export default function Home() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
        const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

        if (!supabaseUrl || !supabaseKey) {
          console.error('Missing Supabase environment variables');
          setIsLoading(false);
          return;
        }

        const supabase = createClient(supabaseUrl, supabaseKey);
        const {
          data: { user },
        } = await supabase.auth.getUser();

        if (user) {
          router.push('/dashboard');
        } else {
          router.push('/login');
        }
      } catch (error) {
        console.error('Auth check failed:', error);
        setIsLoading(false);
      }
    };

    checkAuth();
  }, [router]);

  return (
    <div className="w-full h-screen bg-background flex items-center justify-center">
      {isLoading ? (
        <p className="text-foreground">Redirecting...</p>
      ) : (
        <div className="text-center">
          <p className="text-foreground mb-4">Welcome to CCTV Dashboard</p>
          <p className="text-sm text-muted-foreground">Please configure your Supabase environment variables</p>
        </div>
      )}
    </div>
  );
}
