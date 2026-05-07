import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { createClient } from "@supabase/supabase-js";

function readDotEnvLocal() {
  const p = path.join(process.cwd(), ".env.local");
  if (!fs.existsSync(p)) return {};
  const raw = fs.readFileSync(p, "utf8");
  const out = {};
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const idx = trimmed.indexOf("=");
    if (idx === -1) continue;
    const key = trimmed.slice(0, idx).trim();
    const val = trimmed.slice(idx + 1).trim();
    out[key] = val;
  }
  return out;
}

const env = { ...readDotEnvLocal(), ...process.env };
const url = env.NEXT_PUBLIC_SUPABASE_URL;
const anonKey = env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
const serviceKey = env.SUPABASE_SERVICE_ROLE_KEY;

if (!url || (!anonKey && !serviceKey)) {
  console.error(
    "Missing env. Need NEXT_PUBLIC_SUPABASE_URL and either NEXT_PUBLIC_SUPABASE_ANON_KEY (signup) or SUPABASE_SERVICE_ROLE_KEY (admin create)."
  );
  process.exit(2);
}

const key = serviceKey || anonKey;
const supabase = createClient(url, key, {
  auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false },
});

const rand = crypto.randomBytes(6).toString("hex");
const email = `dummy_${rand}@gmail.com`;
const password = crypto.randomBytes(12).toString("base64url");

if (serviceKey) {
  const { data, error } = await supabase.auth.admin.createUser({
    email,
    password,
    email_confirm: true,
  });
  if (error) {
    console.error(JSON.stringify({ ok: false, mode: "admin", message: error.message, status: error.status }, null, 2));
    process.exit(1);
  }
  console.log(JSON.stringify({ ok: true, mode: "admin", email, password, userId: data.user?.id ?? null }, null, 2));
  process.exit(0);
}

// Fallback: signup (subject to Supabase auth rate limits + email confirmation settings)
const { error: signUpError } = await supabase.auth.signUp({ email, password });
if (signUpError) {
  console.error(
    JSON.stringify(
      { ok: false, mode: "signup", message: signUpError.message, status: signUpError.status, email },
      null,
      2
    )
  );
  process.exit(1);
}

const { data: loginData, error: loginError } = await supabase.auth.signInWithPassword({ email, password });
if (loginError) {
  console.log(JSON.stringify({ ok: true, mode: "signup", email, password, loginOk: false, loginMessage: loginError.message }, null, 2));
  process.exit(0);
}

console.log(
  JSON.stringify(
    { ok: true, mode: "signup", email, password, loginOk: true, userId: loginData.user?.id ?? null },
    null,
    2
  )
);

