// Minimal structured console logging plus money formatting helpers.

import type { Minor, Money } from "../types.ts";

const ts = () => new Date().toISOString();

export const log = {
  info: (msg: string, extra?: unknown) =>
    console.log(`[${ts()}] ${msg}`, extra ?? ""),
  warn: (msg: string, extra?: unknown) =>
    console.warn(`[${ts()}] WARN ${msg}`, extra ?? ""),
  error: (msg: string, extra?: unknown) =>
    console.error(`[${ts()}] ERROR ${msg}`, extra ?? ""),
};

/** Format minor units as a human-readable amount. Assumes 2 decimal places,
 * which holds for USD/EUR/GBP and most currencies used here. */
export function formatMinor(amountMinor: Minor, currency: string): string {
  const major = (amountMinor / 100).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  return `${major} ${currency}`;
}

export function formatMoney(m: Money): string {
  return formatMinor(m.amountMinor, m.currency);
}
