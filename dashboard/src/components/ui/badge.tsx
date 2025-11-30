import { clsx } from "clsx";
import type { PropsWithChildren } from "react";

type BadgeProps = PropsWithChildren<{
  tone?: "success" | "warning" | "info";
  className?: string;
}>;

export function Badge({ children, tone = "info", className }: BadgeProps) {
  const tones: Record<typeof tone, string> = {
    success: "bg-success/10 text-success border border-success/30",
    warning: "bg-warning/10 text-warning border border-warning/30",
    info: "bg-accent/10 text-accent border border-accent/30"
  };
  return <span className={clsx("badge", tones[tone], className)}>{children}</span>;
}
