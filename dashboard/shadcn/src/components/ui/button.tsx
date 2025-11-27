import { clsx } from "clsx";
import type { ButtonHTMLAttributes, PropsWithChildren } from "react";

type ButtonProps = PropsWithChildren<
  ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: "solid" | "outline" | "ghost";
  }
>;

export function Button({ children, className, variant = "solid", ...props }: ButtonProps) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold transition-all";
  const variants: Record<typeof variant, string> = {
    solid: "bg-accent text-white hover:brightness-110 shadow-lg shadow-accent/30",
    outline:
      "border border-slate-700 text-slate-100 hover:border-accent hover:text-white hover:shadow-accent/20",
    ghost: "text-slate-300 hover:text-white hover:bg-white/5"
  };
  return (
    <button className={clsx(base, variants[variant], className)} {...props}>
      {children}
    </button>
  );
}
