import { clsx } from "clsx";
import type { PropsWithChildren, ReactNode } from "react";

type CardProps = PropsWithChildren<{
  title?: string;
  action?: ReactNode;
  className?: string;
}>;

export function Card({ title, action, className, children }: CardProps) {
  return (
    <div className={clsx("card", className)}>
      {(title || action) && (
        <div className="mb-4 flex items-center justify-between">
          {title ? <p className="card-title">{title}</p> : <div />}
          {action}
        </div>
      )}
      {children}
    </div>
  );
}
