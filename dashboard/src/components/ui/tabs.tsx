import { clsx } from "clsx";
import type { PropsWithChildren, ReactNode } from "react";
import { useMemo } from "react";

export type TabItem = {
  id: string;
  label: string;
  content: ReactNode;
};

type TabsProps = PropsWithChildren<{
  items: TabItem[];
  active: string;
  onChange: (id: string) => void;
}>;

export function Tabs({ items, active, onChange }: TabsProps) {
  const activeIndex = useMemo(() => items.findIndex((i) => i.id === active), [items, active]);
  return (
    <div className="w-full">
      <div className="flex gap-2 rounded-xl bg-slate-800/60 p-1">
        {items.map((item) => (
          <button
            key={item.id}
            onClick={() => onChange(item.id)}
            className={clsx(
              "flex-1 rounded-lg px-4 py-2 text-sm font-semibold transition-all",
              active === item.id ? "bg-slate-900 text-white shadow-inner" : "text-slate-300"
            )}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className="mt-4">
        {activeIndex >= 0 ? items[activeIndex].content : <p className="text-slate-400">No tab</p>}
      </div>
    </div>
  );
}
