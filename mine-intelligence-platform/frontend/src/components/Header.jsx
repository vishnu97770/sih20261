import { useEffect, useRef, useState } from "react";
import { Icon } from "./Icon";
import { useAuth } from "../context/AuthContext";
import { api } from "../api/client";

function timeAgo(isoString) {
  const diffMs = Date.now() - new Date(isoString).getTime();
  const minutes = Math.round(diffMs / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return `${days}d ago`;
}

export function Header({ title, description, breadcrumb }) {
  const { user } = useAuth();
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [items, setItems] = useState([]);
  const panelRef = useRef(null);

  const loadNotifications = async () => {
    try {
      const res = await api.notifications();
      setItems(res.notifications || []);
    } catch {
      // Notifications are non-critical - fail silently.
    }
  };

  useEffect(() => {
    loadNotifications();
    const interval = setInterval(loadNotifications, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!notificationsOpen) return;
    const handleClickOutside = (event) => {
      if (panelRef.current && !panelRef.current.contains(event.target)) {
        setNotificationsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [notificationsOpen]);

  const unreadCount = items.filter((item) => !item.read).length;

  const toggleNotifications = async () => {
    const next = !notificationsOpen;
    setNotificationsOpen(next);
    if (next && unreadCount > 0) {
      try {
        await api.markNotificationsRead();
        setItems((prev) => prev.map((item) => ({ ...item, read: true })));
      } catch {
        // Not critical if marking-as-read fails.
      }
    }
  };

  return (
    <header className="sticky top-0 z-20 flex h-16 items-center justify-between border-b border-stone-200 bg-[#fffaf1]/90 px-4 backdrop-blur sm:px-6">
      <div className="min-w-0">
        <div className="flex items-center gap-2 text-xs text-stone-400">
          {breadcrumb?.map((crumb, i) => (
            <span key={i} className="flex items-center gap-1.5">
              {i > 0 && <span>/</span>}
              <span>{crumb}</span>
            </span>
          ))}
        </div>
        <h1 className="truncate text-lg font-semibold text-stone-900">{title}</h1>
        {description && (
          <p className="truncate text-xs text-stone-500">{description}</p>
        )}
      </div>
      <div className="flex shrink-0 items-center gap-3 pl-3">
        <div className="relative" ref={panelRef}>
          <button
            type="button"
            onClick={toggleNotifications}
            className="relative flex h-10 w-10 items-center justify-center rounded-full border border-stone-200 text-stone-500 transition-colors hover:border-amber-300 hover:bg-amber-50 hover:text-amber-700"
            aria-label="Notifications"
            aria-expanded={notificationsOpen}
          >
            <Icon name="bell" className="h-5 w-5" />
            {unreadCount > 0 && (
              <span className="absolute right-2 top-2 h-2 w-2 rounded-full bg-red-500" />
            )}
          </button>
          {notificationsOpen && (
            <div className="absolute right-0 top-12 max-h-96 w-80 overflow-y-auto rounded-2xl border border-stone-200 bg-white p-4 text-sm shadow-lg">
              <div className="font-semibold text-stone-800">Notifications</div>
              {items.length ? (
                <div className="mt-3 space-y-2">
                  {items.map((item) => (
                    <div key={item.id} className="rounded-xl border border-stone-100 bg-stone-50 px-3 py-2">
                      <p className="text-stone-700">{item.message}</p>
                      <p className="mt-1 text-xs text-stone-400">{timeAgo(item.created_at)}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="mt-2 text-stone-500">You're all caught up - no new notifications right now.</p>
              )}
            </div>
          )}
        </div>
        <div className="hidden h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-amber-500 to-amber-700 text-sm font-semibold text-white sm:flex">
          {user?.name?.[0] ?? "U"}
        </div>
      </div>
    </header>
  );
}
