import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Polymas - Diabetes Risk Prediction Dashboard",
  description: "Multi-label ensemble ML predictions for diabetes risk classification",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen">
          <header className="border-b-2 border-border bg-surface-card p-4 shadow-brutal-sm">
            <div className="container mx-auto flex items-center justify-between">
              <h1 className="font-mono text-2xl font-bold tracking-tighter">
                POLY<span className="text-surface-accent">MAS</span>
              </h1>
              <nav className="flex gap-4">
                <a href="/" className="btn-brutal-outline text-xs">
                  Dashboard
                </a>
                <a href="/patients" className="btn-brutal-outline text-xs">
                  Patients
                </a>
                <a href="/clusters" className="btn-brutal-outline text-xs">
                  Clusters
                </a>
                <a href="/explainability" className="btn-brutal-outline text-xs">
                  Explanations
                </a>
              </nav>
            </div>
          </header>
          <main className="container mx-auto p-6">{children}</main>
        </div>
      </body>
    </html>
  );
}
