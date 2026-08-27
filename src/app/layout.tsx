import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vital Monitoring App",
  description: "Research-grade PPG vital signs monitor",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className="antialiased"
      >
        {children}
      </body>
    </html>
  );
}