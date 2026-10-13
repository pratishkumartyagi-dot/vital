import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Resilient Health",
  description: "Camera-based vital signs monitoring for resilient, continuous care",
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