import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/rppg/:path*",
        destination: "http://localhost:8002/:path*",
      },
      {
        source: "/api/health",
        destination: "http://localhost:8002/health",
      },
    ];
  },
};

export default nextConfig;
