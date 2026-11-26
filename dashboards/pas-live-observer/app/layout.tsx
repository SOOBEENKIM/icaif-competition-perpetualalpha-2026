import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SFINX · PAS Live Observer",
  description: "PAS 대회 지갑별 체결·롱숏·수익률 실시간 관측판",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body className="antialiased">{children}</body>
    </html>
  );
}
