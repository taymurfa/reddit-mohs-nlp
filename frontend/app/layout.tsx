import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Mohs Reddit LDA",
  description: "End-to-end Reddit NLP analysis for Mohs surgery discussions",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
