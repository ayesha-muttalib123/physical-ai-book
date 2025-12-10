// src/theme/Root.tsx
import React from 'react';
import { AuthProvider } from '../contexts/AuthContext';

// Default implementation, that you can customize
const Root = ({ children }: { children: React.ReactNode }) => {
  return <AuthProvider>{children}</AuthProvider>;
};

export default Root;