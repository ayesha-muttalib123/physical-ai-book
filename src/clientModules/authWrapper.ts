// src/clientModules/authWrapper.ts
import React from 'react';
import { AuthProvider } from '../contexts/AuthContext';

export const AuthWrapper = ({ children }: { children: React.ReactNode }) => {
  return <AuthProvider>{children}</AuthProvider>;
};