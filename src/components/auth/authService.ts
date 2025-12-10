// src/components/auth/authService.ts
import axios from 'axios';

// Define types
export interface User {
  id: number;
  email: string;
  name: string;
  is_software_focused: boolean | null;
  learning_path: string | null;
}

export interface RegistrationData {
  email: string;
  name: string;
  auth_method: string;
  is_software_focused: boolean | null;
  learning_path: string | null;
}

export interface AuthResponse {
  user_id: number;
  email: string;
  name: string;
}

class AuthService {
  private baseUrl: string = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  async register(userData: RegistrationData): Promise<AuthResponse> {
    try {
      const response = await axios.post(`${this.baseUrl}/register`, userData);
      return response.data;
    } catch (error) {
      console.error('Registration error:', error);
      throw error;
    }
  }

  async login(email: string, password: string): Promise<AuthResponse> {
    // In a real implementation, you would send credentials to your backend
    // For this demo, we'll use our existing register endpoint as a mock login
    try {
      const response = await axios.post(`${this.baseUrl}/register`, {
        email,
        name: email.split('@')[0], // Extract name from email as placeholder
        auth_method: 'email',
        is_software_focused: null,
        learning_path: null
      });
      return response.data;
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  }

  async getUserPreferences(userId: number) {
    try {
      const response = await axios.get(`${this.baseUrl}/users/${userId}/preferences`);
      return response.data;
    } catch (error) {
      console.error('Get user preferences error:', error);
      throw error;
    }
  }

  async updateUserPreferences(userId: number, preferences: any) {
    try {
      const response = await axios.put(`${this.baseUrl}/users/${userId}/preferences`, preferences);
      return response.data;
    } catch (error) {
      console.error('Update user preferences error:', error);
      throw error;
    }
  }
}

export default new AuthService();